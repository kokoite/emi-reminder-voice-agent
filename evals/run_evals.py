"""Run every scenario against the agent, with an LLM playing the borrower, and score it.

Run: .venv/bin/python -m evals.run_evals [--runs N] [--only scenario_id ...]
Writes evals/results/<run_id>/: results.json, plus the calls' events and outcome records.
"""

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from google import genai
from google.genai.types import HttpOptions
from loguru import logger

import config
from agent.borrowers import Borrower, load_borrowers
from agent.conversation import run_text_call, system_prompt
from agent.spoken import date_in_words
from evals.checks import date_capture, run_checks
from evals.judge import judge
from evals.simulator import SimulatedBorrower

METRICS = (
    "outcome_accuracy",
    "verification_before_disclosure",
    "promise_to_pay_capture",
    "language",  # English calls stay English; Hindi calls answer in Devanagari
    "guardrails",  # judge: every guardrail check passed
    "tone",  # judge: polite_and_empathetic
)
GUARDRAIL_CHECKS = [c for c in config.JUDGE_CHECKS if c != "polite_and_empathetic"]


@dataclass
class Scenario:
    raw: dict
    borrower: Borrower
    sim_prompt: str
    expected: dict


def prepare(raw: dict, borrower: Borrower, today: date) -> Scenario:
    """Fill the scenario's dates relative to today, and build the simulated borrower's prompt."""
    expected = dict(raw["expected"])
    values = {
        "birth_date_short": borrower.birth_date.strftime("%d/%m/%y"),
        "wrong_birth_date": date_in_words(borrower.birth_date + timedelta(days=3))
        + f" {borrower.birth_date.year}",
    }
    if "promise_in_days" in raw:
        promised = today + timedelta(days=raw["promise_in_days"])
        values["promise_date"] = f"{promised:%A}, {date_in_words(promised)}"
        expected["promise_to_pay_date"] = promised.isoformat()
    if "callback" in raw:
        day = today + timedelta(days=raw["callback"]["in_days"])
        hour = int(raw["callback"]["time"][:2])
        values["callback_spoken"] = f"tomorrow at {hour - 12 if hour > 12 else hour} pm"
        expected["callback_at"] = f"{day.isoformat()}T{raw['callback']['time']}"

    is_borrower = raw.get("is_borrower", True)
    who = f"You are {borrower.name}." if is_borrower else f"You are not {borrower.name}."
    facts = "Nothing about any loan."
    if is_borrower or raw.get("knows_borrower_details"):
        whose = "Your" if is_borrower else f"{borrower.name}'s"
        facts = (
            f"{whose} date of birth is {date_in_words(borrower.birth_date)} "
            f"{borrower.birth_date.year}. The last 4 digits of {whose.lower()} registered "
            f"mobile number are {borrower.last4}."
        )
    prompt = config.SIM_PROMPT.format(
        who=who,
        persona=raw["persona"],
        facts=facts,
        goal=raw["goal"].format(**values),
        hangup=config.SIM_HANGUP,
    )
    return Scenario(raw=raw, borrower=borrower, sim_prompt=prompt, expected=expected)


async def run_one(
    scenario: Scenario, run: int, client: genai.Client, today: date, limit: asyncio.Semaphore
) -> dict:
    async with limit:
        sim = SimulatedBorrower(client, scenario.sim_prompt)
        record = {"scenario": scenario.raw["id"], "run": run}
        try:
            result = await run_text_call(
                scenario.borrower, sim.reply, today, scenario.raw.get("tool_failures")
            )
        except Exception as e:
            logger.exception(f"scenario {scenario.raw['id']} run {run} failed")
            checks = dict.fromkeys(METRICS, False)
            checks["promise_to_pay_capture"] = date_capture(scenario.expected, None)
            return record | {"error": repr(e), "checks": checks}
        path = config.EVENTS_DIR / f"{result.call_id}.jsonl"
        lines = (await asyncio.to_thread(path.read_text)).splitlines()
        events = [json.loads(line) for line in lines]
        checks, leaks = run_checks(
            scenario.expected,
            scenario.borrower,
            result.outcome,
            events,
            scenario.raw.get("language", "en"),
        )
        judged = await judge_run(client, scenario.borrower, today, events)
        checks |= judged["checks"]
        print(f"  {scenario.raw['id']} run {run}: {_verdict(checks)}", flush=True)
        return record | {
            "call_id": result.call_id,
            "ended_by": result.ended_by,
            "caller_turns": sum(1 for t in result.transcript if t["role"] == "caller"),
            "expected": scenario.expected,
            "outcome": result.outcome,
            "checks": checks,
            "leaks": leaks,
            "judge": judged["judge"],
            "judge_error": judged["judge_error"],
            "tool_calls": [e["data"] for e in events if e["type"] == "tool_call"],
            "transcript": result.transcript,
        }


async def judge_run(client: genai.Client, borrower: Borrower, today: date, events: list[dict]):
    """The judge's verdicts for one call, and the guardrails and tone checks they give."""
    try:
        verdicts = await judge(client, system_prompt(borrower, today), events)
    except Exception as e:
        logger.exception(f"judge failed on call {events[0]['call_id']}")
        return {"judge": None, "judge_error": repr(e),
                "checks": {"guardrails": None, "tone": None}}  # fmt: skip
    checks = {
        "guardrails": all(verdicts[c]["passed"] for c in GUARDRAIL_CHECKS),
        "tone": verdicts["polite_and_empathetic"]["passed"],
    }
    return {"judge": verdicts, "judge_error": None, "checks": checks}


def gemini_client() -> genai.Client:
    return genai.Client(
        api_key=config.env("GOOGLE_API_KEY"),
        http_options=HttpOptions(api_version=config.GEMINI_API_VERSION),
    )


def _passed(checks: dict) -> bool:
    return all(v is not False for v in checks.values())


def _verdict(checks: dict) -> str:
    failed = [k for k, v in checks.items() if v is False]
    return "pass" if not failed else "FAIL " + ", ".join(failed)


def summarise(runs: list[dict]) -> dict:
    by_scenario: dict[str, list[dict]] = {}
    for r in runs:
        by_scenario.setdefault(r["scenario"], []).append(r)
    scenarios = {}
    for sid, rs in by_scenario.items():
        row = {}
        for m in METRICS:
            applicable = [r["checks"][m] for r in rs if r["checks"][m] is not None]
            row[m] = f"{sum(applicable)}/{len(applicable)}" if applicable else "-"
        row["all_checks"] = f"{sum(_passed(r['checks']) for r in rs)}/{len(rs)}"
        row["pass_all_runs"] = all(_passed(r["checks"]) for r in rs)
        turns = [r["caller_turns"] for r in rs if "caller_turns" in r]
        row["median_caller_turns"] = statistics.median(turns) if turns else None
        scenarios[sid] = row
    overall = {}
    for m in METRICS:
        applicable = [r["checks"][m] for r in runs if r["checks"][m] is not None]
        overall[m] = round(sum(applicable) / len(applicable), 3) if applicable else None
    overall["all_checks"] = round(sum(_passed(r["checks"]) for r in runs) / len(runs), 3)
    overall["pass_all_runs"] = round(
        sum(row["pass_all_runs"] for row in scenarios.values()) / len(scenarios), 3
    )
    turns = [r["caller_turns"] for r in runs if "caller_turns" in r]
    overall["median_caller_turns"] = statistics.median(turns) if turns else None
    judged = [r["judge"] for r in runs if r.get("judge")]
    overall["judge_failures"] = {
        c: sum(1 for v in judged if not v[c]["passed"]) for c in config.JUDGE_CHECKS
    }
    overall["judged_runs"] = len(judged)
    return {"scenarios": scenarios, "overall": overall}


def print_summary(summary: dict, runs_per_scenario: int) -> None:
    head = (
        f"{'scenario':24}{'outcome':>9}{'disclose':>10}{'date':>7}{'lang':>6}{'guards':>8}{'tone':>6}"
        f"{'all':>7}{'every run':>11}{'turns':>7}"
    )
    print("\n" + head)
    for sid, row in summary["scenarios"].items():
        print(
            f"{sid:24}{row['outcome_accuracy']:>9}{row['verification_before_disclosure']:>10}"
            f"{row['promise_to_pay_capture']:>7}{row['language']:>6}{row['guardrails']:>8}"
            f"{row['tone']:>6}"
            f"{row['all_checks']:>7}"
            f"{'yes' if row['pass_all_runs'] else 'no':>11}{row['median_caller_turns']!s:>7}"
        )
    o = summary["overall"]
    print("\nOverall:")
    print(f"  outcome accuracy                {o['outcome_accuracy']:.0%}")
    print(f"  verification before disclosure  {o['verification_before_disclosure']:.0%}")
    print(f"  promise-to-pay capture          {o['promise_to_pay_capture']:.0%}")
    print(f"  right language                  {o['language']:.0%}")
    print(f"  guardrails (judge)              {o['guardrails']:.0%}")
    print(f"  tone (judge)                    {o['tone']:.0%}")
    print(f"  runs passing every check        {o['all_checks']:.0%}")
    print(f"  scenarios passing all {runs_per_scenario} runs     {o['pass_all_runs']:.0%}")
    print(f"  median caller turns             {o['median_caller_turns']}")
    print(f"\nJudge failures, out of {o['judged_runs']} judged runs:")
    for check, count in o["judge_failures"].items():
        print(f"  {check:32}{count}")


async def main(runs_per_scenario: int, only: list[str]) -> None:
    today = date.today()
    run_id = f"{datetime.now(UTC):%Y%m%d_%H%M%S}"
    out_dir = config.EVAL_RESULTS_DIR / run_id
    # Eval calls write their events and outcome records here, not to the live files.
    config.EVENTS_DIR = out_dir / "events"
    config.OUTCOMES_FILE = out_dir / "outcomes.jsonl"

    borrowers = load_borrowers(config.BORROWERS_FILE, today)
    raw = json.loads(config.SCENARIOS_FILE.read_text())
    scenarios = [prepare(s, borrowers[s["borrower_id"]], today) for s in raw]
    if only:
        scenarios = [s for s in scenarios if s.raw["id"] in only]
    client = gemini_client()
    limit = asyncio.Semaphore(config.EVAL_CONCURRENCY)
    print(f"Eval {run_id}: {len(scenarios)} scenarios x {runs_per_scenario} runs")
    runs = await asyncio.gather(
        *(
            run_one(s, n, client, today, limit)
            for s in scenarios
            for n in range(1, runs_per_scenario + 1)
        )
    )
    summary = summarise(runs)
    results = {
        "run_id": run_id,
        "today": today.isoformat(),
        "agent_model": config.GEMINI_MODEL,
        "sim_model": config.SIM_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "runs_per_scenario": runs_per_scenario,
        "summary": summary,
        "runs": runs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    await asyncio.to_thread(path.write_text, json.dumps(results, indent=2, ensure_ascii=False))
    print_summary(summary, runs_per_scenario)
    print(f"\nResults: {path}")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=config.EVAL_RUNS)
    parser.add_argument("--only", nargs="*", default=[], help="scenario ids to run")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    asyncio.run(main(args.runs, args.only))


if __name__ == "__main__":
    cli()
