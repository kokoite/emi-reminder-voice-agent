"""Re-run only the judge on a saved eval run, so a judge change is compared on the same calls.

Run: .venv/bin/python -m evals.rejudge <run_id>
Updates evals/results/<run_id>/results.json in place and prints the summary.
"""

import asyncio
import json
import sys
from datetime import date

from loguru import logger

import config
from borrowers import load_borrowers
from evals.run_evals import gemini_client, judge_run, print_summary, summarise


async def main(run_id: str) -> None:
    out_dir = config.EVAL_RESULTS_DIR / run_id
    path = out_dir / "results.json"
    results = json.loads(await asyncio.to_thread(path.read_text))
    today = date.fromisoformat(results["today"])
    borrowers = load_borrowers(config.BORROWERS_FILE, today)
    scenarios = {s["id"]: s for s in json.loads(config.SCENARIOS_FILE.read_text())}
    client = gemini_client()
    limit = asyncio.Semaphore(config.EVAL_CONCURRENCY)

    async def rejudge(run: dict) -> None:
        if "call_id" not in run:
            return  # the call itself failed; nothing to judge
        events_path = out_dir / "events" / f"{run['call_id']}.jsonl"
        lines = (await asyncio.to_thread(events_path.read_text)).splitlines()
        borrower = borrowers[scenarios[run["scenario"]]["borrower_id"]]
        async with limit:
            judged = await judge_run(client, borrower, today, [json.loads(x) for x in lines])
        run["judge"], run["judge_error"] = judged["judge"], judged["judge_error"]
        run["checks"] |= judged["checks"]

    await asyncio.gather(*(rejudge(r) for r in results["runs"]))
    results["judge_model"] = config.JUDGE_MODEL
    results["summary"] = summarise(results["runs"])
    await asyncio.to_thread(path.write_text, json.dumps(results, indent=2, ensure_ascii=False))
    print_summary(results["summary"], results["runs_per_scenario"])
    print(f"\nResults: {path}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    asyncio.run(main(sys.argv[1]))
