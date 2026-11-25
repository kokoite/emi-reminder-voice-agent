"""The judge: grades one eval call on the guardrail and tone checks in config.JUDGE_CHECKS.

It reads the agent's instructions and the call log (caller lines, agent lines, tool calls
with results) and returns pass/fail per check, quoting the agent line when one fails.
"""

import asyncio
import json

from google import genai
from google.genai import types

import config


def call_log(events: list[dict]) -> str:
    """The call as the judge reads it, in the order things happened."""
    lines = []
    for e in events:
        d = e["data"]
        if e["type"] == "user_said":
            lines.append(f"Caller: {d['text']}")
        elif e["type"] == "bot_said":
            lines.append(f"Agent: {d['text']}")
        elif e["type"] == "tool_call":
            args = json.dumps(d["args"], ensure_ascii=False)
            result = json.dumps(d["data"], ensure_ascii=False)
            lines.append(f"[tool {d['name']}({args}) -> success={d['success']}, data={result}]")
    return "\n".join(lines)


def _schema() -> types.Schema:
    verdict = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "passed": types.Schema(type=types.Type.BOOLEAN),
            "evidence": types.Schema(type=types.Type.STRING),
        },
        required=["passed", "evidence"],
    )
    return types.Schema(
        type=types.Type.OBJECT,
        properties=dict.fromkeys(config.JUDGE_CHECKS, verdict),
        required=list(config.JUDGE_CHECKS),
    )


async def judge(client: genai.Client, agent_prompt: str, events: list[dict]) -> dict:
    """{check: {"passed", "evidence", "fail_votes"}} for one call, by majority of
    config.JUDGE_VOTES independent verdicts."""
    votes = await asyncio.gather(
        *(_judge_once(client, agent_prompt, events) for _ in range(config.JUDGE_VOTES))
    )
    return majority(votes)


def majority(votes: list[dict]) -> dict:
    result = {}
    for check in config.JUDGE_CHECKS:
        fails = [v[check] for v in votes if not v[check]["passed"]]
        passed = len(fails) * 2 < len(votes)
        evidence = "" if passed else fails[0]["evidence"]
        result[check] = {"passed": passed, "evidence": evidence, "fail_votes": len(fails)}
    return result


async def _judge_once(client: genai.Client, agent_prompt: str, events: list[dict]) -> dict:
    """One verdict: {check: {"passed": bool, "evidence": str}}."""
    checks = "\n".join(f"- {name}: {rule}" for name, rule in config.JUDGE_CHECKS.items())
    request = f"Agent's instructions:\n{agent_prompt}\n\nCall log:\n{call_log(events)}"
    response = await client.aio.models.generate_content(
        model=config.JUDGE_MODEL,
        contents=request,
        config=types.GenerateContentConfig(
            system_instruction=config.JUDGE_PROMPT.format(checks=checks),
            temperature=config.JUDGE_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=_schema(),
        ),
    )
    return json.loads(response.text)
