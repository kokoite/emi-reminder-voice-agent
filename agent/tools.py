"""The agent's three tools: verify_identity, send_payment_link and end_call.

Each tool is a plain function over one shared CallState. The state holds facts only
code may set (is the caller verified, was a link sent), so the LLM cannot claim them.
Tools check only what the assignment's guardrails require.

Every tool returns {success, data, _instruction}. `_instruction` (from
config.TOOL_INSTRUCTIONS) tells the LLM how to read the result; what to say next is
the system prompt's job.
"""

import asyncio
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

import config
from agent.borrowers import Borrower
from agent.spoken import date_in_words, digits, due_status, rupees

# The borrower responses the assignment lists, as outcomes.
OUTCOMES = (
    "will_pay_on_time",
    "already_paid",
    "promise_to_pay",
    "hardship",
    "dispute",
    "wrong_number",
    "third_party",
    "callback_requested",
    "asked_to_stop",
    "human_requested",
)
# "Escalate to a human ... for disputes, hardship and explicit requests."
ALWAYS_ESCALATE = ("hardship", "dispute")

# Provider-neutral JSON schemas; the LLM service adapter turns these into its own format.
TOOL_SCHEMAS = [
    {
        "name": "verify_identity",
        "description": "Check the caller is the borrower, before sharing any loan details. "
        "Pass what the caller said: their date of birth, the last four digits of their "
        "registered mobile number, or both.",
        "parameters": {
            "type": "object",
            "properties": {
                "birth_date": {
                    "type": "string",
                    "description": "Date of birth as YYYY-MM-DD. Indian callers say the day "
                    "first: 'twelve March eighty eight' is 1988-03-12.",
                },
                "last4": {
                    "type": "string",
                    "description": "Last four digits of the registered mobile number, as "
                    "four digits, e.g. '4821'.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "send_payment_link",
        "description": "Send a payment link by SMS to the borrower's registered mobile number.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "end_call",
        "description": "Says your goodbye, ends the call and records how it went. It is the "
        "only way to end a call. Put your goodbye in goodbye instead of saying it yourself.",
        "parameters": {
            "type": "object",
            "properties": {
                "goodbye": {
                    "type": "string",
                    "description": "Spoken for you just before the call ends: a short, polite "
                    "goodbye; for hardship or a dispute, it may first say you're sorry to hear "
                    "it. Add only what applies: that "
                    "a human agent will contact them (if you escalated); the callback time they "
                    "gave; if someone else answered, a request to have the borrower call Lumen "
                    "Finance back; for a wrong number, an apology. Nothing else: no question, "
                    "no other promise. Never say you will connect or transfer them.",
                },
                "outcome": {
                    "type": "string",
                    "enum": list(OUTCOMES),
                    "description": "will_pay_on_time: will pay on or before the due date. "
                    "promise_to_pay: commits to a specific date after the due date. "
                    "third_party: someone who knows the borrower answered (family, a colleague). "
                    "wrong_number: the person doesn't know the borrower. "
                    "human_requested: the borrower asked to speak to a person for no other "
                    "reason. Hardship and dispute take priority over every other outcome.",
                },
                "promise_to_pay_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD, when the borrower promises to pay on a date.",
                },
                "callback_at": {
                    "type": "string",
                    "description": "When the borrower asked to be called back, India time: "
                    "YYYY-MM-DDTHH:MM, or just YYYY-MM-DD if they gave no time. Only what they "
                    "said; never guess a time.",
                },
                "escalate_to_human": {
                    "type": "boolean",
                    "description": "Flags the case so a human agent contacts the borrower "
                    "later. It does not transfer the call.",
                },
                "notes": {
                    "type": "string",
                    "description": "Short notes on the call.",
                },
            },
            "required": ["goodbye", "outcome", "escalate_to_human"],
        },
    },
]


@dataclass
class CallState:
    """What the tools share during one call. Only tools change it."""

    call_id: str
    borrower: Borrower
    verified: bool = False  # set only by verify_identity
    payment_link_sent: bool = False  # set only by send_payment_link
    outcome: dict | None = None  # the saved outcome record, set only by end_call
    goodbye: str | None = None  # the closing line end_call asked to be spoken
    # Injected failure rate per tool for this call; falls back to config.TOOL_FAILURE_RATE.
    # Per call, so eval calls running side by side don't affect each other.
    failure_rates: dict[str, float] = field(default_factory=dict)


def envelope(tool: str, status: str, success: bool, data: dict | None = None) -> dict:
    return {
        "success": success,
        "data": data or {},
        "_instruction": config.TOOL_INSTRUCTIONS[tool][status],
    }


async def verify_identity(
    state: CallState, birth_date: str | None = None, last4: str | None = None
) -> dict:
    """Verified when at least one value is given and every given value matches the record."""
    b = state.borrower
    given = [(birth_date, b.birth_date.isoformat()), (last4, b.last4)]
    given = [(said, record) for said, record in given if said]
    if not given or any(said != record for said, record in given):
        return envelope("verify_identity", "not_verified", False)
    state.verified = True
    details = {
        "name": b.name,
        "loan_type": b.loan_type,
        "amount_due": rupees(b.emi_amount),
        "due_date": date_in_words(b.due_date),
        "status": due_status(b.days_overdue),
    }
    return envelope("verify_identity", "verified", True, details)


async def send_payment_link(state: CallState) -> dict:
    # The link carries loan details, so it goes only to a verified borrower.
    if not state.verified:
        return envelope("send_payment_link", "not_verified", False)
    state.payment_link_sent = True
    to = f"the mobile number ending {digits(state.borrower.last4)}"
    return envelope("send_payment_link", "sent", True, {"to": to})


async def end_call(
    state: CallState,
    outcome: str,
    goodbye: str,
    escalate_to_human: bool = False,
    promise_to_pay_date: str | None = None,
    callback_at: str | None = None,
    notes: str | None = None,
) -> dict:
    """Write the outcome record. verified and payment_link_sent come from the call state."""
    escalate = escalate_to_human or outcome in ALWAYS_ESCALATE
    try:
        await _save_record(state, outcome, promise_to_pay_date, callback_at, escalate, notes)
    except OSError:
        logger.exception(f"could not save the outcome for call {state.call_id}")
        return envelope("end_call", "service_unavailable", False)
    state.goodbye = goodbye
    return envelope("end_call", "saved", True, {"outcome": outcome, "escalated": escalate})


async def save_record_without_end_call(state: CallState, ended_by: str) -> None:
    """Called by code when a call ends and the agent never recorded an outcome, so every
    call still has a record. The outcome is left empty; notes say why."""
    if state.outcome is not None:
        return
    notes = f"The agent did not call end_call; the call ended by {ended_by}."
    try:
        await _save_record(state, None, None, None, False, notes)
    except OSError:
        logger.exception(f"could not save the outcome for call {state.call_id}")


async def _save_record(
    state: CallState,
    outcome: str | None,
    promise_to_pay_date: str | None,
    callback_at: str | None,
    escalate: bool,
    notes: str | None,
) -> None:
    record = {
        "call_id": state.call_id,
        "borrower_id": state.borrower.borrower_id,
        "at": datetime.now().isoformat(timespec="seconds"),
        "verified": state.verified,
        "outcome": outcome,
        "promise_to_pay_date": promise_to_pay_date,
        "callback_at": callback_at,
        "payment_link_sent": state.payment_link_sent,
        "escalate_to_human": escalate,
        "notes": notes,
    }
    await asyncio.to_thread(_append_line, config.OUTCOMES_FILE, record)
    state.outcome = record


TOOLS = {"verify_identity": verify_identity, "send_payment_link": send_payment_link,
         "end_call": end_call}  # fmt: skip


async def run_tool(
    name: str,
    args: dict[str, Any],
    state: CallState,
    emit: Callable[..., None],
    rng: random.Random | None = None,
) -> dict:
    """Run one tool with its configured delay and failure rate, and record a tool_call event."""
    started = time.monotonic()
    injected = False
    await asyncio.sleep(config.TOOL_DELAY_SECS[name])
    rate = state.failure_rates.get(name, config.TOOL_FAILURE_RATE[name])
    if (rng or random).random() < rate:
        injected = True
        result = envelope(name, "service_unavailable", False, {"injected_failure": True})
    else:
        try:
            result = await TOOLS[name](state, **args)
        except Exception:
            logger.exception(f"tool {name} failed, call {state.call_id}, args {args}")
            result = envelope(name, "service_unavailable", False)
    emit(
        "tool_call",
        name=name,
        args=args,
        success=result["success"],
        data=result["data"],
        injected_failure=injected,
        ms=round((time.monotonic() - started) * 1000),
    )
    return result


def _append_line(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
