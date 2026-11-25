import json
from datetime import date

import pytest

import config
from agent.borrowers import load_borrowers
from agent.tools import (
    TOOL_SCHEMAS,
    TOOLS,
    CallState,
    end_call,
    run_tool,
    save_record_without_end_call,
    send_payment_link,
    verify_identity,
)

# Async tests run through anyio's pytest plugin (anyio comes with pipecat).
pytestmark = pytest.mark.anyio

TODAY = date(2026, 11, 24)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_tool_delay(monkeypatch):
    for name in config.TOOL_DELAY_SECS:
        monkeypatch.setitem(config.TOOL_DELAY_SECS, name, 0.0)


@pytest.fixture
def borrowers():
    return load_borrowers(config.BORROWERS_FILE, TODAY)


@pytest.fixture
def state(borrowers):
    return CallState(call_id="c_test", borrower=borrowers["b_03"])


@pytest.fixture
def outcomes(tmp_path, monkeypatch):
    path = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(config, "OUTCOMES_FILE", path)
    return path


def assert_envelope(result, tool, status):
    assert set(result) == {"success", "data", "_instruction"}
    assert result["_instruction"] == config.TOOL_INSTRUCTIONS[tool][status]


# --- borrowers


def test_due_dates_are_relative_to_today(borrowers):
    assert len(borrowers) == 8
    assert borrowers["b_03"].due_date == date(2026, 11, 4)
    assert borrowers["b_03"].days_overdue == 20
    assert borrowers["b_01"].days_overdue == -3


# --- verify_identity


async def test_verify_pass_returns_loan_details_in_spoken_form(state):
    result = await verify_identity(state, birth_date="1985-11-02")
    assert_envelope(result, "verify_identity", "verified")
    assert result["data"] == {
        "name": "Arjun Mehta",
        "loan_type": "home loan",
        "amount_due": "one lakh twenty four thousand rupees",
        "due_date": "the fourth of November",
        "status": "twenty days overdue",
    }
    assert state.verified


async def test_verify_by_last4(state):
    assert (await verify_identity(state, last4="2290"))["success"] is True


@pytest.mark.parametrize(
    "args",
    [
        {"last4": "1234"},
        {"birth_date": "1985-02-11"},
        {"birth_date": "1985-11-02", "last4": "1234"},  # one of two wrong
        {},
    ],
)
async def test_verify_fails_without_a_full_match(state, args):
    result = await verify_identity(state, **args)
    assert_envelope(result, "verify_identity", "not_verified")
    assert result["data"] == {}
    assert not state.verified


# --- send_payment_link


async def test_link_needs_a_verified_caller(state):
    assert_envelope(await send_payment_link(state), "send_payment_link", "not_verified")
    assert not state.payment_link_sent

    await verify_identity(state, last4="2290")
    result = await send_payment_link(state)
    assert_envelope(result, "send_payment_link", "sent")
    assert result["data"] == {"to": "the mobile number ending two two nine zero"}
    assert state.payment_link_sent


# --- end_call


async def test_outcome_record_takes_facts_from_the_call_state(state, outcomes):
    await verify_identity(state, last4="2290")
    await send_payment_link(state)
    result = await end_call(
        state,
        goodbye="Goodbye.",
        outcome="promise_to_pay",
        promise_to_pay_date="2026-12-01",
        escalate_to_human=False,
    )
    assert_envelope(result, "end_call", "saved")
    assert result["data"] == {"outcome": "promise_to_pay", "escalated": False}
    assert state.goodbye == "Goodbye."
    [record] = [json.loads(line) for line in outcomes.read_text().splitlines()]
    record.pop("at")
    assert record == {
        "call_id": "c_test",
        "borrower_id": "b_03",
        "verified": True,
        "outcome": "promise_to_pay",
        "promise_to_pay_date": "2026-12-01",
        "callback_at": None,
        "payment_link_sent": True,
        "escalate_to_human": False,
        "notes": None,
    }


async def test_callback_time_is_recorded(state, outcomes):
    await end_call(
        state,
        goodbye="Goodbye.",
        outcome="callback_requested",
        callback_at="2026-11-25T18:00",
        escalate_to_human=False,
    )
    assert json.loads(outcomes.read_text())["callback_at"] == "2026-11-25T18:00"


@pytest.mark.parametrize("outcome", ["hardship", "dispute"])
async def test_hardship_and_dispute_always_escalate(state, outcomes, outcome):
    result = await end_call(state, goodbye="Goodbye.", outcome=outcome, escalate_to_human=False)
    assert result["data"]["escalated"] is True
    assert json.loads(outcomes.read_text())["escalate_to_human"] is True


async def test_save_failure_is_reported(state, tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    monkeypatch.setattr(config, "OUTCOMES_FILE", blocker / "outcomes.jsonl")
    result = await end_call(
        state, goodbye="Goodbye.", outcome="already_paid", escalate_to_human=False
    )
    assert_envelope(result, "end_call", "service_unavailable")


# --- run_tool


async def test_run_tool_records_a_tool_call_event(state):
    events = []
    result = await run_tool(
        "verify_identity", {"last4": "2290"}, state, lambda t, **d: events.append((t, d))
    )
    assert result["success"] is True
    [(type_, data)] = events
    assert type_ == "tool_call"
    assert data["name"] == "verify_identity"
    assert data["args"] == {"last4": "2290"}
    assert data["injected_failure"] is False


async def test_injected_failure(state, monkeypatch):
    monkeypatch.setitem(config.TOOL_FAILURE_RATE, "send_payment_link", 1.0)
    state.verified = True
    events = []
    result = await run_tool("send_payment_link", {}, state, lambda t, **d: events.append(d))
    assert_envelope(result, "send_payment_link", "service_unavailable")
    assert result["data"] == {"injected_failure": True}
    assert not state.payment_link_sent
    assert events[0]["injected_failure"] is True


async def test_a_tool_error_becomes_service_unavailable(state):
    result = await run_tool("verify_identity", {"pan": "ABCDE1234F"}, state, lambda *a, **k: None)
    assert_envelope(result, "verify_identity", "service_unavailable")


def test_every_schema_has_a_tool_and_instructions():
    assert [s["name"] for s in TOOL_SCHEMAS] == list(TOOLS)
    for name in TOOLS:
        assert "service_unavailable" in config.TOOL_INSTRUCTIONS[name]


async def test_a_call_without_end_call_still_gets_a_record(state, outcomes):
    await verify_identity(state, last4="2290")
    await save_record_without_end_call(state, "caller_hung_up")
    record = json.loads(outcomes.read_text())
    assert record["outcome"] is None
    assert record["verified"] is True
    assert "caller_hung_up" in record["notes"]


async def test_no_second_record_after_end_call(state, outcomes):
    await end_call(state, goodbye="Goodbye.", outcome="already_paid", escalate_to_human=False)
    await save_record_without_end_call(state, "end_call")
    assert len(outcomes.read_text().splitlines()) == 1


async def test_a_call_can_force_a_tool_failure(state):
    state.verified = True
    state.failure_rates["send_payment_link"] = 1.0
    result = await run_tool("send_payment_link", {}, state, lambda *a, **k: None)
    assert_envelope(result, "send_payment_link", "service_unavailable")
    assert not state.payment_link_sent
