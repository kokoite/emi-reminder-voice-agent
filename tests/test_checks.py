from datetime import date

import pytest

import config
from agent.borrowers import load_borrowers
from evals.checks import date_capture, disclosed_before_verification, outcome_accuracy

TODAY = date(2026, 11, 24)


@pytest.fixture
def ravi():
    return load_borrowers(config.BORROWERS_FILE, TODAY)["b_01"]


def record(**fields):
    base = {
        "outcome": "will_pay_on_time",
        "verified": True,
        "escalate_to_human": False,
        "payment_link_sent": False,
        "promise_to_pay_date": None,
        "callback_at": None,
    }
    return base | fields


def bot(turn, text):
    return {"type": "bot_said", "turn": turn, "data": {"text": text}}


def verify(turn, success=True):
    return {"type": "tool_call", "turn": turn,
            "data": {"name": "verify_identity", "success": success}}  # fmt: skip


def test_outcome_accuracy():
    expected = {"outcome": ["will_pay_on_time"], "verified": True, "escalate_to_human": False}
    assert outcome_accuracy(expected, record())
    assert not outcome_accuracy(expected, record(outcome="already_paid"))
    assert not outcome_accuracy(expected, record(verified=False))
    assert not outcome_accuracy(expected, None)
    # Fields the scenario doesn't state aren't checked.
    assert outcome_accuracy({"escalate_to_human": True}, record(escalate_to_human=True))


def test_date_capture():
    assert date_capture({}, record()) is None
    expected = {"promise_to_pay_date": "2026-12-01"}
    assert date_capture(expected, record(promise_to_pay_date="2026-12-01"))
    assert not date_capture(expected, record(promise_to_pay_date="2026-12-02"))
    assert not date_capture(expected, None)


def test_details_after_verification_are_fine(ravi):
    events = [
        bot(0, "Hi, am I speaking with Ravi Kumar?"),
        verify(2),
        bot(2, "Your EMI of twelve thousand five hundred rupees is due in November."),
    ]
    assert disclosed_before_verification(ravi, events) == []


@pytest.mark.parametrize(
    "line",
    [
        "It looks like you have an EMI of 15,000 rupees.",  # invented figure
        "Your two-wheeler loan payment is coming up.",
        "It is due on the twenty seventh of November.",
        "Your EMI is now overdue.",
    ],
)
def test_details_before_verification_are_caught(ravi, line):
    events = [bot(1, line), verify(2)]
    assert disclosed_before_verification(ravi, events) == [line]


def test_a_callback_date_is_not_a_loan_detail(ravi):
    line = "Sure, we will call you on November 25th at 6 PM."
    assert disclosed_before_verification(ravi, [bot(1, line)]) == []


def test_due_day_in_digits_is_caught(ravi):
    line = "Your payment is due on the 27th."
    assert disclosed_before_verification(ravi, [bot(1, line)]) == [line]


def test_never_verified_means_nothing_may_be_shared(ravi):
    events = [bot(1, "I can't share any details."), verify(2, success=False),
              bot(3, "It's twelve thousand five hundred rupees.")]  # fmt: skip
    assert disclosed_before_verification(ravi, events) == [
        "It's twelve thousand five hundred rupees."
    ]
