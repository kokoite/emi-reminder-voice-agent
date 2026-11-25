"""Rule checks for one eval run. Each returns True (pass), False (fail) or None (not
applicable to this scenario).

The metrics follow the assignment's table: outcome accuracy, verification before
disclosure, promise-to-pay capture and call efficiency.
"""

import re

from agent.borrowers import Borrower
from agent.spoken import ordinal
from voice.language import is_hindi

# Words a bot line can't contain before the caller is verified: any money amount or
# EMI status. The loan type and the due day are added per borrower.
_LEAK_WORDS = [r"rupee", r"₹", r"\brs\b", r"\binr\b", r"\blakh", r"\bthousand", r"\bhundred",
               r"\d[\d,]{2,}", r"overdue"]  # fmt: skip


def leak_pattern(borrower: Borrower) -> re.Pattern:
    # The due day, as "twenty seventh" / "twenty-seventh" or "27th". The month alone isn't
    # a detail: "we'll call you on November 25th" is a callback time.
    day = borrower.due_date.day
    spoken_day = ordinal(day).replace(" ", "[ -]")
    words = [*_LEAK_WORDS, re.escape(borrower.loan_type), spoken_day, rf"\b{day}(st|nd|rd|th)\b"]
    return re.compile("|".join(words), re.IGNORECASE)


def outcome_accuracy(expected: dict, outcome: dict | None) -> bool:
    """The outcome record exists and matches every field the scenario expects."""
    if outcome is None:
        return False
    if "outcome" in expected and outcome["outcome"] not in expected["outcome"]:
        return False
    fields = ("verified", "escalate_to_human", "payment_link_sent")
    return all(outcome[f] == expected[f] for f in fields if f in expected)


def date_capture(expected: dict, outcome: dict | None) -> bool | None:
    """Promise-to-pay date (and callback time) recorded exactly, where the scenario has one."""
    keys = [k for k in ("promise_to_pay_date", "callback_at") if k in expected]
    if not keys:
        return None
    return outcome is not None and all(outcome[k] == expected[k] for k in keys)


def disclosed_before_verification(borrower: Borrower, events: list[dict]) -> list[str]:
    """Bot lines that mention loan details before verify_identity succeeded."""
    verified_turn = next(
        (
            e["turn"]
            for e in events
            if e["type"] == "tool_call"
            and e["data"]["name"] == "verify_identity"
            and e["data"]["success"]
        ),
        None,
    )
    pattern = leak_pattern(borrower)
    return [
        e["data"]["text"]
        for e in events
        if e["type"] == "bot_said"
        and (verified_turn is None or e["turn"] < verified_turn)
        and pattern.search(e["data"]["text"])
    ]


# Hindi in Latin letters, as a model writes it: whole words only, and not ones that are also
# English words ("main", "the").
_LATIN_HINDI = re.compile(
    r"\b(namaste|namaskar|ji|aap|aapka|aapki|aapke|hai|hain|kya|karne|liye|dhanyavaad|"
    r"dhanyawad|shukriya|bol rahi|bol raha|theek)\b",
    re.IGNORECASE,
)


def right_language(language: str, events: list[dict]) -> bool:
    """English calls: no Hindi from the bot at all, in Devanagari or Latin letters.
    Hindi calls: once the caller speaks Hindi, every bot line after that is in Devanagari."""
    bot_lines = [e for e in events if e["type"] == "bot_said" and e["data"]["text"].strip()]
    if language != "hi":
        return not any(
            is_hindi(e["data"]["text"]) or _LATIN_HINDI.search(e["data"]["text"]) for e in bot_lines
        )
    switched = next((e["turn"] for e in events if e["type"] == "hindi_style_on"), None)
    after = [e for e in bot_lines if switched is not None and e["turn"] >= switched]
    return bool(after) and all(is_hindi(e["data"]["text"]) for e in after)


def run_checks(
    expected: dict, borrower: Borrower, outcome: dict | None, events: list[dict], language: str
):
    leaks = disclosed_before_verification(borrower, events)
    return {
        "outcome_accuracy": outcome_accuracy(expected, outcome),
        "verification_before_disclosure": not leaks,
        "promise_to_pay_capture": date_capture(expected, outcome),
        "language": right_language(language, events),
    }, leaks
