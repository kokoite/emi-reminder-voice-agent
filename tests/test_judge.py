import config
from evals.judge import call_log, majority


def test_call_log_keeps_the_order_and_shows_tool_results():
    events = [
        {"type": "call_started", "data": {}},
        {"type": "bot_said", "data": {"text": "Am I speaking with Ravi?"}},
        {"type": "user_said", "data": {"text": "Yes, last four 4821."}},
        {
            "type": "tool_call",
            "data": {
                "name": "verify_identity",
                "args": {"last4": "4821"},
                "success": True,
                "data": {"amount_due": "twelve thousand five hundred rupees"},
            },
        },  # fmt: skip
        {"type": "bot_said", "data": {"text": "Thanks, Ravi."}},
    ]
    assert call_log(events).splitlines() == [
        "Agent: Am I speaking with Ravi?",
        "Caller: Yes, last four 4821.",
        '[tool verify_identity({"last4": "4821"}) -> success=True, '
        'data={"amount_due": "twelve thousand five hundred rupees"}]',
        "Agent: Thanks, Ravi.",
    ]


def test_majority_vote_per_check():
    def vote(failing):
        return {c: {"passed": c not in failing, "evidence": f"line for {c}" if c in failing else ""}
                for c in config.JUDGE_CHECKS}  # fmt: skip

    result = majority([vote({"no_invented_facts"}), vote({"no_invented_facts", "stays_on_topic"}),
                       vote(set())])  # fmt: skip
    assert result["no_invented_facts"] == {
        "passed": False, "evidence": "line for no_invented_facts", "fail_votes": 2}  # fmt: skip
    assert result["stays_on_topic"] == {"passed": True, "evidence": "", "fail_votes": 1}
    assert result["no_threats_or_pressure"]["passed"] is True
