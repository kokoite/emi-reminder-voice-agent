"""Summarise one call's turn-taking: latency, barge-ins, Jev decisions and silences.

Run: .venv/bin/python -m scripts.summarize_call [call_id]   (default: the latest call)
"""

import json
import statistics
import sys
from collections import Counter

import config


def pct(values: list[float], p: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, -(-len(ordered) * p // 100) - 1)]


def fmt(ms: float | None) -> str:
    return "-" if ms is None else f"{ms:.0f} ms"


def main() -> None:
    files = sorted(config.EVENTS_DIR.glob("c_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if len(sys.argv) > 1:
        files = [config.EVENTS_DIR / f"{sys.argv[1]}.jsonl"]
    if not files:
        print("No calls recorded yet.")
        return
    path = files[-1]
    events = [json.loads(line) for line in path.read_text().splitlines()]
    by_type = lambda t: [e["data"] for e in events if e["type"] == t]  # noqa: E731

    turns = [d["total_ms"] for d in by_type("turn_latency") if d["measured_from"] == "user_silence"]
    users = [d["text"] for d in by_type("user_said")]
    barges = by_type("barge")
    resumes = [b for b in barges if b["action"] == "resume"]
    stops = [b for b in barges if b["action"] == "stop"]
    jev = by_type("jev_decision")

    print(f"Call {path.stem}")
    print(f"  user turns {len(users)}, empty transcripts {sum(1 for u in users if not u.strip())}")
    median = statistics.median(turns) if turns else None
    print(f"  latency: median {fmt(median)}, p90 {fmt(pct(turns, 90))} over {len(turns)} turns")
    print(f"  barge-ins: {sum(1 for b in barges if b['action'] == 'pause')} pauses")
    print(f"    resumed {len(resumes)} {dict(Counter(r['reason'] for r in resumes))}")
    print(f"      pause length: median {fmt(pct([r['pause_ms'] for r in resumes], 50))}")
    print(f"    stopped {len(stops)} {dict(Counter(s['reason'] for s in stops))}")
    print(f"      time to stop: median {fmt(pct([s['time_to_stop_ms'] for s in stops], 50))}")
    jev_ms = [j["ms"] for j in jev]
    print(
        f"  Jev: {len(jev)} decisions, median {fmt(pct(jev_ms, 50))}, p90 {fmt(pct(jev_ms, 90))}, "
        f"timeouts {sum(1 for j in jev if j['timed_out'])}"
    )
    for j in jev:
        if "p_interruption" in j:
            score = f"p_interrupt {j['p_interruption']}"
        else:
            score = f"confidence {j.get('confidence')}"  # calls recorded before the change
        print(f"    {j['text']!r} -> {j['label']} ({score}) in {j['ms']} ms")
    print(f"  silence check-ins: {len(by_type('user_idle'))}")
    ended = by_type("call_ended")
    if ended:
        print(f"  ended: {ended[-1]['reason']} after {ended[-1]['duration_ms'] / 1000:.0f} s")


if __name__ == "__main__":
    main()
