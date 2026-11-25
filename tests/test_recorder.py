import asyncio
import json

from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown,
    LatencyContribution,
    LatencyOwnerKind,
    MeasuredFrom,
)

from recorder import EventRecorder


def _read(recorder: EventRecorder) -> list[dict]:
    return [json.loads(line) for line in recorder.path.read_text().splitlines()]


def _breakdown(measured_from: MeasuredFrom = MeasuredFrom.USER_SILENCE) -> LatencyBreakdown:
    parts = [
        ("endpointing_wait", "VAD silence", "config:vad", 0.2),
        ("transcription", "Speech-to-text", "SarvamRealtimeSTTService", 0.18),
        ("llm_inference", "LLM", "GoogleLLMService", 0.52),
        ("speech_synthesis", "Text-to-speech", "ElevenLabsTTSService", 0.31),
    ]
    return LatencyBreakdown(
        contributions=[
            LatencyContribution(
                key=key,
                label=label,
                owner=owner,
                owner_kind=LatencyOwnerKind.SERVICE,
                start_time=0.0,
                duration_secs=secs,
            )
            for key, label, owner, secs in parts
        ],
        measured_from=measured_from,
        total_secs=1.21,
    )


def test_call_is_recorded_with_one_envelope(tmp_path):
    async def call() -> EventRecorder:
        recorder = EventRecorder(tmp_path, borrower_id="b_01")
        await recorder.start(lang="en")
        recorder.user_said("hello")
        recorder.turn_latency(_breakdown())
        recorder.bot_said("Hi, this is Lumen Finance.", interrupted=False)
        await recorder.stop("client_disconnected")
        return recorder

    recorder = asyncio.run(call())
    events = _read(recorder)

    assert [e["type"] for e in events] == [
        "call_started",
        "user_said",
        "turn_latency",
        "bot_said",
        "call_ended",
    ]
    for event in events:
        assert set(event) == {"ts", "t_ms", "call_id", "turn", "type", "data"}
        assert event["call_id"] == recorder.call_id
    assert [e["t_ms"] for e in events] == sorted(e["t_ms"] for e in events)
    assert events[0]["turn"] == 0 and events[1]["turn"] == 1
    assert events[3]["data"] == {
        "text": "Hi, this is Lumen Finance.",
        "words_played": 5,
        "interrupted": False,
    }
    assert events[-1]["data"]["turns"] == 1


def test_turn_latency_keeps_pipecat_parts(tmp_path):
    async def call() -> EventRecorder:
        recorder = EventRecorder(tmp_path)
        await recorder.start()
        recorder.turn_latency(_breakdown())
        await recorder.stop("done")
        return recorder

    data = _read(asyncio.run(call()))[1]["data"]

    assert data["total_ms"] == 1210
    assert data["measured_from"] == "user_silence"
    assert [p["key"] for p in data["parts"]] == [
        "endpointing_wait",
        "transcription",
        "llm_inference",
        "speech_synthesis",
    ]
    assert sum(p["ms"] for p in data["parts"]) == data["total_ms"]


def test_index_lists_each_call_once(tmp_path):
    async def two_calls() -> list[str]:
        ids = []
        for _ in range(2):
            recorder = EventRecorder(tmp_path)
            await recorder.start()
            await recorder.stop("done")
            ids.append(recorder.call_id)
        return ids

    ids = asyncio.run(two_calls())
    index = json.loads((tmp_path / "index.json").read_text())

    assert [c["call_id"] for c in index] == ids
    assert all(c["ended_at"] and c["end_reason"] == "done" for c in index)


def test_only_the_first_call_start_measurement_counts_as_the_greeting(tmp_path):
    async def call() -> EventRecorder:
        recorder = EventRecorder(tmp_path)
        await recorder.start()
        recorder.turn_latency(_breakdown(MeasuredFrom.CLIENT_CONNECTED))
        recorder.user_said("haan")
        recorder.turn_latency(_breakdown(MeasuredFrom.CLIENT_CONNECTED))
        await recorder.stop("done")
        return recorder

    events = _read(asyncio.run(call()))
    labels = [e["data"]["measured_from"] for e in events if e["type"] == "turn_latency"]
    assert labels == ["client_connected", "unanchored"]


def test_turn_after_a_barge_in_is_measured_by_the_recorder(tmp_path):
    import time

    async def call() -> EventRecorder:
        recorder = EventRecorder(tmp_path, vad_stop_secs=0.2)
        await recorder.start()
        recorder.turn_latency(_breakdown(MeasuredFrom.CLIENT_CONNECTED))  # the greeting
        recorder.last_vad_stop = time.monotonic() - 1.0  # caller stopped ~1.2 s ago
        recorder.turn_latency(_breakdown(MeasuredFrom.CLIENT_CONNECTED))  # Pipecat lost the start
        await recorder.stop("done")
        return recorder

    latencies = [e["data"] for e in _read(asyncio.run(call())) if e["type"] == "turn_latency"]
    assert latencies[1]["measured_from"] == "user_silence"
    assert latencies[1]["source"] == "recorder"
    assert 1150 <= latencies[1]["total_ms"] <= 1400
