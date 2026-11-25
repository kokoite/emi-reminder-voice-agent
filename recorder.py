"""Event recorder: writes every call event to events/<call_id>.jsonl.

Each line has the same envelope: ts, t_ms (milliseconds since the call started,
from one monotonic clock), call_id, turn, type and data. events/index.json lists
the calls so the dashboard knows which files to read.

Writes go through one background task, so the pipeline never waits on disk.
"""

import asyncio
import json
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.frames.frames import ErrorFrame, VADUserStoppedSpeakingFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import LatencyBreakdown

# Calls can end at the same moment (the evals run several at once); one lock keeps
# their read-modify-write of index.json from colliding.
_INDEX_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EventRecorder:
    """Records one call. Create it per call, `start()` on connect, `stop()` on hang-up."""

    def __init__(
        self, events_dir: Path, borrower_id: str | None = None, vad_stop_secs: float = 0.2
    ):
        self.events_dir = events_dir
        self.borrower_id = borrower_id
        self.call_id = f"c_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"
        self.turn = 0
        self._greeting_recorded = False
        self._vad_stop_secs = vad_stop_secs
        self.last_vad_stop: float | None = None  # monotonic time, set by the frame observer
        self._t0 = time.monotonic()
        self._started_at = _now_iso()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._writer: asyncio.Task | None = None
        self.observer = _ErrorObserver(self)

    @property
    def path(self) -> Path:
        return self.events_dir / f"{self.call_id}.jsonl"

    async def start(self, lang: str | None = None) -> None:
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.monotonic()
        self._writer = asyncio.create_task(self._write_loop())
        self.emit("call_started", borrower_id=self.borrower_id, lang=lang)
        await asyncio.to_thread(self._update_index, ended_at=None, reason=None)

    async def stop(self, reason: str) -> None:
        if self._writer is None:
            return
        duration_ms = round((time.monotonic() - self._t0) * 1000)
        self.emit("call_ended", duration_ms=duration_ms, reason=reason, turns=self.turn)
        await self._queue.put(None)
        await self._writer
        self._writer = None
        await asyncio.to_thread(self._update_index, ended_at=_now_iso(), reason=reason)

    def emit(self, event_type: str, **data: Any) -> None:
        """Queue one event. Never blocks."""
        event = {
            "ts": _now_iso(),
            "t_ms": round((time.monotonic() - self._t0) * 1000),
            "call_id": self.call_id,
            "turn": self.turn,
            "type": event_type,
            "data": data,
        }
        self._queue.put_nowait(json.dumps(event, ensure_ascii=False))

    def user_said(self, text: str, lang: str | None = None) -> None:
        self.turn += 1
        self.emit("user_said", text=text, lang=lang)

    def bot_said(self, text: str, interrupted: bool) -> None:
        self.emit("bot_said", text=text, words_played=len(text.split()), interrupted=interrupted)

    def turn_latency(self, breakdown: LatencyBreakdown) -> None:
        """Record Pipecat's per-turn breakdown. Its parts add up to the total.

        Pipecat labels a breakdown "client_connected" whenever it lacks the start of
        the caller's turn. That's right for the greeting, but it also happens after
        an interrupted reply, measuring from call start. Those are labelled
        "unanchored" so they aren't mistaken for a greeting or a turn.
        """
        measured_from = str(breakdown.measured_from) if breakdown.measured_from else None
        if measured_from == "client_connected":
            if self._greeting_recorded:
                if self.last_vad_stop is not None:
                    # Pipecat lost the turn start (a turn that began with a barge-in), so
                    # measure it ourselves: caller stopped speaking -> bot audio starts now.
                    caller_stopped = self.last_vad_stop - self._vad_stop_secs
                    self.emit(
                        "turn_latency",
                        total_ms=round((time.monotonic() - caller_stopped) * 1000),
                        measured_from="user_silence",
                        source="recorder",
                        parts=[],
                    )
                    return
                measured_from = "unanchored"
            self._greeting_recorded = True
        self.emit(
            "turn_latency",
            total_ms=round(breakdown.total_secs * 1000),
            measured_from=measured_from,
            parts=[
                {
                    "key": part.key,
                    "label": part.label,
                    "owner": part.owner,
                    "ms": round(part.duration_secs * 1000),
                }
                for part in breakdown.contributions
            ],
        )

    async def _write_loop(self) -> None:
        while (line := await self._queue.get()) is not None:
            try:
                await asyncio.to_thread(self._append, line)
            except OSError as e:
                logger.error(f"Event recorder could not write to {self.path}: {e}")

    def _append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _update_index(self, ended_at: str | None, reason: str | None) -> None:
        with _INDEX_LOCK:
            self._write_index(ended_at, reason)

    def _write_index(self, ended_at: str | None, reason: str | None) -> None:
        index_path = self.events_dir / "index.json"
        calls = json.loads(index_path.read_text()) if index_path.exists() else []
        calls = [c for c in calls if c["call_id"] != self.call_id]
        calls.append(
            {
                "call_id": self.call_id,
                "borrower_id": self.borrower_id,
                "started_at": self._started_at,
                "ended_at": ended_at,
                "end_reason": reason,
                "turns": self.turn,
            }
        )
        tmp = index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(calls, indent=2))
        tmp.replace(index_path)


class _ErrorObserver(BaseObserver):
    """Records pipeline errors as `error` events, and when the caller last stopped speaking."""

    def __init__(self, recorder: EventRecorder):
        super().__init__()
        self._recorder = recorder

    async def on_push_frame(self, data: FramePushed) -> None:
        if isinstance(data.frame, VADUserStoppedSpeakingFrame) and data.first_push:
            self._recorder.last_vad_stop = time.monotonic()
        if isinstance(data.frame, ErrorFrame) and data.first_push:
            self._recorder.emit(
                "error",
                service=str(data.source),
                message=data.frame.error,
                fatal=data.frame.fatal,
            )
