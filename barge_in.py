"""Barge-in: when the caller makes a sound while the bot speaks, pause, decide, then resume or stop.

- OutputGate holds the bot's audio output. Each audio write waits until it has
  played, so holding the write pauses playback within one chunk, and releasing
  it continues from the same word.
- JevBargeInStrategy is a user-turn start strategy. Caller speech always starts
  the caller's turn, so Smart Turn sees every turn from its start. While the bot
  speaks, the turn starts without interrupting: the gate pauses the bot and Jev
  decides. Interruption (or no answer in time, or the caller keeps talking): the
  bot is interrupted and the turn carries on. Backchannel ("haan", "hmm") or no
  words: the bot resumes and the turn's text is dropped, so it ends empty and gets
  no reply.
- BargeInTurnGuard is a user-turn stop strategy beside Smart Turn. A turn can end
  only when every stop strategy agrees, so the guard holds a barge-in turn open
  until Jev has decided and any backchannel text is dropped.
- BotLineObserver keeps what the bot is currently saying, for Jev's context.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from pipecat.classifiers.base_classifier import BaseClassifier, ChoiceQuestion
from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy

Emit = Callable[..., None]


def interrupt_question(instructions: str, options: dict[str, str]) -> ChoiceQuestion:
    """The question Jev answers; options must include "interruption" and "backchannel"."""
    return ChoiceQuestion(instructions=instructions, options=options)


class OutputGate(BaseObserver):
    """Pause and resume the bot's audio playback by holding the transport's audio write.

    On an interruption the old reply's audio is still queued in the output
    transport until the interruption reaches it. Until then the gate drops that
    audio instead of letting it play out in a burst. It returns to normal when it
    sees the output transport pass the interruption on, or after
    `flush_timeout_secs` as a safety net so the bot can never stay mute.
    """

    def __init__(self, flush_timeout_secs: float = 1.0, resume_log_secs: float = 0.0):
        super().__init__()
        self._open = asyncio.Event()
        self._open.set()
        self._flush_timeout_secs = flush_timeout_secs
        self._resume_log_secs = resume_log_secs
        self._dropping_until = 0.0
        self._resumed_at = 0.0
        self._output: Any = None

    @property
    def paused(self) -> bool:
        return not self._open.is_set()

    @property
    def dropping(self) -> bool:
        return time.monotonic() < self._dropping_until

    def install(self, output_transport: Any) -> None:
        self._output = output_transport
        original_write = output_transport.write_audio_frame
        original_push = output_transport.push_frame

        async def gated_write(frame):
            await self._open.wait()
            if self.dropping:
                return True  # Old reply's audio after an interruption: drop it.
            started = time.monotonic()
            result = await original_write(frame)
            if started - self._resumed_at < self._resume_log_secs:
                # Diagnostic: audio after a resume should play at normal speed, so each
                # write should take about as long as the audio it carries.
                audio = getattr(frame, "audio", b"")
                audio_ms = len(audio) / 2 / getattr(frame, "sample_rate", 24000) * 1000
                logger.debug(
                    f"gate: write +{(started - self._resumed_at) * 1000:.0f} ms after resume, "
                    f"{audio_ms:.0f} ms of audio took {(time.monotonic() - started) * 1000:.0f} ms"
                )
            return result

        async def gated_push(frame, *args, **kwargs):
            # The transport pushes the bot's text as its audio is due: sentence
            # announcements, each spoken word, and word progress. Hold them while
            # paused and drop the old reply's, so the saved message and the call page
            # only show what the caller actually heard.
            if isinstance(frame, (AggregatedTextFrame, AggregatedTextProgressFrame)):
                await self._open.wait()
                if self.dropping:
                    return
            await original_push(frame, *args, **kwargs)

        output_transport.write_audio_frame = gated_write
        output_transport.push_frame = gated_push

    def pause(self) -> None:
        self._open.clear()

    def resume(self) -> None:
        self._resumed_at = time.monotonic()
        self._open.set()

    def cancel(self) -> None:
        """The reply was interrupted: drop its queued audio, then play normally again."""
        self._dropping_until = time.monotonic() + self._flush_timeout_secs
        self._open.set()

    async def on_push_frame(self, data: FramePushed) -> None:
        # The output transport has handled the interruption and cleared its queue.
        if (
            self.dropping
            and isinstance(data.frame, InterruptionFrame)
            and data.source is self._output
        ):
            self._dropping_until = 0.0


class BotLineObserver(BaseObserver):
    """Keeps the text of the bot's current reply."""

    def __init__(self):
        super().__init__()
        self.text = ""

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not data.first_push:
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            self.text = ""
        elif isinstance(frame, LLMTextFrame):
            self.text += frame.text
        elif isinstance(frame, TTSSpeakFrame):
            self.text = frame.text


class JevBargeInStrategy(BaseUserTurnStartStrategy):
    """Start every caller turn on VAD; while the bot speaks, pause it and let Jev decide."""

    def __init__(
        self,
        *,
        gate: OutputGate,
        classifier: BaseClassifier,
        question: ChoiceQuestion,
        bot_line: BotLineObserver,
        jev_timeout_secs: float,
        max_decision_secs: float,
        keep_talking_secs: float,
        interrupt_at: float = 0.6,
        doubt_at: float = 0.5,
        bot_line_chars: int = 300,
        transcript_wait_secs: float = 1.5,
        emit: Emit | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._question = question
        self._bot_line_chars = bot_line_chars
        self._transcript_wait_secs = transcript_wait_secs
        self._interrupt_at = interrupt_at
        self._doubt_at = doubt_at
        self._gate = gate
        self._classifier = classifier
        self._bot_line = bot_line
        self._jev_timeout_secs = jev_timeout_secs
        self._max_decision_secs = max_decision_secs
        self._keep_talking_secs = keep_talking_secs
        self._emit = emit or (lambda *args, **kwargs: None)
        self._interrupt: Callable[[], Awaitable[None]] | None = None
        self._ask_caller: Callable[[], Awaitable[None]] | None = None
        self._guard: BargeInTurnGuard | None = None
        self._bot_speaking = False
        self._decision: asyncio.Task | None = None
        self._transcript_wait: asyncio.Task | None = None
        self._reset_speech()
        self._barge_turn = False  # this turn began while the bot was speaking
        self._decided = True  # Jev has decided, and any backchannel text is dropped
        self.backchannel_turn = False  # the turn was only a backchannel
        self._discard_next_final = False

    def set_interrupt(self, interrupt: Callable[[], Awaitable[None]]) -> None:
        """How to interrupt the bot: the user aggregator's broadcast_interruption."""
        self._interrupt = interrupt

    def set_ask_caller(self, ask_caller: Callable[[], Awaitable[None]]) -> None:
        """How to ask "did you say something?" when Jev is unsure."""
        self._ask_caller = ask_caller

    @property
    def turn_end_allowed(self) -> bool:
        return not self._barge_turn or self._decided

    def _reset_speech(self) -> None:
        self._text = ""
        self._final_received = False
        self._vad_active = False
        self._speech_started = 0.0

    async def process_frame(self, frame: Frame):
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._on_speech_started()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_active = False
        elif isinstance(frame, InterimTranscriptionFrame):
            if self._deciding:
                self._text = frame.text
        elif isinstance(frame, TranscriptionFrame):
            await self._on_final_transcript(frame.text)

    @property
    def _deciding(self) -> bool:
        return self._decision is not None and not self._decision.done()

    async def _on_speech_started(self) -> None:
        if self._deciding:
            self._vad_active = True  # Still the same barge-in; the caller resumed speaking.
            return
        if not self._bot_speaking:
            self._barge_turn, self.backchannel_turn = False, False
            await self.report_user_turn_started()
            return
        self._reset_speech()
        self._vad_active = True
        self._speech_started = time.monotonic()
        self._barge_turn, self._decided, self.backchannel_turn = True, False, False
        self._gate.pause()
        self._emit("barge", action="pause")
        # Start the caller's turn now, so Smart Turn follows it from the start, but
        # don't interrupt: the bot is only paused until Jev decides.
        await self.report_user_turn_started(enable_interruptions=False)
        self._decision = asyncio.create_task(self._decide())

    async def _on_final_transcript(self, text: str) -> None:
        # The aggregator has already added this transcript to the turn.
        if self._discard_next_final:
            self._discard_next_final = False
            await self.trigger_reset_aggregation()
            await self._mark_decided()
            return
        if self._deciding:
            self._text = text
            self._final_received = True

    async def _decide(self) -> None:
        start = time.monotonic()
        while True:
            now = time.monotonic()
            if now - start >= self._max_decision_secs:
                return await self._stop(start, "no decision in time")
            if self._vad_active and now - self._speech_started >= self._keep_talking_secs:
                return await self._stop(start, "caller kept talking")
            # Ask Jev once the caller has finished this bit of speech, using whatever
            # the fast stream has transcribed by then, so Jev judges the whole phrase
            # ("what details?") rather than its first word ("what").
            if not self._vad_active and self._text.strip():
                return await self._ask_jev(start)
            if self._final_received and not self._text.strip():
                return await self._resume(start, "no words")
            await asyncio.sleep(0.02)

    async def _ask_jev(self, start: float) -> None:
        text = self._text.strip()
        bot_line = self._bot_line.text.strip()[-self._bot_line_chars :]
        state = f'Bot was saying: "{bot_line}"\nCaller said: "{text}"'
        asked = time.monotonic()
        p_interrupt: float | None = None
        try:
            results = await asyncio.wait_for(
                self._classifier.ask(state, {"kind": self._question}), self._jev_timeout_secs
            )
            p_interrupt = results["kind"].probabilities.get("interruption", 0.0)
            label = self._outcome(p_interrupt)
        except TimeoutError:
            label = "timeout"
        except Exception as e:  # Any Jev failure is treated like a timeout: stop, never talk over.
            logger.warning(f"Jev decision failed: {e}")
            label = "error"
        self._emit(
            "jev_decision",
            text=text,
            label=label,
            p_interruption=None if p_interrupt is None else round(p_interrupt, 2),
            ms=round((time.monotonic() - asked) * 1000),
            timed_out=p_interrupt is None,
        )
        if label == "ignore":
            await self._resume(start, "backchannel")
        elif label == "doubt":
            await self._stop(start, "doubt", ask_caller=True)
        else:
            await self._stop(start, "interruption" if label == "interrupt" else f"jev {label}")

    def _outcome(self, p_interrupt: float) -> str:
        if p_interrupt >= self._interrupt_at:
            return "interrupt"
        if p_interrupt >= self._doubt_at:
            return "doubt"
        return "ignore"

    async def _resume(self, start: float, reason: str) -> None:
        self._gate.resume()
        self.backchannel_turn = True
        pause_ms = round((time.monotonic() - start) * 1000)
        self._emit("barge", action="resume", reason=reason, pause_ms=pause_ms)
        if self._final_received:
            await self.trigger_reset_aggregation()
            await self._mark_decided()
        else:
            # The backchannel's transcript is still on its way; drop it when it lands.
            self._discard_next_final = True
            self._transcript_wait = asyncio.create_task(self._stop_waiting_for_transcript())

    async def _stop_waiting_for_transcript(self) -> None:
        """Safety net: if the transcript never arrives, let the turn end anyway."""
        await asyncio.sleep(self._transcript_wait_secs)
        if self._discard_next_final:
            self._discard_next_final = False
            await self.trigger_reset_aggregation()
            await self._mark_decided()

    async def _stop(self, start: float, reason: str, ask_caller: bool = False) -> None:
        if self._interrupt:
            await self._interrupt()
        self._gate.cancel()
        if ask_caller:
            # Unclear words: drop them so the turn ends empty, then ask instead of guessing.
            if self._final_received:
                await self.trigger_reset_aggregation()
            else:
                self._discard_next_final = True
                self._transcript_wait = asyncio.create_task(self._stop_waiting_for_transcript())
            if self._ask_caller:
                await self._ask_caller()
        self._emit(
            "barge",
            action="stop",
            reason=reason,
            time_to_stop_ms=round((time.monotonic() - start) * 1000),
        )
        await self._mark_decided()

    async def _mark_decided(self) -> None:
        self._decided = True
        if self._guard:
            await self._guard.allow_turn_end()

    async def handle_user_turn_stopped(self):
        await super().handle_user_turn_stopped()
        self._barge_turn, self._decided = False, True

    async def cleanup(self):
        for task in (self._decision, self._transcript_wait):
            if task and not task.done():
                task.cancel()
        self._gate.resume()
        await super().cleanup()


class BargeInTurnGuard(BaseUserTurnStopStrategy):
    """Holds a barge-in turn open until Jev has decided; agrees at once for other turns.

    With Pipecat's default stop mode, a turn ends only when every stop strategy has
    agreed, and new caller speech clears earlier agreements. So the guard agrees
    when a turn starts, after each stretch of speech, and when the decision lands.
    """

    def __init__(self, barge_in: JevBargeInStrategy, **kwargs):
        super().__init__(**kwargs)
        self._barge_in = barge_in
        barge_in._guard = self

    async def allow_turn_end(self) -> None:
        if self._barge_in.turn_end_allowed:
            await self.report_user_turn_stopped()

    async def handle_user_turn_started(self):
        await super().handle_user_turn_started()
        await self.allow_turn_end()

    async def process_frame(self, frame: Frame):
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.allow_turn_end()
