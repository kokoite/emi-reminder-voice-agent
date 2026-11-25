"""Pre-roll for any Pipecat speech-to-text service.

Our VAD needs about 0.2 s of speech before it signals "speech started", so the
signal always arrives after the first part of a word. Services that transcribe
only between the start and end signals (Sarvam in manual mode) would lose it:
short answers like "yes" come back empty and sentences lose their first word.

With pre-roll, audio is sent only while the caller is speaking. During silence
the last few hundred milliseconds are held, and right after "speech started"
they are sent first. This keeps whole words and means we don't pay to
transcribe silence or the time the bot is talking.

While holding audio, a short burst of silence is sent every `keepalive_secs`, so
services that close idle sessions (Sarvam: after 60 s without audio) stay open.

Use: `with_pre_roll(SomeSTTService)(..., pre_roll_secs=0.5)`.
"""

import time
from collections.abc import AsyncGenerator

from pipecat.frames.frames import Frame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection


class PreRollMixin:
    """Gate a speech-to-text service on our VAD, replaying recent audio at speech start."""

    def __init__(
        self,
        *args,
        pre_roll_secs: float,
        keepalive_secs: float = 15.0,
        keepalive_silence_secs: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._pre_roll_secs = pre_roll_secs
        self._keepalive_secs = keepalive_secs
        self._keepalive_silence_secs = keepalive_silence_secs
        self._last_sent = time.monotonic()
        self._held = bytearray()
        self._speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # The service handles the signal first (e.g. sends "speech_start"), so the
        # held audio reaches it inside the speech window.
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame) and not self._speaking:
            self._speaking = True
            held, self._held = bytes(self._held), bytearray()
            if held:
                await self.process_generator(super().run_stt(held))
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if self._speaking:
            self._last_sent = time.monotonic()
            async for frame in super().run_stt(audio):
                yield frame
            return
        if time.monotonic() - self._last_sent >= self._keepalive_secs:
            self._last_sent = time.monotonic()
            silence = bytes(int(self._keepalive_silence_secs * self.send_sample_rate) * 2)
            async for frame in super().run_stt(silence):
                yield frame
        self._held.extend(audio)
        limit = int(self._pre_roll_secs * self.send_sample_rate) * 2  # 16-bit mono
        if len(self._held) > limit:
            del self._held[: len(self._held) - limit]
        yield None


def with_pre_roll(service_cls: type) -> type:
    """Return a subclass of `service_cls` with pre-roll. Works with any STTService."""
    return type(f"PreRoll{service_cls.__name__}", (PreRollMixin, service_cls), {})
