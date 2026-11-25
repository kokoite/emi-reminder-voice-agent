"""Hindi in a call: telling it apart, and switching the agent's Hindi style on when it starts.

The system prompt stays in English. A Hindi example in it pulled English calls into Hindi,
and without one the agent mirrors the caller's Latin-letter Hindi, which the voice reads
with an English accent. So code adds the Hindi-style note only once Sarvam tags a caller's
utterance as Hindi.
"""

import re
from collections.abc import Callable
from typing import Any

from pipecat.frames.frames import TranscriptionFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext


def is_hindi(text: str) -> bool:
    """True if the text has Devanagari letters; the agent writes Hindi in Devanagari."""
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def is_hindi_language(language: Any) -> bool:
    """True for a Hindi language tag from speech-to-text ("hi", "hi-IN", Language.HI_IN)."""
    code = getattr(language, "value", language)
    return isinstance(code, str) and code.lower().startswith("hi")


_ASKS_FOR_HINDI = re.compile(r"\bhindi\b|हिंदी|हिन्दी", re.IGNORECASE)


def wants_hindi(text: str, language: Any = None) -> bool:
    """The caller speaks Hindi (Sarvam's tag, or Devanagari text) or asks for Hindi."""
    return is_hindi_language(language) or is_hindi(text) or bool(_ASKS_FOR_HINDI.search(text))


class HindiSwitch(BaseObserver):
    """The first time the caller speaks or asks for Hindi, add the Hindi-style note."""

    def __init__(self, context: LLMContext, note: str, emit: Callable[..., None]):
        super().__init__()
        self._context = context
        self._note = note
        self._emit = emit
        self.switched = False

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if self.switched or not isinstance(frame, TranscriptionFrame):
            return
        if not wants_hindi(frame.text, frame.language):
            return
        self.switched = True
        self._context.add_message({"role": "user", "content": self._note})
        self._emit("hindi_style_on", language=str(frame.language), text=frame.text)
