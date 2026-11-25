from types import SimpleNamespace

import pytest
from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.transcriptions.language import Language

from voice.language import HindiSwitch

# Async tests run through anyio's pytest plugin (anyio comes with pipecat).
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def pushed(text, language):
    return SimpleNamespace(frame=TranscriptionFrame(text, "caller", "t", language))


async def test_the_note_is_added_once_on_the_first_hindi_utterance():
    context, events = LLMContext(), []
    switch = HindiSwitch(context, "NOTE", lambda t, **d: events.append(t))
    await switch.on_push_frame(pushed("Yes, this is Ravi.", Language.EN_IN))
    assert context.get_messages() == []
    await switch.on_push_frame(pushed("Haan, main Ravi bol raha hoon.", Language.HI_IN))
    await switch.on_push_frame(pushed("Theek hai.", Language.HI_IN))
    assert context.get_messages() == [{"role": "user", "content": "NOTE"}]
    assert events == ["hindi_style_on"]


async def test_asking_for_hindi_in_english_also_switches():
    context = LLMContext()
    switch = HindiSwitch(context, "NOTE", lambda t, **d: None)
    await switch.on_push_frame(pushed("Can we speak in Hindi please?", Language.EN_IN))
    assert context.get_messages() == [{"role": "user", "content": "NOTE"}]


def test_wants_hindi():
    from voice.language import wants_hindi

    assert wants_hindi("I'm more comfortable in Hindi.")
    assert wants_hindi("मैं January में करूँगा।")
    assert wants_hindi("Theek hai", "hi-IN")
    assert not wants_hindi("Yes, this is Ravi.", Language.EN_IN)
