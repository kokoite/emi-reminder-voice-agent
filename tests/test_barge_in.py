import asyncio
from types import SimpleNamespace

from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from voice.barge_in import BotLineObserver, JevBargeInStrategy, OutputGate, interrupt_question


class FakeJev:
    """Answers with a fixed probability that the caller is interrupting."""

    PROBABILITY = {"interruption": 0.9, "backchannel": 0.1}

    def __init__(self, choice: str = "backchannel", delay: float = 0.01, p: float | None = None):
        self.p = p if p is not None else self.PROBABILITY[choice]
        self.delay, self.asked = delay, []

    async def ask(self, state, questions):
        self.asked.append(state)
        await asyncio.sleep(self.delay)
        probabilities = {"interruption": self.p, "backchannel": 1 - self.p}
        return {"kind": SimpleNamespace(probabilities=probabilities)}


class Strategy(JevBargeInStrategy):
    """Records what the strategy reports instead of driving a real aggregator."""

    def __init__(self, jev: FakeJev, **kwargs):
        self.gate = OutputGate()
        self.events: list[tuple] = []
        self.turns_started: list[dict] = []
        self.resets = 0
        self.interrupts = 0
        self.asked_caller = 0
        super().__init__(
            gate=self.gate,
            classifier=jev,
            question=interrupt_question("Interrupting?", {"interruption": "", "backchannel": ""}),
            bot_line=BotLineObserver(),
            transcript_wait_secs=kwargs.pop("transcript_wait_secs", 1.5),
            jev_timeout_secs=kwargs.pop("jev_timeout_secs", 0.2),
            max_decision_secs=kwargs.pop("max_decision_secs", 0.6),
            keep_talking_secs=kwargs.pop("keep_talking_secs", 0.4),
            emit=lambda event, **data: self.events.append((event, data)),
        )
        self.set_interrupt(self._record_interrupt)
        self.set_ask_caller(self._record_ask_caller)

    async def _record_interrupt(self):
        self.interrupts += 1

    async def _record_ask_caller(self):
        self.asked_caller += 1

    async def report_user_turn_started(self, **kwargs):
        self.turns_started.append(kwargs)

    async def trigger_reset_aggregation(self):
        self.resets += 1

    def actions(self) -> list[str]:
        return [data.get("action", data.get("label")) for _, data in self.events]


def _text(text: str, final: bool = False):
    cls = TranscriptionFrame if final else InterimTranscriptionFrame
    return cls(text=text, user_id="", timestamp="")


async def _barge_in(strategy: Strategy, *frames, settle: float = 0.15):
    await strategy.process_frame(BotStartedSpeakingFrame())
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    for frame in frames:
        await strategy.process_frame(frame)
    await asyncio.sleep(settle)


def test_speech_while_the_bot_is_silent_starts_the_turn_at_once():
    strategy = Strategy(FakeJev())
    asyncio.run(strategy.process_frame(VADUserStartedSpeakingFrame()))
    assert strategy.turns_started == [{}]  # a normal turn, with interruptions
    assert not strategy.gate.paused
    assert strategy.turn_end_allowed


def test_backchannel_pauses_then_resumes_and_is_kept_out_of_the_turn():
    strategy = Strategy(FakeJev("backchannel"))

    async def run():
        await _barge_in(strategy, _text("haan"), VADUserStoppedSpeakingFrame())
        assert strategy.gate.paused is False  # resumed
        await strategy.process_frame(_text("haan", final=True))  # arrives after the decision

    asyncio.run(run())
    assert strategy.actions() == ["pause", "ignore", "resume"]
    assert strategy.turns_started == [{"enable_interruptions": False}]  # turn, no interruption
    assert strategy.interrupts == 0
    assert strategy.resets == 1  # the late "haan" transcript was dropped
    assert strategy.backchannel_turn and strategy.turn_end_allowed


def test_interruption_stops_the_bot_and_starts_the_turn():
    strategy = Strategy(FakeJev("interruption"))
    asyncio.run(_barge_in(strategy, _text("ruko ruko"), VADUserStoppedSpeakingFrame()))
    assert strategy.actions() == ["pause", "interrupt", "stop"]
    assert strategy.turns_started == [{"enable_interruptions": False}]
    assert strategy.interrupts == 1  # the bot was interrupted only after Jev decided
    assert not strategy.gate.paused
    assert strategy.turn_end_allowed and not strategy.backchannel_turn


def test_slow_jev_counts_as_an_interruption():
    strategy = Strategy(FakeJev("backchannel", delay=1.0), jev_timeout_secs=0.05)
    asyncio.run(_barge_in(strategy, _text("haan"), VADUserStoppedSpeakingFrame()))
    assert strategy.actions() == ["pause", "timeout", "stop"]
    assert strategy.interrupts == 1


def test_a_cough_with_no_words_resumes():
    strategy = Strategy(FakeJev())
    asyncio.run(_barge_in(strategy, VADUserStoppedSpeakingFrame(), _text("", final=True)))
    assert strategy.actions() == ["pause", "resume"]
    assert strategy.events[-1][1]["reason"] == "no words"
    assert strategy.interrupts == 0
    assert strategy.turn_end_allowed


def test_caller_who_keeps_talking_gets_the_turn_even_without_words_yet():
    strategy = Strategy(FakeJev(), keep_talking_secs=0.1)
    asyncio.run(_barge_in(strategy, settle=0.3))
    assert strategy.actions() == ["pause", "stop"]
    assert strategy.events[-1][1]["reason"] == "caller kept talking"


async def _no_push(frame, *args, **kwargs):
    pass


def test_gate_holds_writes_until_resumed():
    written = []

    async def write(frame):
        written.append(frame)
        return True

    async def run():
        gate = OutputGate()
        transport = SimpleNamespace(write_audio_frame=write, push_frame=_no_push)
        gate.install(transport)
        gate.pause()
        held = asyncio.create_task(transport.write_audio_frame("chunk-1"))
        await asyncio.sleep(0.05)
        assert written == []  # paused: nothing played
        gate.resume()
        await held
        assert written == ["chunk-1"]  # resumed from the same chunk

    asyncio.run(run())


def test_after_an_interruption_old_audio_is_dropped_until_the_output_clears_it():
    written = []

    async def write(frame):
        written.append(frame)
        return True

    async def run():
        gate = OutputGate(flush_timeout_secs=5)
        transport = SimpleNamespace(write_audio_frame=write, push_frame=_no_push)
        gate.install(transport)
        gate.pause()
        held = asyncio.create_task(transport.write_audio_frame("old-1"))
        await asyncio.sleep(0.02)
        gate.cancel()
        await held
        await transport.write_audio_frame("old-2")  # still queued from the old reply
        assert written == []  # neither old chunk plays: no burst

        interruption = SimpleNamespace(frame=InterruptionFrame(), source=transport, first_push=True)
        await gate.on_push_frame(interruption)  # the output transport has cleared its queue
        await transport.write_audio_frame("new-reply")
        assert written == ["new-reply"]

    asyncio.run(run())


def test_drop_mode_ends_on_its_own_as_a_safety_net():
    written = []

    async def write(frame):
        written.append(frame)
        return True

    async def run():
        gate = OutputGate(flush_timeout_secs=0.05)
        transport = SimpleNamespace(write_audio_frame=write, push_frame=_no_push)
        gate.install(transport)
        gate.cancel()
        await asyncio.sleep(0.1)  # interruption never observed
        await transport.write_audio_frame("new-reply")
        assert written == ["new-reply"]

    asyncio.run(run())


def test_words_wait_while_paused_and_old_words_are_dropped():
    from pipecat.frames.frames import TTSTextFrame

    pushed = []

    async def push(frame, *args, **kwargs):
        pushed.append(frame.text if isinstance(frame, TTSTextFrame) else type(frame).__name__)

    async def write(frame):
        return True

    async def run():
        gate = OutputGate(flush_timeout_secs=5)
        transport = SimpleNamespace(write_audio_frame=write, push_frame=push)
        gate.install(transport)

        await transport.push_frame(TTSTextFrame(text="Your", text_type="word"))
        gate.pause()
        emi = TTSTextFrame(text="EMI", text_type="word")
        held = asyncio.create_task(transport.push_frame(emi))
        await asyncio.sleep(0.02)
        assert pushed == ["Your"]  # "EMI" isn't heard yet while paused
        gate.cancel()  # the caller interrupted
        await held
        assert pushed == ["Your"]  # so it never counts as spoken
        await transport.push_frame(InterruptionFrame())  # other frames still pass
        assert pushed == ["Your", "InterruptionFrame"]

        gate.cancel()  # a sentence announcement from the interrupted reply
        sentence = AggregatedTextFrame(text="We can't proceed", text_type="sentence")
        await transport.push_frame(sentence)
        assert pushed == ["Your", "InterruptionFrame"]  # never reaches the call page

    asyncio.run(run())


def test_turn_cannot_end_until_jev_decides():
    strategy = Strategy(FakeJev("backchannel", delay=0.1))

    async def run():
        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(_text("haan"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        assert not strategy.turn_end_allowed  # Jev still deciding: the guard holds the turn
        await asyncio.sleep(0.2)
        assert not strategy.turn_end_allowed  # resumed, but the "haan" transcript isn't in yet
        await strategy.process_frame(_text("haan", final=True))
        assert strategy.turn_end_allowed  # dropped: now the empty turn may end

    asyncio.run(run())


def test_backchannel_turn_ends_even_if_its_transcript_never_arrives():
    strategy = Strategy(FakeJev("backchannel"), transcript_wait_secs=0.05)
    asyncio.run(_barge_in(strategy, _text("haan"), VADUserStoppedSpeakingFrame(), settle=0.2))
    assert strategy.turn_end_allowed
    assert strategy.resets == 1


def test_guard_agrees_at_once_for_normal_turns_and_after_the_decision_for_barge_ins():
    from voice.barge_in import BargeInTurnGuard

    strategy = Strategy(FakeJev("interruption", delay=0.05))
    guard = BargeInTurnGuard(strategy)
    agreed = []

    async def record_agree(**kwargs):
        agreed.append(True)

    guard.report_user_turn_stopped = record_agree

    async def run():
        await guard.handle_user_turn_started()  # normal turn
        assert agreed == [True]

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())  # barge-in turn
        await guard.handle_user_turn_started()
        await strategy.process_frame(_text("wait, what amount?"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await guard.process_frame(VADUserStoppedSpeakingFrame())
        assert agreed == [True]  # held while Jev decides
        await asyncio.sleep(0.15)
        assert agreed == [True, True]  # agreed as soon as the decision landed

    asyncio.run(run())


def test_jev_is_asked_on_the_whole_phrase_once_the_caller_stops():
    jev = FakeJev("interruption")
    strategy = Strategy(jev)

    async def run():
        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(_text("What"))  # fast-stream partial, caller still speaking
        await asyncio.sleep(0.1)
        assert jev.asked == []  # not asked on the first word
        await strategy.process_frame(_text("What details?"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.1)

    asyncio.run(run())
    assert len(jev.asked) == 1 and jev.asked[0].endswith('Caller said: "What details?"')
    assert strategy.actions() == ["pause", "interrupt", "stop"]


def test_doubt_stops_the_bot_drops_the_words_and_asks_the_caller():
    strategy = Strategy(FakeJev(p=0.55))
    asyncio.run(_barge_in(strategy, _text("hmm haan"), VADUserStoppedSpeakingFrame()))
    assert strategy.actions() == ["pause", "doubt", "stop"]
    assert strategy.interrupts == 1
    assert strategy.asked_caller == 1  # "Sorry, did you say something?"
    assert strategy.events[-1][1]["reason"] == "doubt"


def test_below_the_doubt_range_is_ignored_and_the_bot_resumes():
    strategy = Strategy(FakeJev(p=0.45))
    asyncio.run(_barge_in(strategy, _text("haan"), VADUserStoppedSpeakingFrame()))
    assert strategy.actions() == ["pause", "ignore", "resume"]
    assert strategy.interrupts == 0 and strategy.asked_caller == 0
