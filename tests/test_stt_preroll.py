import asyncio

from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from stt_preroll import with_pre_roll

RATE = 16000


class FakeSTT:
    """Stands in for a Pipecat STT service: records what reaches it, in order."""

    def __init__(self):
        self.log: list[tuple[str, bytes]] = []
        self.send_sample_rate = RATE

    async def process_frame(self, frame, direction):
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self.log.append(("speech_start", b""))
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.log.append(("speech_end", b""))

    async def run_stt(self, audio):
        self.log.append(("audio", audio))
        yield None

    async def process_generator(self, generator):
        async for _ in generator:
            pass


def _audio(secs: float, byte: int) -> bytes:
    return bytes([byte]) * (int(secs * RATE) * 2)


async def _feed(stt, audio: bytes):
    async for _ in stt.run_stt(audio):
        pass


def test_silence_is_held_and_replayed_right_after_speech_start():
    stt = with_pre_roll(FakeSTT)(pre_roll_secs=0.5)

    async def run():
        await _feed(stt, _audio(1.0, 1))  # silence: nothing should be sent
        await _feed(stt, _audio(0.2, 2))  # start of the word, before the VAD fires
        await stt.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await _feed(stt, _audio(0.1, 3))  # rest of the word
        await stt.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        await _feed(stt, _audio(1.0, 4))  # silence again: held, not sent

    asyncio.run(run())
    kinds = [kind for kind, _ in stt.log]
    assert kinds == ["speech_start", "audio", "audio", "speech_end"]

    replay = stt.log[1][1]
    assert len(replay) == int(0.5 * RATE) * 2  # exactly the pre-roll, not all the silence
    assert replay.endswith(_audio(0.2, 2))  # and it includes the start of the word
    assert stt.log[2][1] == _audio(0.1, 3)


def test_nothing_is_sent_while_nobody_speaks():
    stt = with_pre_roll(FakeSTT)(pre_roll_secs=0.5)
    asyncio.run(_feed(stt, _audio(3.0, 1)))
    assert stt.log == []


def test_keepalive_sends_a_little_silence_while_holding(monkeypatch):
    import stt_preroll

    clock = [1000.0]
    monkeypatch.setattr(stt_preroll.time, "monotonic", lambda: clock[0])
    stt = with_pre_roll(FakeSTT)(pre_roll_secs=0.5, keepalive_secs=15)

    async def run():
        for _ in range(40):  # 40 s of silence, fed in 1 s chunks
            clock[0] += 1
            await _feed(stt, _audio(1.0, 1))

    asyncio.run(run())
    sent = [audio for kind, audio in stt.log if kind == "audio"]
    assert len(sent) == 2  # at 15 s and 30 s
    assert all(a == bytes(int(0.1 * RATE) * 2) for a in sent)  # 0.1 s of zeros, not the held audio
