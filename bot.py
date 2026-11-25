"""Step 1: bare voice pipeline.

Browser mic → Sarvam (speech-to-text) → Smart Turn (end of turn) → Gemini →
ElevenLabs (text-to-speech) → browser speaker. The bot speaks first, as on an
outbound call. Every turn's latency is recorded for the dashboard.

Run:  .venv/bin/python bot.py -t webrtc
      python3 -m http.server 8000   then open http://localhost:8000/web/call.html
"""

import asyncio
from datetime import date

from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.classifiers.typesafe.jev.classifier import JevClassifier
from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.sarvam.stt import SarvamRealtimeSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

import config
from agent.borrowers import load_borrowers
from agent.conversation import build_llm, build_tools, opening_line, system_prompt
from agent.tools import CallState, save_record_without_end_call
from recorder import EventRecorder
from voice.barge_in import (
    BargeInTurnGuard,
    BotLineObserver,
    JevBargeInStrategy,
    OutputGate,
    interrupt_question,
)
from voice.stt_preroll import with_pre_roll

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


def build_services(
    system_prompt: str,
) -> tuple[SarvamRealtimeSTTService, GoogleLLMService, ElevenLabsTTSService]:
    """Create the three provider services from config; the LLM runs the agent's prompt."""
    stt_kwargs = {}
    stt_class = SarvamRealtimeSTTService
    if config.STT_PRE_ROLL_SECS > 0:
        stt_class = with_pre_roll(SarvamRealtimeSTTService)
        stt_kwargs["pre_roll_secs"] = config.STT_PRE_ROLL_SECS
        stt_kwargs["keepalive_secs"] = config.STT_KEEPALIVE_SECS
        stt_kwargs["keepalive_silence_secs"] = config.STT_KEEPALIVE_SILENCE_SECS
    stt = stt_class(
        api_key=config.env("SARVAM_API_KEY"),
        endpointing="manual",
        settings=SarvamRealtimeSTTService.Settings(
            model=config.SARVAM_MODEL,
            mode=config.SARVAM_MODE,
            stream_type=config.SARVAM_STREAM_TYPE,
            language=config.SARVAM_LANGUAGE,
        ),
        **stt_kwargs,
    )
    llm = build_llm(system_prompt)
    tts = ElevenLabsTTSService(
        api_key=config.env("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(
            voice=config.ELEVENLABS_VOICE_ID,
            model=config.ELEVENLABS_MODEL,
            language=config.ELEVENLABS_LANGUAGE,
            speed=config.ELEVENLABS_SPEED,
        ),
    )
    return stt, llm, tts


async def warm_up_llm(llm: GoogleLLMService) -> None:
    """Send one tiny request through the LLM service's own client to open its connection."""
    try:
        await llm.run_inference(
            LLMContext([{"role": "user", "content": "Reply with OK."}]), max_tokens=64
        )
        logger.debug("Gemini warm-up done")
    except Exception as e:  # Warm-up is best effort; the call works without it.
        logger.warning(f"Gemini warm-up failed: {e}")


async def warm_up_jev(classifier: JevClassifier, question) -> None:
    """Ask Jev one real question to open its connection (the gateway refuses the usual warm-up)."""
    try:
        await classifier.ask("Caller said: okay", {"kind": question})
        logger.debug("Jev warm-up done")
    except Exception as e:  # Best effort, like the Gemini warm-up.
        logger.warning(f"Jev warm-up failed: {e}")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    today = date.today()
    borrower = load_borrowers(config.BORROWERS_FILE, today)[config.VOICE_BORROWER_ID]
    stt, llm, tts = build_services(system_prompt(borrower, today))
    recorder = EventRecorder(
        config.EVENTS_DIR, borrower_id=borrower.borrower_id, vad_stop_secs=config.VAD_STOP_SECS
    )
    state = CallState(call_id=recorder.call_id, borrower=borrower)

    vad = SileroVADAnalyzer(
        params=VADParams(
            confidence=config.VAD_CONFIDENCE,
            start_secs=config.VAD_START_SECS,
            stop_secs=config.VAD_STOP_SECS,
            min_volume=config.VAD_MIN_VOLUME,
        )
    )

    # Barge-in: the gate pauses the bot's audio, Jev decides resume or stop.
    gate = OutputGate(
        flush_timeout_secs=config.GATE_FLUSH_TIMEOUT_SECS,
        resume_log_secs=config.GATE_RESUME_LOG_SECS,
    )
    gate.install(transport.output())
    jev_question = interrupt_question(config.JEV_QUESTION, config.JEV_OPTIONS)
    bot_line = BotLineObserver()
    jev = JevClassifier(api_key=config.env("TYPESAFE_API_KEY"), base_url=config.JEV_BASE_URL)
    barge_in = JevBargeInStrategy(
        gate=gate,
        classifier=jev,
        question=jev_question,
        bot_line=bot_line,
        jev_timeout_secs=config.JEV_TIMEOUT_SECS,
        max_decision_secs=config.BARGE_MAX_DECISION_SECS,
        keep_talking_secs=config.BARGE_KEEP_TALKING_SECS,
        interrupt_at=config.JEV_INTERRUPT_AT,
        doubt_at=config.JEV_DOUBT_AT,
        bot_line_chars=config.JEV_BOT_LINE_CHARS,
        transcript_wait_secs=config.BARGE_TRANSCRIPT_WAIT_SECS,
        emit=recorder.emit,
    )
    # Smart Turn v3.2 ends the caller's turn, with a shorter silence fallback than
    # Pipecat's default.
    smart_turn = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(stop_secs=config.SMART_TURN_STOP_SECS)
    )

    end_reason = "client_disconnected"

    async def on_end_call(success: bool) -> None:
        """end_call saved the outcome: speak its goodbye, then hang up once it has played."""
        nonlocal end_reason
        if success:
            end_reason = "end_call"
            await worker.queue_frames([TTSSpeakFrame(state.goodbye or ""), EndFrame()])

    async def on_slow_tool() -> None:
        """A lookup is taking a while: say the filler line instead of leaving silence."""
        recorder.emit("filler", text=config.FILLER_LINE)
        await worker.queue_frames([TTSSpeakFrame(config.FILLER_LINE, append_to_context=False)])

    context = LLMContext(tools=build_tools(state, recorder.emit, on_end_call, on_slow_tool))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad,
            user_idle_timeout=config.USER_IDLE_SECS,
            # Replaces Pipecat's default start strategies, one of which would start a
            # turn on any transcript while the bot speaks and so turn "haan" into an
            # interruption.
            user_turn_strategies=UserTurnStrategies(
                start=[barge_in],
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn),
                    BargeInTurnGuard(barge_in),
                ],
            ),
        ),
    )

    barge_in.set_interrupt(user_aggregator.broadcast_interruption)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    latency = UserBotLatencyObserver()
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[latency, recorder.observer, bot_line, gate],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    async def ask_caller() -> None:
        await worker.queue_frames([TTSSpeakFrame(config.DOUBT_LINE)])

    barge_in.set_ask_caller(ask_caller)

    @latency.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        recorder.turn_latency(breakdown)

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        if not message.content and barge_in.backchannel_turn:
            return  # A "haan" while the bot spoke: not a turn of the conversation.
        recorder.user_said(message.content or "")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        recorder.bot_said(message.content, message.interrupted)

    idle_count = 0

    @user_aggregator.event_handler("on_user_turn_started")
    async def on_user_turn_started(aggregator, strategy):
        nonlocal idle_count
        idle_count = 0

    @user_aggregator.event_handler("on_user_turn_idle")
    async def on_user_turn_idle(aggregator):
        nonlocal idle_count, end_reason
        idle_count += 1
        recorder.emit("user_idle", n=idle_count)
        if idle_count == 1:
            await worker.queue_frames([TTSSpeakFrame(config.CHECK_IN_LINE)])
        else:
            end_reason = "no_response"
            await worker.queue_frames([TTSSpeakFrame(config.GOODBYE_LINE), EndFrame()])

    warmups: list[asyncio.Task] = []

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Call started: {recorder.call_id}")
        await recorder.start()
        # Outbound call: the bot speaks first, with a fixed line (no LLM wait).
        context.add_message({"role": "user", "content": config.CALL_CONNECTED_NOTE})
        await worker.queue_frames([TTSSpeakFrame(opening_line(borrower))])
        # While the opening plays, open the Gemini and Jev connections so the first
        # real reply and the first barge-in don't pay the setup cost.
        warmups.append(asyncio.create_task(warm_up_llm(llm)))
        warmups.append(asyncio.create_task(warm_up_jev(jev, jev_question)))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    try:
        await runner.run()
    finally:
        for task in warmups:
            task.cancel()
        await jev.cleanup()
        await save_record_without_end_call(state, end_reason)
        await recorder.stop(end_reason)
        logger.info(f"Call ended: {recorder.call_id} ({recorder.turn} turns, {end_reason})")


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point used by Pipecat's runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
