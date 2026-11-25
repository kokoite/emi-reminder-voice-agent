"""The EMI reminder agent: system prompt, opening line and tools, on Pipecat's Gemini service.

`run_text_call` runs one whole call as text, with no audio: our own testing (chat.py) and
the evals use it. The voice bot uses the same prompt, opening and tools.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date

from google.genai.types import HttpOptions
from loguru import logger
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import EndFrame, FunctionCallResultProperties, LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.workers.runner import WorkerRunner

import config
from borrowers import Borrower
from recorder import EventRecorder
from tools import TOOL_SCHEMAS, CallState, run_tool, save_record_without_end_call

# Gets the bot's last line, returns the caller's reply, or None when the caller hangs up.
Caller = Callable[[str], Awaitable[str | None]]


@dataclass
class TextCallResult:
    call_id: str
    ended_by: str  # end_call, caller_hung_up, max_turns or timeout
    transcript: list[dict] = field(default_factory=list)  # {"role": "bot"|"caller", "text"}
    outcome: dict | None = None  # the outcome record, when end_call saved one


def system_prompt(borrower: Borrower, today: date) -> str:
    return config.AGENT_SYSTEM_PROMPT.format(
        name=borrower.name, today=today.strftime(config.TODAY_FORMAT)
    )


def opening_line(borrower: Borrower) -> str:
    return config.AGENT_OPENING.format(name=borrower.name)


def build_llm(prompt: str) -> GoogleLLMService:
    return GoogleLLMService(
        api_key=config.env("GOOGLE_API_KEY"),
        http_options=HttpOptions(api_version=config.GEMINI_API_VERSION),
        settings=GoogleLLMService.Settings(
            model=config.GEMINI_MODEL,
            system_instruction=prompt,
            thinking=GoogleLLMService.ThinkingConfig(**config.GEMINI_THINKING),
        ),
    )


async def with_filler[T](
    work: Awaitable[T], after_secs: float, on_slow: Callable[[], Awaitable[None]] | None
) -> T:
    """Await work; if it is still running after after_secs, call on_slow once (a filler line)."""
    if on_slow is None:
        return await work

    async def filler_when_slow() -> None:
        await asyncio.sleep(after_secs)
        await on_slow()

    filler = asyncio.create_task(filler_when_slow())
    try:
        return await work
    finally:
        filler.cancel()


def build_tools(
    state: CallState,
    emit: Callable[..., None],
    on_end_call: Callable[[bool], Awaitable[None]],
    on_slow_tool: Callable[[], Awaitable[None]] | None = None,
) -> ToolsSchema:
    """The three tools as Pipecat schemas, each handler running the tool on this call's state.

    When end_call succeeds the LLM doesn't run again: the caller of build_tools speaks
    state.goodbye and ends the call. on_end_call(success) tells it end_call has finished.
    on_slow_tool, if given, is called when verify_identity or send_payment_link is slow.
    """

    def handler_for(name: str):
        async def handler(params: FunctionCallParams) -> None:
            work = run_tool(name, dict(params.arguments), state, emit)
            if name == "end_call":
                result = await work
            else:
                result = await with_filler(work, config.FILLER_AFTER_SECS, on_slow_tool)
            if name != "end_call":
                await params.result_callback(result)
                return
            run_llm = None if not result["success"] else False  # failed: let the LLM recover
            await params.result_callback(
                result, properties=FunctionCallResultProperties(run_llm=run_llm)
            )
            await on_end_call(result["success"])

        return handler

    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name=s["name"],
                description=s["description"],
                properties=s["parameters"]["properties"],
                required=s["parameters"]["required"],
                handler=handler_for(s["name"]),
            )
            for s in TOOL_SCHEMAS
        ]
    )


async def run_text_call(
    borrower: Borrower,
    caller: Caller,
    today: date,
    tool_failures: dict[str, float] | None = None,
) -> TextCallResult:
    """Run one call as text: the bot opens, then caller and bot take turns until it ends.
    tool_failures sets an injected failure rate per tool for this call (evals)."""
    recorder = EventRecorder(config.EVENTS_DIR, borrower_id=borrower.borrower_id)
    state = CallState(
        call_id=recorder.call_id, borrower=borrower, failure_rates=dict(tool_failures or {})
    )
    result = TextCallResult(call_id=recorder.call_id, ended_by="max_turns")
    # One LLM response ends at each tool call; the bot's turn is over at the first
    # response that calls no tool, or once end_call has finished (its goodbye is the
    # last line, spoken for the LLM).
    replies: asyncio.Queue[str] = asyncio.Queue()
    parts: list[str] = []
    tool_called = False
    end_call_done: asyncio.Future[bool] | None = None

    async def on_end_call(success: bool) -> None:
        if end_call_done is not None and not end_call_done.done():
            end_call_done.set_result(success)

    async def finish_turn_after_end_call(done: asyncio.Future[bool]) -> None:
        nonlocal tool_called, end_call_done
        if await done:
            replies.put_nowait(" ".join([*parts, state.goodbye or ""]).strip())
            parts.clear()
        else:  # end_call failed; the LLM runs again and its next response ends the turn
            tool_called, end_call_done = False, None

    llm = build_llm(system_prompt(borrower, today))
    opening = opening_line(borrower)
    context = LLMContext(
        messages=[
            {"role": "user", "content": config.CALL_CONNECTED_NOTE},
            {"role": "assistant", "content": opening},
        ],
        tools=build_tools(state, recorder.emit, on_end_call),
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
    worker = PipelineWorker(
        Pipeline([user_aggregator, llm, assistant_aggregator]), idle_timeout_secs=None
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        nonlocal tool_called, end_call_done
        tool_called = True
        if any(f.function_name == "end_call" for f in function_calls):
            end_call_done = asyncio.get_running_loop().create_future()

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        nonlocal tool_called
        if message.content:
            parts.append(message.content)
        if end_call_done is not None:
            # Not awaited here: the end_call handler needs the pipeline to keep moving.
            asyncio.create_task(finish_turn_after_end_call(end_call_done))
            return
        if tool_called:
            tool_called = False
            return
        replies.put_nowait(" ".join(parts))
        parts.clear()

    def bot_said(text: str) -> None:
        result.transcript.append({"role": "bot", "text": text})
        recorder.bot_said(text, interrupted=False)

    await recorder.start()
    running = asyncio.create_task(runner.run())
    try:
        bot_said(opening)
        for _ in range(config.TEXT_MAX_TURNS):
            line = await caller(result.transcript[-1]["text"])
            if line is None:
                result.ended_by = "caller_hung_up"
                break
            result.transcript.append({"role": "caller", "text": line})
            recorder.user_said(line)
            await worker.queue_frame(
                LLMMessagesAppendFrame([{"role": "user", "content": line}], run_llm=True)
            )
            try:
                reply = await asyncio.wait_for(replies.get(), config.TEXT_TURN_TIMEOUT_SECS)
            except TimeoutError:
                logger.error(
                    f"no reply within {config.TEXT_TURN_TIMEOUT_SECS} s, call {state.call_id}"
                )
                result.ended_by = "timeout"
                break
            bot_said(reply)
            if state.outcome:
                result.ended_by = "end_call"
                break
    finally:
        await worker.queue_frame(EndFrame())
        await running
        await save_record_without_end_call(state, result.ended_by)
        result.outcome = state.outcome
        await recorder.stop(result.ended_by)
    return result
