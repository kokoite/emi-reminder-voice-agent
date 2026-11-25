import asyncio

import pytest

from agent import with_filler

# Async tests run through anyio's pytest plugin (anyio comes with pipecat).
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_slow_work_gets_one_filler():
    calls = []

    async def on_slow():
        calls.append("filler")

    async def slow():
        await asyncio.sleep(0.15)
        return "done"

    assert await with_filler(slow(), 0.05, on_slow) == "done"
    assert calls == ["filler"]


async def test_fast_work_gets_no_filler():
    calls = []

    async def on_slow():
        calls.append("filler")

    async def fast():
        return "done"

    assert await with_filler(fast(), 0.05, on_slow) == "done"
    await asyncio.sleep(0.1)  # the cancelled timer must not fire later
    assert calls == []


async def test_no_callback_means_no_filler():
    async def work():
        return 1

    assert await with_filler(work(), 0.0, None) == 1
