"""Step 0 check: one small real call to each service, to confirm the keys work.

Run: .venv/bin/python -m scripts.check_services
"""

import asyncio
import json
import sys
import time
from collections.abc import Awaitable, Callable
from urllib.parse import urlencode

import aiohttp
import websockets
from google import genai
from google.genai import types
from pipecat.classifiers.base_classifier import ChoiceQuestion
from pipecat.classifiers.typesafe.jev.classifier import JevClassifier

import config

TIMEOUT_SECS = 15


async def check_sarvam() -> str:
    """Open the realtime STT socket and wait for Sarvam to confirm the session."""
    params = {
        "language_code": config.SARVAM_LANGUAGE or "auto",
        "stream_type": config.SARVAM_STREAM_TYPE,
        "endpointing": "manual",
        "encoding": "linear16",
        "sample_rate": 16000,
        "model": config.SARVAM_MODEL,
        "mode": config.SARVAM_MODE,
    }
    url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?{urlencode(params)}"
    headers = {"API-SUBSCRIPTION-KEY": config.env("SARVAM_API_KEY")}
    async with websockets.connect(url, additional_headers=headers) as ws:
        message = json.loads(await ws.recv())
        await ws.send(json.dumps({"event": "end"}))
    if message.get("event") != "session.begin":
        raise RuntimeError(f"unexpected first message: {message}")
    return f"session opened ({config.SARVAM_MODEL}, {config.SARVAM_MODE})"


async def check_gemini() -> str:
    """Ask the configured model for a one-word reply with thinking off."""
    client = genai.Client(
        api_key=config.env("GOOGLE_API_KEY"),
        http_options=types.HttpOptions(api_version=config.GEMINI_API_VERSION),
    )
    response = await client.aio.models.generate_content(
        model=config.GEMINI_MODEL,
        contents="Reply with the single word OK.",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(**config.GEMINI_THINKING),
        ),
    )
    return f"{config.GEMINI_MODEL} replied {response.text.strip()!r}"


async def check_elevenlabs() -> str:
    """Look up the configured voice, which checks both the key and the voice ID."""
    voice_id = config.ELEVENLABS_VOICE_ID
    url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
    headers = {"xi-api-key": config.env("ELEVENLABS_API_KEY")}
    async with aiohttp.ClientSession() as session, session.get(url, headers=headers) as resp:
        body = await resp.json(content_type=None)
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {body}")
    return f"voice found: {body.get('name', voice_id)}"


async def check_jev() -> str:
    """Ask Jev one interruption-vs-backchannel question, as step 2 will."""
    classifier = JevClassifier(api_key=config.env("TYPESAFE_API_KEY"), base_url=config.JEV_BASE_URL)
    question = ChoiceQuestion(
        instructions="While the bot was speaking, the caller said something. Classify it.",
        options={
            "interruption": "The caller wants the bot to stop and listen.",
            "backchannel": "The caller is only showing they are listening.",
        },
    )
    try:
        results = await classifier.ask(
            "Bot: Your EMI of twelve thousand rupees is due on the fifth. Caller: haan",
            {"kind": question},
        )
    finally:
        await classifier.cleanup()
    result = results["kind"]
    return f"'haan' → {result.choice} (confidence {result.confidence:.2f})"


CHECKS: dict[str, Callable[[], Awaitable[str]]] = {
    "Sarvam": check_sarvam,
    "Gemini": check_gemini,
    "ElevenLabs": check_elevenlabs,
    "Jev": check_jev,
}


async def run_check(name: str, check: Callable[[], Awaitable[str]]) -> bool:
    start = time.perf_counter()
    try:
        detail = await asyncio.wait_for(check(), TIMEOUT_SECS)
        ok = True
    except Exception as e:  # Report every failure; one bad service shouldn't hide the others.
        detail = f"{type(e).__name__}: {e}"
        ok = False
    ms = (time.perf_counter() - start) * 1000
    print(f"{'OK  ' if ok else 'FAIL'} {name:<11} {ms:6.0f} ms  {detail}")
    return ok


async def main() -> int:
    missing = config.missing_env()
    if missing:
        print("Missing in lumen/.env: " + ", ".join(missing))
        return 1
    results = await asyncio.gather(*(run_check(n, c) for n, c in CHECKS.items()))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
