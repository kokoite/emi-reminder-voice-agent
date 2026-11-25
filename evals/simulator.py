"""The simulated borrower: an LLM playing one scenario's persona against the agent.

It is not part of the voice call, so it calls Gemini through the Google SDK directly.
"""

from google import genai
from google.genai import types

import config


class SimulatedBorrower:
    """Keeps its own side of the conversation and answers each bot line in character."""

    def __init__(self, client: genai.Client, system_prompt: str):
        self._client = client
        self._config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(**config.SIM_THINKING),
        )
        self._history: list[types.Content] = []

    async def reply(self, bot_line: str) -> str | None:
        """The borrower's answer to the bot's line, or None when they hang up."""
        self._history.append(types.Content(role="user", parts=[types.Part(text=bot_line)]))
        response = await self._client.aio.models.generate_content(
            model=config.SIM_MODEL, contents=self._history, config=self._config
        )
        text = (response.text or "").strip()
        self._history.append(types.Content(role="model", parts=[types.Part(text=text)]))
        if not text or config.SIM_HANGUP in text:
            return None
        return text
