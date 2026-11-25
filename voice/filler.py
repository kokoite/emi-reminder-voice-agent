"""The filler line said while a slow lookup runs, in the language the call is in."""

import config


def is_hindi(text: str) -> bool:
    """True if the text has Devanagari letters; the agent writes Hindi in Devanagari."""
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def filler_line(last_bot_text: str) -> str:
    return config.FILLER_LINES["hi" if is_hindi(last_bot_text) else "en"]
