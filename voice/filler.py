"""The filler line said while a slow lookup runs, in the language the call is in."""

import config
from voice.language import is_hindi


def filler_line(last_bot_text: str) -> str:
    return config.FILLER_LINES["hi" if is_hindi(last_bot_text) else "en"]
