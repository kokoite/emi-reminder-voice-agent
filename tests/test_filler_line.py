import config
from voice.filler import filler_line


def test_filler_follows_the_bot_language():
    assert filler_line("Could you tell me your date of birth?") == config.FILLER_LINES["en"]
    assert filler_line("क्या आप अपनी date of birth बता सकते हैं?") == config.FILLER_LINES["hi"]
    assert filler_line("") == config.FILLER_LINES["en"]
