"""Talk to the agent as a borrower, in text. For our own testing; not part of the voice call.

Run: .venv/bin/python chat.py b_03
Type the borrower's replies; an empty line hangs up. Lines can also be piped in:
  .venv/bin/python chat.py b_03 < lines.txt
"""

import asyncio
import json
import sys
from datetime import date

from loguru import logger

import config
from agent import run_text_call
from borrowers import load_borrowers


async def main(borrower_id: str) -> None:
    today = date.today()
    borrower = load_borrowers(config.BORROWERS_FILE, today)[borrower_id]
    interactive = sys.stdin.isatty()
    print(f"Calling {borrower.name} ({borrower.borrower_id}, {borrower.scenario})")

    async def caller(bot_line: str) -> str | None:
        print(f"\nBot:    {bot_line}")
        line = await asyncio.to_thread(sys.stdin.readline)
        line = line.strip()
        if not interactive and line:
            print(f"Caller: {line}")
        return line or None

    result = await run_text_call(borrower, caller, today)
    if result.ended_by != "caller_hung_up":
        print(f"\nBot:    {result.transcript[-1]['text']}")
    print(f"\nEnded by {result.ended_by}. Events: {config.EVENTS_DIR / result.call_id}.jsonl")
    print("Outcome:", json.dumps(result.outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "b_01"))
