# Lumen Finance: EMI reminder voice agent

An outbound voice agent for a fictional lender. It calls a borrower about an upcoming or
overdue EMI, checks it is speaking to the right person, reminds them of the amount and due
date, finds out what they will do, offers a payment link, and ends with a structured outcome
record. It runs on a laptop: you talk to it from the browser.

The eval suite and its findings are in [evals/REPORT.md](evals/REPORT.md).

## Run it

You need Python 3.12, [uv](https://docs.astral.sh/uv/), and four API keys.

```bash
cd lumen
uv sync                      # installs the pinned dependencies into .venv
cp .env.example .env         # then fill in the four keys
```

| Key | Used for |
|---|---|
| `SARVAM_API_KEY` | speech-to-text (Sarvam realtime) |
| `GOOGLE_API_KEY` | the agent's LLM, and the eval borrower and judge (Gemini) |
| `ELEVENLABS_API_KEY` | text-to-speech (ElevenLabs) |
| `TYPESAFE_API_KEY` | deciding whether the caller is interrupting (TypeSafe Jev) |
| `TYPESAFE_BASE_URL` | optional: a gateway URL for Jev; empty means TypeSafe's own API |

**Voice call**

```bash
.venv/bin/python bot.py -t webrtc            # the bot, on :7860
python3 -m http.server 8000                  # serves the call page, from lumen/
```

Open http://localhost:8000/web/call.html and press **Call**. The bot calls the borrower set in
`config.VOICE_BORROWER_ID` (default `b_01`). Mock borrowers, with what to say to verify:

| id | Name | Date of birth | Last 4 digits | Loan | EMI (₹) | Due |
|---|---|---|---|---|---|---|
| b_01 | Ravi Kumar | 12 Mar 1988 | 4821 | two-wheeler | 12,500 | in 3 days |
| b_02 | Priya Sharma | 25 Jul 1992 | 7314 | personal | 18,750 | 5 days overdue |
| b_03 | Arjun Mehta | 2 Nov 1985 | 2290 | home | 1,24,000 | 20 days overdue |
| b_04 | Neha Iyer | 30 Jan 1990 | 5567 | consumer durable | 4,999 | in 2 days |
| b_05 | Vikram Singh | 14 Sep 1979 | 8803 | car | 21,300 | 8 days overdue |
| b_06 | Sunita Devi | 6 May 1983 | 1142 | gold | 6,200 | in 4 days |
| b_07 | Rahul Verma | 19 Dec 1995 | 3375 | education | 9,850 | 2 days overdue |
| b_08 | Anjali Gupta | 8 Apr 1987 | 6610 | personal | 15,000 | tomorrow |

Due dates are stored relative to today (`data/borrowers.json`), so the data never goes stale.
Each call writes its events to `events/<call_id>.jsonl` and its outcome record to
`outcomes.jsonl`. http://localhost:8000/web/dashboard.html shows per-turn latency;
`.venv/bin/python -m scripts.summarize_call [call_id]` prints one call's latency, barge-ins
and silences, and `.venv/bin/python -m scripts.check_services` checks every API key.

**Text chat with the same agent** (for testing; type as the borrower, an empty line hangs up)

```bash
.venv/bin/python chat.py b_03
```

**Evals**

```bash
.venv/bin/python -m evals.run_evals                 # 15 scenarios x 3 runs, ~5 minutes
.venv/bin/python -m evals.run_evals --only hardship --runs 1
.venv/bin/python -m evals.rejudge <run_id>          # re-run only the judge on a saved run
```

**Checks**: `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest`

## How a call is structured

1. **Opening (code).** A fixed line, "Hi, this is Lumen Finance. Am I speaking with {name}?",
   goes straight to text-to-speech, so the caller hears it in about 0.4 s.
2. **Verify (tool).** The agent asks for the date of birth or the last 4 digits of the
   registered mobile and calls `verify_identity`. Code compares them with the record; the
   LLM never sees the stored values, and loan details reach it only when the check passes.
3. **Remind.** The verify result gives the amount, due date and status already in spoken
   form ("twelve thousand five hundred rupees", "the twenty seventh of November"), so neither
   the LLM nor the voice has to format numbers.
4. **Find out what they will do**, and respond: pay on time, already paid, a later date,
   hardship, dispute, call back, someone else answering, a wrong number, rude or off topic,
   a request for a human, or a request to stop.
5. **Offer the payment link** (`send_payment_link`, refused by code until verified).
6. **End (tool).** `end_call` takes the outcome and the goodbye line. Code speaks the
   goodbye, hangs up after it has played, and writes the outcome record:

```json
{"call_id": "...", "borrower_id": "b_03", "verified": true, "outcome": "hardship",
 "promise_to_pay_date": null, "callback_at": null, "payment_link_sent": false,
 "escalate_to_human": true, "notes": "Borrower lost his job; escalated."}
```

What code decides and what the LLM decides:

| Code (deterministic) | LLM (judgement) |
|---|---|
| Whether the caller is verified; `verified` and `payment_link_sent` in the record | What to say, and in which language |
| No link before verification | Which outcome fits, and the promise or callback date |
| Escalation always set for hardship and dispute | Escalating on an explicit request for a human |
| Speaking the goodbye and hanging up after `end_call` | When the conversation is over |
| A record is still saved if the call ends any other way (hang-up, silence) | |
| Silence: "are you still there?" after 6 s, then a goodbye | |
| A lookup slower than 1 s: "One moment, please." while it runs | |

## Architecture

```
browser mic ─► SmallWebRTC ─► Silero VAD ─► Sarvam realtime STT ─► user turn (Smart Turn v3.2,
   ▲                                                                barge-in decided by Jev)
   │                                                                        │
   └── output gate ◄── ElevenLabs Flash v2.5 ◄── Gemini 3.5 Flash-Lite ◄────┘
       (pause/resume                              + 3 tools (agent/)
        on barge-in)
event recorder (every stage) ─► events/<call_id>.jsonl ─► web/dashboard.html
```

```
lumen/
├── bot.py                 run the voice call (Pipecat 1.17 pipeline and call lifecycle)
├── chat.py                run a text chat with the same agent
├── config.py              every setting: models, prompts, thresholds, timeouts
├── recorder.py            the event log, written off the audio path
├── agent/                 the conversation
│   ├── conversation.py    prompt, opening, tools on Pipecat's Gemini service, text calls
│   ├── tools.py           verify_identity, send_payment_link, end_call, per-call state
│   ├── borrowers.py       mock borrowers, due dates relative to today
│   └── spoken.py          numbers and dates in words (Indian system: lakh, crore)
├── voice/                 pieces of the voice pipeline
│   ├── barge_in.py        interruptions: pause, ask Jev, then resume, stop or ask
│   └── stt_preroll.py     hold audio while nobody speaks, replay 0.5 s before speech
├── evals/                 scenarios, simulated borrower, checks, judge, runner, REPORT.md
├── web/                   call.html (call page), dashboard.html (latency)
├── scripts/               check_services.py, summarize_call.py
├── data/borrowers.json    the mock borrowers
├── tests/                 pytest for the pure logic
└── IMPROVEMENTS.md        logic we chose not to build, as next steps
```

## Key decisions and trade-offs

- **Pipecat.** Turn-taking, interruptions and per-stage metrics are built in, and the text
  evals run the agent through the same Pipecat Gemini service and tool handling as the voice
  call, so what the evals test is what ships.
- **Speech-to-text: Sarvam realtime (saaras v4, codemix).** It handles English, Hindi and
  Hinglish. We send audio only while someone speaks (plus 0.5 s replayed before speech
  starts, so short words like "haan" aren't clipped) and a short silence every 15 s, because
  Sarvam closes a session after 60 s without audio.
- **End of turn: Silero VAD + Smart Turn v3.2,** with Smart Turn's silence fallback cut from
  3 s to 1.2 s. Mid-sentence pauses don't trigger a reply.
- **Interruptions: pause, then decide.** When the caller speaks over the bot, its audio pauses
  at once. Jev reads the partial transcript and the bot's line and gives the probability it
  is a real interruption: at least 0.6 stops the bot; 0.5 to 0.6 stops and asks "did you say
  something?"; below 0.5 resumes ("haan", "okay" don't interrupt). On a Jev timeout or error
  the bot stops: it never talks over the caller.
- **LLM: Gemini 3.5 Flash-Lite, thinking off.** Picked for speed. We ran the same eval on
  2.5 Flash, 3.5 Flash-Lite and 3.6 Flash: the failure types were the same, and the prompt
  fixes mattered far more than the model (see the report).
- **Voice: ElevenLabs Flash v2.5** over a WebSocket, Hindi-capable voice, speed 0.9.
- **A tool result is `{success, data, _instruction}`.** `_instruction` explains how to read
  the result (what may be shared), never what to say next; behaviour lives in the prompt.
- **The prompt names the tool for every action** ("thank them, then end_call"). Lines like
  "end the call politely" or "escalate to a human" made the model act in words: say goodbye
  without hanging up, or say "let me connect you" when it can't transfer. This was the
  biggest single fix (report, failure 2).
- **The goodbye is an argument of `end_call`**, spoken by code, and described as what it is
  (a short goodbye, plus the callback time or the human follow-up when they apply) rather than
  a list of banned phrases; bans didn't hold.
- **No dead air during lookups.** The mock tools take 2.5 s, like a real lookup
  (`TOOL_DELAY_SECS`). If one hasn't answered after 1 s, code says "One moment, please."
  (`FILLER_AFTER_SECS`, `FILLER_LINES`; in Hindi "एक मिनट, please." when the bot is
  speaking Hindi); a fast tool gets no filler.
- **Recovery when a tool fails.** A failed tool returns a plain "temporary fault" result;
  the prompt says to tell the caller plainly and try again or move on. Failures can be
  injected per tool (`TOOL_FAILURE_RATE`, or per call in the evals).
- **Scope.** Only what the assignment asks. Attempt limits, promise-date windows, an
  escalation queue and post-call outcome extraction are listed in `IMPROVEMENTS.md`.

## Latency (caller stops speaking → bot audio starts)

| Measured | Median | Notes |
|---|---|---|
| Step 2 test calls, short test prompt, Gemini 2.5 Flash | ~1.4 s (best 1.23 s) | no tools |
| Full agent call, Gemini 3.5 Flash-Lite (7 turns) | 2.1 s (p90 2.8 s) | turns with a tool call need two LLM requests back to back; measured before the 2.5 s mock lookup delay and the filler were added |

Where the time goes on the full agent call: Gemini 0.6 to 1.2 s per request (two requests
on a tool turn), up to 1.1 s waiting for the end of a one-word turn ("Yes."), about 0.1 s for
the final transcript and 0.2 s to the first audio. What reduced it: connections opened
during the greeting (Gemini, Jev), the fixed opening line, streaming speech-to-text and
text-to-speech, Smart Turn's 1.2 s fallback instead of 3 s, and thinking turned off. A slow
lookup now gets the filler line instead of silence. Next: end-of-turn detection for one-word
answers.

## Known limitations

- **Hindi:** the agent switches to Hindi and speaks everyday Hinglish written in Devanagari,
  with amounts, dates and English terms in English letters ("आपका EMI twelve thousand five
  hundred rupees है, जो the twenty eighth of November को due है"), as a woman, matching the
  voice. Written in Latin letters, the voice pronounced Hindi with an English accent;
  Devanagari fixed that in live calls. Translating the figures itself, the agent once said a
  wrong date, so they stay in English words. This is an instruction, not a guarantee: Hindi
  forms produced by code (`agent/spoken.py`) would be. It can still slip, e.g. writing a name
  wrongly in Devanagari or saying goodbye twice.
- **Verification can't tell who is speaking:** a family member who knows the date of birth
  passes. The prompt says not to verify someone who said they aren't the borrower.
- **Run-to-run variation:** see the report; the agent passes 38 to 42 of 42 eval calls.
- Evals are text-level; voice behaviour (latency, interruptions) is measured from live calls.
