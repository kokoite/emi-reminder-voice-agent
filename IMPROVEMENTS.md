# Possible improvements

Logic we chose not to build because the assignment doesn't ask for it. Each is a
sensible next step for a real lender; we list it here instead of adding it to the code.

## Tools

- **Escalation queue for the human agent.** Today escalation is the `escalate_to_human`
  flag in the outcome record, as the assignment allows. A real hand-off would also write a
  queue item with a short query (what the human needs to resolve) and the whole
  conversation, so the human doesn't have to call back cold.
- **Verification attempt limit.** Today the agent keeps asking until the caller verifies,
  since the assignment doesn't say to stop. A real lender would allow a set number of
  tries, then end the call politely with a `verification_failed` outcome, so a caller
  can't guess the birth date or digits.
- **Clearer verification results.** Tell an unreadable answer (a date in the wrong order,
  three digits) apart from a wrong one, and don't count it as a failed attempt.
- **Promise-to-pay window.** Accept a promise date only from today up to a set number of
  days ahead, and tell the agent the latest allowed date.
- **Callback time check.** Reject a callback time that is in the past.
- **One outcome per call.** Refuse a second `end_call` so the record can't be overwritten
  or duplicated.
- **"No response" outcome.** When the caller stays silent and the bot hangs up, code
  writes the outcome record itself (`no_response`), since the LLM never calls `end_call`.

## Agent

- **Conversation as a graph of steps (Pipecat Flows).** Today one system prompt lists the
  call steps and the agent moves through them freely. Pipecat Flows splits the call into
  nodes (verify, remind, resolve, close), each with its own instructions and tools, so the
  order can't be skipped. Worth it if the single prompt keeps missing steps in the evals.
- **Post-call outcome extraction.** Today the live agent fills the outcome record through
  `end_call` while it is saying goodbye, and sometimes skips it or picks a doubtful label;
  code then saves a record with the outcome left empty. A second LLM could fill the record
  after the call from the full transcript, with only that job to do.
