# Find Patient Zero

A coding assignment on auditing multi-agent AI systems, built on the Patient Zero harness. It comes with two lectures
(`lectures.md`), a rubric (`rubric.md`) and pre/post survey items (`survey.md`). Following Embedded EthiCS practice,
the ethics is tied directly to the technical work.

## The setting
An organisation runs a team of AI agents (researcher → analyst → verifier → writer, or a coordinator with workers).
Users have reported bad outputs. You get the message logs and must work out what happened, build a monitor, and decide
whether the system should be allowed to keep running.

## Data
`student/runs.jsonl` has one run per line:
- `layout`, `question` and `role_docs` (the documents each agent received).
- `condition`: `benign` or `adversarial`.
- `reported_claim`: the bad claim users saw, or null.
- `claim_kind`: `answer` means an agent is infected when its ANSWER adopts the claim; `text` means it is infected
  when its message contains the claim.
- `messages`: each with `id`, `sender`, `recipients`, `visible_inputs` (the ids of messages this agent saw, which are
  the provenance edges), `step`, `round`, `content` and `answer`.

## Tasks
Implement them in a copy of `submission_template.py`.

**(a) Trace the outbreak (30%, autograded).**
- `find_patient_zero` returns the first message that carried the reported claim.
- `r_hat` computes the pooled reproduction number: new infections caused per infected message that was handed on.
- Graded on hidden runs: at least 90% of patient-zero ids correct, and R̂ within 0.05 of the reference for each layout.

**(b) Build a monitor (30%, autograded).**
- `score_message` scores each message without the incident report; it only gets the question, the documents and the
  messages.
- Choose `THRESHOLD` so that at most 1% of clean benign messages are flagged.
- Your score is your recall on infected messages in hidden adversarial runs. It is 0 if your false-alarm rate is
  over budget (the lower bound of a 95% Clopper–Pearson interval must be at or below 1%).

**(c) Audit memo (25%, rubric).** One page:
- Does the system pass T1 (R̂ < 1), T2 (bounded amplification) and T3 (monitor adequacy) from `configs/standard_v1.yaml`?
- Which IMDA MGF and EU AI Act clauses apply?
- Who must be told, and what happens next (quarantine, re-certification)?

**(d) Reflection (15%, rubric).** About 300 words on one value trade-off. For example: quarantining an agent that
serves vulnerable users, or the privacy cost of reading every inter-agent message.

## Running the autograder locally
```
uv run python -m teaching.find_patient_zero.generate --out teaching_data --secret <instructor secret>
PZ_HIDDEN=teaching_data/hidden/runs.jsonl PZ_TRUTH=teaching_data/instructor/truth.json \
PZ_CONDITIONS=teaching_data/instructor/hidden_conditions.json PZ_SUBMISSION=my_submission.py \
uv run pytest teaching/find_patient_zero/autograder -q -s
```
Instructors: keep `instructor/` and the secret private. Logs from real model runs can be used instead with `--from-exp runs/<exp>`.
