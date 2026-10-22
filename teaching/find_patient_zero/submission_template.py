"""Find Patient Zero: your submission. Fill in the three functions and THRESHOLD; keep the signatures.

Data (one run per line in runs.jsonl): run_id, layout, question, role_docs {role: [{id, text}]},
reported_claim (the bad claim users saw, or null), claim_kind ("answer": an agent is infected if its ANSWER adopts
the claim; "text": infected if its message contains the claim), and messages: [{id, sender, recipients,
visible_inputs (ids of messages this agent saw), doc_inputs, step, round, phase, content, answer}].
"""

from __future__ import annotations


def find_patient_zero(messages: list[dict], reported_claim: str | None, claim_kind: str | None) -> str | None:
    """Part (a): the id of the first message that carried the reported claim, or None if no message did."""
    raise NotImplementedError


def r_hat(runs: list[tuple[list[dict], str | None, str | None]]) -> float:
    """Part (a): pooled reproduction number over these runs: new infections caused / infected messages handed on
    (non-empty recipients). An agent counts as newly infected only on its first infected message, and patient zero's
    own agent never counts as a new infection."""
    raise NotImplementedError


THRESHOLD = 0.5  # Part (b): flag a message when score_message(...) > THRESHOLD. Keep false alarms <= 1% on benign runs.


def score_message(message: dict, context: dict) -> float:
    """Part (b): suspicion score for one message. `context` has question, role_docs and the run's messages; it does NOT
    include reported_claim (a deployer's monitor has no incident report)."""
    raise NotImplementedError
