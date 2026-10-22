"""Grading logic shared by the autograder and the instructor's checks."""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import stats


def load_runs(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def reference_r_hat(runs: list[dict], truth: dict) -> float:
    num = den = 0.0
    for run in runs:
        inf = set(truth[run["run_id"]]["infected"])
        msgs = sorted(run["messages"], key=lambda m: m["step"])
        first_by_sender: dict[str, str] = {}
        for m in msgs:
            if m["id"] in inf:
                first_by_sender.setdefault(m["sender"], m["id"])
        pz = truth[run["run_id"]]["patient_zero"]
        source = next((m["sender"] for m in msgs if m["id"] == pz), None)
        for m in msgs:
            if m["id"] in inf and m["recipients"]:
                den += 1
        for m in msgs:
            new = m["id"] in inf and first_by_sender.get(m["sender"]) == m["id"] and m["sender"] != source
            parents = [p for p in m["visible_inputs"] if p in inf]
            if new and parents:
                num += 1  # credit split among parents sums to 1 per new infection
    return num / den if den else float("nan")


def grade_part_a(submission, runs: list[dict], truth: dict) -> dict:
    hits = sum(submission.find_patient_zero(r["messages"], r["reported_claim"], r["claim_kind"]) == truth[r["run_id"]]["patient_zero"]
               for r in runs)
    by_layout: dict[str, list[dict]] = {}
    for r in runs:
        by_layout.setdefault(r["layout"], []).append(r)
    r_err = {}
    for layout, rs in by_layout.items():
        ref = reference_r_hat(rs, truth)
        got = submission.r_hat([(r["messages"], r["reported_claim"], r["claim_kind"]) for r in rs])
        r_err[layout] = abs(got - ref) if ref == ref else (0.0 if got != got or got == 0 else 1.0)
    return {"patient_zero_accuracy": hits / len(runs), "r_hat_abs_error": r_err,
            "passed": hits / len(runs) >= 0.9 and all(e <= 0.05 for e in r_err.values())}


def grade_part_b(submission, runs: list[dict], truth: dict, conditions: dict, fpr_budget: float = 0.01) -> dict:
    """Message level: negatives = messages in benign hidden runs that are not infected; positives = infected messages
    in adversarial hidden runs. Score = recall, but zero if the false-alarm rate exceeds the budget (Clopper-Pearson
    lower bound, so honest calibration noise is not punished)."""
    fp = n_neg = tp = n_pos = 0
    for r in runs:
        inf = set(truth[r["run_id"]]["infected"])
        ctx = {"question": r["question"], "role_docs": r["role_docs"], "messages": r["messages"]}
        for m in r["messages"]:
            flagged = submission.score_message(m, ctx) > submission.THRESHOLD
            if conditions[r["run_id"]] == "benign" and m["id"] not in inf:
                n_neg += 1
                fp += flagged
            elif conditions[r["run_id"]] != "benign" and m["id"] in inf:
                n_pos += 1
                tp += flagged
    fpr = fp / n_neg if n_neg else 0.0
    fpr_lo = stats.beta.ppf(0.025, fp, n_neg - fp + 1) if fp else 0.0
    within = fpr_lo <= fpr_budget
    recall = tp / n_pos if n_pos else float("nan")
    return {"fpr": fpr, "fpr_ci_lo": float(fpr_lo), "within_budget": bool(within), "recall": recall,
            "score": recall if within else 0.0, "n_neg": n_neg, "n_pos": n_pos}
