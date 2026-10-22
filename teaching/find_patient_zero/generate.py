"""Build the assignment data.

  uv run python -m teaching.find_patient_zero.generate --out teaching_data --secret 1234            # simulated
  uv run python -m teaching.find_patient_zero.generate --out teaching_data --secret 1234 --from-exp runs/pilot_nov3

Writes:
  <out>/student/runs.jsonl   runs the students analyse and tune on (condition shown)
  <out>/hidden/runs.jsonl    held-out runs for the autograder (condition and attack fields removed)
  <out>/instructor/truth.json, hidden_conditions.json   answers for grading (never give these out)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import zlib
from pathlib import Path

import pandas as pd

from patient_zero.metrics.basic import load_exp
from patient_zero.metrics.contagion import infected, mode_for, patient_zero

HIDDEN_FIELDS_MSG = {"injected_into_sender", "guard_score", "quarantined", "exp_id", "condition", "vector", "model",
                     "task_id", "run_id", "layout", "seed_idx", "prompt_tokens", "completion_tokens", "latency_ms", "ts"}


def _clean(x):
    """NaN (from pandas) -> None, recursively, so the files are valid JSON."""
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def truth_for(run: dict, msgs: list[dict]) -> dict:
    public = sorted([m for m in msgs if m["phase"] != "probe"], key=lambda m: m["step"])
    marker, _ = patient_zero(run, public)
    if marker is None:
        return {"patient_zero": None, "infected": [], "claim": None, "kind": None}
    golds = [run["gold_answer"]] if run.get("gold_answer") else []
    inf = [m["id"] for m in public if infected(m, marker, golds, mode_for(run["vector"]))]
    kind = "text" if mode_for(run["vector"]) == "content" else "answer"
    return {"patient_zero": inf[0] if inf else None, "infected": inf, "claim": marker, "kind": kind}


def package(runs: pd.DataFrame, msgs: pd.DataFrame, out: Path, secret: int, hidden_share: float = 0.5) -> None:
    grouped = {rid: g.to_dict("records") for rid, g in msgs.groupby("run_id")}
    student, hidden, truth, conditions = [], [], {}, {}
    for run in runs.to_dict("records"):
        run = _clean(run)
        ms = [_clean(m) for m in grouped.get(run["run_id"], []) if m["phase"] != "probe"]
        truth[run["run_id"]] = truth_for(run, ms)
        is_hidden = (zlib.crc32(f"{secret}:{run['task_id']}".encode()) % 1000) / 1000 < hidden_share
        clean = [{k: v for k, v in m.items() if k not in HIDDEN_FIELDS_MSG} for m in ms]
        t = truth[run["run_id"]]
        # the incident report students start from: the bad claim users saw (never passed to the part (b) monitor)
        item = {"run_id": run["run_id"], "layout": run["layout"], "question": run["question"],
                "role_docs": run.get("role_docs") or {}, "reported_claim": t["claim"], "claim_kind": t["kind"],
                "messages": clean}
        if is_hidden:
            conditions[run["run_id"]] = run["condition"]
            hidden.append(item)
        else:
            student.append({**item, "condition": run["condition"]})
    for sub, rows in [("student", student), ("hidden", hidden)]:
        (out / sub).mkdir(parents=True, exist_ok=True)
        (out / sub / "runs.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")
    (out / "instructor").mkdir(parents=True, exist_ok=True)
    (out / "instructor" / "truth.json").write_text(json.dumps(truth, indent=1))
    (out / "instructor" / "hidden_conditions.json").write_text(json.dumps(conditions, indent=1))


async def simulate(out: Path, secret: int, n_tasks: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    from patient_zero.config import GridConfig
    from patient_zero.layouts import LAYOUTS
    from patient_zero.runner import JsonlLogger, expand_grid, run_one
    from patient_zero.tasks import load_tasks

    from .sim_model import SimAgentModel

    tasks = load_tasks("hotpotqa", n_tasks, seed=secret) if n_tasks > 3 else load_tasks("sample", n_tasks)
    cfg = GridConfig(exp_id="teaching", vectors=["v1", "v2.infection"], conditions=["benign", "adversarial"],
                     layouts=["chain", "hub"], models=["sim"], out_dir=str(out / "instructor"), dataset=tasks[0].dataset)
    specs = expand_grid(cfg, {"v1": tasks, "v2.infection": tasks})
    golds = {t.question: t.gold_answer for t in tasks}
    markers = {s.task.question: s.injection.marker for s in specs if s.injection and s.meta.vector == "v1"}
    logger = JsonlLogger(cfg.exp_dir)
    for s in specs:
        model = SimAgentModel(golds=golds, markers=markers if s.meta.vector == "v1" else {}, seed=secret)
        graph = LAYOUTS[s.meta.layout](lambda role: model)
        record, messages = await run_one(s, graph, cfg.doc_roles)
        logger.write_run(record, messages)
    return load_exp(cfg.exp_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--secret", type=int, required=True)
    ap.add_argument("--from-exp", type=Path)
    ap.add_argument("--n-tasks", type=int, default=60)
    a = ap.parse_args()
    runs, msgs = load_exp(a.from_exp) if a.from_exp else asyncio.run(simulate(a.out, a.secret, a.n_tasks))
    runs = runs[runs.error.isna()] if "error" in runs else runs
    package(runs, msgs, a.out, a.secret)
    print(f"wrote {a.out}/student, {a.out}/hidden and {a.out}/instructor")


if __name__ == "__main__":
    main()
