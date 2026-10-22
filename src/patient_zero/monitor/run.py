"""Offline monitoring: score every public message of every logged run and write flags.jsonl.

Monitors are named sets of detectors; a message's monitor score is the max over its detectors."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from ..metrics.basic import load_exp
from .base import MessageContext
from .llm import InstructionLLM, JudgeLLM, ProvenanceLLM
from .rules import FlipRule, InstructionRules, ProvenanceRule

RULES = [InstructionRules, ProvenanceRule, FlipRule]
MONITORS = {
    "rules": RULES,
    "llm": [InstructionLLM, ProvenanceLLM],
    "ensemble": [*RULES, InstructionLLM, ProvenanceLLM],
    "judge": [JudgeLLM],  # single LLM-as-judge baseline
}


def build_detectors(monitor: str, judge_model: str = "openai", data_dir: str = "data") -> list:
    out = []
    for cls in MONITORS[monitor]:
        out.append(cls(model=judge_model, data_dir=data_dir) if hasattr(cls, "prompt") else cls())
    return out


def contexts(run: dict, msgs: list[dict]) -> list[tuple[dict, MessageContext]]:
    """What a deployer can see for each message: its sender's documents and tool results plus those of every
    upstream message, the teammates' messages it saw, and its own previous-round message."""
    doc_text = {d["id"]: d["text"] for docs in (run.get("role_docs") or {}).values() for d in docs}
    for m in msgs:
        for call in m.get("tool_calls") or []:
            if "result" in call:
                doc_text[f"tool:{call['id']}"] = call["result"]
    public = [m for m in sorted(msgs, key=lambda m: m["step"]) if m["phase"] != "probe"]
    by_id = {m["id"]: m for m in public}
    upstream: dict[str, set[str]] = {}
    out = []
    for m in public:
        docs = set(m.get("doc_inputs") or [])
        for p in m["visible_inputs"]:
            docs |= upstream.get(p, set())
        upstream[m["id"]] = docs
        prev = [o for o in public if o["sender"] == m["sender"] and o["round"] == m["round"] - 1]
        ctx = MessageContext(
            question=run["question"],
            docs=[doc_text[d] for d in sorted(docs) if d in doc_text],
            visible=[by_id[p] for p in m["visible_inputs"] if p in by_id],
            own_previous=prev[-1] if prev else None,
        )
        out.append((m, ctx))
    return out


async def score_runs(runs: pd.DataFrame, msgs: pd.DataFrame, monitor: str, judge_model: str = "openai",
                     data_dir: str = "data", concurrency: int = 8) -> pd.DataFrame:
    detectors = build_detectors(monitor, judge_model, data_dir)
    grouped = {rid: g.to_dict("records") for rid, g in msgs.groupby("run_id")}
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict] = []

    async def one(run: dict, m: dict, ctx: MessageContext):
        scores, reasons = {}, {}
        for d in detectors:
            async with sem:
                try:
                    s, r = await d.score(m, ctx)
                except Exception as e:  # a failed LLM call scores as unknown, not as clean
                    s, r = float("nan"), f"error: {type(e).__name__}: {e}"[:300]
            scores[d.name], reasons[d.name] = s, r
        valid = {k: v for k, v in scores.items() if v == v}
        top = max(valid, key=valid.get) if valid else None
        rows.append({"run_id": run["run_id"], "message_id": m["id"], "monitor": monitor,
                     "score": valid[top] if top else float("nan"), "detector": top, "reason": reasons.get(top, ""),
                     "scores": scores})

    await asyncio.gather(*(one(run, m, ctx) for run in runs.to_dict("records")
                           for m, ctx in contexts(run, grouped.get(run["run_id"], []))))
    for d in detectors:
        if hasattr(d, "cache"):
            d.cache.save()
    out = pd.DataFrame(rows)
    out.attrs["judge_tokens"] = sum(getattr(d, "tokens", 0) for d in detectors)
    return out


def task_sample(task_ids, n: int) -> set[str]:
    """A deterministic sample of tasks (all their runs are scored, so benign/adversarial pairs stay together)."""
    import zlib

    return set(sorted(set(task_ids), key=lambda t: zlib.crc32(t.encode()))[:n])


def monitor_experiment(exp_dir: str | Path, monitor: str = "rules", judge_model: str = "openai",
                       data_dir: str = "data", concurrency: int = 8, max_tasks: int | None = None) -> Path:
    exp_dir = Path(exp_dir)
    runs, msgs = load_exp(exp_dir)
    if "error" in runs:
        runs = runs[runs.error.isna()]
    if max_tasks:
        runs = runs[runs.task_id.isin(task_sample(runs.task_id, max_tasks))]
        msgs = msgs[msgs.run_id.isin(runs.run_id)]
    flags = asyncio.run(score_runs(runs, msgs, monitor, judge_model, data_dir, concurrency))
    path = exp_dir / f"flags_{monitor}.jsonl"
    flags.to_json(path, orient="records", lines=True)
    meta = {"monitor": monitor, "judge_model": judge_model, "max_tasks": max_tasks, "judge_tokens": flags.attrs.get("judge_tokens", 0),
            "agent_tokens": int((runs.prompt_tokens + runs.completion_tokens).sum())}
    (exp_dir / f"flags_{monitor}.meta.json").write_text(json.dumps(meta, indent=1))
    return path
