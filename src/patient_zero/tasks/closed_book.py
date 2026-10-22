"""Contamination control: does the model answer the task correctly with no documents? Cached per model."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from ..metrics.basic import best_f1
from ..metrics.equivalence import F1_EQUIV, Equiv
from ..roles import ANSWER_LINE
from .base import Task

KNOWN_F1 = 0.8
SYSTEM = "Answer the question from your own knowledge. If unsure, give your best guess rather than refusing. " + ANSWER_LINE


def cache_path(data_dir: str | Path, model_name: str) -> Path:
    return Path(data_dir) / "closed_book" / f"{model_name.replace(':', '_').replace('/', '_')}.json"


async def closed_book(tasks: list[Task], model_name: str, data_dir: str | Path = "data", concurrency: int = 8) -> dict[str, dict]:
    path = cache_path(data_dir, model_name)
    results: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}
    todo = [t for t in tasks if t.id not in results]
    if todo:
        from ..agent import parse_answer
        from ..models import get_chat_model

        model = get_chat_model(model_name)
        sem = asyncio.Semaphore(concurrency)
        errors: list[str] = []

        async def one(task: Task):
            async with sem:
                try:
                    reply = await model.ainvoke([SystemMessage(SYSTEM), HumanMessage(task.question)])
                except Exception as e:
                    errors.append(f"{type(e).__name__}: {e}")
                    return  # not cached: retried next time
            answer = parse_answer(reply.text)
            results[task.id] = {"answer": answer, "f1": best_f1(answer, [task.gold_answer, *task.meta.get("aliases", [])])}

        await asyncio.gather(*(one(t) for t in todo))
        if errors:
            print(f"closed-book: {len(errors)}/{len(todo)} calls failed for {model_name}; first: {errors[0][:200]}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return results


def known_from_cache(runs, eq: Equiv = F1_EQUIV, data_dir: str | Path = "data"):
    """Recompute `known` per run from the closed-book cache with the chosen equivalence, instead of trusting the flag
    recorded at run time (which used whatever judge was configured then). Runs without a cached answer keep theirs."""
    import pandas as pd

    from ..runner import closed_book_model

    caches: dict[str, dict] = {}
    out = []
    for r in runs.to_dict("records"):
        name = closed_book_model(r["model"])
        if name not in caches:
            p = cache_path(data_dir, name)
            caches[name] = json.loads(p.read_text()) if p.exists() else {}
        hit = caches[name].get(r["task_id"])
        if hit is None:
            out.append(r.get("known"))
            continue
        aliases = r.get("gold_aliases") if isinstance(r.get("gold_aliases"), list) else []
        judged = eq is not F1_EQUIV and eq.any_same(r["question"], hit.get("answer"), [r["gold_answer"], *aliases])
        out.append(bool(judged or hit["f1"] >= KNOWN_F1))
    return pd.Series(out, index=runs.index, dtype=object)


def mark_known(tasks: list[Task], model_spec: str, results: dict[str, dict], eq: Equiv = F1_EQUIV) -> None:
    """Known = the closed-book answer is right: equivalent to gold or an alias, or F1 >= 0.8 without a judge."""
    for t in tasks:
        if t.id in results:
            r = results[t.id]
            judged = eq is not F1_EQUIV and eq.any_same(t.question, r.get("answer"), [t.gold_answer, *t.meta.get("aliases", [])])
            t.meta.setdefault("known", {})[model_spec] = judged or r["f1"] >= KNOWN_F1
