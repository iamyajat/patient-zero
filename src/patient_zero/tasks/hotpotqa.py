"""HotpotQA (distractor setting): each task ships with 10 paragraphs, 2 of them gold."""

from __future__ import annotations

import random

from .base import Task


def load_hotpotqa(n: int, seed: int = 0, split: str = "validation", exclude_yes_no: bool = True) -> list[Task]:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    tasks: list[Task] = []
    for i in idx:
        row = ds[i]
        # yes/no comparison questions have no entity to swap for V1, so they are left out everywhere
        if exclude_yes_no and row["answer"].lower() in {"yes", "no"}:
            continue
        ctx = row["context"]
        docs = [f"{t}: {' '.join(s)}" for t, s in zip(ctx["title"], ctx["sentences"])]
        tasks.append(
            Task(
                id=f"hotpot_{row['id']}",
                dataset="hotpotqa",
                question=row["question"],
                docs=docs,
                gold_answer=row["answer"],
                meta={"type": row["type"], "level": row["level"]},
            )
        )
        if len(tasks) == n:
            break
    return tasks
