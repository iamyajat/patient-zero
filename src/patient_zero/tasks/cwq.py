"""ComplexWebQuestions items from MINT (only those with at least one verified misinformation passage).

CWQ has no source documents: the benign condition gives the researcher MINT's length-matched filler text,
the adversarial condition replaces it with a misinformation passage (attacks/mint.py)."""

from __future__ import annotations

import ast
import random

from .base import Task


def load_cwq(n: int, seed: int = 0) -> list[Task]:
    from datasets import load_dataset

    from ..attacks.mint import mint_index

    verified = mint_index("cwq")
    rows = [r for r in load_dataset("jonasbecker/MINT", "items_cwq", split="test") if r["MINT_ID"] in verified]
    random.Random(seed).shuffle(rows)
    tasks = []
    for r in rows[:n]:
        raw = r["gold_answers"]
        aliases = ast.literal_eval(raw) if isinstance(raw, str) and raw != "None" else (raw or [])
        tasks.append(
            Task(
                id=f"cwq_{r['MINT_ID'].rsplit('_', 1)[-1]}",
                dataset="cwq",
                question=r["sentence"],
                docs=[r["filler_information"]],
                gold_answer=r["gold_text"],
                meta={"mint_id": r["MINT_ID"], "aliases": [a for a in aliases if a != r["gold_text"]]},
            )
        )
    return tasks
