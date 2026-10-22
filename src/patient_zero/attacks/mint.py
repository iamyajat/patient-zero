"""V1 with MINT misinformation passages (Becker et al. 2026, huggingface.co/datasets/jonasbecker/MINT, Apache-2.0).

Only passages MINT marks verified=True are used. CWQ passages carry no intended wrong answer, so the marker is the
validating judge's `implied_answer`. Validation was done by two LLM judges, not human annotators.
"""

from __future__ import annotations

import random
import zlib
from functools import cache

from ..tasks.base import Task
from .base import Injection

STYLES = ["false_fact", "clickbait", "conspiracy", "framing", "hoax", "other", "propaganda", "rumor", "satire"]


def _implied(validation: dict) -> str | None:
    for judge in validation.get("judges", []):
        j = validation.get(judge) or {}
        if j.get("is_false") and j.get("supports_target") and j.get("implied_answer"):
            return j["implied_answer"]
    return None


@cache
def mint_index(split: str = "cwq") -> dict[str, list[dict]]:
    """mint_id -> verified passages [{style, passage, marker}]."""
    from datasets import load_dataset

    index: dict[str, list[dict]] = {}
    for row in load_dataset("jonasbecker/MINT", "passages", split=split):
        v = row["validation"]
        marker = _implied(v) if v.get("verified") else None
        if marker:
            index.setdefault(row["mint_id"], []).append({"style": row["category"], "passage": row["passage"], "marker": marker})
    return index


class MintAttack:
    vector = "v1"

    def __init__(self, tasks: list[Task], styles: list[str] | None = None, target_role: str = "researcher", index=None):
        self.styles = styles or STYLES
        self.target_role = target_role
        self._index = index

    @property
    def index(self) -> dict[str, list[dict]]:
        return self._index if self._index is not None else mint_index()

    def apply(self, task: Task) -> Injection:
        options = [p for p in self.index.get(task.meta["mint_id"], []) if p["style"] in self.styles]
        if not options:
            raise ValueError(f"no verified MINT passage in styles {self.styles} for {task.id}")
        pick = random.Random(zlib.crc32(task.id.encode())).choice(sorted(options, key=lambda p: p["style"]))
        return Injection(
            target_role=self.target_role,
            marker=pick["marker"],
            docs=[pick["passage"]],  # replaces the length-matched filler the benign condition gets
            meta={"gold": task.gold_answer, "false": pick["marker"], "style": f"mint:{pick['style']}", "source": "mint"},
        )
