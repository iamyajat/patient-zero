"""Task interface (owned by B: add datasets by returning lists of Task)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Task:
    id: str
    dataset: str
    question: str
    docs: list[str]
    gold_answer: str
    meta: dict = field(default_factory=dict)


class TaskSource(Protocol):
    def __call__(self, n: int, seed: int = 0) -> list[Task]: ...
