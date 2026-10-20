"""Attack interface (owned by B). An attack turns a clean task into an Injection aimed at one agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..tasks.base import Task


@dataclass
class Injection:
    target_role: str
    marker: str  # string whose spread through later agents is tracked (e.g. the false answer)
    docs: list[str] | None = None  # replaces the target's documents
    extra_docs: list[str] = field(default_factory=list)  # appended to the target's documents
    system_override: str | None = None  # appended to the target's system prompt
    meta: dict = field(default_factory=dict)


class Attack(Protocol):
    vector: str

    def apply(self, task: Task) -> Injection: ...
