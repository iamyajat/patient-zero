"""Monitor interface (C + A). Runs offline over logged messages, so it never blocks the swarm.

The real monitor (rule-based checks plus an LLM grader) reports catch rate on V1-V3 adversarial runs
and false-alarm rate on benign runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..runlog import MessageRecord


@dataclass
class Flag:
    message_id: str
    rule: str
    score: float


class Monitor(Protocol):
    def inspect(self, messages: list[MessageRecord]) -> list[Flag]: ...


class NullMonitor:
    def inspect(self, messages: list[MessageRecord]) -> list[Flag]:
        return []
