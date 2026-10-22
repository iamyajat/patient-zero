"""Monitor interfaces (C + A). A monitor scores messages as a deployer could: it sees the question, the documents
each agent received, tool results and the messages, never the gold answer or the injection.

Offline: `monitor.run.monitor_experiment` scores logged runs into flags.jsonl. Online: an `OnlineGuard` scores each
message as it is produced and quarantines it above a calibrated threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..runlog import MessageRecord


@dataclass
class Flag:
    message_id: str
    rule: str
    score: float  # 0..1; thresholds are calibrated on benign traffic, not fixed here
    reason: str = ""


@dataclass
class MessageContext:
    question: str
    docs: list[str]  # documents/tool results the sender and its upstream agents saw
    visible: list[dict] = field(default_factory=list)  # teammates' messages the sender saw
    own_previous: dict | None = None  # the sender's own previous-round message (debates)


class Detector(Protocol):
    name: str

    async def score(self, msg: dict, ctx: MessageContext) -> tuple[float, str]: ...


class Monitor(Protocol):
    def inspect(self, messages: list[MessageRecord]) -> list[Flag]: ...


class NullMonitor:
    def inspect(self, messages: list[MessageRecord]) -> list[Flag]:
        return []


class OnlineGuard(Protocol):
    """Scores each message as it is produced; above `threshold` it is quarantined (no other agent sees it)."""

    name: str
    threshold: float

    async def score(self, message: MessageRecord, docs: list[str], question: str = "") -> float: ...
