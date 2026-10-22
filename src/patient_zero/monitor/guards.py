"""Online guards: the same detectors, run as each message is produced, with a threshold taken from calibration."""

from __future__ import annotations

from ..runlog import MessageRecord
from .base import MessageContext
from .run import build_detectors


class DetectorGuard:
    def __init__(self, monitor: str, threshold: float, judge_model: str = "openai", data_dir: str = "data"):
        self.name = monitor
        self.threshold = threshold
        self.detectors = build_detectors(monitor, judge_model, data_dir)

    async def score(self, message: MessageRecord, docs: list[str], question: str = "") -> float:
        ctx = MessageContext(question=question, docs=docs)
        msg = message.model_dump()
        scores = [(await d.score(msg, ctx))[0] for d in self.detectors]
        return max(scores, default=0.0)
