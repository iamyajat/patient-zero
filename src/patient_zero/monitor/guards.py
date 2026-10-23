"""Online guards: the same detectors, run as each message is produced, with a threshold taken from calibration."""

from __future__ import annotations

from ..runlog import MessageRecord
from .base import MessageContext
from .run import build_detectors


class DetectorGuard:
    """With `thresholds` ({detector: t}, from calibration with combine="union"), the guard returns 1.0 when any
    detector exceeds its own threshold and 0.0 otherwise, and quarantines at threshold 0.5. Without them it falls back
    to a single threshold on the max score."""

    def __init__(self, monitor: str, threshold: float | None = None, judge_model: str = "openai", data_dir: str = "data",
                 thresholds: dict[str, float] | None = None):
        self.name = monitor
        self.thresholds = thresholds
        self.threshold = 0.5 if thresholds else threshold
        self.detectors = build_detectors(monitor, judge_model, data_dir)

    async def score(self, message: MessageRecord, docs: list[str], question: str = "") -> float:
        ctx = MessageContext(question=question, docs=docs)
        msg = message.model_dump()
        scores = {d.name: (await d.score(msg, ctx))[0] for d in self.detectors}
        if self.thresholds:
            return float(any(scores.get(k, 0.0) > t for k, t in self.thresholds.items()))
        return max(scores.values(), default=0.0)
