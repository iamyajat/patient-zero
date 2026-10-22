from .base import Detector, Flag, MessageContext, Monitor, NullMonitor, OnlineGuard


def make_guard(name: str | None, threshold: float | None = None, judge_model: str = "openai", data_dir: str = "data"):
    """An online guard from a monitor name (monitor.run.MONITORS). Use the threshold `pz calibrate` reports."""
    if name is None:
        return None
    from .guards import DetectorGuard
    from .run import MONITORS

    if name not in MONITORS:
        raise ValueError(f"unknown guard {name!r}; available: {sorted(MONITORS)}")
    if threshold is None:
        raise ValueError("a guard needs guard_threshold (from `pz calibrate`, at the target false-alarm rate)")
    return DetectorGuard(name, threshold, judge_model, data_dir)
