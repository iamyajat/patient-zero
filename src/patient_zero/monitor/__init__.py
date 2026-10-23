from .base import Detector, Flag, MessageContext, Monitor, NullMonitor, OnlineGuard


def make_guard(name: str | None, threshold: float | None = None, judge_model: str = "openai", data_dir: str = "data",
               thresholds: dict[str, float] | None = None):
    """An online guard from a monitor name (monitor.run.MONITORS). Use the threshold `pz calibrate` reports."""
    if name is None:
        return None
    from .guards import DetectorGuard
    from .run import MONITORS

    if name not in MONITORS:
        raise ValueError(f"unknown guard {name!r}; available: {sorted(MONITORS)}")
    if threshold is None and not thresholds:
        raise ValueError("a guard needs guard_threshold or per-detector guard_thresholds (from calibration)")
    return DetectorGuard(name, threshold, judge_model, data_dir, thresholds)
