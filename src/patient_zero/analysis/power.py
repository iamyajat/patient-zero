"""Sample-size planning from pilot rates."""

from __future__ import annotations

import math

from scipy import stats


def n_two_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-group n to detect p1 vs p2 (unpaired, two-sided)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def n_mcnemar(p10: float, p01: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Pairs needed for McNemar's test given discordant probabilities P(benign ok, adv wrong)=p10 and the reverse
    p01 (Connor 1987)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pd_, diff = p10 + p01, p10 - p01
    return math.ceil((za * math.sqrt(pd_) + zb * math.sqrt(pd_ - diff**2)) ** 2 / diff**2)


def half_width(p: float, n: int, alpha: float = 0.05) -> float:
    return stats.norm.ppf(1 - alpha / 2) * math.sqrt(p * (1 - p) / n)
