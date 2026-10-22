"""Intervals and tests used by the audit: Wilson and Clopper-Pearson intervals, a task-resampling bootstrap for
ratio statistics (R-hat), McNemar's test on paired outcomes, and a task-clustered logistic GEE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Estimate:
    value: float
    lo: float
    hi: float
    n: int

    def as_dict(self) -> dict:
        return {"value": self.value, "lo": self.lo, "hi": self.hi, "n": self.n}


def wilson(k: int, n: int, alpha: float = 0.05) -> Estimate:
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Estimate(float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half)), n)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Estimate:
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return Estimate(k / n, float(lo), float(hi), n)


def bootstrap_ratio(df: pd.DataFrame, num: str, den: str, cluster: str = "task_id", b: int = 2000,
                    seed: int = 0, alpha: float = 0.05) -> Estimate:
    """sum(num)/sum(den) with a percentile CI from resampling whole clusters (tasks) with replacement."""
    g = df.groupby(cluster)[[num, den]].sum()
    total_den = g[den].sum()
    if total_den == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), len(g))
    value = g[num].sum() / total_den
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(b, len(g)))
    nums, dens = g[num].to_numpy()[idx].sum(axis=1), g[den].to_numpy()[idx].sum(axis=1)
    ratios = nums[dens > 0] / dens[dens > 0]
    lo, hi = np.quantile(ratios, [alpha / 2, 1 - alpha / 2]) if len(ratios) else (float("nan"), float("nan"))
    return Estimate(float(value), float(lo), float(hi), len(g))


def bootstrap_diff(df: pd.DataFrame, value: str, group: str, a, b_level, cluster: str = "task_id", b: int = 2000,
                   seed: int = 0, alpha: float = 0.05) -> Estimate:
    """mean(value | group==a) - mean(value | group==b_level), resampling tasks (keeps the pairing)."""
    wide = df.pivot_table(index=cluster, columns=group, values=value, aggfunc="mean")[[a, b_level]].dropna()
    d = (wide[a] - wide[b_level]).to_numpy()
    if len(d) == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(b, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(d.mean()), float(lo), float(hi), len(d))


def mcnemar(df: pd.DataFrame, outcome: str, condition: str = "condition", a="benign", b_level="adversarial",
            pair: tuple[str, ...] = ("task_id", "layout", "model", "seed_idx")) -> dict:
    """Exact McNemar test on paired binary outcomes (same task, layout, model and seed in both conditions)."""
    from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

    wide = df.pivot_table(index=list(pair), columns=condition, values=outcome, aggfunc="first")[[a, b_level]].dropna()
    wide = wide.astype(bool)
    table = pd.crosstab(wide[a], wide[b_level]).reindex(index=[False, True], columns=[False, True], fill_value=0)
    res = _mcnemar(table.to_numpy(), exact=True)
    return {"n_pairs": len(wide), "b": int(table.loc[True, False]), "c": int(table.loc[False, True]),
            "statistic": float(res.statistic), "p_value": float(res.pvalue)}


def gee_logit(df: pd.DataFrame, formula: str, cluster: str = "task_id"):
    """Logistic GEE with an exchangeable working correlation within tasks (stand-in for a mixed-effects model)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    return smf.gee(formula, groups=cluster, data=df, family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()
