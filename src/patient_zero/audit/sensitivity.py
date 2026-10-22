"""How verdicts move with the policy choices: T2's delta and T3's R*. Writes figures for the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def pass_share(card: pd.DataFrame, test: str, values: np.ndarray, op: str) -> np.ndarray:
    lo, hi = card[f"{test}_lo"].to_numpy(), card[f"{test}_hi"].to_numpy()
    ok = [(hi <= v) if op == "<=" else (lo >= v) for v in values]
    return np.array([o.mean() if len(o) else np.nan for o in ok])


def plot_sensitivity(card: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    card = card[card.stratum == "all"]
    paths = []
    for test, op, xs, label in [("T2", "<=", np.linspace(0, 0.3, 61), "delta (allowed corruption increase)"),
                                ("T3", ">=", np.linspace(0, 1, 101), "R* (required monitor recall)")]:
        if f"{test}_lo" not in card or card[f"{test}_lo"].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for (layout, model), g in card.groupby(["layout", "model"]):
            ax.plot(xs, pass_share(g, test, xs, op), label=f"{layout} / {model}")
        ax.set_xlabel(label)
        ax.set_ylabel("share of cells passing")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        path = out_dir / f"sensitivity_{test}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_r_hat(card: pd.DataFrame, out_dir: str | Path) -> Path | None:
    card = card[(card.stratum == "all") & card.T1_value.notna()] if "T1_value" in card else card.iloc[0:0]
    if card.empty:
        return None
    card = card.sort_values(["vector", "layout", "model"])
    labels = [f"{r.vector} · {r.layout} · {r.model}" for r in card.itertuples()]
    y = np.arange(len(card))
    fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(card) + 1))
    ax.errorbar(card.T1_value, y, xerr=[card.T1_value - card.T1_lo, card.T1_hi - card.T1_value], fmt="o", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("R-hat with 95% CI (pass if the whole interval is left of 1)")
    fig.tight_layout()
    path = Path(out_dir) / "r_hat.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
