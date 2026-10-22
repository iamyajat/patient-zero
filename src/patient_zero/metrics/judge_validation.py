"""Validate the answer-equivalence judge against human labels before trusting it.

export_answer_labels: a CSV of answer pairs the judge actually has to decide (the cheap path settles the rest),
  led by every pair where the first judge (prompt v1) and F1 disagreed, then a random sample stratified by F1.
score_answer_labels: agreement and Cohen's kappa with the human labels for each candidate judge, and for the F1
  rule as the baseline a judge must beat. Disagreement pairs are capped at half the sheet; each pair's stratum is kept
  in a sidecar file labellers don't see, and overall agreement is re-weighted to the full population of judged pairs.
  The pre-registration requires >= 90% weighted agreement.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from pathlib import Path

import pandas as pd

from ..monitor.labels import cohen_kappa
from .basic import f1, load_exp, normalize
from .equivalence import FALLBACK_F1, AnswerJudge, experiment_pairs, pair_key

COLUMNS = ["pair_id", "question", "answer_a", "answer_b", "f1", "label", "notes"]
GUIDE = ("label = same if a fair marker who accepts one answer would accept the other (aliases, qualifiers, units, formats, "
         "extra correct detail); different if they point to different things or one drops the asked-for part. Leave notes if unsure.")


def _legacy_key(q: str, a: str, b: str) -> str:
    """Cache key of the v1 prompt (before keys were versioned)."""
    x, y = sorted([normalize(a), normalize(b)])
    return hashlib.sha256(f"{q}|{x}|{y}".encode()).hexdigest()


def export_answer_labels(exp_dirs: list[str | Path], out: str | Path, n: int = 100, seed: int = 0,
                         prior_judge: str | None = "qwen", data_dir: str | Path = "data") -> Path:
    pairs = []
    for d in exp_dirs:
        runs, msgs = load_exp(d)
        pairs += experiment_pairs(runs, msgs)
    judge = AnswerJudge("__none__", data_dir)
    todo = judge.pending(pairs)  # only pairs the cheap path cannot decide
    prior = AnswerJudge(prior_judge, data_dir).verdicts if prior_judge else {}
    disagree, rest = [], []
    for q, a, b in todo:
        v = prior.get(_legacy_key(q, a, b))
        (disagree if v is not None and v != (f1(a, b) >= FALLBACK_F1) else rest).append((q, a, b))
    rng = random.Random(seed)
    rng.shuffle(rest)
    buckets = {"zero": [p for p in rest if f1(p[1], p[2]) == 0], "low": [p for p in rest if 0 < f1(p[1], p[2]) < FALLBACK_F1],
               "mid": [p for p in rest if f1(p[1], p[2]) >= FALLBACK_F1]}
    rng.shuffle(disagree)
    chosen = disagree[: n // 2]
    strata = {pair_key(*p): "disagree" for p in chosen}
    while len(chosen) < n and any(buckets.values()):
        for b in buckets.values():
            if b and len(chosen) < n:
                chosen.append(p := b.pop())
                strata[pair_key(*p)] = "rest"
    rng.shuffle(chosen)  # labellers must not see which pairs were disagreements
    rows = [{"pair_id": pair_key(q, a, b), "question": q, "answer_a": a, "answer_b": b, "f1": round(f1(a, b), 2),
             "label": "", "notes": ""} for q, a, b in chosen]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
    out.with_suffix(".guide.txt").write_text(GUIDE + "\n")
    population = {"disagree": len(disagree), "rest": len(rest)}
    pd.DataFrame([{"pair_id": k, "stratum": v, "population": population[v]} for k, v in strata.items()]).to_csv(
        out.with_suffix(".strata.csv"), index=False)  # keep away from labellers
    n_dis = sum(v == "disagree" for v in strata.values())
    print(f"{len(rows)} pairs: {n_dis} of {len(disagree)} v1-judge/F1 disagreements, {len(rows) - n_dis} of {len(rest)} others -> {out}")
    return out


def merge_labels(labels_csv: str | Path, sheet_csv: str | Path) -> pd.DataFrame:
    """Labels joined to the exported sheet by pair_id, so text mangled by a spreadsheet app ("October 1922" ->
    "Oct-22", curly quotes) never reaches a judge. One or more `label*` columns; with several, `label` is the
    consensus where annotators agree, and blank where they disagree (to be resolved by discussion)."""
    raw = pd.read_csv(labels_csv)
    cols = [c for c in raw.columns if c.startswith("label")]
    norm = raw[cols].apply(lambda c: c.astype(str).str.strip().str.lower().where(c.notna(), ""))
    sheet = pd.read_csv(sheet_csv).drop(columns=["label", "notes"], errors="ignore")
    out = sheet.merge(pd.concat([raw[["pair_id"]], norm, raw.get("notes", pd.Series(dtype=str)).rename("notes")], axis=1),
                      on="pair_id", how="inner")
    if len(cols) == 1:
        out["label"] = out[cols[0]]
    else:
        agree = out[cols].nunique(axis=1) == 1
        out["label"] = out[cols[0]].where(agree, "")
    out.attrs["annotators"] = cols
    return out


def annotator_agreement(df: pd.DataFrame) -> dict:
    cols = df.attrs.get("annotators") or [c for c in df.columns if c.startswith("label_")]
    if len(cols) < 2:
        return {}
    a, b = ((df[c] == "same").astype(int).tolist() for c in cols[:2])
    return {"annotators": cols[:2], "pairs": len(df), "agreement": sum(x == y for x, y in zip(a, b)) / len(a),
            "kappa": cohen_kappa(a, b), "disagreements": int(sum(x != y for x, y in zip(a, b)))}


def score_answer_labels(csv: str | Path, models: list[str], data_dir: str | Path = "data", concurrency: int = 4,
                        sheet: str | Path | None = None) -> pd.DataFrame:
    df = merge_labels(csv, sheet) if sheet else pd.read_csv(csv)
    pairs = list(zip(df.question, df.answer_a, df.answer_b))
    labelled = df.label.astype(str).str.strip().str.lower().isin(["same", "different"])
    human = (df.label.astype(str).str.strip().str.lower() == "same").astype(int)
    preds = {"F1 >= 0.5 (baseline)": [int(f1(a, b) >= FALLBACK_F1) for _, a, b in pairs]}
    for m in models:
        judge = AnswerJudge(m, data_dir)
        asyncio.run(judge.fill(pairs, concurrency))
        eq = judge.equiv()
        preds[m] = [int(eq.same(q, a, b)) for q, a, b in pairs]
    strata_path = Path(sheet or csv).with_suffix(".strata.csv")
    strata = pd.read_csv(strata_path).set_index("pair_id") if strata_path.exists() else None
    rows = []
    for name, p in preds.items():
        row = {"judge": name, "pairs": len(df)}
        if labelled.any():
            h, pr = human[labelled].tolist(), pd.Series(p)[labelled].tolist()
            row.update({"labelled": len(h), "agreement": sum(x == y for x, y in zip(h, pr)) / len(h), "kappa": cohen_kappa(h, pr)})
            if strata is not None:
                hit = (human == pd.Series(p)) & labelled
                weighted, total = 0.0, 0
                for stratum, g in strata.groupby("stratum"):
                    ids = df.pair_id.isin(g.index) & labelled
                    if ids.any():
                        acc = hit[ids].mean()
                        row[f"agreement_{stratum}"] = acc
                        weighted += acc * g.population.iloc[0]
                        total += g.population.iloc[0]
                row["agreement_weighted"] = weighted / total if total else float("nan")
        row["says_same"] = sum(p) / len(p)
        rows.append(row)
    out = pd.DataFrame(rows)
    # judges' raw verdicts next to the labels, for reading disagreements
    for name, p in preds.items():
        df[f"pred: {name}"] = ["same" if x else "different" for x in p]
    df.to_csv(Path(csv).with_suffix(".scored.csv"), index=False)
    out.attrs["annotators"] = annotator_agreement(df) if sheet else {}
    return out
