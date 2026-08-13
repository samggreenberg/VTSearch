"""Text sort (zero clicks) vs the trained detector (clicks over time).

Reports, per (dataset, embedder): what typing the category name gets you for
free under the GMM cut, what the detector gets at 10/25/50/100/150 votes, and
the crossover - the first vote count at which the clicked detector beats the
typed query, per cell, as a median over cells (with the share that never do).
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

CHECKPOINTS = [10, 25, 50, 100, 150]


def load_detector(results: str) -> pd.DataFrame:
    files = [
        f for f in glob.glob(f"{results}/cells/task_*.csv") if not any(k in f for k in ("sweep", "cutdiag", "cutincl"))
    ]
    frames = []
    for f in files:
        d = pd.read_csv(f)
        if len(d):
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def cost_at(sub: pd.DataFrame, t: int) -> float:
    """Detector cost at the last step <= t (what the user would see having voted t times)."""
    s = sub[sub["t"] <= t]
    return float(s.sort_values("t")["cost"].iloc[-1]) if len(s) else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    det = load_detector(args.results)
    txt = pd.read_csv(args.text)

    have = txt[txt["supports_text"] == 1].copy()
    nots = txt[txt["supports_text"] == 0]
    if len(nots):
        pairs = sorted({(r.dataset, r.embedder) for r in nots.itertuples()})
        print(f"NO TEXT TOWER (reported as n/a, not dropped): {pairs}")

    rows = []
    for r in have.itertuples():
        sub = det[
            (det.dataset == r.dataset)
            & (det.embedder == r.embedder)
            & (det.category == r.category)
            & (det.seed == r.seed)
        ]
        if sub.empty:
            rows.append(
                {
                    "dataset": r.dataset,
                    "embedder": r.embedder,
                    "category": r.category,
                    "seed": r.seed,
                    "text_cost": r.text_cost,
                    "detector_ran": 0,
                }
            )
            continue
        sub = sub.sort_values("t")
        beat = sub[sub["cost"] <= r.text_cost]
        rows.append(
            {
                "dataset": r.dataset,
                "embedder": r.embedder,
                "category": r.category,
                "seed": r.seed,
                "detector_ran": 1,
                "prevalence": r.prevalence,
                "text_cost": r.text_cost,
                "text_AP": r.text_AP,
                "text_auroc": r.text_auroc,
                "text_oracle_cost": r.text_oracle_cost,
                **{f"det_cost_t{t}": cost_at(sub, t) for t in CHECKPOINTS},
                "det_AP_final": float(sub["average_precision"].iloc[-1]),
                "crossover_t": int(beat["t"].iloc[0]) if len(beat) else -1,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    ok = out[out.detector_ran == 1].copy()
    ok["arm"] = ok.dataset + " x " + ok.embedder
    print("\n" + "=" * 110)
    print("TEXT SORT (0 clicks, GMM cut) vs TRAINED DETECTOR - cost = fpr + fnr, lower is better")
    print("=" * 110)
    agg = ok.groupby("arm").agg(
        cells=("text_cost", "size"),
        text_cost=("text_cost", "mean"),
        text_AP=("text_AP", "mean"),
        **{f"det_t{t}": (f"det_cost_t{t}", "mean") for t in CHECKPOINTS},
        det_AP=("det_AP_final", "mean"),
    )
    print(agg.round(4).to_string())

    print("\n" + "=" * 110)
    print("CROSSOVER - clicks needed for the detector to beat the typed query")
    print("=" * 110)
    cross = ok.groupby("arm").apply(
        lambda g: pd.Series(
            {
                "cells": len(g),
                "never_beats": int((g.crossover_t < 0).sum()),
                "pct_never": round(100 * (g.crossover_t < 0).mean(), 1),
                "median_t": (
                    float(np.median(g.loc[g.crossover_t >= 0, "crossover_t"])) if (g.crossover_t >= 0).any() else np.nan
                ),
                "min_t": (int(g.loc[g.crossover_t >= 0, "crossover_t"].min()) if (g.crossover_t >= 0).any() else -1),
            }
        ),
        include_groups=False,
    )
    print(cross.to_string())

    print("\n" + "=" * 110)
    print("PER-CATEGORY DETAIL (text vs detector at 150 votes)")
    print("=" * 110)
    detail = (
        ok.groupby(["arm", "category"])
        .agg(
            prev=("prevalence", "mean"),
            text_cost=("text_cost", "mean"),
            text_AP=("text_AP", "mean"),
            det_t150=("det_cost_t150", "mean"),
            det_AP=("det_AP_final", "mean"),
            crossover=("crossover_t", "median"),
        )
        .round(4)
    )
    print(detail.to_string())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
