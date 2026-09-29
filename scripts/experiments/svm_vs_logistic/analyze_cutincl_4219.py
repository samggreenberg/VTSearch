"""#4219: the head comparison at every Inclusion stop, from the ``__cutincl`` side frame.

The shipped cut (fold-anchored, ``mid_tilt``, kappa 0.3, ``qmean``) was tuned on
SVM scores, and a user moves the Inclusion slider.  Each arm's own trajectory is
re-cut at every stop in ``LOGREG_CUT_INCL_KS``; this pairs the arms on
(category, seed, click, stop) and reports ``other - svm`` per stop, as a mean
over clicks 1-150 per cell, with an SE clustered on category.

Instrument check first: the k = 0 re-cut must reproduce the live cut, since the
live rule at inclusion 0 is the same rule.  A mismatch means the side frame is
not measuring the shipped cut and nothing below it should be read.

    python analyze_cutincl_4219.py --root /expscratch/$USER/logreg-4219 --out DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))

KEYS = ["category", "seed", "t"]
METRICS = ("cut_cost", "cut_regret", "k_oracle_cost", "cut_fnr", "cut_fpr")


def load_cutincl(arm_dir: Path) -> pd.DataFrame:
    parts = []
    for f in sorted((arm_dir / "cells").glob("task_*__cutincl.csv*")):
        if f.stat().st_size == 0:
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        parts.append(df[[*KEYS, "inclusion_k", "cut_rule", *METRICS]])
    if not parts:
        raise SystemExit(f"{arm_dir}: no __cutincl rows")
    return pd.concat(parts, ignore_index=True)


def clustered(x: pd.Series) -> tuple[float, float, int]:
    g = x.groupby(level="category").mean()
    return float(x.mean()), float(g.std(ddof=1) / np.sqrt(len(g))), int(len(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arms", default="lrconv,svmc01")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import _cells_io

    arms = ["svm", *[a for a in args.arms.split(",") if a]]
    frames = {a: load_cutincl(args.root / a / "results") for a in arms}

    # Instrument check: k = 0 re-cut vs the live base row's cost, per arm.
    checks = []
    for a in arms:
        base, _ = _cells_io.load_arm(args.root / a / "results")
        k0 = frames[a][frames[a]["inclusion_k"] == 0]
        m = k0.merge(base[[*KEYS, "cost"]], on=KEYS, how="inner")
        checks.append({"arm": a, "n": len(m), "max_abs_diff": float((m["cut_cost"] - m["cost"]).abs().max())})
    chk = pd.DataFrame(checks)
    chk.to_csv(args.out / "cutincl_k0_check.csv", index=False)
    print("k=0 re-cut vs live cost:\n" + chk.to_string(index=False))

    rows = []
    for other in arms[1:]:
        m = frames["svm"].merge(frames[other], on=[*KEYS, "inclusion_k"], suffixes=("_s", "_o"))
        for k, g in m.groupby("inclusion_k"):
            cell = g.groupby(["category", "seed"]).mean(numeric_only=True)
            for metric in METRICS:
                mean, se, n = clustered(cell[f"{metric}_o"] - cell[f"{metric}_s"])
                rows.append({"arm": other, "inclusion_k": k, "metric": metric, "mean": mean, "se": se, "n": n})
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "cutincl_paired.csv", index=False)
    out["s"] = out.apply(lambda r: f"{r['mean']:+.4f}±{r['se']:.4f}", axis=1)
    print("\nother - svm, mean over clicks per cell, SE clustered on category:")
    print(out.pivot_table(index=["arm", "inclusion_k"], columns="metric", values="s", aggfunc="first").to_string())

    levels = pd.concat(
        [
            f.groupby("inclusion_k")[["cut_cost", "k_oracle_cost"]].mean().assign(arm=a).reset_index()
            for a, f in frames.items()
        ]
    )
    levels.to_csv(args.out / "cutincl_levels.csv", index=False)
    figure(out, levels, args.out / "cutincl_by_stop.png")
    return 0


def figure(paired: pd.DataFrame, levels: pd.DataFrame, path: Path) -> None:
    """Left: each arm's cost at its own cut and at the best cut, per stop.  Right: paired
    ``other - svm`` cost and regret per stop, +-2 SE."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"svm": "#1f77b4", "lrconv": "#d9730d", "svmc01": "#7b52ab"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, g in levels.groupby("arm"):
        a1.plot(g["inclusion_k"], g["cut_cost"], "o-", color=colors.get(arm), label=f"{arm}: shipped cut")
        a1.plot(g["inclusion_k"], g["k_oracle_cost"], "--", color=colors.get(arm), alpha=0.6, label=f"{arm}: best cut")
    a1.set_yscale("log")
    a1.set_xlabel("Inclusion stop k  (negative = false alarms cost more)")
    a1.set_ylabel("cost at k (log scale; mean over clicks and cells)")
    a1.set_title("Every head, shipped cut vs the best cut", fontsize=10)
    a1.legend(fontsize=7, ncol=2)
    for arm, g in paired.groupby("arm"):
        for metric, style in (("cut_cost", "o-"), ("cut_regret", "s:")):
            h = g[g["metric"] == metric].sort_values("inclusion_k")
            a2.errorbar(
                h["inclusion_k"],
                h["mean"],
                yerr=2 * h["se"],
                fmt=style,
                color=colors.get(arm),
                capsize=3,
                label=f"{arm} − svm: {'cost' if metric == 'cut_cost' else 'regret (the cut)'}",
            )
    a2.axhline(0, color="black", lw=0.8)
    a2.set_yscale("symlog", linthresh=0.02)
    a2.set_xlabel("Inclusion stop k")
    a2.set_ylabel("paired difference from the shipped SVM (±2 SE)")
    a2.set_title("Below 0 = better than the shipped SVM", fontsize=10)
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
