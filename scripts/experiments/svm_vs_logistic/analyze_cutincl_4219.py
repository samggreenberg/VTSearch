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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
