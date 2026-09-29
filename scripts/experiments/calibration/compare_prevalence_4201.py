#!/usr/bin/env python3
"""#4184 / #4201: the same seven rungs at two pool prevalences, paired per cell.

The natural grid and the ``CALIB_HAYSTACK_PREVALENCE`` grid share every cell's
test set (``thin_haystack`` only drops simulation-half negatives), so a rung's
cost at the two prevalences is a paired comparison.  For each rung and arm this
reports, over the clicks where the app shows a detector (``app_trained == 1``,
t = 1..150):

* mean cost, and the regret / oracle-cost split of it: oracle cost is what the
  model could do with the best cut, regret is what the rule's cut gave away;
* the median pool percentile the cut sits at, and FPR / FNR;
* the median high-component weight of the haystack mixture (``__cutdiag``).

It also reports each rung's Δcost (5% − natural) per cell, with its SE and the
share of cells where the thinned pool helped.

    python compare_prevalence_4201.py \\
        --natural /expscratch/$USER/progression-4184 \\
        --thinned /expscratch/$USER/progression-4184-h0.05 --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _cells_io  # noqa: E402
from analyze_progression_4184 import RUNGS  # noqa: E402

KEY = ["category", "seed"]
COLS = ["cost", "oracle_cost", "regret", "fpr", "fnr", "threshold_percentile"]


def shown(root: Path, rung: str) -> pd.DataFrame:
    """One rung's rows on the clicks where a detector is on screen."""
    df, _ = _cells_io.load_arm(root / rung / "results")
    return df[(df["app_trained"] == 1) & df["t"].between(1, 150)]


def w_hi(root: Path, rung: str) -> float:
    """Median high-component weight over every cutdiag row of the rung."""
    files = _cells_io.side_frame_files(root / rung / "results" / "cells", "__cutdiag")
    vals = [pd.read_csv(f, usecols=["w_hi"])["w_hi"].dropna() for f in files]
    return float(pd.concat(vals).median()) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--natural", type=Path, required=True)
    ap.add_argument("--thinned", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary, paired = [], []
    for rung in RUNGS:
        per_arm = {}
        for arm, root in (("natural", args.natural), ("h0.05", args.thinned)):
            df = shown(root, rung)
            row = {"rung": rung, "arm": arm, **df[COLS].mean().to_dict()}
            row["cut_pct_median"] = float(df["threshold_percentile"].median())
            row["w_hi_median"] = w_hi(root, rung)
            summary.append(row)
            per_arm[arm] = df.groupby(KEY)["cost"].mean()
        both = pd.concat(per_arm, axis=1).dropna()
        d = both["h0.05"] - both["natural"]
        paired.append(
            {
                "rung": rung,
                "n": len(d),
                "delta_mean": d.mean(),
                "delta_se": d.std(ddof=1) / np.sqrt(len(d)),
                "share_thinned_better": float((d < 0).mean()),
            }
        )
    s = pd.DataFrame(summary)
    p = pd.DataFrame(paired)
    s.to_csv(args.out / "prevalence_summary.csv", index=False, float_format="%.6g")
    p.to_csv(args.out / "prevalence_paired.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print()
    print(p.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
