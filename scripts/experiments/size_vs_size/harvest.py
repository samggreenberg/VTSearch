#!/usr/bin/env python3
"""Size vs size (#4160): is training on small objects bad because of what it
learns, or because of what it finds?

    python harvest.py --exp <exp dir> --analysis <analysis dir> --out <dir>

A small-object hunt finds few positives: the text sort ranks small instances
low, so some runs reach click 150 having voted Good on 0-2 images. This splits
the train-S row by that harvest:

* ``harvest.csv``: positives found (``n_good`` at the last click) per run.
* ``harvest_share.csv``: per training size, the share of runs finding < 3.
* ``harvest_contrasts.csv``: train S minus train M on each pure test size,
  paired on (class, seed) and clustered by class, over all runs and again
  restricted to the (class, seed) pairs whose S run found >= 3 positives.
  The restriction is on the S run only, and the M run of the same pair stays
  in, so the pairing survives. It still selects pairs where the S hunt went
  well, which is the point, and it is read as "given a small-object user who
  found a few positives", not as a population rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

TRAIN = {"small": "S", "medium": "M", "large": "L", "mix-equal": "SML=", "mix-natural": "SMLn"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True, help="analyze.py's output dir (cells_final.csv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--click", type=int, default=150)
    ap.add_argument("--min-found", type=int, default=3)
    args = ap.parse_args()

    import _cells_io

    df, _ = _cells_io.load_arm(args.exp / "results")
    last = df[["category", "seed", "t", "n_good"]].sort_values("t").groupby(["category", "seed"]).last().reset_index()
    cb = last["category"].str.partition("@")
    last = pd.DataFrame({"cls": cb[0], "seed": last["seed"], "train": cb[2].map(TRAIN), "n_good": last["n_good"]})
    args.out.mkdir(parents=True, exist_ok=True)
    last.to_csv(args.out / "harvest.csv", index=False)

    cf = pd.read_csv(args.analysis / "cells_final.csv")
    cf = cf[cf["click"] == args.click]
    m = cf.merge(last, on=["cls", "seed", "train"], how="left")
    # A run that never wrote a row found no Good and Bad pair: it harvested < 2.
    m["n_good"] = m["n_good"].fillna(0)

    runs = m[m["test"] == "S"].drop_duplicates(["cls", "seed", "train"])
    share = (
        runs.assign(low=runs["n_good"] < args.min_found)
        .groupby("train")
        .agg(runs=("low", "size"), share_below=("low", "mean"), median_found=("n_good", "median"))
        .reset_index()
    )
    share.to_csv(args.out / "harvest_share.csv", index=False)
    print(share.round(3).to_string(index=False))

    rows = []
    for test in ("S", "M", "L"):
        w = m[m["test"] == test].pivot_table(index=["cls", "seed"], columns="train", values=["cost", "auroc", "n_good"])
        for label, ok in (
            ("all runs", w[("n_good", "S")] >= 0),
            (f"S found >= {args.min_found}", w[("n_good", "S")] >= args.min_found),
        ):
            for metric in ("cost", "auroc"):
                d = (w[(metric, "S")] - w[(metric, "M")])[ok].dropna()
                pc = d.groupby(level="cls").mean()
                se = float(pc.std(ddof=1) / np.sqrt(len(pc)))
                rows.append(
                    {
                        "test": test,
                        "subset": label,
                        "metric": metric,
                        "S_minus_M": float(pc.mean()),
                        "se": se,
                        "resolvable": bool(abs(pc.mean()) >= 2 * se),
                        "n_classes": len(pc),
                        "n_pairs": len(d),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "harvest_contrasts.csv", index=False)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
