"""Did the extra votes buy anything? A 150-vote grid against its 250-vote re-run.

The horizon run is the *same cells carried further* — same categories, seeds,
splits and startup exemplar — so this compares two things a single grid cannot:

* **within the long run**, the deep window at the old horizon (votes 101-150)
  against the same-width window at the new one (201-250), paired per cell. This
  is the answer to "is the user better off clicking longer", and pairing removes
  the between-category variance that dominates a 3-seed sample.
* **across the two runs**, the overlapping steps, which must be *identical*. That
  is the premise the whole comparison rests on: if steps 1-150 do not reproduce,
  the long run is a re-draw and the deltas below mean nothing.

It also counts the runs that only *started* because of the longer horizon — a
cell that finds its first positive at vote 232 was reported as a starved run by
the short grid, which is a statement about the horizon rather than the method.

    python analyze_horizon.py <short_results> <long_results> [out.txt]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from _cells_io import describe_load
from bench_cells import load_cells

#: Compared windows: the old horizon's deep regime, and the same width at the new
#: one. Equal widths, so the comparison is not confounded by averaging more steps.
OLD_WINDOW = (101, 150)
NEW_WINDOW = (201, 250)
CELL = ["dataset", "embedder", "category", "seed"]
METRICS = ["cost", "fpr", "fnr", "average_precision", "auroc", "n_good"]


def window_means(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return df[(df["t"] >= lo) & (df["t"] <= hi)].groupby(CELL)[METRICS].mean()


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    short_dir, long_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    pd.set_option("display.width", 220)
    short, prov_s = load_cells(short_dir)
    long_, prov_l = load_cells(long_dir)
    if short.empty or long_.empty:
        print("one side has no cells")
        return 1

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 110)
    emit(f"HORIZON COMPARISON — {int(short['t'].max())} votes against {int(long_['t'].max())}")
    emit("=" * 110)
    for prov in (prov_s, prov_l):
        emit(f"{prov['results']}: {describe_load(prov)}")

    # --- the premise: the overlap must reproduce -----------------------------
    cols = ["cost", "fpr", "fnr", "average_precision", "auroc", "n_good", "threshold"]
    a = short.set_index(CELL + ["t"])[cols].sort_index()
    b = long_.set_index(CELL + ["t"])[cols].sort_index()
    common = a.index.intersection(b.index)
    worst = (a.loc[common] - b.loc[common]).abs().max().max()
    emit()
    emit(f"OVERLAP CHECK: {len(common)} shared steps, largest absolute difference {worst:.3g}")
    emit(
        "  -> the long run reproduces the short one step for step"
        if worst < 1e-9
        else "  -> ** THE RUNS DIVERGE: the comparison below is between different draws, not horizons **"
    )

    # --- runs that only started because of the longer horizon ----------------
    first = long_.groupby(CELL)["t"].min()
    late = first[first > OLD_WINDOW[1]]
    emit()
    emit(f"RUNS THAT ONLY STARTED PAST VOTE {OLD_WINDOW[1]}: {len(late)}")
    for cellkey, t0 in late.items():
        emit(f"  first positive around vote {int(t0)}: {' / '.join(str(x) for x in cellkey)}")
    if len(late):
        emit("  (each of these was reported as a starved run at the shorter horizon)")

    # --- what the extra votes bought -----------------------------------------
    old = window_means(long_, *OLD_WINDOW)
    new = window_means(long_, *NEW_WINDOW)
    paired = old.index.intersection(new.index)
    emit()
    emit(
        f"WHAT VOTES {OLD_WINDOW[1] + 1}-{NEW_WINDOW[1]} BOUGHT — paired per cell, mean +- SE over {len(paired)} cells"
    )
    emit("negative cost / positive AP = the longer horizon is better; |mean| < 2*SE is not resolvable")
    rows = []
    for metric in METRICS:
        d = new.loc[paired, metric] - old.loc[paired, metric]
        rows.append(
            {
                "metric": metric,
                f"votes {OLD_WINDOW[0]}-{OLD_WINDOW[1]}": round(float(old.loc[paired, metric].mean()), 3),
                f"votes {NEW_WINDOW[0]}-{NEW_WINDOW[1]}": round(float(new.loc[paired, metric].mean()), 3),
                "change": f"{d.mean():+.3f} +-{d.sem():.3f}",
                "resolvable": "yes" if abs(d.mean()) > 2 * d.sem() else "no",
            }
        )
    emit(pd.DataFrame(rows).to_string(index=False))

    emit()
    emit("BY ARM")
    per_arm = []
    for (ds, emb), grp in old.loc[paired].groupby(level=["dataset", "embedder"]):
        idx = grp.index
        d_cost = new.loc[idx, "cost"] - old.loc[idx, "cost"]
        d_pos = new.loc[idx, "n_good"] - old.loc[idx, "n_good"]
        per_arm.append(
            {
                "dataset": ds,
                "embedder": emb,
                "cells": len(idx),
                "cost_old": round(float(old.loc[idx, "cost"].mean()), 3),
                "cost_new": round(float(new.loc[idx, "cost"].mean()), 3),
                "d_cost": f"{d_cost.mean():+.3f} +-{d_cost.sem():.3f}",
                "d_positives": f"{d_pos.mean():+.1f}",
            }
        )
    emit(pd.DataFrame(per_arm).to_string(index=False))

    # --- the runs that never worked ------------------------------------------
    emit()
    stuck_old = int((old.loc[paired, "cost"] > 0.9).sum())
    stuck_new = int((new.loc[paired, "cost"] > 0.9).sum())
    emit(f"RUNS STILL NOT WORKING (window cost > 0.9): {stuck_old} at the old horizon -> {stuck_new} at the new one")
    moved = new.loc[paired, "cost"] - old.loc[paired, "cost"]
    emit(
        f"cells improved by >0.01: {int((moved < -0.01).sum())}, "
        f"unchanged: {int((moved.abs() <= 0.01).sum())}, worse by >0.01: {int((moved > 0.01).sum())}"
    )

    if out:
        out.write_text("\n".join(lines) + "\n")
        print(f"\n(written to {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
