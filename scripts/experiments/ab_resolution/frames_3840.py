#!/usr/bin/env python
"""Collapse trajectory-A/B grids to one compact per-step frame (#3840).

    python frames_3840.py --grid sklearn=/expscratch/.../gmm-3585/ab_baseline/results \\
                          --grid native=/expscratch/.../gmm-3585/ab_native/results \\
                          --out /expscratch/$USER/abres-3840/frames/steps.csv.gz

Every grid is read with :func:`analyze_ab.load_base_rows` - the loader the A/B
numbers every study quotes were computed with - so a cell here is exactly the
cell those studies paired, and a per-cell mean here reproduces their Δ to the
last digit (``resolution_3840.py`` asserts that against the published
``ab_paired_cells.csv`` rather than trusting it).

Only the columns the resolution analysis reads are kept.  The raw grids hold
~3,200 rows per cell-step family because every GMM variant and sweep is a row;
the base rows are one per (cell, step), which is what makes a
subsample-thousands-of-times analysis cheap.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))

import common  # noqa: E402

common.setup_env()

import pandas as pd  # noqa: E402
from analyze_ab import load_base_rows  # noqa: E402

KEEP = (
    "grid",
    "arm",
    "dataset",
    "embedder",
    "style",
    "category",
    "seed",
    "t",
    "n_votes",
    "app_trained",
    "cost",
    "fnr",
    "fpr",
    "regret",
    "average_precision",
    "threshold",
    "n_flagged",
    "elapsed_seconds",
)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", action="append", required=True, help="NAME=results_dir, repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)

    frames = []
    for spec in args.grid:
        name, _, path = spec.partition("=")
        if not path:
            print(f"--grid wants NAME=DIR, got {spec!r}", file=sys.stderr)
            return 2
        df = load_base_rows(Path(path), name)
        if df.empty:
            print(f"{name}: no rows under {path}", file=sys.stderr)
            return 1
        df["grid"] = name
        # One base row per (cell, step) is the invariant everything downstream
        # leans on; a duplicate would silently double-weight a step.
        dup = df.duplicated(["arm", "category", "seed", "t"]).sum()
        if dup:
            print(f"{name}: {dup} duplicate (cell, step) base rows", file=sys.stderr)
            return 1
        frames.append(df[[c for c in KEEP if c in df.columns]])
        print(f"{name}: {len(df)} steps, {df.groupby(['arm', 'category', 'seed']).ngroups} cells", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
