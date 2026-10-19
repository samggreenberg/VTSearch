#!/usr/bin/env python
"""Are a re-run's cells the run's cells? Every file of each task, compared value for value.

A paired study that reuses an earlier run's sessions as its control has to show that the control
would come out the same at the study's commit (#4777 reuses #4762's 6,744 sessions). This re-reads
each ``task_NNNN*.csv`` a re-run wrote and the run's file of the same name, drops the wall-clock
columns (``*_seconds``: timings, not results), and reports any cell whose remaining values differ.

    python cells_identical.py --rerun <rerun>/results/cells --run <run>/results/cells

Exit status 0 only when every re-run file has a twin with identical values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[[c for c in df.columns if not c.endswith("_seconds")]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rerun", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()
    files = sorted(args.rerun.glob("task_*.csv*"))
    if not files:
        print(f"no task files under {args.rerun}")
        return 2
    bad = 0
    tasks = set()
    for f in files:
        plain = f.name.removesuffix(".gz")
        tasks.add(plain.split("__")[0].removesuffix(".csv"))
        # The twin in either spelling: a CALIB_CELLS_GZIP re-run of a plain run, or the reverse.
        twin = next((t for t in (args.run / plain, args.run / f"{plain}.gz") if t.exists()), args.run / f.name)
        if not twin.exists():
            print(f"MISSING {f.name}: no file of that name in the run")
            bad += 1
            continue
        a, b = frame(f), frame(twin)
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            print(f"DIFFER  {f.name}: shape {a.shape} vs {b.shape}")
            bad += 1
            continue
        neq = ~((a == b) | (a.isna() & b.isna()))
        if neq.to_numpy().any():
            cols = [c for c in a.columns if neq[c].any()]
            print(f"DIFFER  {f.name}: {int(neq.to_numpy().sum())} values in {cols[:8]}")
            bad += 1
    print(f"{len(files)} files of {len(tasks)} tasks: {len(files) - bad} identical, {bad} not")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
