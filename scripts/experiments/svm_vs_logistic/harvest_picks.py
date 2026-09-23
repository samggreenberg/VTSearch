"""Concatenate one Stage B arm's per-task ``picks`` side frames into one file.

    python harvest_picks.py /expscratch/$USER/svmlog-3197/stageB/svm

Writes ``<arm dir>/picks_all.csv.gz`` (the file ``stage_a.py --replay-root``
reads).  Keeps only the columns a replay needs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

COLS = ["dataset", "embedder", "category", "seed", "style", "t", "phase", "picked_id", "picked_label"]


def main() -> int:
    arm = Path(sys.argv[1])
    files = sorted((arm / "results" / "cells").glob("task_*__picks.csv"))
    frames = [pd.read_csv(f, usecols=lambda c: c in COLS) for f in files if f.stat().st_size > 0]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(arm / "picks_all.csv.gz", index=False)
    print(
        f"{arm.name}: {len(files)} files, {len(df)} picks, "
        f"{df.groupby(['dataset', 'embedder', 'category', 'seed']).ngroups} cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
