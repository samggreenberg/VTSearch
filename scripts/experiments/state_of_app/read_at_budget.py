"""A run's cells read at a smaller click budget (#5141): every row, pick and rank frame after it dropped.

A State of the App review clicks to 150; a study may run the same app to 200 (#5134's arms, for the click cap,
#4903). Cut at 150, such a run is what a review would have recorded of those sessions up to its budget, minus the
closing check, which the stopping table (``stops_by_preset.py``) and the objective at every click (``by_click.py``)
do not read. Writes a results directory beside the source's: the main frames, ``__picks`` and ``__rankframes`` cut,
``prepare_info.json`` and ``crops`` linked, ``grid_shape.json`` with the new ``max_steps``. Then ``analyze.py``
runs on it as on any review run.

    python read_at_budget.py --src <run>/results --dst <out>/results --budget 150
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

#: The side frames the review's analyzer reads; the rest are left behind.
SIDE_FRAMES = ("__picks", "__rankframes")


def _cut(job: tuple[str, str], budget: int) -> int:
    src, dst = job
    df = pd.read_csv(src, low_memory=False)
    if "t" in df.columns:
        df = df[pd.to_numeric(df["t"], errors="coerce") <= budget]
    df.to_csv(dst, index=False, compression="gzip" if dst.endswith(".gz") else None)
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="the run's results directory")
    ap.add_argument("--dst", type=Path, required=True, help="the results directory to write")
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    (a.dst / "cells").mkdir(parents=True, exist_ok=True)
    for name in ("prepare_info.json", "crops"):
        if not (a.dst / name).exists():
            os.symlink(os.path.realpath(a.src / name), a.dst / name)
    shape = json.loads((a.src / "grid_shape.json").read_text())
    shape["max_steps"] = a.budget
    shape["note"] = f"{a.src} read at click {a.budget}, every row after it dropped (read_at_budget.py)"
    (a.dst / "grid_shape.json").write_text(json.dumps(shape, indent=1))
    files = sorted(
        f
        for f in (a.src / "cells").glob("task_*.csv*")
        if "__" not in f.name or any(side in f.name for side in SIDE_FRAMES)
    )
    jobs = [(str(f), str(a.dst / "cells" / f.name)) for f in files]
    cut = partial(_cut, budget=a.budget)
    if a.procs > 1:
        with Pool(a.procs) as pool:
            rows = pool.map(cut, jobs)
    else:
        rows = [cut(job) for job in jobs]
    print(f"{a.src} -> {a.dst}: {len(files)} files, {sum(rows)} rows at or before click {a.budget}")


if __name__ == "__main__":
    main()
