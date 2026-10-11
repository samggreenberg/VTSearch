#!/usr/bin/env python
"""#4776's second candidate, priced: the head started from the Goods' centroid against today's head.

Two session sets on the same people (FHIBE) or classes and seeds (COCO): the arm
(``CALIB_HEAD=linear_svm_prior``, every fit takes it, so Test, the picks, the folds and the line
all move) and its control, today's app (the shadow run's untagged rows, which are today's sessions
exactly). Unlike a hand-over rule the sessions differ, so they pair run for run, not click for
click.

What a Find returns at every vote 1..150 is read as the face review reads it
(``sota_face_tables.read_find``): Test's untagged rows, the end-of-run check's left out, carried
between rows, 0 before the first. Every run is in every average (#4631). Per set: the session mean
of Find's F-beta and of AP over votes 1-50, 51-150 and 1-150, arm minus control, mean and SE over
runs, the share of runs it wins and loses, and the curves.

Usage::

    python prior_4776.py --pair NAME=ARM_CELLS::CONTROL_CELLS[::DATASET] ... --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import _cells_io  # noqa: E402
from _cells_paths import main_frame_files  # noqa: E402

HORIZON = 150
COLS = [
    "t",
    "phase",
    "dataset",
    "category",
    "seed",
    "gmm_variant",
    "schedule",
    "pool_variant",
    "fbeta",
    "average_precision",
    "n_flagged",
]
SPANS = ((1, 50), (51, 150), (1, 150))


def _carry(ts: np.ndarray, vals: np.ndarray, fill: float) -> np.ndarray:
    out = np.full(HORIZON, np.nan)
    for t, v in zip(ts, vals, strict=True):
        if 1 <= t <= HORIZON:
            out[int(t) - 1] = v
    s = pd.Series(out).ffill()
    return s.fillna(fill).to_numpy() if fill == fill else s.to_numpy()


def read_runs(path: str, dataset: str | None) -> list[dict]:
    """Find's F-beta, AP and kept count at every vote, per run in this cell."""
    try:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return []
    df = df[(df["phase"].fillna("").astype(str) != "ceiling") & (df["t"] > 0)]
    df = df[~_cells_io.check_rows(df)]
    out = []
    for (ds, cat, seed), run in df.groupby(["dataset", "category", "seed"], sort=False):
        if dataset and ds != dataset:
            continue
        b = _cells_io._base_rows(run).groupby("t").last()
        if b.empty:
            continue
        ts = b.index.to_numpy()
        out.append(
            {
                "key": (ds, cat, int(seed)),
                "F": _carry(ts, b["fbeta"].to_numpy(float), 0.0),
                "AP": _carry(ts, b["average_precision"].to_numpy(float), float("nan")),
                "kept": _carry(ts, b["n_flagged"].to_numpy(float), float("nan")),
            }
        )
    return out


def load(cells: str, dataset: str | None, workers: int) -> dict:
    files = [str(f) for f in main_frame_files(cells)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        got = [r for rs in pool.map(partial(read_runs, dataset=dataset), files, chunksize=8) for r in rs]
    return {r["key"]: r for r in got}


def _se(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pair", action="append", required=True, help="NAME=ARM_CELLS::CONTROL_CELLS[::DATASET]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for spec in args.pair:
        name, _, rest = spec.partition("=")
        parts = rest.split("::")
        arm_dir, ctl_dir, dataset = parts[0], parts[1], (parts[2] if len(parts) > 2 else None)
        arm, ctl = load(arm_dir, dataset, args.workers), load(ctl_dir, dataset, args.workers)
        keys = sorted(set(arm) & set(ctl))
        if not keys:
            print(f"{name}: no paired runs", file=sys.stderr)
            continue
        row: dict = {
            "set": name,
            "runs": len(keys),
            "arm_only": len(set(arm) - set(ctl)),
            "control_only": len(set(ctl) - set(arm)),
        }
        for metric in ("F", "AP"):
            A = np.stack([arm[k][metric] for k in keys])
            C = np.stack([ctl[k][metric] for k in keys])
            for lo, hi in SPANS:
                a, c = np.nanmean(A[:, lo - 1 : hi], axis=1), np.nanmean(C[:, lo - 1 : hi], axis=1)
                d = a - c
                row[f"{metric}_ctl_{lo}_{hi}"] = float(np.nanmean(c))
                row[f"{metric}_arm_{lo}_{hi}"] = float(np.nanmean(a))
                row[f"d{metric}_{lo}_{hi}"] = float(np.nanmean(d))
                row[f"d{metric}_{lo}_{hi}_se"] = _se(d)
                if metric == "F":
                    row[f"win_{lo}_{hi}"] = float((d > 1e-9).mean())
                    row[f"loss_{lo}_{hi}"] = float((d < -1e-9).mean())
            for t in range(HORIZON):
                curves.append(
                    {
                        "set": name,
                        "metric": metric,
                        "vote": t + 1,
                        "control": float(np.nanmean(C[:, t])),
                        "arm": float(np.nanmean(A[:, t])),
                        "diff": float(np.nanmean(A[:, t] - C[:, t])),
                        "diff_se": _se(A[:, t] - C[:, t]),
                    }
                )
        K = np.stack([arm[k]["kept"] for k in keys]), np.stack([ctl[k]["kept"] for k in keys])
        for t in (8, 10, 25, 50, 150):
            row[f"kept_arm_median_{t}"] = float(np.nanmedian(K[0][:, t - 1]))
            row[f"kept_ctl_median_{t}"] = float(np.nanmedian(K[1][:, t - 1]))
        rows.append(row)
        print(f"{name}: {len(keys)} paired runs", file=sys.stderr)
    pd.DataFrame(rows).round(5).to_csv(args.out / "prior_ab.csv", index=False)
    pd.DataFrame(curves).round(5).to_csv(args.out / "prior_curves.csv", index=False)
    print(f"wrote prior_ab.csv, prior_curves.csv -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
