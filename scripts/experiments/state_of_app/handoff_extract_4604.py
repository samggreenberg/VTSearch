"""#4604: pull the per-step and per-pick columns the hand-off pricing needs out of a State of the App run.

Writes <out>/steps_<tag>.csv.gz (one row per run x ordinary step: the detector the harness trained at that
step, shown or not) and <out>/picks_<tag>.csv.gz (one row per run x click pick), for one path's embedder.

    python handoff_extract_4604.py --exp <run dir> --embedder siglip --seeds 10 --tag binary_b1 --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool

import pandas as pd

STEP_COLS = [
    "dataset", "category", "seed", "embedder", "t", "phase", "app_trained", "threshold", "precision", "recall",
    "fpr", "n_test_pos", "n_test_neg", "n_good", "n_bad", "smart", "stable", "floor_count", "beta",
    "gmm_variant", "schedule", "pool_variant",
]  # fmt: skip
PICK_COLS = [
    "category", "seed", "embedder", "t", "phase", "picked_id", "picked_label", "picked_seed_rank",
    "picked_seed_score", "picked_detector_score", "acq_threshold", "n_good", "n_bad",
]  # fmt: skip


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().isin(("", "nan", "None"))


def one(args: tuple[str, str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    main, embedder, seeds = args
    head = pd.read_csv(main, nrows=0).columns
    df = pd.read_csv(main, usecols=[c for c in STEP_COLS if c in head], low_memory=False)
    df = df[(df["embedder"] == embedder) & (df["seed"] < seeds)]
    for col in ("gmm_variant", "schedule"):
        if col in df.columns:
            df = df[_blank(df[col])]
    if "pool_variant" in df.columns:
        df = df[df["pool_variant"].fillna("").astype(str).str.strip().isin(("", "max"))]
    df = df[df["phase"].fillna("").astype(str).str.strip() != "check"]
    df = df.drop(columns=[c for c in ("gmm_variant", "schedule", "pool_variant") if c in df.columns])
    pk_path = main[: -len(".csv")] + "__picks.csv"
    pk = pd.DataFrame(columns=PICK_COLS)
    if os.path.exists(pk_path) and os.path.getsize(pk_path):
        pk = pd.read_csv(pk_path)
        pk = pk[[c for c in PICK_COLS if c in pk.columns]]
        pk = pk[(pk["embedder"] == embedder) & (pk["seed"] < seeds)]
        pk = pk[pk["phase"].fillna("").astype(str).str.strip() != "check"]
    return df, pk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--seeds", type=int, required=True, help="keep seeds < this (the analysis's n_seeds)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(f"{a.exp}/results/cells/task_*.csv") if "__" not in os.path.basename(f))
    with Pool(a.procs) as pool:
        parts = pool.map(one, [(f, a.embedder, a.seeds) for f in files], chunksize=8)
    steps = pd.concat([p[0] for p in parts if len(p[0])], ignore_index=True)
    picks = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    steps.to_csv(f"{a.out}/steps_{a.tag}.csv.gz", index=False)
    picks.to_csv(f"{a.out}/picks_{a.tag}.csv.gz", index=False)
    n_runs = steps.groupby(["category", "seed"]).ngroups
    n_pk = picks.groupby(["category", "seed"]).ngroups
    print(
        f"{a.tag}: {len(files)} files, {n_runs} runs with steps, {n_pk} with picks, {len(steps)} steps, {len(picks)} picks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
