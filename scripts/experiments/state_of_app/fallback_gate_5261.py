#!/usr/bin/env python3
"""#5261: count the clicks where a trained head drew no labels line, so the fused cut decided.

Since #4452 a trained head's line under a balance is the labels line; the fused cut (``_fused_threshold``) draws it,
and Autopilot leaves the labels line's even-odds target for the line - 4 re-cut, only where ``fit_labels_line``
returns nothing. A calibration base row (``gmm_variant`` empty) records that state since #4878: ``line_head_count``
is 0 or 1 where a labels line was drawn and -1 where none was, and its ``threshold_provenance`` is ``balance`` only
where the line came from the labels. This reads every recorded State of the App / acquisition cell and counts, per
mode, beta and vote count, the trained-head base rows with no line or a provenance other than ``balance``.

  fallback_gate_5261.py scan --runs RUNS.txt --out DIR [--procs N]
  fallback_gate_5261.py tables --out DIR

``scan`` takes one run directory per line (each holding ``results/cells/task_NNNN.csv[.gz]``) and writes
``agg.csv`` (base-row counts by run, dataset, style, head, embedder, beta, t, tier, provenance, line_head_count) and
``odd.csv`` (every trained-head base row in the fallback state, with its F-beta and ``oracle_fbeta`` to price it).
``tables`` prints the per-mode, per-beta, per-vote-count rates from ``agg.csv``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from multiprocessing import Pool

import pandas as pd

COLS = [
    "seed", "dataset", "category", "style", "head", "embedder", "beta", "t", "n_good", "n_bad", "phase",
    "detector_tier", "app_trained", "gmm_variant", "threshold_provenance", "line_head_count", "threshold", "fbeta",
    "oracle_fbeta",
]  # fmt: skip
KEYS = ["run", "dataset", "style", "head", "embedder", "beta", "t", "detector_tier", "threshold_provenance",
        "line_head_count"]  # fmt: skip
CELL_RE = re.compile(r"task_\d+\.csv(\.gz)?$")
VOTE_BINS = [0, 10, 25, 50, 100, 150, 10**6]
VOTE_LABELS = ["1-10", "11-25", "26-50", "51-100", "101-150", ">150"]


def _one(path: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    run = path.split("/results/cells/")[0].rstrip("/")
    try:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    except Exception as exc:  # noqa: BLE001 - one unreadable cell is reported, not fatal
        return None, None, f"{path}: {exc}"
    base = df[df["gmm_variant"].isna()].copy()
    base["run"] = run
    keys = [k for k in KEYS if k in base.columns]
    agg = (
        base.fillna({"threshold_provenance": "NA", "head": "NA", "embedder": "NA"})
        .groupby(keys, dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    trained = base["detector_tier"] == "trained"
    odd = base[trained & ((base["line_head_count"] == -1) | (base["threshold_provenance"] != "balance"))]
    return agg, odd, None


def scan(runs_file: str, out: str, procs: int) -> None:
    runs = [line.strip() for line in open(runs_file) if line.strip()]
    files = [f for r in runs for f in glob.glob(f"{r}/results/cells/task_*.csv*") if CELL_RE.search(f)]
    print(f"{len(runs)} runs, {len(files)} cells", flush=True)
    with Pool(procs) as pool:
        res = pool.map(_one, files, chunksize=20)
    agg = pd.concat([a for a, _, _ in res if a is not None])
    agg = agg.groupby([c for c in agg.columns if c != "n"], dropna=False)["n"].sum().reset_index()
    odd = [o for _, o, _ in res if o is not None and len(o)]
    errors = [e for _, _, e in res if e]
    os.makedirs(out, exist_ok=True)
    agg.to_csv(f"{out}/agg.csv", index=False)
    (pd.concat(odd) if odd else pd.DataFrame(columns=COLS)).to_csv(f"{out}/odd.csv", index=False)
    print(f"errors {len(errors)}", *errors[:10], sep="\n")
    print(f"fallback rows {sum(len(o) for o in odd)}")


def tables(out: str) -> None:
    agg = pd.read_csv(f"{out}/agg.csv")
    print(pd.crosstab([agg.detector_tier, agg.threshold_provenance], agg.line_head_count, values=agg.n, aggfunc="sum")
          .fillna(0).astype(int).to_string())  # fmt: skip
    tr = agg[agg.detector_tier == "trained"].copy()
    tr["mode"] = tr["style"].map(lambda s: "binary" if s == "whole_image" else "region")
    tr["votes"] = pd.cut(tr.t, VOTE_BINS, labels=VOTE_LABELS)
    tr["fallback"] = ((tr.line_head_count == -1) | (tr.threshold_provenance != "balance")) * tr.n
    by = tr.groupby(["mode", "beta", "votes"], observed=True).agg(clicks=("n", "sum"), fallback=("fallback", "sum"))
    print(by.to_string())
    print(tr.groupby("mode").agg(clicks=("n", "sum"), fallback=("fallback", "sum"), runs=("run", "nunique"),
                                 first_t=("t", "min")).to_string())  # fmt: skip
    print(tr.groupby(["mode", "style"]).n.sum().to_string())
    print(tr.groupby("dataset").n.sum().to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--runs", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    t = sub.add_parser("tables")
    t.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "scan":
        scan(args.runs, args.out, args.procs)
    else:
        tables(args.out)


if __name__ == "__main__":
    main()
