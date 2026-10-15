#!/usr/bin/env python3
"""#4216: positives found per run, read off finished simulated-user cells (no new runs).

    python scan_found.py --root /expscratch/$USER --dirs run_dirs.txt --out <scan dir> [--procs 32]

``run_dirs.txt`` lists one results directory per line, relative to ``--root`` (each holds
``cells/task_NNNN.csv[.gz]``).  For every main cell file this writes one row per
(category, seed, embedder) to ``<scan dir>/<label>.csv``:

* ``found150``: positives found by vote 150, i.e. ``n_good`` at the last voting row with
  ``t <= 150``.  The end-of-run spot check's rows (``phase == "check"``) are not clicks and a
  supervised-skyline row (``gmm_variant`` ``skyline_*``, ``t = 0``) is not a vote; both are
  dropped.  Before #4736's harness a run wrote no row until its first Good, so a run with no
  row by vote k had found 0 by then.
* ``found_<k>``: the same at votes 5, 10, 25, 50, 75, 100, 125 and 150 (the harvest curve).
* ``phase150``: Autopilot's phase at the last voting row by vote 150; ``left_good_t``: the
  first vote outside the Good phase (-1 if the run never left it).

A main file with no voting row at all is written as ``is_empty = 1`` with no cell: the run
never found a Good in the whole session.  ``starved.py`` maps it back to its cell from the
task index.  The label is the run directory's name, prefixed by its study directory when that
is not ``state-of-the-app``.
"""

from __future__ import annotations

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

WANT = ["category", "seed", "t", "n_good", "embedder", "style", "phase", "n_test_pos", "gmm_variant"]
MAIN = re.compile(r"task_\d+\.csv(\.gz)?$")
CHECKPOINTS = (5, 10, 25, 50, 75, 100, 125, 150)


def _empty(path: str) -> list[dict]:
    return [{"file": os.path.basename(path), "is_empty": 1}]


def one(path: str) -> list[dict]:
    try:
        df = pd.read_csv(path, usecols=lambda c: c in WANT)
    except pd.errors.EmptyDataError:
        return _empty(path)
    if "phase" in df:
        df = df[df["phase"].fillna("").astype(str) != "check"]
    if "gmm_variant" in df:
        df = df[~df["gmm_variant"].fillna("").astype(str).str.startswith("skyline")]
    df = df[df["t"] >= 1]
    if df.empty:
        return _empty(path)
    out = []
    keys = [k for k in ("category", "seed", "embedder") if k in df]
    for key, g in df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(keys, key))
        g = g.sort_values("t", kind="stable")
        g150 = g[g["t"] <= 150]
        not_good = g.loc[g["phase"].astype(str) != "good", "t"] if "phase" in g else pd.Series(dtype=int)
        rec.update(
            file=os.path.basename(path),
            is_empty=0,
            found150=int(g150["n_good"].max()) if len(g150) else 0,
            found_end=int(g["n_good"].max()),
            first_row_t=int(g["t"].min()),
            max_t=int(g["t"].max()),
            t3=int(g.loc[g["n_good"] >= 3, "t"].min()) if (g["n_good"] >= 3).any() else -1,
            n_test_pos=float(g["n_test_pos"].iloc[0]) if "n_test_pos" in g else float("nan"),
            style=str(g["style"].iloc[0]) if "style" in g else "",
            phase150=str(g150["phase"].iloc[-1]) if len(g150) and "phase" in g150 else "",
            left_good_t=int(not_good.min()) if len(not_good) else -1,
        )
        per_t = g.drop_duplicates("t", keep="last").set_index("t")["n_good"]
        for k in CHECKPOINTS:
            prior = per_t[per_t.index <= k]
            rec[f"found_{k}"] = int(prior.iloc[-1]) if len(prior) else 0
        out.append(rec)
    return out


def label_of(rel: str) -> str:
    base = rel.rstrip("/")
    base = base[: -len("/results")] if base.endswith("/results") else base
    parent = os.path.basename(os.path.dirname(base))
    name = os.path.basename(base)
    return name if parent == "state-of-the-app" else f"{parent}__{name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="what run_dirs.txt's paths are relative to")
    ap.add_argument("--dirs", type=Path, required=True, help="one results dir per line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rels = [ln.strip() for ln in a.dirs.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    with Pool(a.procs) as pool:
        for rel in rels:
            cells = a.root / rel / "cells"
            files = sorted(str(f) for f in cells.glob("task_*") if MAIN.search(f.name))
            rows = [r for rs in pool.map(one, files, chunksize=8) for r in rs]
            df = pd.DataFrame(rows)
            label = label_of(rel)
            df.insert(0, "arm", label)
            df.to_csv(a.out / f"{label}.csv", index=False)
            n_empty = int(df["is_empty"].sum()) if "is_empty" in df else 0
            print(f"{label}: {len(files)} files, {len(df)} rows, {n_empty} with no voting row", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
