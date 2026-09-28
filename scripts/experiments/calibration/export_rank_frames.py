#!/usr/bin/env python3
"""Export compact "rank frames" for precision-floor studies that run off the GRID.

A precision-floor rule is judged by one thing: of the top *k* items of a corpus,
how many are positive.  That needs only **where the positives sit in the
corpus's ranking**, not the scores - about 50 integers per frame instead of the
~0.6 MB precision frame (``CALIB_PFRAME_STEPS``) it is cut from.  So a study
that audits the top of the ranking (random verification, #4224 follow-ups) can
run anywhere, from a CSV in the repo.

One row per (world, cell, seed, vote checkpoint):

* ``n_corpus``, ``n_pos`` - the corpus: the test half of the cell (the 5% world
  thins its negatives to 5%, as #4220's "same" scenario did);
* ``pos_ranks`` - space-separated 0-based ranks of the positives in the corpus
  sorted by the final model's score, descending (ties broken stably);
* ``n_votes``, ``n_vote_pos``, ``n_cal_pos`` - the session so far;
* for X in 25/50/75%, the #4220 estimator's cut as the library ships it
  (``vtscore.training.thresholds.precision_floor_cut``, reference pool
  including the voted items) and with a consistent reference pool (voted items
  removed, as the fold haystacks do): ``*_status_xNN`` and ``*_k_xNN`` (the
  number of corpus items returned; 0 when nothing is promised).

    python export_rank_frames.py --world 0.44%=DIR --world 5%=DIR --world 0.1%=DIR --out OUT_DIR [--jobs N]

``DIR`` is an arm's ``results`` directory holding ``cells/``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _cells_paths import main_frame_files, pframe_files  # noqa: E402

FLOORS = (0.25, 0.5, 0.75)
#: Worlds whose corpus is thinned to a target prevalence, as #4220's "same"
#: scenario did (keep every positive, sample negatives with the same stream).
THIN_TO = {"5%": 0.05}


def _cut(floor, corpus, pool, orderings, haystacks):
    from vtscore.training.thresholds.precision_floor import precision_floor_cut  # noqa: PLC0415

    c = precision_floor_cut(floor, corpus, pool, orderings, haystacks)
    k = int(np.count_nonzero(corpus >= c.threshold)) if c.status.value == "promised" else 0
    return c.status.value, k


def cell_rows(job: tuple[str, str, str]) -> list[dict]:
    world, npz_path, main_path = job
    first = pd.read_csv(main_path, nrows=1)
    if first.empty:  # starved: no detector, nothing to rank
        return []
    cat, seed = str(first["category"].iloc[0]), int(first["seed"].iloc[0])
    z = np.load(npz_path)
    rows = []
    for st in sorted({k.split("/")[0] for k in z.files}, key=lambda s: int(s[1:])):

        def g(k, st=st):
            return z[f"{st}/{k}"]

        test_s = g("test_scores").astype(np.float64)
        test_y = g("test_labels").astype(np.int64)
        if world in THIN_TO:
            target = THIN_TO[world]
            pos, neg = np.flatnonzero(test_y == 1), np.flatnonzero(test_y == 0)
            keep_n = int(round(len(pos) * (1 - target) / target))
            rng = np.random.default_rng([seed, 4220, len(test_s)])
            idx = np.sort(np.concatenate([pos, rng.choice(neg, size=min(keep_n, len(neg)), replace=False)]))
            test_s, test_y = test_s[idx], test_y[idx]
        order = np.argsort(-test_s, kind="stable")
        ranks = np.flatnonzero(test_y[order] == 1)
        pool = g("pool_scores").astype(np.float64)
        votes = g("vote_scores").astype(np.float64)
        keep = np.ones(len(pool), dtype=bool)
        for v in votes:  # one pool entry per voted item, matched on its in-sample score
            hit = np.flatnonzero(keep & (np.abs(pool - v) <= 1e-7))
            if len(hit):
                keep[hit[0]] = False
        cf, hf = g("fold_cal_fold"), g("fold_hay_fold")
        folds = sorted(set(cf.tolist()))
        orderings = [(g("fold_cal_scores")[cf == k], g("fold_cal_labels")[cf == k]) for k in folds]
        haystacks = [g("fold_hay_scores")[hf == k] for k in folds]
        row = {
            "world": world,
            "category": cat,
            "band": cat.split("@")[-1],
            "seed": seed,
            "t": int(g("t")),
            "n_corpus": len(test_s),
            "n_pos": int(test_y.sum()),
            "pos_ranks": " ".join(map(str, ranks.tolist())),
            "n_votes": len(votes),
            "n_vote_pos": int(g("vote_labels").sum()),
            "n_cal_pos": int(g("fold_cal_labels").sum()),
        }
        for x in FLOORS:
            tag = f"x{int(round(x * 100))}"
            for name, ref in (("shipped", pool), ("consistent", pool[keep])):
                status, k = _cut(x, test_s, ref, orderings, haystacks)
                row[f"{name}_status_{tag}"] = status
                row[f"{name}_k_{tag}"] = k
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--world", action="append", required=True, help="label=results_dir (holding cells/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.world:
        world, _, d = spec.partition("=")
        cells = Path(d) / "cells"
        mains = {p.name.split(".")[0]: p for p in main_frame_files(cells)}
        jobs = [
            (world, str(p), str(mains[p.name.split("__")[0]]))
            for p in pframe_files(cells)
            if p.name.split("__")[0] in mains
        ]
        rows: list[dict] = []
        if args.jobs > 1:
            with ProcessPoolExecutor(args.jobs) as ex:
                for r in ex.map(cell_rows, jobs, chunksize=4):
                    rows += r
        else:
            for j in jobs:
                rows += cell_rows(j)
        df = pd.DataFrame(rows).sort_values(["category", "seed", "t"])
        name = "rank_frames_" + world.replace("%", "pct").replace(".", "p") + ".csv.gz"
        df.to_csv(args.out / name, index=False)
        print(f"{world}: {len(df)} frames from {len(jobs)} cells -> {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
