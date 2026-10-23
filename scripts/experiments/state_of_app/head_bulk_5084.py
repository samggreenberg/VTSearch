#!/usr/bin/env python3
"""#5084: the head's count of what stands out counts the bulk's heavy upper tail; price bulk models for it offline.

The head's count line (#4786, #5039) keeps a multiple of what stands out in the head's scores: #4603's count, the
corpus logits above ``median + 4 spreads`` less what a Gaussian bulk puts there (``_count_estimate``). The heads'
bulks are heavier than a Gaussian above 3 spreads, so the count reads the bulk's tail as positives (#5084).

  head_bulk_5084.py extract --results DIR [--results DIR ...] --out DIR [--procs N]

``extract`` keeps, for every snapshot of a Binary / Region snapshot run (``fold_scale_4936.sh``,
``CALIB_SAVE_TEST_SCORES_AT``) on the head's count line (the class model holds fewer than ``HEAD_COUNT_GOODS``
held-out Goods, and a typed query on a single-vector space is past its hold), the withheld half's scores best first,
their truth and the folds' held-out Goods' and Bads' scores, in ``head_count_5039.py``'s format (``<run>.npz`` +
``<run>.meta.csv``), so its ``price`` / ``grid`` and this study's pricers read them. It replaces #5039's
``extract``, which needed #4937's per-snapshot table. Face runs go through ``head_count_5039.py extract-faces``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fold_scale_4936 import PATHS, _ident, load_snapshots  # noqa: E402
from head_count_5039 import _save  # noqa: E402


def _run_cell(args) -> list:
    """The head-count snapshots of one Binary / Region cell, as ``head_count_5039._save`` takes them."""
    from vtscore.detectors.label_quota import query_holds
    from vtscore.eval.live_threshold_rules import head_count_recall
    from vtscore.training.thresholds.labels_line import HEAD_COUNT_GOODS, _finite_unit, head_count_cut

    npz, run = args
    ident = _ident(os.path.dirname(npz), npz)
    out, seen = [], set()
    for key, snap in sorted(load_snapshots(npz).items()):
        prefix = key.rsplit("/", 1)[0]
        path = PATHS.get(prefix)
        model = snap.get("model")
        step = int(snap["t"])  # a step can be filed under several snapshot clicks; keep one copy of each
        if path is None or not model or int(model["n_pos"]) >= HEAD_COUNT_GOODS or (path, step) in seen:
            continue
        votes = int((np.asarray(snap["sim_vote"]) >= 0).sum())
        if query_holds(votes, 0, patch_space=path == "region"):
            continue
        seen.add((path, step))
        s = _finite_unit(np.asarray(snap["scores"], dtype=np.float64))
        y = np.asarray(snap["labels"], dtype=np.int8)
        if y.sum() == 0:
            continue
        order = np.argsort(-s, kind="stable")
        fl = np.asarray(snap["fold_labels"], dtype=np.int8)
        fs = np.asarray(snap["fold_scores"], dtype=np.float64)
        with head_count_recall("off", "off"):  # #4603's count alone, the line before #5039
            thr = head_count_cut(s[order], float(snap["beta"]))
        meta = {"run": run, **ident, "path": path, "t": step, "votes": votes,
                "k_today": int((s >= thr).sum())}  # fmt: skip
        out.append((meta, s[order], y[order], fs[fl == 1], fs[fl == 0]))
    return out


def cmd_extract(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    for res in a.results:
        run = os.path.basename(os.path.dirname(res.rstrip("/")))
        files = sorted(glob.glob(os.path.join(res, "cells", "task_*__testscores.npz")))
        with ProcessPoolExecutor(max_workers=a.procs) as ex:
            items = [it for part in ex.map(_run_cell, [(f, run) for f in files], chunksize=4) for it in part]
        if not items:
            print(f"{run}: {len(files)} cells, no head-count snapshots", flush=True)
            continue
        meta = _save(a.out, run, items)
        print(f"{run}: {len(files)} cells, {len(meta)} head-count snapshots", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--results", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=8)
    p.set_defaults(func=cmd_extract)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
