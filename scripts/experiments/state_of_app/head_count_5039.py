#!/usr/bin/env python3
"""#5039: price counts for the head's count line on #4937's snapshots, at every preset.

While the class model holds fewer than ``HEAD_COUNT_GOODS`` held-out Goods, the labels line at every balance is the
head's count line (#4786, #4798): ``beta ** 0.708`` times what stands out in the head's scores (``head_count_cut``).
#4937 found that count keeps 0.14 of the F-beta-best set at beta 4 on Binary Photo, 0.57 at beta 1 and 1.2 at beta
1/4. ``head_count_cut`` is a pure function of the corpus's scores, so any count is read here offline on the same heads:
Find's line on the withheld half, as #4937 read it.

  head_count_5039.py extract --snapshots CSV --results DIR [--results DIR ...] --out DIR [--procs N]

``extract`` keeps, for every snapshot on the head's count line (``mall/snapshots.csv.gz``'s ``head_count``, outside
the typed query's hold), the withheld half's scores best first and their truth, and the folds' held-out Goods' and
Bads' scores (the class model's evidence), one ``<run>.npz`` per run, and checks
that today's count reproduces #4937's ``today_kept`` on every one.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fold_scale_4936 import PATHS, _ident, load_snapshots  # noqa: E402

KEY = ["run", "idx", "path", "t"]


def _cell_arrays(args) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    """``(key, scores best first, truth in that order, held-out Goods' and Bads' scores)`` per wanted snapshot."""
    from vtscore.training.thresholds.labels_line import _finite_unit, head_count_cut

    npz, run, wanted = args
    snaps = load_snapshots(npz)
    ident = _ident(os.path.dirname(npz), npz)
    out = []
    for prefix, path in PATHS.items():
        for key, snap in snaps.items():
            if not key.startswith(prefix + "/"):
                continue
            t = int(snap["t"])
            if (path, t) not in wanted:
                continue
            s = _finite_unit(np.asarray(snap["scores"], dtype=np.float64))
            y = np.asarray(snap["labels"], dtype=np.int8)
            order = np.argsort(-s, kind="stable")
            s_desc, y_desc = s[order], y[order]
            thr = head_count_cut(s_desc, float(snap["beta"]))
            fl = np.asarray(snap["fold_labels"], dtype=np.int8)
            fs = np.asarray(snap["fold_scores"], dtype=np.float64)
            meta = {"run": run, **ident, "path": path, "t": t, "k_today": int((s >= thr).sum())}
            out.append((meta, s_desc, y_desc, fs[fl == 1], fs[fl == 0]))
    # A step can be filed under several clicks; keep one copy of each.
    seen, uniq = set(), []
    for item in out:
        k = (item[0]["path"], item[0]["t"])
        if k not in seen:
            seen.add(k)
            uniq.append(item)
    return uniq


def cmd_extract(a) -> None:
    mall = pd.read_csv(a.snapshots, low_memory=False)
    mall = mall[(mall["head_count"] == True) & (mall["query_hold"] != True)]  # noqa: E712
    os.makedirs(a.out, exist_ok=True)
    for res in a.results:
        run = os.path.basename(os.path.dirname(res.rstrip("/")))
        mine = mall[mall["run"] == run]
        wanted_by_idx: dict[int, set] = {}
        for idx, path, t in mine[["idx", "path", "t"]].itertuples(index=False):
            wanted_by_idx.setdefault(int(idx), set()).add((str(path), int(t)))
        jobs = []
        for npz in sorted(glob.glob(os.path.join(res, "cells", "task_*__testscores.npz"))):
            idx = int(os.path.basename(npz).split("_")[1])
            if idx in wanted_by_idx:
                jobs.append((npz, run, wanted_by_idx[idx]))
        print(f"{run}: {len(jobs)} cells, {len(mine)} snapshots", flush=True)
        metas, ss, ys, gs, bs = [], [], [], [], []
        with ProcessPoolExecutor(max_workers=a.procs) as ex:
            for items in ex.map(_cell_arrays, jobs, chunksize=4):
                for meta, s, y, g, b in items:
                    metas.append(meta)
                    ss.append(s)
                    ys.append(y)
                    gs.append(g)
                    bs.append(b)

        def offsets(parts):
            return np.concatenate([[0], np.cumsum([p.size for p in parts])]).astype(np.int64)

        off = offsets(ss)
        np.savez(
            os.path.join(a.out, f"{run}.npz"),
            s=np.concatenate(ss),
            y=np.concatenate(ys),
            off=off,
            goods=np.concatenate(gs),
            goods_off=offsets(gs),
            bads=np.concatenate(bs),
            bads_off=offsets(bs),
        )
        meta = pd.DataFrame(metas)
        meta["row"] = np.arange(len(meta))
        chk = meta.merge(mine[KEY + ["today_kept"]], on=KEY, how="left")
        bad = int((chk["k_today"] != chk["today_kept"]).sum())
        missing = len(mine) - len(meta)
        print(
            f"{run}: wrote {len(meta)} snapshots ({off[-1]:,} scores); today_kept mismatches {bad}; missing {missing}"
        )
        meta.to_csv(os.path.join(a.out, f"{run}.meta.csv"), index=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--snapshots", required=True)
    e.add_argument("--results", action="append", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--procs", type=int, default=8)
    a = p.parse_args(argv)
    {"extract": cmd_extract}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
