#!/usr/bin/env python
"""Would a minimum Good count, or a later first check, fix the weak check (#4778)? Its paired effect, split.

#4778 asked, if the check loses on faces, whether d' should need a minimum number of Goods before it can call
separation weak: face sessions hold about four Goods when it fires, so "weak" might be the label count speaking.
A gate on Goods is only worth having if the checks it would remove are the ones that do not pay. So each paired
pair of session sets (the weak check on, and the end-of-run check only, otherwise the same sessions) is split by
the Goods in hand when the first prompted check starts, and the paired effect, check on minus check off, is read
per stratum as the session mean of the objective over votes 1-50 and 1-150 (carried between recorded votes, 0
before the first).

Two sources of the objective, by the pair's third field:

* a dataset name (the FHIBE cells of today): Test's untagged rows' ``fbeta`` (``prior_4776.read_runs``);
* ``thresholds`` (COCO runs from before that column, #4496's): ``analysis-binary/thresholds.csv`` beside the
  cells, ``thr_fbeta`` at each recorded step, as #4496's own comparison read it.

    python weak_check_goods_4778.py NAME=CHECK_CELLS::NO_CHECK_CELLS::DATASET|thresholds ... [--by vote] [--out CSV]

``--by vote`` splits by the vote the first check started at instead: a later earliest vote
(``WEAK_CHECK_MIN_VOTES``, ``CALIB_WEAK_MIN_T``) is the other lever, and it only removes the early checks.

A check's round logs its picks at the round's last vote (``weak_check_4778.read_picks``), so the Goods in hand
are counted over the votes cast before the round's first.
"""

from __future__ import annotations

import argparse
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from prior_4776 import load

HORIZON = 150
STRATA = {
    "goods": (("never prompted", None, None), ("<4", 0, 3), ("4-5", 4, 5), ("6-9", 6, 9), ("10-19", 10, 19),
              (">=20", 20, 10**9)),
    "vote": (("never prompted", None, None), ("11-29", 0, 29), ("30-49", 30, 49), ("50-79", 50, 79),
             (">=80", 80, 10**9)),
}  # fmt: skip
SPANS = ((1, 50), (1, 150))


def load_thresholds(cells: str) -> dict:
    """``thr_fbeta`` at each recorded step of a COCO run set's analysis, carried; 0 before the first."""
    th = pd.read_csv(Path(cells).parent.parent / "analysis-binary" / "thresholds.csv")
    th = th[th["point"] == "step"].sort_values("t")
    out = {}
    for (cat, seed), g in th.groupby(["category", "seed"]):
        f = np.full(HORIZON, np.nan)
        for t, v in zip(g["t"], g["thr_fbeta"], strict=True):
            if 1 <= t <= HORIZON:
                f[int(t) - 1] = v if np.isfinite(v) else 0.0
        out[("coco_better", cat, int(seed))] = {"F": pd.Series(f).ffill().fillna(0.0).to_numpy()}
    return out


def first_check(path: str) -> list[tuple]:
    """Per run in a pick log: (key, Goods in hand at the first prompted vote, prompted votes, that vote).

    The Goods and the vote are None for a run never prompted.
    """
    try:
        pk = pd.read_csv(path, usecols=["dataset", "category", "seed", "t", "phase", "picked_label"])
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return []
    pk = pk[pk["t"] <= HORIZON]
    out = []
    for (ds, cat, seed), r in pk.groupby(["dataset", "category", "seed"], sort=False):
        r = r.assign(vote=r["t"] - r.groupby("t")["t"].transform("size") + r.groupby("t").cumcount() + 1)
        r = r.sort_values("vote", kind="stable")
        good = (r["picked_label"] == 1).to_numpy()
        prompt = (r["phase"].astype(str) == "prompt").to_numpy()
        i = int(np.argmax(prompt))
        goods = int((np.cumsum(good) - good)[i]) if prompt.any() else None
        vote = int(r["vote"].iloc[i]) if prompt.any() else None
        out.append(((ds, cat, int(seed)), goods, int(prompt.sum()), vote))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pairs", nargs="+", help="NAME=CHECK_CELLS::NO_CHECK_CELLS::DATASET|thresholds")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--by",
        choices=tuple(STRATA),
        default="goods",
        help="split by Goods in hand, or by the vote, at the first check",
    )
    args = ap.parse_args()
    rows = []
    for spec in args.pairs:
        name, _, rest = spec.partition("=")
        chk, nochk, src = rest.split("::")
        if src == "thresholds":
            A, C = load_thresholds(chk), load_thresholds(nochk)
        else:
            A, C = load(chk, src, args.workers), load(nochk, src, args.workers)
        with ProcessPoolExecutor(args.workers) as pool:
            files = glob.glob(f"{chk}/*__picks.csv")
            fc = {k: (g if args.by == "goods" else v, n) for got in pool.map(first_check, files, chunksize=16)
                  for k, g, n, v in got}  # fmt: skip
        keys = sorted(set(A) & set(C) & set(fc))
        for label, lo, hi in STRATA[args.by]:
            ks = [
                k for k in keys if (fc[k][0] is None if lo is None else fc[k][0] is not None and lo <= fc[k][0] <= hi)
            ]
            if not ks:
                continue
            row = {"set": name, f"{args.by}_at_first_check": label, "runs": len(ks), "share": len(ks) / len(keys),
                   "prompted_votes_mean": float(np.mean([fc[k][1] for k in ks]))}  # fmt: skip
            for lo_, hi_ in SPANS:
                d = np.array([A[k]["F"][lo_ - 1 : hi_].mean() - C[k]["F"][lo_ - 1 : hi_].mean() for k in ks])
                row[f"d_{lo_}_{hi_}"] = float(d.mean())
                row[f"d_{lo_}_{hi_}_se"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            rows.append(row)
            print(f"{name:12s} {label:15s} n={len(ks):5d}  check on minus off: 1-50 {row['d_1_50']:+.3f} "
                  f"+- {row['d_1_50_se']:.3f}  1-150 {row['d_1_150']:+.3f} +- {row['d_1_150_se']:.3f}")  # fmt: skip
    if args.out:
        pd.DataFrame(rows).round(4).to_csv(args.out, index=False)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
