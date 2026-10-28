#!/usr/bin/env python
"""#4797: the weak-separation check at today's head on COCO Better's Binary and Region paths, paired.

Two session sets per path and beta from ``weak_check_4797.sh``: ``app`` (the app as shipped: the weak check once a
session, the Done check, the closing check) and ``end`` (``CALIB_SPOT_CHECK=end``: no weak check, the rest kept). The
same cells and seeds, so the runs pair one for one, and a pair's picks agree until the app's first weak check starts.

The objective is read as #4778 read it (``prior_4776.read_runs``): Test's untagged rows' ``fbeta`` at every vote
1..150, the closing check's rows left out, carried between rows, 0 before the first. Every run is in every average
(#4631). Per path and beta, check on minus check off (app minus end):

* the session mean over votes 1-50, 51-150 and 1-150, and F at vote 150, mean and SE clustered on the class;
* the share of sessions the weak check prompts, the vote it starts at, and the Goods in hand then (counted over the
  votes cast before the check's first round, which logs its picks at the round's last vote);
* the same effect split by those Goods and by that vote (#4778's strata);
* per vote, both arms' mean and the paired difference (``weak_4797_curves.csv``), and every pair's numbers
  (``weak_4797_per_run.csv``), for the figures and the examples;
* the pairing check: the first vote where a pair's picks differ, which must not come before the app's first weak
  check (a departure before it is drift, not the check).

    python weak_check_4797.py --root <state-of-the-app> --date 2026-10-28 [--tag T] --out <dir> [--docs <dir>]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fhibe"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
from _cells_paths import side_frame_files  # noqa: E402
from prior_4776 import load  # noqa: E402
from weak_check_goods_4778 import STRATA  # noqa: E402

HORIZON = 150
PATHS = ("binary", "region")
BETAS = (("025", 0.25), ("1", 1.0), ("4", 4.0))
SPANS = ((1, 50), (51, 150), (1, 150))


def read_picks(path: str) -> dict:
    """Per run in one pick log: its picks by vote, and its first weak check (vote, Goods before it, weak votes)."""
    try:
        pk = pd.read_csv(
            path, usecols=["dataset", "category", "seed", "t", "phase", "check_kind", "picked_id", "picked_label"]
        )
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return {}
    pk = pk[pk["t"] <= HORIZON]
    pk["phase"] = pk["phase"].fillna("").astype(str)
    pk["check_kind"] = pk["check_kind"].fillna("").astype(str)
    out = {}
    for (ds, cat, seed), r in pk.groupby(["dataset", "category", "seed"], sort=False):
        r = r.assign(vote=r["t"] - r.groupby("t")["t"].transform("size") + r.groupby("t").cumcount() + 1)
        r = r.sort_values("vote", kind="stable")
        weak = (r["phase"] == "prompt") & (r["check_kind"] == "weak")
        good = (r["picked_label"] == 1).to_numpy()
        first = int(r.loc[weak, "vote"].iloc[0]) if weak.any() else None
        out[(ds, cat, int(seed))] = {
            "picks": r["picked_id"].to_numpy(),
            "first": first,
            "goods": int(good[(r["vote"] < first).to_numpy()].sum()) if first is not None else None,
            "weak_votes": int(weak.sum()),
            "goods_150": int(good.sum()),
        }
    return out


def load_picks(cells: str, workers: int) -> dict:
    files = [str(f) for f in side_frame_files(cells, "__picks")]
    with ProcessPoolExecutor(workers) as pool:
        return {k: v for got in pool.map(read_picks, files, chunksize=8) for k, v in got.items()}


def departure(a: np.ndarray, b: np.ndarray) -> int | None:
    """The first vote (1-based) where two pick sequences differ; None if one is a prefix of the other."""
    n = min(len(a), len(b))
    diff = np.nonzero(a[:n] != b[:n])[0]
    return int(diff[0]) + 1 if len(diff) else None


def se_cat(keys: list, d: np.ndarray) -> float:
    """SE of the mean, clustered on the class: the spread of the per-class means over the classes."""
    per = pd.Series(d, index=[k[1] for k in keys]).groupby(level=0).mean()
    return float(per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else float("nan")


def effect(keys: list, A: dict, C: dict) -> dict:
    row = {}
    for lo, hi in SPANS:
        d = np.array([A[k]["F"][lo - 1 : hi].mean() - C[k]["F"][lo - 1 : hi].mean() for k in keys])
        row[f"d_{lo}_{hi}"], row[f"d_{lo}_{hi}_se"] = float(d.mean()), se_cat(keys, d)
    d = np.array([A[k]["F"][HORIZON - 1] - C[k]["F"][HORIZON - 1] for k in keys])
    row["d_F150"], row["d_F150_se"] = float(d.mean()), se_cat(keys, d)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--date", default="2026-10-28")
    ap.add_argument("--tag", default="")
    ap.add_argument("--paths", default=",".join(PATHS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", type=Path, help="also write the tables here")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary, splits, curves, per_run = [], [], [], []
    for path in args.paths.split(","):
        for btag, beta in BETAS:
            cells = {
                arm: args.root / f"{args.date}-weak4797{args.tag}-{arm}-{path}-b{btag}" / "results" / "cells"
                for arm in ("app", "end")
            }
            if not all(c.is_dir() for c in cells.values()):
                continue
            A, C = (
                load(str(cells["app"]), "coco_better", args.workers),
                load(str(cells["end"]), "coco_better", args.workers),
            )
            PA, PC = load_picks(str(cells["app"]), args.workers), load_picks(str(cells["end"]), args.workers)
            keys = sorted(set(A) & set(C) & set(PA) & set(PC))
            if not keys:
                continue
            # Pairing: a pair may differ only from the app's first weak check on.
            early = sum(
                1
                for k in keys
                if (dep := departure(PA[k]["picks"], PC[k]["picks"])) is not None
                and (PA[k]["first"] is None or dep < PA[k]["first"])
            )
            prompted = [k for k in keys if PA[k]["first"] is not None]
            row = {
                "path": path,
                "beta": beta,
                "runs": len(keys),
                "runs_app_only": len(set(A) - set(C)),
                "runs_end_only": len(set(C) - set(A)),
                "early_departures": early,
                "prompted_share": len(prompted) / len(keys),
                "first_vote_median": float(np.median([PA[k]["first"] for k in prompted])) if prompted else np.nan,
                "goods_at_first_median": float(np.median([PA[k]["goods"] for k in prompted])) if prompted else np.nan,
                "goods_at_first_lt6": float(np.mean([PA[k]["goods"] < 6 for k in prompted])) if prompted else np.nan,
                "weak_votes_mean_prompted": float(np.mean([PA[k]["weak_votes"] for k in prompted]))
                if prompted
                else np.nan,
                "goods150_app": float(np.mean([PA[k]["goods_150"] for k in keys])),
                "goods150_end": float(np.mean([PC[k]["goods_150"] for k in keys])),
                "F150_end": float(np.mean([C[k]["F"][HORIZON - 1] for k in keys])),
                "meanF_1_150_end": float(np.mean([C[k]["F"].mean() for k in keys])),
            }
            row.update(effect(keys, A, C))
            summary.append(row)
            fa, fc_ = np.array([A[k]["F"] for k in keys]), np.array([C[k]["F"] for k in keys])
            for v in range(HORIZON):
                d = fa[:, v] - fc_[:, v]
                curves.append({"path": path, "beta": beta, "vote": v + 1, "F_app": fa[:, v].mean(),
                               "F_end": fc_[:, v].mean(), "d": d.mean(), "d_se": se_cat(keys, d)})  # fmt: skip
            for k in keys:
                per_run.append({"path": path, "beta": beta, "category": k[1], "seed": k[2],
                                "first_vote": PA[k]["first"], "goods_at_first": PA[k]["goods"],
                                "goods150_app": PA[k]["goods_150"], "goods150_end": PC[k]["goods_150"],
                                "d_1_50": A[k]["F"][:50].mean() - C[k]["F"][:50].mean(),
                                "d_1_150": A[k]["F"].mean() - C[k]["F"].mean(),
                                "F150_app": A[k]["F"][-1], "F150_end": C[k]["F"][-1]})  # fmt: skip
            print(
                f"{path:6s} b{btag:3s} n={len(keys):4d} early_dep={early} prompted={row['prompted_share']:.0%} "
                f"first~{row['first_vote_median']:.0f} goods~{row['goods_at_first_median']:.0f} "
                f"(<6 {row['goods_at_first_lt6']:.0%})  app-end 1-50 {row['d_1_50']:+.4f}±{row['d_1_50_se']:.4f} "
                f"1-150 {row['d_1_150']:+.4f}±{row['d_1_150_se']:.4f} F150 {row['d_F150']:+.4f}±{row['d_F150_se']:.4f}",
                flush=True,
            )
            for by, field in (("goods", "goods"), ("vote", "first")):
                for label, lo, hi in STRATA[by]:
                    ks = [
                        k
                        for k in keys
                        if (
                            PA[k][field] is None
                            if lo is None
                            else PA[k][field] is not None and lo <= PA[k][field] <= hi
                        )
                    ]
                    if ks:
                        splits.append({"path": path, "beta": beta, "by": by, "stratum": label, "runs": len(ks),
                                       "share": len(ks) / len(keys), **effect(ks, A, C)})  # fmt: skip
    tables = (("weak_4797_summary.csv", summary), ("weak_4797_splits.csv", splits), ("weak_4797_curves.csv", curves),
              ("weak_4797_per_run.csv", per_run))  # fmt: skip
    for name, rows in tables:
        df = pd.DataFrame(rows).round(4)
        df.to_csv(args.out / name, index=False)
        if args.docs:
            args.docs.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.docs / name, index=False)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
