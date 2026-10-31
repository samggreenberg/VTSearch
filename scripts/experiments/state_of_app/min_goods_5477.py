#!/usr/bin/env python
"""#5477: let a session with fewer than 5 Goods be told Done. Today's app against three arms, paired.

Smart and Stable stay red until a session holds ``MIN_PER_CLASS`` (5) Goods, so a target with 3 or 4 to find is
never told Done. Four session sets per bench and beta, the same cells and seeds, so the runs pair one for one:

* ``ctl``: the app as shipped;
* ``min3``: ``CALIB_MIN_GOOD=3``, the lights need 3 Goods (Autopilot's Good target);
* ``dry8`` / ``dry16``: ``CALIB_HARD_DRY_RUN=8`` / ``16``, the lights need only the Goods in hand once that many
  learned picks in a row held no Good.

Benches: COCO Better's Binary path (``min_goods_5477.sh``; 144 cells x 5 seeds a beta) and the face path on FHIBE
(``fhibe/sota_face.sh`` with SOTA_FACE_TAG=5477<arm>; 562 people x 2 sizes, K = 1 and 4).

The objective is read as #5473 read it (``prior_4776.read_runs``): Test's untagged rows' ``fbeta`` at every vote
1..150, the check rows left out, carried between rows, 0 before the first. Every run is in every average (#4631).
Per set, each arm minus ``ctl``, SE clustered on the class (the person on faces):

* the stop: the first vote Autopilot says Done (``phase == "done"``), or 150 for a run it never tells to stop; the
  share told to stop by 150 and the clicks to the stop;
* F at the stop (what the user leaves with when they stop where Autopilot says), and F at votes 50 and 150 and the
  session mean over 1-150 (what a user who keeps clicking gets);
* to come: F at 150 minus F at the stop, per arm, what stopping where told leaves unfound;
* the same split by the Goods the control holds at 150 (under 5, which the arms are for, and 5 or more);
* the pairing check: a pair's picks may differ only from the first vote its phases differ.

Writes ``mingood_5477_summary.csv``, ``mingood_5477_curves.csv`` (per vote, each arm's mean F, the paired
difference, and the share told to stop by then) and ``mingood_5477_per_run.csv``.

    python min_goods_5477.py --coco-root <state-of-the-app> --face-runs <fhibe runs> --out <dir> [--docs <dir>]
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
from _cells_paths import main_frame_files  # noqa: E402
from prior_4776 import load  # noqa: E402
from weak_check_4797 import departure, load_picks, se_cat  # noqa: E402

HORIZON = 150
BETAS = (("025", 0.25), ("1", 1.0), ("4", 4.0))
ARMS = ("min3", "dry8", "dry16")
AT = (50, 150)


def read_phases(path: str) -> dict:
    """Per run in one cell: the first Done vote, the phase at every vote, and the Goods held by vote 150."""
    try:
        df = pd.read_csv(path, usecols=["dataset", "category", "seed", "t", "phase", "n_good"], low_memory=False)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return {}
    df = df[(df["t"] >= 1) & (df["t"] <= HORIZON)]
    df["phase"] = df["phase"].fillna("").astype(str)
    out = {}
    for (ds, cat, seed), r in df.groupby(["dataset", "category", "seed"], sort=False):
        # A vote's phase is its first row's (the check rows come after); "done" anywhere in the vote is Done.
        by_t = r.groupby("t")["phase"].agg(list)
        done = [t for t, ph in by_t.items() if "done" in ph]
        out[(ds, cat, int(seed))] = {
            "done": int(min(done)) if done else None,
            "phases": {int(t): ph[0] for t, ph in by_t.items()},
            "goods": int(r["n_good"].max()),
        }
    return out


def load_phases(cells: str, workers: int) -> dict:
    files = [str(f) for f in main_frame_files(cells)]
    with ProcessPoolExecutor(workers) as pool:
        return {k: v for got in pool.map(read_phases, files, chunksize=8) for k, v in got.items()}


def stop_of(ph: dict) -> int:
    return ph["done"] if ph["done"] is not None else HORIZON


def phase_split(a: dict, c: dict) -> int | None:
    """The first vote two runs' phases differ, or None."""
    diff = [t for t in sorted(set(a) & set(c)) if a[t] != c[t]]
    return diff[0] if diff else None


def effect(keys: list, A: dict, C: dict, PA: dict, PC: dict) -> dict:
    """The arm (A) minus ctl (C) over *keys*: the stop, F there, F to come, and F at fixed votes."""
    row: dict = {"runs": len(keys)}
    if not keys:
        return row
    sa = np.array([stop_of(PA[k]) for k in keys], float)
    sc = np.array([stop_of(PC[k]) for k in keys], float)
    fa = np.array([A[k]["F"][int(s) - 1] for k, s in zip(keys, sa, strict=True)])
    fc = np.array([C[k]["F"][int(s) - 1] for k, s in zip(keys, sc, strict=True)])
    ea = np.array([A[k]["F"][-1] for k in keys])
    ec = np.array([C[k]["F"][-1] for k in keys])
    row.update(
        stopped_ctl=float(np.mean([PC[k]["done"] is not None for k in keys])),
        stopped_arm=float(np.mean([PA[k]["done"] is not None for k in keys])),
        stop_ctl=float(sc.mean()),
        stop_arm=float(sa.mean()),
        d_stop=float((sa - sc).mean()),
        d_stop_se=se_cat(keys, sa - sc),
        sooner=float(np.mean(sa < sc)),
        later=float(np.mean(sa > sc)),
        F_stop_ctl=float(fc.mean()),
        F_stop_arm=float(fa.mean()),
        d_F_stop=float((fa - fc).mean()),
        d_F_stop_se=se_cat(keys, fa - fc),
        worse_F_stop_005=float(np.mean(fa - fc < -0.05)),
        better_F_stop_005=float(np.mean(fa - fc > 0.05)),
        to_come_ctl=float((ec - fc).mean()),
        to_come_arm=float((ea - fa).mean()),
        to_come_arm_se=se_cat(keys, ea - fa),
        to_come_arm_005=float(np.mean(ea - fa > 0.05)),
    )
    for v in AT:
        d = np.array([A[k]["F"][v - 1] - C[k]["F"][v - 1] for k in keys])
        row[f"F{v}_ctl"] = float(np.mean([C[k]["F"][v - 1] for k in keys]))
        row[f"d_F{v}"], row[f"d_F{v}_se"] = float(d.mean()), se_cat(keys, d)
    d = np.array([A[k]["F"].mean() - C[k]["F"].mean() for k in keys])
    row["d_1_150"], row["d_1_150_se"] = float(d.mean()), se_cat(keys, d)
    return row


def compare(bench: str, label: str, beta: float, cells: dict, workers: int, out: dict) -> None:
    """One set: load every arm, check the pairing, and append its summary, curve and per-run rows to *out*."""
    F = {a: load(str(c), None, workers) for a, c in cells.items()}
    P = {a: load_phases(str(c), workers) for a, c in cells.items()}
    K = {a: load_picks(str(c), workers) for a, c in cells.items()}
    C, PC, KC = F["ctl"], P["ctl"], K["ctl"]
    for arm in (a for a in ARMS if a in cells):
        A, PA, KA = F[arm], P[arm], K[arm]
        keys = sorted(set(A) & set(C) & set(PA) & set(PC) & set(KA) & set(KC))
        if not keys:
            print(f"{bench} {label} {arm}: no paired runs", file=sys.stderr)
            continue
        # Pairing: a pair's picks may part only once its phases have (the arm's lights are all it changes).
        early = 0
        for k in keys:
            dep = departure(KA[k]["picks"], KC[k]["picks"])
            split = phase_split(PA[k]["phases"], PC[k]["phases"])
            if dep is not None and (split is None or dep < split):
                early += 1
        scarce = [k for k in keys if PC[k]["goods"] < 5]
        for split_name, ks in (
            ("all", keys),
            ("ctl under 5 Goods", scarce),
            ("ctl 5+ Goods", sorted(set(keys) - set(scarce))),
        ):
            row = {"bench": bench, "set": label, "beta": beta, "arm": arm, "split": split_name}
            if split_name == "all":
                row.update(
                    arm_only=len(set(A) - set(C)),
                    ctl_only=len(set(C) - set(A)),
                    early_departures=early,
                    scarce_share=len(scarce) / len(keys),
                )
            row.update(effect(ks, A, C, PA, PC))
            out["summary"].append(row)
        s = out["summary"][-3]
        print(
            f"{bench:5s} {label:6s} b{beta:<4g} {arm:5s} n={len(keys):4d} early_dep={early} "
            f"<5G={s['scarce_share']:.0%} stopped {s['stopped_ctl']:.0%}->{s['stopped_arm']:.0%} "
            f"stop {s['stop_ctl']:.0f}->{s['stop_arm']:.0f} F@stop {s['F_stop_ctl']:.3f}->{s['F_stop_arm']:.3f} "
            f"({s['d_F_stop']:+.4f}±{s['d_F_stop_se']:.4f}) to come {s['to_come_ctl']:.3f}->{s['to_come_arm']:.3f} "
            f"F150 {s['d_F150']:+.4f}±{s['d_F150_se']:.4f}",
            flush=True,
        )
        fa, fc = np.array([A[k]["F"] for k in keys]), np.array([C[k]["F"] for k in keys])
        sa = np.array([stop_of(PA[k]) if PA[k]["done"] is not None else np.inf for k in keys])
        sc = np.array([stop_of(PC[k]) if PC[k]["done"] is not None else np.inf for k in keys])
        for v in range(1, HORIZON + 1):
            d = fa[:, v - 1] - fc[:, v - 1]
            out["curves"].append({"bench": bench, "set": label, "beta": beta, "arm": arm, "vote": v,
                                  "F_ctl": fc[:, v - 1].mean(), "F_arm": fa[:, v - 1].mean(), "d": d.mean(),
                                  "d_se": se_cat(keys, d), "stopped_ctl": float((sc <= v).mean()),
                                  "stopped_arm": float((sa <= v).mean())})  # fmt: skip
        for k in keys:
            out["per_run"].append({"bench": bench, "set": label, "beta": beta, "arm": arm, "dataset": k[0],
                                   "category": k[1], "seed": k[2], "goods_ctl": PC[k]["goods"],
                                   "goods_arm": PA[k]["goods"], "done_ctl": PC[k]["done"], "done_arm": PA[k]["done"],
                                   "F_stop_ctl": C[k]["F"][stop_of(PC[k]) - 1],
                                   "F_stop_arm": A[k]["F"][stop_of(PA[k]) - 1],
                                   "F150_ctl": C[k]["F"][-1], "F150_arm": A[k]["F"][-1],
                                   "departure": departure(KA[k]["picks"], KC[k]["picks"]),
                                   "phase_split": phase_split(PA[k]["phases"], PC[k]["phases"])})  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--coco-root", type=Path, help="the state-of-the-app root holding <date>-mingood5477-<arm>-b<beta>")
    ap.add_argument("--coco-date", default="2026-10-30")
    ap.add_argument("--face-runs", type=Path, help="the FHIBE runs dir holding <date>-5477<arm>-ik-k<K>-b<beta>")
    ap.add_argument("--face-date", default="2026-10-30")
    ap.add_argument("--ks", default="1,4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", type=Path, help="also write the tables here")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out: dict[str, list] = {"summary": [], "curves": [], "per_run": []}
    for btag, beta in BETAS:
        if args.face_runs:
            for k in (int(x) for x in args.ks.split(",")):
                cells = {
                    arm: args.face_runs / f"{args.face_date}-5477{arm}-ik-k{k}-b{btag}" / "results" / "cells"
                    for arm in ("ctl", *ARMS)
                }
                if all(c.is_dir() for c in cells.values()):
                    compare("face", f"K{k}", beta, cells, args.workers, out)
        if args.coco_root:
            cells = {
                arm: args.coco_root / f"{args.coco_date}-mingood5477-{arm}-b{btag}" / "results" / "cells"
                for arm in ("ctl", *ARMS)
            }
            if all(c.is_dir() for c in cells.values()):
                compare("coco", "binary", beta, cells, args.workers, out)
    for name, rows in out.items():
        df = pd.DataFrame(rows).round(4)
        df.to_csv(args.out / f"mingood_5477_{name}.csv", index=False)
        if args.docs:
            args.docs.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.docs / f"mingood_5477_{name}.csv", index=False)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
