#!/usr/bin/env python
"""#5473: does Autopilot need its Diversity step? Today's app against ``CALIB_SPAN_GREEN=0``, paired.

Two session sets per bench and beta: ``ctl`` (the app as shipped: Span green at 40 evidence-bearing Coverage Atlas
nodes) and ``nodiv`` (Span green from the first vote, so Autopilot never enters New, the Diversity step, and says
Done once Smart and Stable are green or the labels line's level is met). The same cells and seeds, so the runs pair
one for one, and a pair's picks agree until the control's first New pick.

Benches: COCO Better's Binary path (``skip_diversity_5473.sh``; 144 cells x 5 seeds a beta) and the face path on
FHIBE (``fhibe/sota_face.sh`` with SOTA_FACE_TAG=5473ctl / 5473nodiv; 562 people x 2 sizes, K = 1 and 4).

The objective is read as #4778 and #4797 read it (``prior_4776.read_runs``): Test's untagged rows' ``fbeta`` at
every vote 1..150, the check rows left out, carried between rows, 0 before the first. Every run is in every average
(#4631). Per set, nodiv minus ctl, SE clustered on the class (the person on faces):

* the stop: the first vote Autopilot says Done (``phase == "done"``), or 150 for a run it never tells to stop; the
  share told to stop by 150 and the clicks to the stop;
* F at the stop (what the user leaves with when they stop where Autopilot says), and F at votes 50 and 150 and the
  session means over 1-50, 51-150 and 1-150 (what a user who keeps clicking gets);
* matched clicks: both arms read at nodiv's stop, and both at the control's, which parts the stop rule from the
  picks;
* the same split by whether the control entered New, the only sessions the arm can change;
* the pairing check: the first vote where a pair's picks differ must not come before the control's first New vote.

Writes ``nodiv_5473_summary.csv``, ``nodiv_5473_curves.csv`` (per vote, both arms' mean F, the paired difference,
and the share told to stop by then) and ``nodiv_5473_per_run.csv``.

    python skip_diversity_5473.py --coco-root <state-of-the-app> --face-runs <fhibe runs> --out <dir> [--docs <dir>]
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
SPANS = ((1, 50), (51, 150), (1, 150))
AT = (25, 50, 100, 150)


def read_phases(path: str) -> dict:
    """Per run in one cell: the first vote Autopilot says Done, the votes spent in New, and the first New vote."""
    try:
        df = pd.read_csv(path, usecols=["dataset", "category", "seed", "t", "phase"], low_memory=False)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return {}
    df = df[(df["t"] >= 1) & (df["t"] <= HORIZON)]
    df["phase"] = df["phase"].fillna("").astype(str)
    out = {}
    for (ds, cat, seed), r in df.groupby(["dataset", "category", "seed"], sort=False):
        by_t = r.groupby("t")["phase"].agg(set)
        done = [t for t, ph in by_t.items() if "done" in ph]
        new = [t for t, ph in by_t.items() if "new" in ph]
        out[(ds, cat, int(seed))] = {
            "done": int(min(done)) if done else None,
            "first_new": int(min(new)) if new else None,
            "new_votes": len(new),
        }
    return out


def load_phases(cells: str, workers: int) -> dict:
    files = [str(f) for f in main_frame_files(cells)]
    with ProcessPoolExecutor(workers) as pool:
        return {k: v for got in pool.map(read_phases, files, chunksize=8) for k, v in got.items()}


def stop_of(ph: dict) -> int:
    return ph["done"] if ph["done"] is not None else HORIZON


def effect(keys: list, A: dict, C: dict, PA: dict, PC: dict) -> dict:
    """nodiv (A) minus ctl (C) over *keys*: the stop, F there, and F at fixed votes."""
    row: dict = {"runs": len(keys)}
    if not keys:
        return row
    sa = np.array([stop_of(PA[k]) for k in keys], float)
    sc = np.array([stop_of(PC[k]) for k in keys], float)
    fa = np.array([A[k]["F"][int(s) - 1] for k, s in zip(keys, sa, strict=True)])
    fc = np.array([C[k]["F"][int(s) - 1] for k, s in zip(keys, sc, strict=True)])
    row.update(
        stopped_ctl=float(np.mean([PC[k]["done"] is not None for k in keys])),
        stopped_nodiv=float(np.mean([PA[k]["done"] is not None for k in keys])),
        stop_ctl=float(sc.mean()),
        stop_nodiv=float(sa.mean()),
        d_stop=float((sa - sc).mean()),
        d_stop_se=se_cat(keys, sa - sc),
        F_stop_ctl=float(fc.mean()),
        F_stop_nodiv=float(fa.mean()),
        d_F_stop=float((fa - fc).mean()),
        d_F_stop_se=se_cat(keys, fa - fc),
        worse_F_stop_005=float(np.mean(fa - fc < -0.05)),
        better_F_stop_005=float(np.mean(fa - fc > 0.05)),
    )
    # Matched clicks: each arm read at the other's stop, so the stop rule and the picks come apart.
    ca = np.array([C[k]["F"][int(s) - 1] for k, s in zip(keys, sa, strict=True)])
    ac = np.array([A[k]["F"][int(s) - 1] for k, s in zip(keys, sc, strict=True)])
    row.update(
        d_F_at_nodiv_stop=float((fa - ca).mean()),
        d_F_at_nodiv_stop_se=se_cat(keys, fa - ca),
        d_F_at_ctl_stop=float((ac - fc).mean()),
        d_F_at_ctl_stop_se=se_cat(keys, ac - fc),
    )
    for v in AT:
        d = np.array([A[k]["F"][v - 1] - C[k]["F"][v - 1] for k in keys])
        row[f"F{v}_ctl"] = float(np.mean([C[k]["F"][v - 1] for k in keys]))
        row[f"d_F{v}"], row[f"d_F{v}_se"] = float(d.mean()), se_cat(keys, d)
    for lo, hi in SPANS:
        d = np.array([A[k]["F"][lo - 1 : hi].mean() - C[k]["F"][lo - 1 : hi].mean() for k in keys])
        row[f"d_{lo}_{hi}"], row[f"d_{lo}_{hi}_se"] = float(d.mean()), se_cat(keys, d)
    return row


def compare(bench: str, label: str, beta: float, cells: dict, workers: int, out: dict) -> None:
    """One set: load both arms, check the pairing, and append its summary, curve and per-run rows to *out*."""
    A, C = (load(str(cells[a]), None, workers) for a in ("nodiv", "ctl"))
    PA, PC = (load_phases(str(cells[a]), workers) for a in ("nodiv", "ctl"))
    KA, KC = (load_picks(str(cells[a]), workers) for a in ("nodiv", "ctl"))
    keys = sorted(set(A) & set(C) & set(PA) & set(PC) & set(KA) & set(KC))
    if not keys:
        print(f"{bench} {label}: no paired runs", file=sys.stderr)
        return
    # Pairing: the arm may differ only from the control's first New vote on (the arm's stop check may start then).
    early = sum(
        1
        for k in keys
        if (dep := departure(KA[k]["picks"], KC[k]["picks"])) is not None
        and (PC[k]["first_new"] is None or dep < PC[k]["first_new"])
    )
    entered = [k for k in keys if PC[k]["first_new"] is not None]
    for split, ks in (
        ("all", keys),
        ("ctl entered New", entered),
        ("ctl never in New", sorted(set(keys) - set(entered))),
    ):
        row = {"bench": bench, "set": label, "beta": beta, "split": split}
        if split == "all":
            row.update(
                arm_only=len(set(A) - set(C)),
                ctl_only=len(set(C) - set(A)),
                early_departures=early,
                entered_new_share=len(entered) / len(keys),
                new_votes_mean_entered=float(np.mean([PC[k]["new_votes"] for k in entered])) if entered else np.nan,
                first_new_median=float(np.median([PC[k]["first_new"] for k in entered])) if entered else np.nan,
            )
        row.update(effect(ks, A, C, PA, PC))
        out["summary"].append(row)
    s = out["summary"][-3]
    print(
        f"{bench:5s} {label:10s} b{beta:<4g} n={len(keys):4d} early_dep={early} inNew={s['entered_new_share']:.0%} "
        f"stopped {s['stopped_ctl']:.0%}->{s['stopped_nodiv']:.0%} stop {s['stop_ctl']:.0f}->{s['stop_nodiv']:.0f} "
        f"F@stop {s['F_stop_ctl']:.3f} {s['d_F_stop']:+.4f}±{s['d_F_stop_se']:.4f} "
        f"F150 {s['d_F150']:+.4f}±{s['d_F150_se']:.4f} 1-150 {s['d_1_150']:+.4f}±{s['d_1_150_se']:.4f}",
        flush=True,
    )
    fa, fc = np.array([A[k]["F"] for k in keys]), np.array([C[k]["F"] for k in keys])
    sa = np.array([stop_of(PA[k]) if PA[k]["done"] is not None else np.inf for k in keys])
    sc = np.array([stop_of(PC[k]) if PC[k]["done"] is not None else np.inf for k in keys])
    for v in range(1, HORIZON + 1):
        d = fa[:, v - 1] - fc[:, v - 1]
        out["curves"].append({"bench": bench, "set": label, "beta": beta, "vote": v, "F_ctl": fc[:, v - 1].mean(),
                              "F_nodiv": fa[:, v - 1].mean(), "d": d.mean(), "d_se": se_cat(keys, d),
                              "stopped_ctl": float((sc <= v).mean()), "stopped_nodiv": float((sa <= v).mean())})  # fmt: skip
    for k in keys:
        out["per_run"].append({"bench": bench, "set": label, "beta": beta, "dataset": k[0], "category": k[1],
                               "seed": k[2], "first_new_ctl": PC[k]["first_new"], "new_votes_ctl": PC[k]["new_votes"],
                               "done_ctl": PC[k]["done"], "done_nodiv": PA[k]["done"],
                               "F_stop_ctl": C[k]["F"][stop_of(PC[k]) - 1], "F_stop_nodiv": A[k]["F"][stop_of(PA[k]) - 1],
                               "F150_ctl": C[k]["F"][-1], "F150_nodiv": A[k]["F"][-1],
                               "departure": departure(KA[k]["picks"], KC[k]["picks"])})  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--coco-root", type=Path, help="the state-of-the-app root holding <date>-nodiv5473-<arm>-b<beta>")
    ap.add_argument("--coco-date", default="2026-10-30")
    ap.add_argument("--face-runs", type=Path, help="the FHIBE runs dir holding <date>-5473<arm>-ik-k<K>-b<beta>")
    ap.add_argument("--face-date", default="2026-10-30")
    ap.add_argument("--ks", default="4,1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", type=Path, help="also write the tables here")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out: dict[str, list] = {"summary": [], "curves": [], "per_run": []}
    for btag, beta in BETAS:
        if args.coco_root:
            cells = {
                arm: args.coco_root / f"{args.coco_date}-nodiv5473-{d}-b{btag}" / "results" / "cells"
                for arm, d in (("ctl", "app"), ("nodiv", "nodiv"))
            }
            if all(c.is_dir() for c in cells.values()):
                compare("coco", "binary", beta, cells, args.workers, out)
        if args.face_runs:
            for k in (int(x) for x in args.ks.split(",")):
                cells = {
                    arm: args.face_runs / f"{args.face_date}-5473{arm}-ik-k{k}-b{btag}" / "results" / "cells"
                    for arm in ("ctl", "nodiv")
                }
                if all(c.is_dir() for c in cells.values()):
                    compare("face", f"K{k}", beta, cells, args.workers, out)
    for name, rows in (("summary", out["summary"]), ("curves", out["curves"]), ("per_run", out["per_run"])):
        df = pd.DataFrame(rows).round(4)
        df.to_csv(args.out / f"nodiv_5473_{name}.csv", index=False)
        if args.docs:
            args.docs.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.docs / f"nodiv_5473_{name}.csv", index=False)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
