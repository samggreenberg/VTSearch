#!/usr/bin/env python3
"""#4222: does "stay in TextTop until G Goods" help a low-prevalence session?

Reads the eight ``launch_textgood_4222.sh`` arms (G = 3/6/10/20 in the natural
0.44% world and a 0.1% world) and reports, per arm, at votes 25/50/100/150:

* **harvest** - Good votes so far, share of cells that reached G, and the
  votes the opening spent before the learned phases began;
* **trust** - positives among the calibration votes, the share of cells past
  the #4220 gate (>= 10), and the #4220 estimator's promises (fold-rank +
  logistic + bootstrap lower bound + EM) at X = 25/50/75%: broken of those
  made, made at all, and recall against the oracle;
* **detector quality** - average precision and oracle cost of the model the
  session has, paired per cell against the same world's G = 3 control.

The decision rule is fixed in #4222: the smallest G that maximizes the share
of cells past the gate by vote 50, provided AP at vote 150 is not below the
control's by more than 2 paired SE, in both worlds.

    python analyze_textgood_4222.py --base /expscratch/$USER/textgood-4222 --out OUT [--jobs N]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _cells_io  # noqa: E402
import analyze_pframes_4220 as P  # noqa: E402
from _cells_paths import pframe_files, side_frame_files  # noqa: E402

CHECKPOINTS = (25, 50, 100, 150)
GATE = 10
FLOORS = (0.25, 0.5, 0.75)
LEARNED = {"hard", "done", "new"}
KEY = ["category", "seed"]


def arm_dirs(base: Path) -> list[tuple[str, str, int, Path]]:
    """``(arm, world, G, cells_dir)`` for every arm directory present."""
    out = []
    for d in sorted(base.iterdir()):
        cells = d / "results" / "cells"
        if not cells.is_dir() or "-g" not in d.name:
            continue
        world, g = d.name.rsplit("-g", 1)
        out.append((d.name, world, int(g), cells))
    return out


def harvest(cells: Path) -> pd.DataFrame:
    """One row per (cell, checkpoint): Good votes so far, whether G was met, opening length."""
    rows = []
    for f in side_frame_files(cells, "__picks"):
        p = pd.read_csv(f, usecols=["category", "seed", "t", "phase", "picked_label"]).sort_values("t")
        if p.empty:
            continue
        learned = p[p["phase"].isin(LEARNED)]
        opening = int(learned["t"].min() - 1) if len(learned) else int(p["t"].max())
        cum = p["picked_label"].cumsum().to_numpy()
        ts = p["t"].to_numpy()
        for c in CHECKPOINTS:
            upto = cum[ts <= c]
            rows.append(
                {
                    "category": p["category"].iloc[0],
                    "seed": int(p["seed"].iloc[0]),
                    "t": c,
                    "goods": int(upto[-1]) if len(upto) else 0,
                    "opening_votes": opening,
                }
            )
    return pd.DataFrame(rows)


def quality(cells: Path) -> pd.DataFrame:
    """AP and oracle cost at each checkpoint (the step's own detector), per cell."""
    df, _ = _cells_io.load_arm(cells.parent)
    if df.empty:
        return df
    df = df[df["t"].isin(CHECKPOINTS)]
    return df.groupby([*KEY, "t"], as_index=False)[["average_precision", "oracle_cost", "cost"]].mean()


def promise_job(job: tuple[str, str, str, str]) -> list[dict]:
    """The #4220 estimator on one cell's frames: fold-rank, logistic, lower bound + EM."""
    P.ESTIMATORS = ("foldrank",)
    P.FITS = ("logistic",)
    P.FLOORS = FLOORS
    P.THIN_TO = {}
    rows, _ = P.cell_rows(job)
    return [r for r in rows if r["reading"] == "lcb+em"]


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str) -> tuple[float, float, int]:
    m = a.merge(b, on=[*KEY, "t"], suffixes=("_a", "_b"))
    d = (m[f"{col}_a"] - m[f"{col}_b"]).dropna()
    return (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"), len(d))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    arms = arm_dirs(args.base)
    if not arms:
        raise SystemExit(f"no arms under {args.base}")

    harv, qual, prom = [], [], []
    jobs = []
    for arm, world, g, cells in arms:
        h = harvest(cells)
        h[["arm", "world", "G"]] = arm, world, g
        harv.append(h)
        q = quality(cells)
        if not q.empty:
            q[["arm", "world", "G"]] = arm, world, g
            qual.append(q)
        jobs += [
            (arm, j[1], j[2], j[3]) for j in P.jobs_for(arm, cells) if j[1] in {str(x) for x in pframe_files(cells)}
        ]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for r in ex.map(promise_job, jobs, chunksize=8):
                prom += r
    else:
        for j in jobs:
            prom += promise_job(j)
    H = pd.concat(harv, ignore_index=True)
    Q = pd.concat(qual, ignore_index=True) if qual else pd.DataFrame()
    R = pd.DataFrame(prom)
    if R.empty:
        raise SystemExit("no precision frames - were the arms launched with CALIB_PFRAME_STEPS?")
    world_of = {a: (w, g) for a, w, g, _ in arms}
    R["world"] = R["arm"].map(lambda a: world_of[a][0])
    R["G"] = R["arm"].map(lambda a: world_of[a][1])

    # --- harvest -----------------------------------------------------------------------
    H["reached_G"] = H["goods"] >= H["G"]
    hs = H.groupby(["world", "G", "t"]).agg(
        cells=("goods", "size"),
        mean_goods=("goods", "mean"),
        reached_G=("reached_G", "mean"),
        median_opening_votes=("opening_votes", "median"),
    )
    hs.reset_index().to_csv(args.out / "harvest.csv", index=False, float_format="%.4g")

    # --- trust ---------------------------------------------------------------------------
    cal = R.drop_duplicates(["arm", *KEY, "t"])[["world", "G", *KEY, "t", "n_cal_pos"]]
    cal["past_gate"] = cal["n_cal_pos"] >= GATE
    ts = cal.groupby(["world", "G", "t"]).agg(mean_cal_pos=("n_cal_pos", "mean"), past_gate=("past_gate", "mean"))
    ts.reset_index().to_csv(args.out / "trust_gate.csv", index=False, float_format="%.4g")
    R["gated"] = R["n_cal_pos"] >= GATE
    made = R[~R["empty"] & R["gated"]]
    ps = (
        R[R["gated"]]
        .groupby(["world", "G", "t", "X"])
        .agg(
            frames=("empty", "size"),
            made=("empty", lambda e: float((~e).mean())),
            recall=("recall", "mean"),
            oracle=("oracle_recall", "mean"),
        )
        .join(made.groupby(["world", "G", "t", "X"]).agg(broken_of_made=("violated", "mean")))
    )
    ps.reset_index().to_csv(args.out / "promises_gated.csv", index=False, float_format="%.4g")
    # What a user gets out of the promise overall: a kept promise, per cell, at X.
    R["kept"] = R["gated"] & ~R["empty"] & ~R["violated"]
    kept = R.groupby(["world", "G", "t", "X"]).agg(share_cells_with_kept_promise=("kept", "mean"))
    kept.reset_index().to_csv(args.out / "promises_kept.csv", index=False, float_format="%.4g")

    # --- detector quality, paired against each world's G=3 --------------------------------
    qrows = []
    if not Q.empty:
        for world in sorted(Q["world"].unique()):
            base = Q[(Q["world"] == world) & (Q["G"] == 3)]
            for g in sorted(Q[Q["world"] == world]["G"].unique()):
                arm = Q[(Q["world"] == world) & (Q["G"] == g)]
                for t in CHECKPOINTS:
                    for col in ("average_precision", "oracle_cost"):
                        m, se, n = paired(arm[arm["t"] == t], base[base["t"] == t], col)
                        qrows.append(
                            {
                                "world": world,
                                "G": g,
                                "t": t,
                                "metric": col,
                                "delta_vs_G3": m,
                                "se": se,
                                "n": n,
                                "level": float(arm[arm["t"] == t][col].mean()),
                            }
                        )
    qd = pd.DataFrame(qrows)
    qd.to_csv(args.out / "quality_paired.csv", index=False, float_format="%.4g")

    # --- the fixed decision rule -----------------------------------------------------------
    verdict = {}
    for world in sorted(ts.reset_index()["world"].unique()):
        t50 = ts.reset_index().query("world == @world and t == 50").set_index("G")["past_gate"]
        ok = []
        for g in t50.index:
            row = qd.query("world == @world and G == @g and t == 150 and metric == 'average_precision'")
            safe = row.empty or not (row["delta_vs_G3"].iloc[0] < -2 * row["se"].iloc[0])
            ok.append((g, float(t50[g]), bool(safe)))
        eligible = [x for x in ok if x[2]]
        best = max(x[1] for x in eligible) if eligible else float("nan")
        verdict[world] = {
            "past_gate_by_50": {str(g): round(v, 3) for g, v, _ in ok},
            "ap_safe": {str(g): s for g, _, s in ok},
            "smallest_G_at_max": min(g for g, v, _ in eligible if v >= best - 1e-9) if eligible else None,
        }
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    pd.set_option("display.width", 220)
    print(hs.round(2).to_string())
    print(ts.round(2).to_string())
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
