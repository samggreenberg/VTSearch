#!/usr/bin/env python3
"""#4256: does calibrating the precision floor only on learned-sort votes keep its promise?

Reads precision frames that carry ``fold_cal_phase`` (the phase that surfaced
each calibration vote) and runs the library estimator,
``vtscore.training.thresholds.precision_floor_cut``, four ways per frame:

* **evidence** - ``all`` calibration votes (as the app would today), or
  ``learned`` only: votes picked at or after the session's first learned-phase
  pick (``hard`` / ``new`` / ``done``).  Split by *time*, not by phase name:
  Autopilot's Good phase also recurs after the opening as a *learned* Good
  pick (the top of the model's ranking), under the same name, so dropping
  every ``good`` vote would drop learned-sort votes too;
* **reference pool** - ``shipped`` (the pool the corpus is ranked against
  includes the voted items, as the library ships) or ``consistent`` (voted
  items removed, as the fold haystacks already are).

Each promise is scored against the frame's truth (the unvoted test half):
promises made, broken of those made, recall against the oracle at X, and the
calibration positives each evidence set leaves for the gate.

    python analyze_provenance_4256.py --arm g3=DIR --arm g6=DIR --arm g20=DIR --out OUT [--jobs N]

At ~16 s a cell, a full grid is worth sharding: ``--shard I/N`` writes only
shard I's rows (``rows_shard_I.csv.gz``), and ``--merge`` summarizes every
shard already in ``OUT``.
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

from _cells_paths import main_frame_files, pframe_files, side_frame_files  # noqa: E402

FLOORS = (0.25, 0.5, 0.75)
#: Phases only a learned sort surfaces; the first pick in one ends the opening.
LEARNED_PHASES = frozenset({"hard", "new", "done"})


def learned_mask(votes: np.ndarray, pick_t: dict, first_learned_t: float) -> np.ndarray:
    """True for each calibration vote picked at or after the first learned-phase pick."""
    return np.array([pick_t.get(int(v), -1) >= first_learned_t for v in votes.tolist()], dtype=bool)


def evidence(fr: dict, which: str, pick_t: dict, first_learned_t: float) -> tuple[list, list, int]:
    """``(fold_orderings, fold_haystacks, calibration positives)`` for one evidence set."""
    s, y, f = fr["fold_cal_scores"], fr["fold_cal_labels"], fr["fold_cal_fold"]
    keep = np.ones(len(s), dtype=bool) if which == "all" else learned_mask(fr["fold_cal_vote"], pick_t, first_learned_t)
    folds = sorted(set(f.tolist()))
    orderings = [(s[(f == k) & keep], y[(f == k) & keep]) for k in folds]
    haystacks = [fr["fold_hay_scores"][fr["fold_hay_fold"] == k] for k in folds]
    return orderings, haystacks, int(y[keep].sum())


def reference_pool(fr: dict, which: str) -> np.ndarray:
    pool = fr["pool_scores"].astype(np.float64)
    if which == "shipped":
        return pool
    keep = np.ones(len(pool), dtype=bool)
    for v in fr["vote_scores"].astype(np.float64):  # one pool entry per voted item
        hit = np.flatnonzero(keep & (np.abs(pool - v) <= 1e-7))
        if len(hit):
            keep[hit[0]] = False
    return pool[keep]


def frame_rows(meta: dict, fr: dict, pick_t: dict, first_learned_t: float) -> list[dict]:
    from vtscore.training.thresholds.precision_floor import precision_floor_cut  # noqa: PLC0415

    test_s = fr["test_scores"].astype(np.float64)
    test_y = fr["test_labels"].astype(np.int64)
    order = np.argsort(-test_s, kind="stable")
    ys = test_y[order]
    k = np.arange(1, len(ys) + 1)
    prec, rec = np.cumsum(ys) / k, np.cumsum(ys) / max(int(ys.sum()), 1)
    oracle = {x: (float(rec[np.flatnonzero(prec >= x).max()]) if (prec >= x).any() else 0.0) for x in FLOORS}
    if len(fr["fold_cal_vote"]) != len(fr["fold_cal_scores"]):
        return []  # frame without provenance (grouped path): cannot filter
    n_opening = int((~learned_mask(fr["fold_cal_vote"], pick_t, first_learned_t)).sum())
    rows = []
    for ev in ("all", "learned"):
        orderings, haystacks, n_pos = evidence(fr, ev, pick_t, first_learned_t)
        for pool_name in ("shipped", "consistent"):
            ref = reference_pool(fr, pool_name)
            for x in FLOORS:
                c = precision_floor_cut(x, test_s, ref, orderings, haystacks)
                promised = c.status.value == "promised"
                sel = test_s >= c.threshold if promised else np.zeros(len(test_s), dtype=bool)
                n_ret = int(sel.sum())
                achieved = float(test_y[sel].mean()) if n_ret else float("nan")
                rows.append(
                    {
                        **meta,
                        "t": int(fr["t"]),
                        "evidence": ev,
                        "pool": pool_name,
                        "X": x,
                        "n_cal": sum(len(o[0]) for o in orderings),
                        "n_cal_opening": n_opening,
                        "n_cal_pos": n_pos,
                        "status": c.status.value,
                        "returned": n_ret,
                        "achieved": achieved,
                        "recall": float(test_y[sel].sum() / max(int(test_y.sum()), 1)),
                        "oracle_recall": oracle[x],
                        "broken": bool(promised and n_ret and achieved < x),
                    }
                )
    return rows


def cell_rows(job: tuple[str, str, str, str]) -> list[dict]:
    arm, npz_path, main_path, picks_path = job
    first = pd.read_csv(main_path, nrows=1)
    if first.empty or not picks_path:
        return []
    picks = pd.read_csv(picks_path, usecols=["t", "phase", "picked_id"])
    pick_t = dict(zip(picks["picked_id"].astype(int), picks["t"].astype(int), strict=False))
    learned = picks[picks["phase"].isin(LEARNED_PHASES)]
    first_learned_t = float(learned["t"].min()) if len(learned) else float("inf")
    meta = {
        "arm": arm,
        "category": str(first["category"].iloc[0]),
        "seed": int(first["seed"].iloc[0]),
        "band": str(first["category"].iloc[0]).split("@")[-1],
    }
    z = np.load(npz_path)
    out = []
    for st in sorted({k.split("/")[0] for k in z.files}, key=lambda s: int(s[1:])):
        fr = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(st + "/")}
        if "fold_cal_vote" not in fr:
            continue
        out += frame_rows(meta, fr, pick_t, first_learned_t)
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(gated=df["n_cal_pos"] >= 10, made=df["status"] == "promised")
    g = df.groupby(["arm", "evidence", "pool", "X"])
    s = g.agg(
        frames=("made", "size"),
        past_gate=("gated", "mean"),
        promises_made=("made", "mean"),
        mean_cal_pos=("n_cal_pos", "mean"),
        mean_cal_opening=("n_cal_opening", "mean"),
        recall=("recall", "mean"),
        oracle=("oracle_recall", "mean"),
    )
    made = df[df["made"]].groupby(["arm", "evidence", "pool", "X"])
    s = s.join(made.agg(made_n=("broken", "size"), broken_of_made=("broken", "mean")))
    s["broken_se"] = np.sqrt(s["broken_of_made"] * (1 - s["broken_of_made"]) / s["made_n"])
    return s.reset_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="label=results_dir (holding cells/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--shard", help="I/N: analyze every Nth cell from I, write its rows only")
    ap.add_argument("--merge", action="store_true", help="summarize the shards already in --out")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        shards = sorted(args.out.glob("rows_shard_*.csv.gz"))
        return report(pd.concat([pd.read_csv(p) for p in shards]), args.out, {"shards": len(shards)})
    jobs = []
    for spec in args.arm:
        arm, _, d = spec.partition("=")
        cells = Path(d) / "cells"
        mains = {p.name.split(".")[0]: p for p in main_frame_files(cells)}
        picks = {p.name.split("__")[0]: p for p in side_frame_files(cells, "__picks")}
        jobs += [
            (arm, str(p), str(mains[p.name.split("__")[0]]), str(picks.get(p.name.split("__")[0], "")))
            for p in pframe_files(cells)
            if p.name.split("__")[0] in mains
        ]
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        jobs = jobs[i::n]
    rows: list[dict] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for r in ex.map(cell_rows, jobs, chunksize=4):
                rows += r
    else:
        for j in jobs:
            rows += cell_rows(j)
    if not rows and args.shard:
        return 0  # every cell in this shard starved; --merge finds nothing to read
    if not rows:
        raise SystemExit("no frames with fold_cal_vote - were the arms run with the #4256 harness?")
    df = pd.DataFrame(rows)
    if args.shard:
        df.to_csv(args.out / f"rows_shard_{args.shard.split('/')[0]}.csv.gz", index=False, float_format="%.5g")
        return 0
    return report(df, args.out, {"cells": len(jobs)})


def report(df: pd.DataFrame, out: Path, provenance: dict) -> int:
    df.to_csv(out / "provenance_rows.csv.gz", index=False, float_format="%.5g")
    s = summarize(df)
    s.to_csv(out / "provenance_summary.csv", index=False, float_format="%.4g")
    by_t = summarize(df.assign(arm=df["arm"] + "@t" + df["t"].astype(str)))
    by_t.to_csv(out / "provenance_by_t.csv", index=False, float_format="%.4g")
    (out / "provenance.json").write_text(json.dumps({**provenance, "rows": len(df)}, indent=2) + "\n")
    pd.set_option("display.width", 220)
    print(s[s["X"] == 0.5].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
