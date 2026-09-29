#!/usr/bin/env python3
"""#4222 follow-up: does the dry stop (#4254) beat a fixed Good target in both low pools?

Reads opening arms per pool, each ``pool/label=results_dir``, where ``label`` is
``g<G>`` (a fixed Good target) or ``g<G>d<W>`` (the dry stop ``g<G>+dry1/<W>``:
the text sort until G Goods, or until W picks in a row hold none).  ``g3`` is
each pool's control, today's app.  Reports, per arm:

* **the opening** - its median length in votes, and how it ended: met G, ran
  dry, or never handed over in 150 votes;
* **harvest** - Good votes by 25/50/100/150, and the share of hunts that found
  fewer than 3 positives in 150 votes, per size band (#4216: 28% of ``@small``
  hunts starve), paired against the control;
* **detector quality** - average precision and oracle cost per checkpoint,
  paired per cell against the pool's control, overall and per size band.

The decision rule is fixed on #4222 before the run: the arm with the largest
mean paired AP gain over g3 at vote 150, averaged over the pools, among arms
not below g3 by more than 2 paired SE in any pool.  Tie within one SE: the
simpler arm (a fixed G over a dry stop, the smaller G).

    python analyze_drystop_4222.py --arm 0.44%/g3=DIR --arm 0.44%/g20d8=DIR ... --out OUT
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_textgood_4222 as T  # noqa: E402
from _cells_paths import side_frame_files  # noqa: E402

CHECKPOINTS = T.CHECKPOINTS
KEY = T.KEY
LEARNED = T.LEARNED
CONTROL = "g3"
STARVED = 3  # fewer positives than this in 150 votes is a starved hunt (#4216)
ARM_RE = re.compile(r"^g(\d+)(?:d(\d+))?$")


def parse_arm(spec: str) -> tuple[str, str, int, int, Path]:
    """``pool/label=dir`` -> ``(pool, label, G, dry window or 0, cells dir)``."""
    head, _, d = spec.partition("=")
    pool, _, label = head.rpartition("/")
    m = ARM_RE.match(label)
    if not pool or not m or not d:
        raise SystemExit(f"--arm wants pool/g<G>[d<W>]=DIR, got {spec!r}")
    return pool, label, int(m.group(1)), int(m.group(2) or 0), Path(d) / "cells"


def sessions(cells: Path, g: int) -> pd.DataFrame:
    """One row per (cell, checkpoint): Goods so far, and how the opening ended."""
    rows = []
    for f in side_frame_files(cells, "__picks"):
        p = pd.read_csv(f, usecols=["category", "seed", "t", "phase", "picked_label"]).sort_values("t")
        if p.empty:
            continue
        learned = p[p["phase"].isin(LEARNED)]
        opening = int(learned["t"].min() - 1) if len(learned) else int(p["t"].max())
        cum = p["picked_label"].cumsum().to_numpy()
        ts = p["t"].to_numpy()
        at_handover = int(cum[ts <= opening][-1]) if (ts <= opening).any() else 0
        if not len(learned):
            ended = "never"
        elif at_handover >= g:
            ended = "met_G"
        else:
            ended = "ran_dry"
        cat = str(p["category"].iloc[0])
        for c in CHECKPOINTS:
            upto = cum[ts <= c]
            rows.append(
                {
                    "category": cat,
                    "band": cat.split("@")[-1],
                    "seed": int(p["seed"].iloc[0]),
                    "t": c,
                    "goods": int(upto[-1]) if len(upto) else 0,
                    "opening_votes": opening,
                    "ended": ended,
                }
            )
    return pd.DataFrame(rows)


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str, on: list[str]) -> tuple[float, float, int]:
    m = a.merge(b, on=on, suffixes=("_a", "_b"))
    d = (m[f"{col}_a"] - m[f"{col}_b"]).dropna()
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return float(d.mean()) if len(d) else float("nan"), se, len(d)


def simplicity(label: str) -> tuple[int, int]:
    """Sort key for a tie: a fixed G before a dry stop, then the smaller G."""
    m = ARM_RE.match(label)
    assert m
    return (1 if m.group(2) else 0, int(m.group(1)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="pool/label=results_dir (holding cells/)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    arms = [parse_arm(s) for s in args.arm]
    pools = sorted({a[0] for a in arms})
    for pool in pools:
        if (pool, CONTROL) not in {(a[0], a[1]) for a in arms}:
            raise SystemExit(f"pool {pool} has no {CONTROL} control arm")

    S, Q = {}, {}
    for pool, label, g, _w, cells in arms:
        s = sessions(cells, g)
        if s.empty:
            raise SystemExit(f"no pick logs under {cells}")
        S[pool, label] = s
        q = T.quality(cells)
        q["band"] = q["category"].str.split("@").str[-1]
        Q[pool, label] = q

    # --- the opening and harvest ----------------------------------------------------------
    open_rows, harv_rows, starve_rows = [], [], []
    for (pool, label), s in S.items():
        last = s[s["t"] == 150]
        ended = last["ended"].value_counts(normalize=True)
        open_rows.append(
            {
                "pool": pool,
                "arm": label,
                "cells": len(last),
                "median_opening_votes": float(last["opening_votes"].median()),
                "met_G": float(ended.get("met_G", 0.0)),
                "ran_dry": float(ended.get("ran_dry", 0.0)),
                "never_handed_over": float(ended.get("never", 0.0)),
            }
        )
        for t in CHECKPOINTS:
            harv_rows.append({"pool": pool, "arm": label, "t": t, "mean_goods": float(s[s["t"] == t]["goods"].mean())})
        ctrl = S[pool, CONTROL]
        for band in ["all", *sorted(last["band"].unique())]:
            a = last if band == "all" else last[last["band"] == band]
            b = ctrl[ctrl["t"] == 150]
            b = b if band == "all" else b[b["band"] == band]
            a = a.assign(starved=(a["goods"] < STARVED).astype(float))
            b = b.assign(starved=(b["goods"] < STARVED).astype(float))
            m, se, n = paired(a, b, "starved", KEY)
            starve_rows.append(
                {
                    "pool": pool,
                    "arm": label,
                    "band": band,
                    "cells": len(a),
                    "starved": float(a["starved"].mean()),
                    "delta_vs_g3": m,
                    "se": se,
                    "n_paired": n,
                }
            )
    pd.DataFrame(open_rows).to_csv(args.out / "opening.csv", index=False, float_format="%.4g")
    pd.DataFrame(harv_rows).to_csv(args.out / "harvest.csv", index=False, float_format="%.4g")
    pd.DataFrame(starve_rows).to_csv(args.out / "starved.csv", index=False, float_format="%.4g")

    # --- detector quality, paired against each pool's control -------------------------------
    qrows = []
    for (pool, label), q in Q.items():
        ctrl = Q[pool, CONTROL]
        for band in ["all", *sorted(q["band"].unique())]:
            a = q if band == "all" else q[q["band"] == band]
            b = ctrl if band == "all" else ctrl[ctrl["band"] == band]
            for t in CHECKPOINTS:
                for col in ("average_precision", "oracle_cost"):
                    m, se, n = paired(a[a["t"] == t], b[b["t"] == t], col, [*KEY, "t"])
                    qrows.append(
                        {
                            "pool": pool,
                            "arm": label,
                            "band": band,
                            "t": t,
                            "metric": col,
                            "level": float(a[a["t"] == t][col].mean()),
                            "delta_vs_g3": m,
                            "se": se,
                            "n": n,
                        }
                    )
    qd = pd.DataFrame(qrows)
    qd.to_csv(args.out / "quality_paired.csv", index=False, float_format="%.4g")

    # --- the fixed decision rule -------------------------------------------------------------
    ap150 = qd[(qd["band"] == "all") & (qd["t"] == 150) & (qd["metric"] == "average_precision")]
    labels = sorted({label for _p, label in Q}, key=simplicity)
    table = {}
    for label in labels:
        rows = ap150[ap150["arm"] == label].set_index("pool")
        if set(rows.index) != set(pools):
            continue  # an arm must be run in every pool to be eligible
        safe = all(not (rows.loc[p, "delta_vs_g3"] < -2 * rows.loc[p, "se"]) for p in pools)
        table[label] = {
            "mean_gain": float(rows["delta_vs_g3"].mean()),
            "se": float(np.sqrt((rows["se"].fillna(0) ** 2).sum()) / len(pools)),
            "safe": bool(safe),
            "per_pool": {p: round(float(rows.loc[p, "delta_vs_g3"]), 4) for p in pools},
        }
    eligible = {k: v for k, v in table.items() if v["safe"]}
    best = max(eligible.values(), key=lambda v: v["mean_gain"]) if eligible else None
    winner = None
    if best is not None:
        close = [k for k, v in eligible.items() if v["mean_gain"] >= best["mean_gain"] - best["se"]]
        winner = min(close, key=simplicity)
    verdict = {"arms": table, "winner": winner}
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(open_rows).round(3).to_string(index=False))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
