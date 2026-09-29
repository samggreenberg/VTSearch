"""#4219 / #4213: the head comparison by category difficulty, and the acquisition mechanism.

Difficulty is each category's mean average precision over clicks 1-150 on the
#4114 ``linear`` arm (the old early-stopped logistic head).  That trajectory is
independent of both arms compared here; stratifying on the SVM's own AP would
bias the easy stratum against the other head by regression to the mean.

Writes, into ``--out``:

* ``binary_quartiles.csv`` - binary voting, ``lrconv - svm`` (and any other
  arm) per difficulty quartile: cost, oracle cost, regret, AP (mean over
  clicks per cell, SE clustered on category);
* ``region_strata.csv`` - the same for the region grid's hard and easy
  strata (the grid IS those two quartiles, ``region_categories.txt``), plus
  Goods found at clicks 40 and 150;
* ``region_acquisition.csv`` - per arm and click bucket on the easy stratum:
  the acquisition cut's pool percentile and score, and the Good rate of each
  Autopilot phase's picks.

    python strata_4219.py --difficulty /expscratch/$USER/logreg-4114/linear/results \\
        --binary /expscratch/$USER/logreg-4219 --region /expscratch/$USER/logreg-4213 \\
        --categories docs/experiments/2026-09-28-head-switch-4219/region_categories.txt --out DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))

import analyze_stage_b as B  # noqa: E402

METRICS = ("cost", "oracle_cost", "regret", "average_precision")
BUCKETS = [0, 20, 50, 100, 150]


def fmt(r: dict) -> str:
    return f"{r['mean']:+.4f} ± {r['se']:.4f}" if r.get("n") else ""


def by_stratum(df: pd.DataFrame, stratum: dict[str, str], arms: list[str]) -> list[dict]:
    rows = []
    for m in METRICS:
        u = B.aulc(df, m)
        u["stratum"] = u["category"].map(stratum)
        for s, g in u.groupby("stratum"):
            for arm in arms:
                r = B.paired(g, m, arm)
                rows.append({"stratum": s, "arm": arm, "metric": m, "window": "aulc", **r})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", required=True, type=Path, help="the #4114 linear arm's results dir")
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--region", required=True, type=Path)
    ap.add_argument("--categories", required=True, type=Path, help="region_categories.txt")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import _cells_io

    lin, _ = _cells_io.load_arm(args.difficulty)
    ap_by_cat = lin.groupby("category")["average_precision"].mean()
    quart = pd.qcut(ap_by_cat, 4, labels=["q1 hard", "q2", "q3", "q4 easy"]).astype(str).to_dict()

    # Binary: every arm but svm, by quartile.
    df, _ = B.load(args.binary)
    arms = [a for a in B.ARMS if a in set(df["arm"]) and a != "svm"]
    out = pd.DataFrame(by_stratum(df, quart, arms))
    out.to_csv(args.out / "binary_quartiles.csv", index=False)
    print("binary, other - svm by difficulty quartile (mean over clicks):")
    print(
        out.assign(s=out.apply(fmt, axis=1))
        .pivot_table(index=["arm", "stratum"], columns="metric", values="s", aggfunc="first")
        .to_string()
    )

    # Region: the grid's own two strata.
    lines = args.categories.read_text().splitlines()
    hard = set(lines[lines.index("# hard") + 1 : lines.index("# easy")])
    easy = set(lines[lines.index("# easy") + 1 :])
    stratum_of = {c: "hard" for c in hard} | {c: "easy" for c in easy}
    rdf, _ = B.load(args.region)
    rows = by_stratum(rdf, stratum_of, ["lrconv"])
    for t in (40, 150):
        snap = B.at_checkpoint(rdf, t)
        snap["stratum"] = snap["category"].map(stratum_of)
        for s, g in snap.groupby("stratum"):
            r = B.paired(g, "n_good", "lrconv")
            lv = g.groupby("arm")["n_good"].mean()
            rows.append(
                {
                    "stratum": s,
                    "arm": "lrconv",
                    "metric": "n_good",
                    "window": f"t{t}",
                    **r,
                    "level_svm": float(lv.get("svm", np.nan)),
                    "level_other": float(lv.get("lrconv", np.nan)),
                }
            )
    rout = pd.DataFrame(rows)
    rout.to_csv(args.out / "region_strata.csv", index=False)
    print("\nregion, lrconv - svm by stratum:")
    print(
        rout.assign(s=rout.apply(fmt, axis=1))
        .pivot_table(index=["stratum", "window"], columns="metric", values="s", aggfunc="first")
        .to_string()
    )

    # Region mechanism, easy stratum: where the acquisition cut sits, and what its picks are.
    acq = []
    for arm in ("svm", "lrconv"):
        base, _ = _cells_io.load_arm(args.region / arm / "results")
        base = base[base["category"].isin(easy)]
        base["bucket"] = pd.cut(base["t"], BUCKETS).astype(str)
        lv = base.groupby("bucket")[["acq_pool_percentile", "acq_threshold", "threshold"]].mean()
        picks = pd.concat(
            [pd.read_csv(f) for f in sorted((args.region / arm / "results" / "cells").glob("task_*__picks.csv*"))],
            ignore_index=True,
        )
        picks = picks[picks["category"].isin(easy)]
        picks["bucket"] = pd.cut(picks["t"], BUCKETS).astype(str)
        good = picks.groupby("bucket")["picked_label"].mean().rename("good_rate_all")
        hard_rate = (
            picks[picks["phase"] == "hard"].groupby("bucket")["picked_label"].mean().rename("good_rate_hard_phase")
        )
        acq.append(lv.join(good).join(hard_rate).assign(arm=arm).reset_index())
    aq = pd.concat(acq)
    aq.to_csv(args.out / "region_acquisition.csv", index=False)
    print("\nregion, easy stratum, acquisition by click bucket:")
    print(aq.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
