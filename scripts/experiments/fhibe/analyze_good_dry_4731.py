#!/usr/bin/env python
"""#4731: does a Good phase that runs dry get a few-photo person to a detector, and at what cost?

Reads the FHIBE arms ``launch.sh`` ran on one identity file (``ctl`` = the app, then the arms), pairs
them session for session (identity x dataset), and writes, per photo-count stratum and dataset:

* the F-beta curve over clicks 1-150, every session filled at every click (the last row carried
  through a spot check's jump), and its mean over 1-150 and 1-50: the area under the curve the
  owner reads as one number (``eval-horizon``), paired against ``ctl``;
* how many sessions show a detector (``app_trained``) by click 150, and with a trained head;
* where the Good phase ran dry, and whether the arm agreed with ``ctl`` pick for pick until then.

F-beta is the run's own (``fbeta``, the shipped line at the run's balance) on the withheld half,
read off each click's base row: what Test gives at that click, which is the Goods' centroid under
the label quota and the trained head from it on. The click count ``t`` includes the example vote.

Usage::

    python analyze_good_dry_4731.py --runs ctl=<dir> dry16=<dir> dry16q16=<dir> \\
        --strata <identities.txt.strata.tsv> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 150
SHORT = 50
#: Phases before the app shows a detector.  ``example`` is the K example votes.
OPENING = {"example", "good", "bad", "more"}
_COLS = [
    "dataset",
    "category",
    "t",
    "phase",
    "app_trained",
    "detector_tier",
    "gmm_variant",
    "fbeta",
    "average_precision",
    "n_flagged",
    "n_good",
    "n_bad",
    "n_test_pos",
    "strategy",
]


def load_rows(run: Path) -> pd.DataFrame:
    """Every cell's base rows (the shipped line, one per click), clicks 1..HORIZON."""
    frames = []
    for f in sorted(glob.glob(str(run / "results" / "cells" / "task_[0-9][0-9][0-9][0-9].csv"))):
        d = pd.read_csv(f, usecols=lambda c: c in _COLS, low_memory=False)
        d = d[(d["strategy"] == "autopilot") & d["gmm_variant"].isna()]
        frames.append(d.drop(columns=["gmm_variant", "strategy"]))
    if not frames:
        raise SystemExit(f"no cells under {run}")
    rows = pd.concat(frames, ignore_index=True)
    return rows[rows["t"] <= HORIZON]


def load_picks(run: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(f, usecols=["dataset", "category", "t", "phase", "picked_id", "picked_label"])
        for f in sorted(glob.glob(str(run / "results" / "cells" / "task_*__picks.csv")))
    ]
    return pd.concat(frames, ignore_index=True)


def session_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per session: its filled F-beta curve and what it showed."""
    out = []
    grid = np.arange(1, HORIZON + 1)
    for (ds, cat), g in rows.groupby(["dataset", "category"], sort=False):
        g = g.drop_duplicates("t", keep="last").set_index("t").sort_index()
        # Fill every click: a spot check answers several picks at once and skips clicks; carry the
        # last row forward (the set on screen until the next one).  Clicks before the first row
        # (none under the example opening) score 0.
        curve = g["fbeta"].reindex(grid).ffill().fillna(0.0).to_numpy()
        shown = g[g["app_trained"] == 1]
        trained = shown[shown["detector_tier"] == "trained"]
        left = g[~g["phase"].isin(OPENING)]
        out.append(
            {
                "dataset": ds,
                "category": cat,
                "fb_mean": float(curve.mean()),
                "fb_short": float(curve[:SHORT].mean()),
                "fb_150": float(curve[-1]),
                "shown_at": float(shown.index.min()) if len(shown) else math.nan,
                "trained_shown_at": float(trained.index.min()) if len(trained) else math.nan,
                "left_opening_at": float(left.index.min()) if len(left) else math.nan,
                "goods_150": int(g["n_good"].iloc[-1]),
                "n_test_pos": int(g["n_test_pos"].iloc[0]),
                "flagged_150": float(g["n_flagged"].iloc[-1]),
                "curve": curve,
            }
        )
    return pd.DataFrame(out)


def dry_fire(picks: pd.DataFrame, dry_run: int) -> pd.DataFrame:
    """Per session: the click a Good walk of *dry_run* misses with a Good in hand would end, or NaN."""
    out = []
    for (ds, cat), g in picks.groupby(["dataset", "category"], sort=False):
        goods, misses, fire = 0, 0, math.nan
        for p in g.sort_values("t").itertuples():
            if p.phase not in ("example", "good"):
                break
            if p.picked_label == 1:
                goods, misses = goods + 1, 0
            elif goods >= 1:
                misses += 1
                if misses >= dry_run:
                    fire = float(p.t)
                    break
        out.append({"dataset": ds, "category": cat, "fires_at": fire})
    return pd.DataFrame(out)


def agree_until(a: pd.DataFrame, b: pd.DataFrame, upto: pd.DataFrame) -> pd.DataFrame:
    """Per session: did arms *a* and *b* pick the same media through click ``fires_at``?"""
    key = ["dataset", "category"]
    m = a.merge(b, on=key + ["t"], suffixes=("_a", "_b")).merge(upto, on=key)
    m = m[m["t"] <= m["fires_at"]]
    same = m.groupby(key).apply(lambda g: bool((g["picked_id_a"] == g["picked_id_b"]).all()), include_groups=False)
    return same.rename("same_until_fire").reset_index()


def paired(base: pd.DataFrame, arm: pd.DataFrame, col: str) -> tuple[float, float, int]:
    m = base.merge(arm, on=["dataset", "category"], suffixes=("_c", "_a"))
    d = (m[f"{col}_a"] - m[f"{col}_c"]).to_numpy()
    if len(d) == 0:
        return math.nan, math.nan, 0
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else math.nan
    return float(d.mean()), se, len(d)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, help="arm=<run dir>; the first is the control")
    ap.add_argument("--strata", type=Path, required=True)
    ap.add_argument("--dry-run", type=int, default=16, help="the arms' Good dry run, to locate where it fires")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    arms = dict(spec.split("=", 1) for spec in args.runs)
    ctl_name = next(iter(arms))
    strata = pd.read_csv(args.strata, sep="\t").rename(columns={"identity": "category"})
    args.out.mkdir(parents=True, exist_ok=True)

    sessions: dict[str, pd.DataFrame] = {}
    picks: dict[str, pd.DataFrame] = {}
    for name, run in arms.items():
        sessions[name] = session_table(load_rows(Path(run))).merge(strata, on="category", how="left")
        picks[name] = load_picks(Path(run))
        print(f"{name}: {len(sessions[name])} sessions")

    # Every arm must hold the same sessions, or the pairing is a selection.
    keysets = {n: set(zip(s["dataset"], s["category"])) for n, s in sessions.items()}
    common = set.intersection(*keysets.values())
    for n, ks in keysets.items():
        if ks != common:
            print(f"WARNING {n}: {len(ks - common)} sessions not in every arm; pairing on the {len(common)} common")
    for n in sessions:
        s = sessions[n]
        sessions[n] = s[[(d, c) in common for d, c in zip(s["dataset"], s["category"])]].reset_index(drop=True)

    fire = dry_fire(picks[ctl_name], args.dry_run)
    summary = []
    for n, s in sessions.items():
        s = s.merge(fire, on=["dataset", "category"])
        for (stratum, ds), g in s.groupby(["stratum", "dataset"]):
            row = {
                "arm": n,
                "stratum": stratum,
                "dataset": ds,
                "n": len(g),
                "fb_mean": g["fb_mean"].mean(),
                "fb_short": g["fb_short"].mean(),
                "fb_150": g["fb_150"].mean(),
                "shown_by_150": g["shown_at"].notna().mean(),
                "trained_by_150": g["trained_shown_at"].notna().mean(),
                "median_shown_at": g["shown_at"].median(),
                "dry_fires_in_ctl": g["fires_at"].notna().mean(),
                "median_flagged_150": g["flagged_150"].median(),
            }
            if n != ctl_name:
                c = sessions[ctl_name]
                c = c[(c["stratum"] == stratum) & (c["dataset"] == ds)]
                for col in ("fb_mean", "fb_short", "fb_150"):
                    row[f"d_{col}"], row[f"se_{col}"], _ = paired(c, g, col)
                fired = g[g["fires_at"].notna()]
                row["d_fb_mean_fired"], row["se_fb_mean_fired"], row["n_fired"] = paired(c, fired, "fb_mean")
            summary.append(row)
    table = pd.DataFrame(summary)
    table.to_csv(args.out / "summary.csv", index=False)

    # Curves: mean F-beta at every click, per arm x stratum x dataset.
    curves = []
    for n, s in sessions.items():
        for (stratum, ds), g in s.groupby(["stratum", "dataset"]):
            mean_curve = np.mean(np.stack(g["curve"].to_numpy()), axis=0)
            curves += [
                {"arm": n, "stratum": stratum, "dataset": ds, "t": t + 1, "fbeta": float(v)}
                for t, v in enumerate(mean_curve)
            ]
    pd.DataFrame(curves).to_csv(args.out / "curves.csv", index=False)

    # Per-session table (no curves) for spot reads.
    pd.concat([s.drop(columns=["curve"]).assign(arm=n) for n, s in sessions.items()]).to_csv(
        args.out / "sessions.csv", index=False
    )

    # Pick-for-pick agreement with the control until the dry run fires.
    agree = []
    for n in arms:
        if n == ctl_name:
            continue
        a = agree_until(picks[ctl_name], picks[n], fire[fire["fires_at"].notna()])
        agree.append({"arm": n, "fired_sessions": len(a), "same_until_fire": int(a["same_until_fire"].sum())})
    pd.DataFrame(agree).to_csv(args.out / "agreement.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
        print(pd.DataFrame(agree).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
