#!/usr/bin/env python
"""The face review's own tables (#4762): headline, strata, Find at every vote, and the hand-over.

``sota_face_analyze.sh`` runs the State of the App's analyzer per session set; this reads its
``cells.csv`` files, the people's strata (``person_strata.py``) and the run cells, and writes the
tables the face report adds to the photo reviews' shape. Every run is in every average (#4631).

Written under ``--out`` (aggregates only, so they can be committed):

* ``headline.csv`` - per starting photos (K) x size x beta: the session's objective (F-beta of the
  withheld set at the app's line) at click 0 and at 10/25/50/100/150 clicks and after the check,
  the returned set's median size, AP from click 0 to the end and the ceiling's, headroom and what
  the clicks bought, positives found.
* ``strata.csv`` - the same per stratum of the people: age bracket, pronoun, skin tone, ancestry,
  face-size band, photo count. Mean and SE over people.
* ``find_by_vote.csv`` - what a Find returns at every vote (Test, the export): mean F-beta and AP,
  the median kept, the share of runs on the Goods' centroid, the share whose vote was a
  prompted spot check's pick (#4496), and the share that has entered one by that vote. This is where the dip is read.
* ``handover.csv`` - per K x size x beta, paired per run: Find's F-beta, AP and kept count at the
  last vote that gave the centroid against the first that gave the trained head.

``--worst N`` also prints the N people with the lowest final objective per set, by subject ID,
for the report to point at (the owner: the report may name subject IDs).

Usage::

    python sota_face_tables.py --analysis <dir from sota_face_analyze.sh> --runs <release runs dir> \\
        --date <date> --strata <person_strata.csv> --out <report dir> [--worst 10]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import _cells_io  # noqa: E402

KS = (1, 4)
SIZES = ("1024", "640")
BETAS = {0.25: "025", 1.0: "1", 4.0: "4"}
HORIZON = 150
CHECKPOINTS = (10, 25, 50, 100, 150)
STRATA = ("age_bracket", "pronoun", "skin_tone", "ancestry", "face_band", "n_photos")


def _se(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def headline_row(c: pd.DataFrame) -> dict:
    """One session set's headline, over every run."""
    row = {"runs": len(c), "never_trained": int(c["never_trained"].fillna(0).astype(bool).sum())}
    row["F_click0"] = c["text_thr_fbeta"].mean()
    for t in CHECKPOINTS:
        row[f"F_{t}"] = c[f"thr_fbeta_{t}"].mean()
    row["F_unchecked"] = c["thr_fbeta_unchecked"].mean()
    row["F_after_check"] = c["thr_fbeta_final"].mean()
    row["F_after_check_se"] = _se(c["thr_fbeta_final"])
    row["returned_median"] = c["thr_returned_final"].median()
    row["precision_after_check"] = c["thr_precision_final"].mean()
    row["recall_after_check"] = c["thr_recall_final"].mean()
    row["AP_click0"] = c["text_ap"].mean()
    for t in (25, 50):
        row[f"AP_{t}"] = c[f"ap_{t}"].mean()
    row["AP_final"] = c["final_ap"].mean()
    row["AP_ceiling"] = c["ceiling_ap"].mean()
    row["headroom"] = c["headroom"].mean()
    row["clicks_bought"] = c["clicks_bought"].mean()
    row["positives_found"] = c["positives_found"].mean()
    return row


def read_find(path: str) -> dict | None:
    """One run's Find at every vote 1..HORIZON (Test's rows, carried), its tier and its prompted votes."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except (pd.errors.EmptyDataError, OSError):
        return None
    if df.empty:
        return None
    b = _cells_io._base_rows(df)
    b = b[~_cells_io.check_rows(b)]
    b = b[b["t"] <= HORIZON].groupby("t").last()
    if b.empty:
        return None
    idx = range(1, HORIZON + 1)
    f = b["fbeta"].reindex(idx).ffill().fillna(0.0).to_numpy()
    ap = b["average_precision"].reindex(idx).ffill().to_numpy(dtype=float)
    kept = b["n_flagged"].reindex(idx).ffill().to_numpy(dtype=float)
    tier = b["detector_tier"].reindex(idx).ffill().fillna("").astype(str).to_numpy()
    pk_path = Path(path.replace(".csv", "__picks.csv"))
    prompt = np.zeros(HORIZON, dtype=bool)
    if pk_path.exists() and pk_path.stat().st_size:
        pk = pd.read_csv(pk_path, usecols=["t", "phase"])
        ts = pk.loc[pk["phase"].astype(str) == "prompt", "t"].to_numpy(dtype=int)
        prompt[[t - 1 for t in ts if 1 <= t <= HORIZON]] = True
    # The hand-over: the last vote on the centroid and the first on the trained head.
    hand = {}
    cen = np.where(tier == "centroid")[0]
    tra = np.where(tier == "trained")[0]
    if len(cen) and len(tra) and tra.min() > cen.max():
        i, j = cen.max(), tra.min()
        hand = {
            "t_centroid": i + 1,
            "t_trained": j + 1,
            "F_centroid": f[i],
            "F_trained": f[j],
            "AP_centroid": ap[i],
            "AP_trained": ap[j],
            "kept_centroid": kept[i],
            "kept_trained": kept[j],
        }
    return {
        "dataset": str(b["dataset"].iloc[0]),
        "category": str(b["category"].iloc[0]),
        "f": f,
        "ap": ap,
        "kept": kept,
        "centroid": tier == "centroid",
        "prompt": prompt,
        "hand": hand,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True, help="the release's derived runs directory")
    ap.add_argument("--date", required=True)
    ap.add_argument("--strata", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worst", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    strata = pd.read_csv(args.strata, dtype={"skin_tone": str})
    heads, strat_rows, by_vote, hand_rows = [], [], [], []
    for k in KS:
        for size in SIZES:
            ds = f"fhibe_faces_{size}"
            for beta, tag in BETAS.items():
                cpath = args.analysis / f"k{k}-b{tag}-{size}" / "cells.csv"
                if not cpath.exists():
                    continue
                c = pd.read_csv(cpath).merge(strata, on="category", how="left")
                key = {"K": k, "size": size, "beta": beta}
                heads.append({**key, **headline_row(c)})
                for col in STRATA:
                    for level, g in c.groupby(col):
                        strat_rows.append(
                            {
                                **key,
                                "stratum": col,
                                "level": level,
                                "people": len(g),
                                "F_after_check": g["thr_fbeta_final"].mean(),
                                "F_after_check_se": _se(g["thr_fbeta_final"]),
                                "F_25": g["thr_fbeta_25"].mean(),
                                "AP_final": g["final_ap"].mean(),
                                "AP_ceiling": g["ceiling_ap"].mean(),
                            }
                        )
                if args.worst:
                    w = c.sort_values("thr_fbeta_final").head(args.worst)
                    print(f"\nlowest final objective, K={k} {size} beta {beta:g}:")
                    for r in w.itertuples():
                        print(
                            f"  {r.category}  F {r.thr_fbeta_final:.2f}  AP {r.final_ap:.2f}  ceiling AP {r.ceiling_ap:.2f}"
                            f"  ({r.age_bracket}, {r.pronoun}, skin {r.skin_tone}, {r.ancestry}, {r.n_photos} photos)"
                        )
                # Find at every vote, from the run cells.
                cells = args.runs / f"{args.date}-sota-k{k}-b{tag}" / "results" / "cells"
                files = [str(f) for f in _cells_io.main_frame_files(cells)]
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    got = [g for g in pool.map(read_find, files, chunksize=8) if g is not None and g["dataset"] == ds]
                if not got:
                    continue
                F = np.stack([g["f"] for g in got])
                AP = np.stack([g["ap"] for g in got])
                KEPT = np.stack([g["kept"] for g in got])
                CEN = np.stack([g["centroid"] for g in got])
                PR = np.stack([g["prompt"] for g in got])
                for t in range(HORIZON):
                    by_vote.append(
                        {
                            **key,
                            "vote": t + 1,
                            "runs": len(got),
                            "F": float(F[:, t].mean()),
                            "AP": float(np.nanmean(AP[:, t])),
                            "kept_median": float(np.nanmedian(KEPT[:, t])),
                            "share_centroid": float(CEN[:, t].mean()),
                            "share_prompt": float(PR[:, t].mean()),
                            # Runs that have entered a prompted spot check by this vote: a round's picks
                            # land on one vote, so the per-vote share is a comb.
                            "share_prompted_by": float(PR[:, : t + 1].any(axis=1).mean()),
                        }
                    )
                h = pd.DataFrame([g["hand"] for g in got if g["hand"]])
                if len(h):
                    d_f, d_ap = h["F_trained"] - h["F_centroid"], h["AP_trained"] - h["AP_centroid"]
                    hand_rows.append(
                        {
                            **key,
                            "runs": len(h),
                            "t_trained_median": h["t_trained"].median(),
                            "F_centroid": h["F_centroid"].mean(),
                            "F_trained": h["F_trained"].mean(),
                            "dF": d_f.mean(),
                            "dF_se": _se(d_f),
                            "share_F_drops": float((d_f < 0).mean()),
                            "AP_centroid": h["AP_centroid"].mean(),
                            "AP_trained": h["AP_trained"].mean(),
                            "dAP": d_ap.mean(),
                            "dAP_se": _se(d_ap),
                            "kept_centroid_median": h["kept_centroid"].median(),
                            "kept_trained_median": h["kept_trained"].median(),
                        }
                    )
    pd.DataFrame(heads).round(4).to_csv(args.out / "headline.csv", index=False)
    pd.DataFrame(strat_rows).round(4).to_csv(args.out / "strata.csv", index=False)
    pd.DataFrame(by_vote).round(4).to_csv(args.out / "find_by_vote.csv", index=False)
    pd.DataFrame(hand_rows).round(4).to_csv(args.out / "handover.csv", index=False)
    print(f"\nwrote headline.csv, strata.csv, find_by_vote.csv, handover.csv -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
