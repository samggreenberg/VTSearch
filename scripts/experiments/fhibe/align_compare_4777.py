#!/usr/bin/env python
"""Aligned face crops against today's, session for session (#4777).

Each aligned arm (``sota_face.sh`` under ``SOTA_FACE_ARM``) ran the face review's 6,744 sessions on
crops turned upright: the same people, splits, starting photos and seeds as the plain arm, since
the cells share ids and labels. So every session pairs with one plain session by (person, size,
K, beta), and the arm's effect is read as the mean paired difference, its SE over people.

Per K x size x beta, for each arm against the plain one:

* the owner's single number, the **session mean**: each run's objective (F-beta of the withheld set
  at the app's line, at the run's own beta) averaged over clicks 1-150, filled as ``by_click.py``
  fills it (the example sort until the app shows a detector, then the last value carried);
* click 0 (the example sort's own objective and AP), the objective after the check, final AP;
* **failed sessions**: final AP below ``--fail`` while the person's own full-label ceiling reaches
  ``--fail-ceiling`` (the review's bar: AP < 0.1 against a ceiling of 0.64-1.00; a person even the
  ceiling cannot find is the embedder's limit, not a failed session), with how many the arm fixed
  and how many it made.

    python align_compare_4777.py --plain <analysis> --plain-baseline <baseline dir> \\
        --arm roll=<analysis>,<baseline dir> [--arm sim=...] --out <dir> [--watch <prefix,...>]

``--plain`` and each arm's analysis dir are ``sota_face_analyze.sh`` outputs
(``k<K>-b<tag>-<size>/{cells,curves}.csv``); the baseline dirs hold ``example_baseline_k<K>.csv``.
``--plain-tag`` names the control's own arm when it is not the plain crop (#4788 pairs its
``side`` arm against #4777's ``roll``: ``--plain-tag roll``).
#5321 pairs its ``ik`` arm (the imagekit provider's crops and vectors) against ``roll`` the same way.
``--watch`` names subject-id prefixes (the review's 12 failures) to list run by run. Writes
``compare.csv`` (aggregates) and ``failures.csv`` (runs that failed in either arm, by subject id:
owner-only, never committed).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "state_of_app"))
from by_click import RUN, filled_runs, typed_query  # noqa: E402

KS = (1, 4)
SIZES = ("1024", "640")
BETAS = {0.25: "025", 1.0: "1", 4.0: "4"}
HORIZON = 150


def session_set(analysis: Path, baseline: Path, k: int, size: str, beta: float, faces: str) -> pd.DataFrame:
    """One session set's runs: session mean, click 0, after the check, final and ceiling AP."""
    d = analysis / f"k{k}-b{BETAS[beta]}-{size}"
    cells = pd.read_csv(d / "cells.csv")
    curves = pd.read_csv(d / "curves.csv")
    base = pd.read_csv(baseline / f"example_baseline_k{k}.csv")
    base = base[base["dataset"] == f"{faces}_{size}"]
    # The imagekit arm's sessions are filed under its own embedder (#5321); every other arm is FaceNet's.
    embedder = "imagekit_face" if faces.endswith("_ik") else "face"
    filled, _ = filled_runs(curves, cells, typed_query(base, embedder, beta))
    clicks = [t for t in filled.columns if 1 <= int(t) <= HORIZON]
    out = cells.set_index(RUN)[["text_thr_fbeta", "text_ap", "thr_fbeta_final", "final_ap", "ceiling_ap"]].copy()
    out["session_mean"] = filled[clicks].mean(axis=1).reindex(out.index)
    return out


def failed(runs: pd.DataFrame, fail: float, ceiling: float, side: str = "") -> pd.Series:
    """Runs that end below *fail* AP although their ceiling reaches *ceiling*."""
    return (runs[f"final_ap{side}"] < fail) & (runs[f"ceiling_ap{side}"] >= ceiling)


def paired(plain: pd.DataFrame, arm: pd.DataFrame, fail: float, ceiling: float) -> dict:
    j = plain.join(arm, lsuffix="_plain", rsuffix="_arm", how="inner")
    row = {"runs": len(j), "unpaired": len(plain) + len(arm) - 2 * len(j)}
    for col in ("session_mean", "text_thr_fbeta", "text_ap", "thr_fbeta_final", "final_ap"):
        a, p = j[f"{col}_arm"], j[f"{col}_plain"]
        d = (a - p).dropna()
        row[f"{col}_plain"] = round(float(p.mean()), 4)
        row[f"{col}_arm"] = round(float(a.mean()), 4)
        row[f"d_{col}"] = round(float(d.mean()), 4)
        row[f"se_{col}"] = round(float(d.std(ddof=1) / np.sqrt(len(d))), 4) if len(d) > 1 else float("nan")
    fp, fa = failed(j, fail, ceiling, "_plain"), failed(j, fail, ceiling, "_arm")
    row.update(
        {
            "fails_plain": int(fp.sum()),
            "fails_arm": int(fa.sum()),
            "fixed": int((fp & ~fa).sum()),
            "made": int((~fp & fa).sum()),
            "ceiling_ap": round(float(j["ceiling_ap_plain"].mean()), 4),
        }
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plain", required=True, type=Path)
    ap.add_argument("--plain-baseline", required=True, type=Path)
    ap.add_argument("--plain-tag", default="", help="the control's arm tag (fhibe_faces_<tag>_*), empty for plain")
    ap.add_argument("--arm", action="append", required=True, help="tag=<analysis dir>,<baseline dir>")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fail", type=float, default=0.1)
    ap.add_argument("--fail-ceiling", type=float, default=0.5)
    ap.add_argument("--watch", default="", help="comma-separated subject-id prefixes to list run by run")
    args = ap.parse_args()
    arms = {}
    for spec in args.arm:
        tag, dirs = spec.split("=", 1)
        a, b = dirs.split(",", 1)
        arms[tag] = (Path(a), Path(b))
    watch = [w for w in args.watch.split(",") if w]
    rows, failures = [], []
    for k in KS:
        for size in SIZES:
            for beta in BETAS:
                faces = f"fhibe_faces_{args.plain_tag}" if args.plain_tag else "fhibe_faces"
                plain = session_set(args.plain, args.plain_baseline, k, size, beta, faces)
                for tag, (a, b) in arms.items():
                    arm = session_set(a, b, k, size, beta, f"fhibe_faces_{tag}")
                    rows.append(
                        {"K": k, "size": size, "beta": beta, "arm": tag}
                        | paired(plain, arm, args.fail, args.fail_ceiling)
                    )
                    j = plain.join(arm, lsuffix="_plain", rsuffix="_arm", how="inner").reset_index()
                    keep = failed(j, args.fail, args.fail_ceiling, "_plain") | failed(
                        j, args.fail, args.fail_ceiling, "_arm"
                    )
                    if watch:
                        keep |= j["category"].astype(str).str[:8].isin([w[:8] for w in watch])
                    for r in j[keep].to_dict("records"):
                        failures.append(
                            {"K": k, "size": size, "beta": beta, "arm": tag, "subject": r["category"]}
                            | {
                                c: round(float(r[c]), 4)
                                for c in (
                                    "text_ap_plain",
                                    "text_ap_arm",
                                    "final_ap_plain",
                                    "final_ap_arm",
                                    "ceiling_ap_plain",
                                    "session_mean_plain",
                                    "session_mean_arm",
                                )
                            }
                        )
    args.out.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    try:
        pd.DataFrame(rows).to_csv(args.out / "compare.csv", index=False)
        pd.DataFrame(failures).to_csv(args.out / "failures.csv", index=False)
    finally:
        os.umask(old)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = [
            "K",
            "size",
            "beta",
            "arm",
            "runs",
            "session_mean_plain",
            "d_session_mean",
            "se_session_mean",
            "d_text_ap",
            "d_final_ap",
            "fails_plain",
            "fails_arm",
            "fixed",
            "made",
        ]
        print(pd.DataFrame(rows)[cols].to_string(index=False))
    print(f"-> {args.out / 'compare.csv'}, {args.out / 'failures.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
