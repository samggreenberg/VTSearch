#!/usr/bin/env python
"""The face review with and without the rotation retry, session for session (#5373).

Both arms ran the face review's sessions on one commit. ``ctl`` read the face cells as they were
built, ``rot`` cells rebuilt with :func:`vtscore.media.face.imagekit_bridge.process_upright`, which
re-detects a face under Face Quality 18 in the photo turned 90/180/270 and keeps a turned face that
gains 2 quality. Same people, starting photos and withheld halves; what differs is the vector (and
crop) of each face the retry turned.

A face *turned* is one whose Face Quality differs between the two cells. Per K x size x beta,
from ``align_compare_4777.session_set``: session mean over clicks 1-150, click 0,
after the check, final AP; each arm's mean, the paired difference and its SE over people, for
every person, for the people with a turned face, and (K = 1) for sessions that START from one.

    python face_rotation_5373_compare.py --ctl <analysis>,<baseline> --rot <analysis>,<baseline> \\
        --ctl-cells <cell dir> --rot-cells <cell dir> --runs <runs dir> --ctl-set <set> --out <dir>

Writes ``paired.csv`` and ``turned.csv`` (aggregates, no subject ids).
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from align_compare_4777 import BETAS, KS, SIZES, session_set

COLS = ("session_mean", "text_thr_fbeta", "thr_fbeta_final", "final_ap")


def turned_faces(ctl: Path, rot: Path, size: str) -> tuple[set, dict]:
    """Media ids whose quality moved, and each id's person, from the two face cells of *size*."""
    name = f"fhibe_faces_ik_{size}__face.pkl"
    a = pickle.loads((ctl / name).read_bytes())  # noqa: S301  # our own pile cell
    b = pickle.loads((rot / name).read_bytes())  # noqa: S301  # our own rebuilt cell
    q = "Face Quality"
    turned = {m for m in a.keys() & b.keys() if a[m]["custom_metadata"][q] != b[m]["custom_metadata"][q]}
    person = {m: r.get("category") or "" for m, r in a.items()}
    print(f"{size}: {len(a)} ctl faces, {len(b)} rot, {len(a.keys() ^ b.keys())} in one only, {len(turned)} turned")
    return turned, person


def starts(runs: Path, run_set: str, size: str, beta_tag: str) -> dict[str, int]:
    """K = 1: each person's starting face (the example pick) in one ctl session set."""
    out = {}
    for f in sorted((runs / f"{run_set}-k1-b{beta_tag}" / "results" / "cells").glob("task_*__picks.csv.gz")):
        p = pd.read_csv(f, usecols=["dataset", "category", "phase", "picked_id"])
        ex = p[(p["phase"] == "example") & (p["dataset"] == f"fhibe_faces_ik_{size}")]
        if len(ex):
            out[str(ex["category"].iloc[0])] = int(ex["picked_id"].iloc[0])
    return out


def pair(j: pd.DataFrame) -> dict:
    row = {"runs": len(j)}
    for col in COLS:
        a, p = j[f"{col}_arm"], j[f"{col}_plain"]
        d = (a - p).dropna()
        row[f"{col}_ctl"] = round(float(p.mean()), 4)
        row[f"{col}_rot"] = round(float(a.mean()), 4)
        row[f"d_{col}"] = round(float(d.mean()), 4)
        row[f"se_{col}"] = round(float(d.std(ddof=1) / np.sqrt(len(d))), 4) if len(d) > 1 else float("nan")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ctl", required=True)
    ap.add_argument("--rot", required=True)
    ap.add_argument("--ctl-cells", required=True, type=Path)
    ap.add_argument("--rot-cells", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--ctl-set", required=True, help="the control's run-dir prefix, <date>-<tag>-ik")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    ctl_a, ctl_b = (Path(p) for p in args.ctl.split(",", 1))
    rot_a, rot_b = (Path(p) for p in args.rot.split(",", 1))

    rows, turned_rows = [], []
    for size in SIZES:
        turned, person = turned_faces(args.ctl_cells, args.rot_cells, size)
        touched = {person[m] for m in turned if person.get(m)}
        turned_rows.append({"size": size, "faces_turned": len(turned), "people_touched": len(touched)})
        for k in KS:
            for beta, btag in BETAS.items():
                c = session_set(ctl_a, ctl_b, k, size, beta, "fhibe_faces_ik")
                r = session_set(rot_a, rot_b, k, size, beta, "fhibe_faces_ik")
                j = c.join(r, lsuffix="_plain", rsuffix="_arm", how="inner").reset_index()
                j["touched"] = j["category"].isin(touched)
                groups = {"all": j, "touched": j[j["touched"]], "untouched": j[~j["touched"]]}
                if k == 1:
                    st = starts(args.runs, args.ctl_set, size, btag)
                    groups["start turned"] = j[j["category"].map(lambda p: st.get(p) in turned)]
                for name, g in groups.items():
                    rows.append({"K": k, "size": size, "beta": beta, "people": name} | pair(g))
    args.out.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    try:
        pd.DataFrame(rows).to_csv(args.out / "paired.csv", index=False)
        pd.DataFrame(turned_rows).to_csv(args.out / "turned.csv", index=False)
    finally:
        os.umask(old)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        show = ["K", "size", "beta", "people", "runs"] + [f"{p}{c}" for c in COLS for p in ("d_", "se_")]
        print(pd.DataFrame(turned_rows).to_string(index=False))
        print(pd.DataFrame(rows)[show].to_string(index=False))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
