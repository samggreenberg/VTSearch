#!/usr/bin/env python
"""Does turning the face crop upright help FaceNet find the person from one photo? (#4777)

The screen before the sessions. Every subject crop of a person takes a turn as the one starting
photo, and the example sort (cosine to that crop, as the app ranks one example) is scored by the
AP of the person's other crops among every other media in the cell: everyone else's faces and the
strays labelled for nobody. That is click 0 of a one-photo session for every photo a session could
start from, not only the one each session drew.

Arms pair by media id: the aligned cells (``fhibe_faces_roll_*``, ``fhibe_faces_sim_*``) hold the
same detections, ids and labels as ``fhibe_faces_*`` (the loader's ``align``), so each query is the
same photo in every arm. The script refuses arms whose ids or labels differ.

Read per arm and size: the mean AP over queries, the share of queries that fail (AP below
``--fail``, the review's bar for a failed session), the paired difference from the plain arm with
its SE over people (queries of one person are not independent), and the same by the starting
photo's roll (``align_landmarks_4777.py``'s ``subjects_1024.csv``: the angle the 5-point fit
removes, read at 1024 px for both sizes, since it is the same photo).

    python align_screen_4777.py --identities <ids.txt> --rolls <dir>/subjects_1024.csv --out <dir> \\
        [--fail 0.1] [--sizes 1024,640] [--arms faces,faces_roll,faces_sim] \\
        [--starts 1024=<one-photo run's cells dir> --starts 640=<...>]

``--starts`` marks the photo each one-photo session of a review started from (its vote-1
``example`` pick), so the screen also reads the review's own starts: ``start_*`` columns, over
those queries only.

``--identities`` restricts the queries to the bench's people (the review's 562); the pool is the
whole cell either way. Writes ``screen.csv``, ``screen_by_roll.csv`` (aggregates) and
``queries.csv`` (one row per query and arm, with subject ids: owner-only, never committed).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))
import common  # noqa: E402

common.setup_env()

import numpy as np  # noqa: E402

ROLL_BANDS = ((0, 10), (10, 20), (20, 45), (45, 90), (90, 181))


def load_cell(ds: str) -> tuple[list[int], np.ndarray, list[str]]:
    """``(ids, unit vectors, category)`` of a face cell, sorted by id."""
    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    from _cells_io import load_medias  # noqa: PLC0415

    medias = load_medias(EMBEDDINGS_DIR / f"{ds}__face.pkl", repair=True)
    ids = sorted(medias)
    vecs = np.stack([np.asarray(media_embedding(medias[i]), dtype=np.float64) for i in ids])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    cats = [medias[i].get("category") or "" for i in ids]
    return ids, vecs, cats


def query_aps(vecs: np.ndarray, cats: list[str], queries: list[int], chunk: int = 512) -> np.ndarray:
    """AP of each query row's own category among every other row, by cosine."""
    cat_arr = np.asarray(cats)
    out = np.empty(len(queries))
    for s in range(0, len(queries), chunk):
        q = np.asarray(queries[s : s + chunk])
        sims = vecs[q] @ vecs.T
        sims[np.arange(len(q)), q] = -np.inf  # the query itself is not in its own ranking
        order = np.argsort(-sims, axis=1, kind="stable")
        for r, qi in enumerate(q):
            rel = (cat_arr[order[r]] == cat_arr[qi]) & (order[r] != qi)
            hits = np.cumsum(rel)
            npos = int(rel.sum())
            out[s + r] = float((hits[rel] / (np.flatnonzero(rel) + 1)).sum() / npos) if npos else np.nan
    return out


def session_starts(cells: Path) -> set[int]:
    """The photo each session in *cells* started from: its vote-1 ``example`` pick."""
    import pandas as pd  # noqa: PLC0415

    starts = set()
    for f in sorted(cells.glob("task_*__picks.csv")):
        df = pd.read_csv(f, usecols=["t", "phase", "picked_id"])
        starts.update(int(v) for v in df.loc[(df["t"] == 1) & (df["phase"] == "example"), "picked_id"])
    return starts


def person_se(diff: np.ndarray, people: np.ndarray) -> float:
    """SE of the mean of *diff* with people as the unit (cluster means, then their SE)."""
    means = np.asarray([diff[people == p].mean() for p in np.unique(people)])
    return float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--identities", required=True, help="the bench's identity file (one subject id per line)")
    ap.add_argument("--rolls", required=True, help="subjects_1024.csv from align_landmarks_4777.py template")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fail", type=float, default=0.1)
    ap.add_argument("--sizes", default="1024,640")
    ap.add_argument("--arms", default="faces,faces_roll,faces_sim")
    ap.add_argument("--starts", action="append", default=[], help="size=<cells dir of a one-photo run>")
    args = ap.parse_args()
    starts = {int(k): session_starts(Path(v)) for k, v in (spec.split("=", 1) for spec in args.starts)}

    bench = {
        ln.strip() for ln in Path(args.identities).read_text().splitlines() if ln.strip() and not ln.startswith("#")
    }
    roll = {int(r["pid"]): abs(float(r["roll_deg"])) for r in csv.DictReader(open(args.rolls))}
    arms = [a for a in args.arms.split(",") if a]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    summary, by_roll, per_query = [], [], []
    try:
        for size in [int(s) for s in args.sizes.split(",") if s]:
            cells = {arm: load_cell(f"fhibe_{arm}_{size}") for arm in arms}
            ids0, _, cats0 = cells[arms[0]]
            for arm, (ids, _, cats) in cells.items():
                if ids != ids0 or cats != cats0:
                    raise SystemExit(f"{arm} at {size} does not pair with {arms[0]}: ids or labels differ")
            count = defaultdict(int)
            for c in cats0:
                count[c] += 1
            queries = [i for i, c in enumerate(cats0) if c in bench and count[c] >= 2]
            people = np.asarray([cats0[i] for i in queries])
            rolls = np.asarray([roll.get(ids0[i], np.nan) for i in queries])
            is_start = np.asarray([ids0[i] in starts.get(size, ()) for i in queries])
            aps = {arm: query_aps(cells[arm][1], cats0, queries) for arm in arms}
            common.log(f"{size}: {len(queries):,} queries over {len(set(people)):,} people, pool {len(ids0):,}")
            base = aps[arms[0]]
            for arm in arms:
                a = aps[arm]
                d = a - base
                summary.append(
                    {
                        "size": size,
                        "arm": arm,
                        "queries": len(a),
                        "people": len(set(people)),
                        "mean_ap": round(float(a.mean()), 4),
                        "person_mean_ap": round(float(np.mean([a[people == p].mean() for p in np.unique(people)])), 4),
                        "fail_share": round(float((a < args.fail).mean()), 4),
                        "fails": int((a < args.fail).sum()),
                        "d_mean_ap": round(float(d.mean()), 4),
                        "d_se": round(person_se(d, people), 4) if arm != arms[0] else 0.0,
                        "fails_fixed": int(((base < args.fail) & (a >= args.fail)).sum()),
                        "fails_made": int(((base >= args.fail) & (a < args.fail)).sum()),
                    }
                )
                if is_start.any():
                    m = is_start
                    summary[-1] |= {
                        "start_queries": int(m.sum()),
                        "start_mean_ap": round(float(a[m].mean()), 4),
                        "start_fails": int((a[m] < args.fail).sum()),
                        "start_d_mean_ap": round(float(d[m].mean()), 4),
                        "start_d_se": round(float(d[m].std(ddof=1) / np.sqrt(m.sum())), 4) if arm != arms[0] else 0.0,
                    }
                for lo, hi in ROLL_BANDS:
                    m = (rolls >= lo) & (rolls < hi)
                    if not m.any():
                        continue
                    by_roll.append(
                        {
                            "size": size,
                            "arm": arm,
                            "roll_band": f"{lo}-{hi}",
                            "queries": int(m.sum()),
                            "mean_ap": round(float(a[m].mean()), 4),
                            "fail_share": round(float((a[m] < args.fail).mean()), 4),
                            "d_mean_ap": round(float(d[m].mean()), 4),
                            "d_se": round(person_se(d[m], people[m]), 4) if arm != arms[0] and m.sum() > 2 else "",
                        }
                    )
            for k, qi in enumerate(queries):
                per_query.append(
                    {
                        "size": size,
                        "pid": ids0[qi],
                        "subject": cats0[qi],
                        "roll_deg": rolls[k],
                        "start": bool(is_start[k]),
                    }
                    | {f"ap_{arm}": round(float(aps[arm][k]), 5) for arm in arms}
                )
        for name, rows in (("screen.csv", summary), ("screen_by_roll.csv", by_roll), ("queries.csv", per_query)):
            with (out / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    finally:
        os.umask(old)
    for r in summary:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
