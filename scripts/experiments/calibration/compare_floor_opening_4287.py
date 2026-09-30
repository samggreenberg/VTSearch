#!/usr/bin/env python3
"""Compare the #4267 spot check across two openings' rank frames (#4287).

``analyze_floor_candidate_4267.py`` prices the check on one set of rank frames.
This reads its ``best_attempt.csv`` for two sets (today's g3 opening and the
opening #4282 shipped) and adds what a cross-arm comparison needs:

* **absolute recall**: ``recall_share`` divides by each arm's own oracle, which a
  better ranking also raises, so the ratio alone can hide a gain;
* **paired, deterministic** per-session quantities with a standard error: the
  precision of the unchecked starting candidate (the top K of the ranking) and
  the oracle's recall at X.  Sessions pair on (category, seed, t); the two
  #4222 arms share cells and seeds, and starve the same sessions.

    python compare_floor_opening_4287.py --old DIR --new DIR --old-an DIR --new-an DIR --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_floor_candidate_4267 as F
import analyze_random_verification as A

WORLDS = {w: f for w, f in A.WORLDS.items() if w != "5%"}  # the #4222 grid has no 5% pool
FLOORS = (0.10, 0.25, 0.50, 0.75, 0.90)
SLICES = ("all", "t=150", "band=small")
KEY = ["category", "seed", "t"]


def per_frame(frames_dir: Path, world: str, x: float) -> pd.DataFrame:
    fr = A.load_frames(frames_dir, {world: WORLDS[world]})[world]
    k0 = F.schedule_for(x)[0]
    ok = A.oracle_k(fr, x)
    n_corpus = fr.meta["n_corpus"].to_numpy()
    k0s = np.minimum(k0, n_corpus)
    hits0 = np.array([(r < k).sum() for r, k in zip(fr.ranks, k0s)])
    orec = np.array([(r < k).sum() for r, k in zip(fr.ranks, ok)]) / fr.meta["n_pos"].to_numpy()
    out = fr.meta[KEY].copy()
    out["unchecked_prec"] = hits0 / k0s
    out["oracle_recall"] = orec
    out["n_vote_pos"] = fr.meta["n_vote_pos"].to_numpy()
    return out


def paired(old: pd.DataFrame, new: pd.DataFrame, col: str) -> tuple[float, float, float, float, int]:
    m = old.merge(new, on=KEY, suffixes=("_o", "_n"))
    d = m[f"{col}_n"] - m[f"{col}_o"]
    return m[f"{col}_o"].mean(), m[f"{col}_n"].mean(), d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    for a in ("old", "new", "old-an", "new-an", "out"):
        ap.add_argument(f"--{a}", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ba = {arm: pd.read_csv(p / "best_attempt.csv") for arm, p in (("old", args.old_an), ("new", args.new_an))}
    rows = []
    for world in WORLDS:
        for x in FLOORS:
            pf = {arm: per_frame(d, world, x) for arm, d in (("old", args.old), ("new", args.new))}
            for sl in SLICES:
                sel = {arm: A_slice(pf[arm], sl) for arm in pf}
                row = {"world": world, "X": x, "slice": sl}
                for arm in ("old", "new"):
                    b = ba[arm]
                    r = b[(b.world == world) & np.isclose(b.X, x) & (b.slice == sl)].iloc[0]
                    orec = pf[arm][sel[arm]]["oracle_recall"].mean()
                    row |= {
                        f"confirmed_{arm}": r.confirmed,
                        f"votes_{arm}": r.votes_mean,
                        f"returned_{arm}": r.returned,
                        f"precision_{arm}": r.precision,
                        f"reached_x_{arm}": r.reached_x,
                        f"recall_{arm}": r.recall_share * orec,
                        f"coverage_{arm}": r.coverage,
                        f"range_width_{arm}": r.range_width,
                    }
                for col in ("unchecked_prec", "oracle_recall", "n_vote_pos"):
                    o, n, d, se, npair = paired(pf["old"][sel["old"]], pf["new"][sel["new"]], col)
                    row |= {f"{col}_old": o, f"{col}_new": n, f"d_{col}": d, f"se_d_{col}": se}
                row["n_paired"] = npair
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "compare.csv", index=False, float_format="%.4g")
    print(f"wrote {args.out / 'compare.csv'} ({len(df)} rows)")


def A_slice(df: pd.DataFrame, sl: str) -> pd.Series:
    if sl == "all":
        return pd.Series(True, index=df.index)
    if sl.startswith("t="):
        return df["t"] == int(sl[2:])
    return df["category"].str.endswith("@" + sl.split("=")[1])


if __name__ == "__main__":
    main()
