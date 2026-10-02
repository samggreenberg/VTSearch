#!/usr/bin/env python
"""#4427: how far the balance's line sits from the best cut, over clicks, off the pricing arms' rank frames.

Each rank frame (``task_*__rankframes.csv``) records, at a click *t*, the ranks of the test corpus's
positives in the model's ranking (``test_pos_ranks``), their number (``n_test_pos``), the corpus size and
the count the line would keep on that corpus at each beta (``test_line_k_b05 / b1 / b2``: the mixture's
F-beta argmax under the balance's cap, 32 at beta <= 1 and 128 at 2).  The best cut's size is the argmax
of F-beta over every k.  Per arm, beta and click this prints the mean kept size against the best size,
how often the cap binds (kept == cap), how often the best cut is deeper than the cap, and the share of the
best F-beta at the kept size and at the cap x 2 / x 4 (what a higher cap could buy if the mixture's argmax
were there).

    python ceiling_gap.py --arms b1-xctl b1-x0.5 --betas 1 --out ceiling_gap.md
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/expscratch/sgreenberg/p-aware-acq-4409")
CAP = {0.5: 32, 1.0: 32, 2.0: 128}
TAG = {0.5: "b05", 1.0: "b1", 2.0: "b2"}
STEPS = (10, 25, 50, 75, 100, 125, 150)


def fbeta_curve(ranks: np.ndarray, n_pos: int, n: int, beta: float) -> np.ndarray:
    """F-beta at every k = 1..n for positives at 0-based *ranks* (sorted)."""
    k = np.arange(1, n + 1)
    tp = np.searchsorted(ranks, k, side="left")  # positives with rank < k
    b2 = beta * beta
    return (1.0 + b2) * tp / (b2 * n_pos + k)


def frame_rows(arm: str, betas: list[float]) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(ROOT / arm / "results" / "cells" / "task_*__rankframes.csv"))):
        d = pd.read_csv(f, dtype={"test_pos_ranks": str, "pool_pos_ranks": str}, low_memory=False)
        d = d[(d["kind"] == "step") & d["t"].isin(STEPS)]
        for r in d.itertuples(index=False):
            n_pos = int(r.n_test_pos)
            if n_pos <= 0 or not isinstance(r.test_pos_ranks, str) or not r.test_pos_ranks.strip():
                continue
            ranks = np.sort(np.fromiter((int(x) for x in r.test_pos_ranks.split()), dtype=np.int64))
            n = int(r.n_test)
            for beta in betas:
                curve = fbeta_curve(ranks, n_pos, n, beta)
                best_k = int(np.argmax(curve)) + 1
                best = float(curve[best_k - 1])
                kept = int(getattr(r, f"test_line_k_{TAG[beta]}"))
                cap = CAP[beta]
                at = lambda kk: float(curve[min(max(kk, 1), n) - 1])  # noqa: E731
                rows.append(
                    {
                        "arm": arm,
                        "beta": beta,
                        "t": int(r.t),
                        "category": r.category,
                        "seed": int(r.seed),
                        "kept": kept,
                        "best_k": best_k,
                        "cap": cap,
                        "cap_binds": int(kept >= cap),
                        "best_past_cap": int(best_k > cap),
                        "best_past_2cap": int(best_k > 2 * cap),
                        "share_kept": at(kept) / best if best > 0 else np.nan,
                        "share_cap": at(cap) / best if best > 0 else np.nan,
                        "share_2cap": at(2 * cap) / best if best > 0 else np.nan,
                        "share_4cap": at(4 * cap) / best if best > 0 else np.nan,
                        "share_best_k_capped_2": at(min(best_k, 2 * cap)) / best if best > 0 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    out = ["# The line against the best cut, over clicks (#4427; off the #4409 pricing frames)\n"]
    for (arm, beta), g in df.groupby(["arm", "beta"]):
        out.append(
            f"## {arm}, beta {beta:g} (cap {CAP[beta]}): {g['category'].nunique()} categories x {g['seed'].nunique()} seeds\n"
        )
        out.append(
            "| click | kept | best k (mean / median) | cap binds | best > cap | best > 2 cap | share @kept | @cap | @2 cap | @4 cap | @min(best, 2 cap) |"
        )
        out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for t, h in g.groupby("t"):
            out.append(
                f"| {t} | {h['kept'].mean():.1f} | {h['best_k'].mean():.1f} / {h['best_k'].median():.0f} | "
                f"{h['cap_binds'].mean():.2f} | {h['best_past_cap'].mean():.2f} | {h['best_past_2cap'].mean():.2f} | "
                f"{h['share_kept'].mean():.3f} | {h['share_cap'].mean():.3f} | {h['share_2cap'].mean():.3f} | "
                f"{h['share_4cap'].mean():.3f} | {h['share_best_k_capped_2'].mean():.3f} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["b0.5-xctl", "b0.5-x0.5", "b1-xctl", "b1-x0.5", "b2-xctl", "b2-x0.5"])
    ap.add_argument("--betas", nargs="+", type=float, default=None, help="default: the arm's own beta")
    ap.add_argument("--out", type=Path, default=ROOT / "ceiling_gap.md")
    args = ap.parse_args()
    frames = []
    for arm in args.arms:
        own = float(arm.split("-")[0][1:])
        betas = args.betas or [own]
        frames.append(frame_rows(arm, betas))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    text = report(df)
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
