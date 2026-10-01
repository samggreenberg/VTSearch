#!/usr/bin/env python3
"""Figures for the F-beta line pricing (#4411), from ``analyze_fbeta_line_4411.py``'s ``summary.csv``.

* ``fb_share_by_size.png`` - one panel per world x beta: each rule's F-beta as
  a share of the best F-beta of any cut, against corpus size, with the floor's
  walk and no-vote line beside the F-beta rules.
* ``count_vs_oracle.png`` - the count each rule returns against the oracle's,
  per world, at beta = 1: where a rule over- or under-returns.

    python figure_fbeta_line_4411.py --run OUT --out FIGDIR [--t 150]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SIZES = ("32", "320", "3200", "full", "x10")
WORLDS = ("0.1%", "0.44%", "5%")
RULES = ("fb-gmm", "fb-walk", "fb-bands", "floor-walk", "floor-novote")
COLORS = {
    "fb-gmm": "#1f77b4",
    "fb-walk": "#2ca02c",
    "fb-bands": "#17becf",
    "floor-walk": "#d62728",
    "floor-novote": "#ff7f0e",
}
STYLE = {"floor-walk": "--", "floor-novote": "--"}


def share_by_size(s: pd.DataFrame, out: Path, t: int) -> None:
    betas = sorted(s["beta"].unique())
    fig, axes = plt.subplots(len(betas), len(WORLDS), figsize=(4.3 * len(WORLDS), 3.3 * len(betas)), sharey=True)
    for i, beta in enumerate(betas):
        for j, world in enumerate(WORLDS):
            ax = axes[i, j]
            g = s[(s["t"] == t) & (s["beta"] == beta) & (s["world"] == world)]
            for rule in RULES:
                r = g[g["rule"] == rule].set_index("size").reindex(SIZES)
                ax.errorbar(
                    range(len(SIZES)), r["fb_share"], yerr=r["se_fb_share"], label=rule, color=COLORS[rule],
                    ls=STYLE.get(rule, "-"), marker="o", ms=3, capsize=2,
                )  # fmt: skip
            ax.set_xticks(range(len(SIZES)), SIZES)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
            ax.set_title(f"{world}, beta = {beta:g}", fontsize=10)
            if j == 0:
                ax.set_ylabel("F-beta / best F-beta")
            if i == len(betas) - 1:
                ax.set_xlabel("corpus size")
    axes[0, -1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fb_share_by_size.png", dpi=110)
    plt.close(fig)


def count_vs_oracle(s: pd.DataFrame, out: Path, t: int, beta: float = 1.0) -> None:
    fig, axes = plt.subplots(1, len(WORLDS), figsize=(4.3 * len(WORLDS), 3.6))
    for j, world in enumerate(WORLDS):
        ax = axes[j]
        g = s[(s["t"] == t) & (s["beta"] == beta) & (s["world"] == world)]
        for rule in RULES:
            r = g[g["rule"] == rule].set_index("size").reindex(SIZES)
            ax.plot(range(len(SIZES)), r["k"] / r["oracle_k"].clip(lower=1), label=rule, color=COLORS[rule],
                    ls=STYLE.get(rule, "-"), marker="o", ms=3)  # fmt: skip
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(SIZES)), SIZES)
        ax.grid(alpha=0.3)
        ax.set_title(f"{world}, beta = {beta:g}: count returned / oracle's", fontsize=10)
        ax.set_xlabel("corpus size")
    axes[0].set_ylabel("returned / oracle count (log)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "count_vs_oracle.png", dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t", type=int, default=150)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(args.run / "summary.csv", dtype={"size": str})
    share_by_size(s, args.out, args.t)
    count_vs_oracle(s, args.out, args.t)
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
