#!/usr/bin/env python
"""#4758's figure: what Find returns under the quota in a typed-query opening, the Goods' centroid or the typed query.

Reads ``curves.csv`` from ``analyze_find_typed_query_4758.py`` and writes ``fig_find_typed_query.png``: one row per
balance (beta 1/4, 1, 4). The left column is F-beta at each vote over the runs whose Find gives the centroid there:
the centroid's set and the typed query's own set on the same runs, with the share of runs under the quota as grey
bars. The right column is each design minus today over every run (zero wherever the designs agree): the typed query
under the quota, and also before the first Good.

Usage::

    python figures_find_typed_query_4758.py --curves <analysis>/curves.csv --out <dir> [--corpus labelled] [--max-t 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_centroid_count_4759 import BETAS, GRID, INK, INK2, SURFACE, _style  # noqa: E402

CENTROID, QUERY, OPEN = "#2a78d6", "#eb6834", "#f2a541"


def figure(curves: pd.DataFrame, corpus: str, out: Path, max_t: int, name: str) -> Path:
    d = curves[(curves["corpus"] == corpus) & (curves["t"] <= max_t)]
    fig, axes = plt.subplots(len(BETAS), 2, figsize=(9.2, 2.7 * len(BETAS)), squeeze=False, sharex=True)
    for i, (beta, bname) in enumerate(BETAS.items()):
        c = d[d["beta"] == beta].sort_values("t")
        ax, bx = axes[i]
        share = ax.twinx()
        share.bar(c["t"], c["share_under_quota"], color=GRID, width=0.8, zorder=0)
        share.set_ylim(0, 1)
        share.set_yticks([0, 0.5, 1])
        share.tick_params(colors=INK2, labelsize=7)
        share.spines["right"].set_visible(True)
        share.spines["right"].set_color(INK2)
        if i == 1:
            share.set_ylabel("share of runs under the quota", fontsize=8)
        ax.set_zorder(share.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(c["t"], c["query_fbeta_before_quota"], color=QUERY, lw=2.2, label="typed query's own set")
        ax.plot(c["t"], c["centroid_fbeta_under_quota"], color=CENTROID, lw=2.2, label="Goods' centroid (today)")
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{bname}: F-beta", fontsize=9)
        bx.plot(c["t"], c["d_query_open"], color=OPEN, lw=1.4, label="typed query, also before the first Good")
        bx.plot(c["t"], c["d_query_quota"], color=QUERY, lw=2.2, label="typed query under the quota")
        bx.axhline(0, color=INK2, lw=0.8)
        bx.grid(axis="y", color=GRID, lw=0.8)
        bx.set_ylabel(f"{bname}: minus today", fontsize=9)
        if i == 0:
            ax.set_title("On the votes Find gives the centroid", fontsize=10, color=INK, loc="left")
            bx.set_title("Every run: what Find returns, minus today", fontsize=10, color=INK, loc="left")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
            bx.legend(frameon=False, fontsize=8, loc="upper right")
        for a in (ax, bx):
            a.set_xlim(0.5, max_t + 0.5)
            a.xaxis.set_major_locator(MaxNLocator(integer=True))
            if i == len(BETAS) - 1:
                a.set_xlabel("vote")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus", default="labelled", help="which centroid line: labelled (the app's Find) or withheld")
    ap.add_argument("--max-t", type=int, default=20)
    ap.add_argument("--name", default="fig_find_typed_query.png")
    args = ap.parse_args(argv)
    _style()
    print(figure(pd.read_csv(args.curves), args.corpus, args.out, args.max_t, args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
