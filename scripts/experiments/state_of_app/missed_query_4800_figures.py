#!/usr/bin/env python
"""#4800's figure: on the Region path, the region centroid against the typed query once the query has missed.

Reads ``curves.csv`` from ``missed_query_4800.py`` and writes ``fig_missed_query.png``: one row per balance (beta
1/4, 1, 4). The left column is F-beta at each vote over the runs whose typed query has missed there (the Bads quota
met before 3 Goods): the typed query's own set, which Find gives today, and the centroid built from the Goods' own
boxes, on the same runs, with the share of runs in that state as grey bars. The right column is each design minus
today over every run (zero wherever the designs agree): the centroid once the query has missed, and the better of the
two sets at every vote under the quota (the headroom, which no rule reads).

Usage::

    python missed_query_4800_figures.py --curves <analysis>/curves.csv --out <dir> [--corpus labelled] [--max-t 20]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
from figures_centroid_count_4759 import BETAS, GRID, INK, INK2, SURFACE, _style  # noqa: E402

CENTROID, QUERY, ORACLE = "#2a78d6", "#eb6834", "#9aa3ad"


def figure(curves: pd.DataFrame, corpus: str, out: Path, max_t: int, name: str) -> Path:
    d = curves[(curves["corpus"] == corpus) & (curves["t"] <= max_t)]
    fig, axes = plt.subplots(len(BETAS), 2, figsize=(9.2, 2.7 * len(BETAS)), squeeze=False, sharex=True)
    for i, (beta, bname) in enumerate(BETAS.items()):
        c = d[d["beta"] == beta].sort_values("t")
        ax, bx = axes[i]
        share = ax.twinx()
        share.bar(c["t"], c["share_missed"], color=GRID, width=0.8, zorder=0)
        share.set_ylim(0, 1)
        share.set_yticks([0, 0.5, 1])
        share.tick_params(colors=INK2, labelsize=7)
        share.spines["right"].set_visible(True)
        share.spines["right"].set_color(INK2)
        if i == 1:
            share.set_ylabel("share of runs whose query has missed", fontsize=8)
        ax.set_zorder(share.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(
            c["t"], c["query_fbeta_missed"], color=QUERY, lw=2.2, marker="o", ms=3, label="typed query's set (today)"
        )
        ax.plot(c["t"], c["centroid_fbeta_missed"], color=CENTROID, lw=2.2, marker="o", ms=3, label="region centroid")
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{bname}: F-beta", fontsize=9)
        bx.plot(c["t"], c["d_oracle"], color=ORACLE, lw=1.4, ls="--", label="the better set at every vote (headroom)")
        bx.plot(c["t"], c["d_missed"], color=CENTROID, lw=2.2, label="the centroid once the query has missed")
        bx.axhline(0, color=INK2, lw=0.8)
        bx.grid(axis="y", color=GRID, lw=0.8)
        bx.set_ylabel(f"{bname}: minus today", fontsize=9)
        if i == 0:
            ax.set_title("On the runs whose typed query has missed", fontsize=10, color=INK, loc="left")
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
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus", default="labelled", help="which centroid line: labelled (the app's Find) or withheld")
    ap.add_argument("--max-t", type=int, default=20)
    ap.add_argument("--name", default="fig_missed_query.png")
    args = ap.parse_args(argv)
    _style()
    print(figure(pd.read_csv(args.curves), args.corpus, args.out, args.max_t, args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
