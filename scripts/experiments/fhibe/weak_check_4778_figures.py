#!/usr/bin/env python
"""The weak-check study's figures (#4778), from ``weak_check_4778.py``'s ``curves.csv``.

* ``fig_objective_by_vote.png`` - the objective each session set shows at every vote (F-beta of the
  withheld set at the app's line, filled as ``by_click.py`` fills it), mean over the 562 people: today's
  app solid, the end-of-run check only dashed (top row); and the share of today's sessions casting a
  prompted check's vote at that vote (bottom row, its own axis rather than a second one). One column per
  starting-photo count; colour is the preset.
* ``fig_session_diff.png`` - the paired difference, check off minus today, at every vote, with a +-1 SE
  band over people. One panel per starting-photo count.

Colours are the validated categorical slots 1-3 in fixed order (slides/STYLE.md, as
``sota_face_figures.py``), every line labelled at its end.

    python weak_check_4778_figures.py --compare <weak_check_4778.py out dir> --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sota_face_figures import BETA, GRID, INK2, KS, _end_labels, _style, photos  # noqa: E402


def fig_objective(curves: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 6.0), sharex=True, sharey="row", gridspec_kw={"height_ratios": [3, 1.2]}
    )
    for col, k in enumerate(KS):
        top, bottom = axes[0, col], axes[1, col]
        ends, share_ends = [], []
        for beta, (color, name) in BETA.items():
            c = curves[(curves["K"] == k) & (curves["beta"].astype(float) == beta)].sort_values("vote")
            if c.empty:
                continue
            top.plot(c["vote"], c["control"], color=color, lw=1.6)
            top.plot(c["vote"], c["arm"], color=color, lw=1.6, ls="--")
            ends.append((c["vote"].max() + 3, float(c["control"].iloc[-1]), f"{name} today", color))
            ends.append((c["vote"].max() + 3, float(c["arm"].iloc[-1]), f"{name} check off", color))
            bottom.plot(c["vote"], c["share_casting_prompt"], color=color, lw=1.4)
            share_ends.append((c["vote"].max() + 3, float(c["share_casting_prompt"].iloc[-1]), name, color))
        _end_labels(top, ends, gap=0.022)
        _end_labels(bottom, share_ends, gap=0.06)
        top.set_title(photos(k), color=INK2, fontsize=10)
        for ax in (top, bottom):
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_xlim(0, 182)
        bottom.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        bottom.set_xlabel("vote")
    axes[0, 0].set_ylabel("objective the session shows")
    axes[1, 0].set_ylabel("today's sessions\ncasting a check vote")
    fig.text(0.99, 0.005, "solid: today (the weak check); dashed: the end-of-run check only", ha="right",
             fontsize=8, color=INK2)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out / "fig_objective_by_vote.png", dpi=150)
    plt.close(fig)


def fig_diff(curves: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, k in zip(axes, KS):
        ends = []
        for beta, (color, name) in BETA.items():
            c = curves[(curves["K"] == k) & (curves["beta"].astype(float) == beta)].sort_values("vote")
            if c.empty:
                continue
            ax.plot(c["vote"], c["diff"], color=color, lw=1.6)
            ax.fill_between(c["vote"], c["diff"] - c["diff_se"], c["diff"] + c["diff_se"], color=color, alpha=0.15,
                            lw=0)  # fmt: skip
            ends.append((c["vote"].max() + 3, float(c["diff"].iloc[-1]), name, color))
        _end_labels(ax, ends, gap=0.006)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(photos(k), color=INK2, fontsize=10)
        ax.set_xlabel("vote")
        ax.set_xlim(0, 165)
    axes[0].set_ylabel("objective: check off minus today")
    fig.text(0.99, 0.01, "mean over the 562 people, ±1 SE band", ha="right", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig_session_diff.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--compare", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(args.compare / "curves.csv")
    fig_objective(curves, args.out)
    fig_diff(curves, args.out)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
