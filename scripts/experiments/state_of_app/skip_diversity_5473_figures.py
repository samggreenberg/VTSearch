#!/usr/bin/env python
"""#5473's figures, from ``skip_diversity_5473.py``'s tables: one row per bench, one column per beta.

* ``fig_stopped_by_vote.png``: the share of sessions Autopilot has told to stop by each vote, the app and no Diversity;
* ``fig_diff_by_vote.png``: F at every vote, no Diversity minus the app, paired, with a 95% band clustered on the class;
* ``fig_stop_tradeoff.png``: every session the app took into New: the clicks no Diversity saved to the stop, against
  what it changed in F at the stop.

    python skip_diversity_5473_figures.py --tables <dir> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

APP, NODIV = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, MUTED, GRID = "#3d3d3a", "#73726c", "#e5e4df"
ROWS = (("coco", "binary", "COCO Binary"), ("face", "K4", "Faces, 4 photos"), ("face", "K1", "Faces, 1 photo"))
BETAS = (0.25, 1.0, 4.0)
BLABEL = {0.25: "β ¼", 1.0: "β 1", 4.0: "β 4"}


def grid(title: str, rows: list, sharex: bool = True):
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.2 * len(rows)), sharex=sharex, squeeze=False,
                             constrained_layout=True)  # fmt: skip
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    for ax in axes.flat:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    for i, (_, _, name) in enumerate(rows):
        for j, b in enumerate(BETAS):
            axes[i, j].set_title(f"{name}, {BLABEL[b]}", color=INK, fontsize=10, loc="left")
    return fig, axes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cur = pd.read_csv(args.tables / "nodiv_5473_curves.csv")
    run = pd.read_csv(args.tables / "nodiv_5473_per_run.csv")
    rows = [r for r in ROWS if ((cur["bench"] == r[0]) & (cur["set"] == r[1])).any()]

    def sel(frame: pd.DataFrame, bench: str, s: str, b: float) -> pd.DataFrame:
        return frame[(frame["bench"] == bench) & (frame["set"] == s) & (frame["beta"] == b)]

    fig, axes = grid("Share of sessions Autopilot has told to stop, by vote: the app and no Diversity step", rows)
    for i, (bench, s, _) in enumerate(rows):
        for j, b in enumerate(BETAS):
            ax, c = axes[i, j], sel(cur, bench, s, b)
            if c.empty:
                continue
            ax.plot(c["vote"], c["stopped_ctl"], color=APP, linewidth=2, label="the app (Span at 40)")
            ax.plot(c["vote"], c["stopped_nodiv"], color=NODIV, linewidth=2, label="no Diversity (Span at 0)")
            ax.set_ylim(0, 1)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    for ax in axes[-1]:
        ax.set_xlabel("vote", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("share told to stop", color=MUTED, fontsize=9)
    fig.savefig(args.out / "fig_stopped_by_vote.png", dpi=110)
    plt.close(fig)

    fig, axes = grid(
        "F-beta at every vote, no Diversity minus the app, paired (band: 95%, clustered on the class)", rows
    )
    for i, (bench, s, _) in enumerate(rows):
        for j, b in enumerate(BETAS):
            ax, c = axes[i, j], sel(cur, bench, s, b)
            if c.empty:
                continue
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.fill_between(c["vote"], c["d"] - 1.96 * c["d_se"], c["d"] + 1.96 * c["d_se"], color=NODIV, alpha=0.18,
                            linewidth=0)  # fmt: skip
            ax.plot(c["vote"], c["d"], color=NODIV, linewidth=2)
    for ax in axes[-1]:
        ax.set_xlabel("vote", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("F-beta, no Diversity minus app", color=MUTED, fontsize=9)
    fig.savefig(args.out / "fig_diff_by_vote.png", dpi=110)
    plt.close(fig)

    fig, axes = grid("Sessions the app took into Diversity: clicks saved to the stop, and F at the stop", rows,
                     sharex=False)  # fmt: skip
    for i, (bench, s, _) in enumerate(rows):
        for j, b in enumerate(BETAS):
            ax = axes[i, j]
            r = sel(run, bench, s, b)
            r = r[r["first_new_ctl"].notna()]
            if r.empty:
                continue
            stop_c = r["done_ctl"].fillna(150).to_numpy()
            stop_a = r["done_nodiv"].fillna(150).to_numpy()
            saved = stop_c - stop_a
            dF = (r["F_stop_nodiv"] - r["F_stop_ctl"]).to_numpy()
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.scatter(saved, dF, s=10, color=NODIV, alpha=0.5, linewidths=0)
            ax.text(0.02, 0.95, f"{len(r)} sessions; median saved {np.median(saved):.0f} clicks, "
                    f"F {dF.mean():+.3f}", transform=ax.transAxes, fontsize=8, color=INK, va="top")  # fmt: skip
    for ax in axes[-1]:
        ax.set_xlabel("clicks saved to the stop (app minus no Diversity)", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("F at the stop, no Diversity minus app", color=MUTED, fontsize=9)
    fig.savefig(args.out / "fig_stop_tradeoff.png", dpi=110)
    plt.close(fig)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
