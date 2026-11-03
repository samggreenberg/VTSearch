#!/usr/bin/env python
"""#5477's figures, from ``min_goods_5477.py``'s tables: one row per bench, one column per beta.

* ``fig_stopped_by_vote.png``: the share of sessions Autopilot has told Done by each vote, the app before #5477 and
  each arm;
* ``fig_stop_tradeoff.png``: each arm's mean stop against its F at the stop minus the app's, paired, with a 95% bar
  clustered on the class (the app before #5477 sits at zero).

    python min_goods_5477_figures.py --tables <dir> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots 1-4, in arm order.
COLOR = {"ctl": "#2a78d6", "min3": "#eb6834", "dry8": "#1baf7a", "dry16": "#eda100"}
LABEL = {
    "ctl": "before #5477 (5 Goods)",
    "min3": "min3 (3 Goods)",
    "dry8": "dry8 (waive after 8)",
    "dry16": "dry16 (waive after 16, shipped)",
}
ARMS = ("min3", "dry8", "dry16")
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
    cur = pd.read_csv(args.tables / "mingood_5477_curves.csv")
    summ = pd.read_csv(args.tables / "mingood_5477_summary.csv")
    summ = summ[summ["split"] == "all"]
    rows = [r for r in ROWS if ((cur["bench"] == r[0]) & (cur["set"] == r[1])).any()]

    def sel(frame: pd.DataFrame, bench: str, s: str, b: float) -> pd.DataFrame:
        return frame[(frame["bench"] == bench) & (frame["set"] == s) & (frame["beta"] == b)]

    fig, axes = grid("Share of sessions Autopilot has told Done, by vote", rows)
    for i, (bench, s, _) in enumerate(rows):
        for j, b in enumerate(BETAS):
            ax, c = axes[i, j], sel(cur, bench, s, b)
            if c.empty:
                continue
            ctl = c[c["arm"] == ARMS[0]]
            ax.plot(ctl["vote"], ctl["stopped_ctl"], color=COLOR["ctl"], linewidth=2, label=LABEL["ctl"])
            for arm in ARMS:
                a = c[c["arm"] == arm]
                ax.plot(a["vote"], a["stopped_arm"], color=COLOR[arm], linewidth=2, label=LABEL[arm])
            ax.set_ylim(0, 1)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    for ax in axes[-1]:
        ax.set_xlabel("vote", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("share told Done", color=MUTED, fontsize=9)
    fig.savefig(args.out / "fig_stopped_by_vote.png", dpi=110)
    plt.close(fig)

    fig, axes = grid("Mean stop against F at the stop, each arm minus the app before #5477 (bar: 95%, clustered "
                     "on the class)", rows, sharex=False)  # fmt: skip
    for i, (bench, s, _) in enumerate(rows):
        for j, b in enumerate(BETAS):
            ax, m = axes[i, j], sel(summ, bench, s, b)
            if m.empty:
                continue
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.plot([m["stop_ctl"].iloc[0]], [0], "o", markersize=8, color=COLOR["ctl"], label=LABEL["ctl"])
            for arm in ARMS:
                r = m[m["arm"] == arm]
                ax.errorbar(r["stop_arm"], r["d_F_stop"], yerr=1.96 * r["d_F_stop_se"], fmt="o", markersize=8,
                            color=COLOR[arm], elinewidth=2, capsize=0, label=LABEL[arm])  # fmt: skip
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("mean stop (vote; 150 when never told Done)", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("F at the stop, arm minus before", color=MUTED, fontsize=9)
    fig.savefig(args.out / "fig_stop_tradeoff.png", dpi=110)
    plt.close(fig)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
