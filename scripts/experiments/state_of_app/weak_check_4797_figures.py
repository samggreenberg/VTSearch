#!/usr/bin/env python
"""#4797's figures, from ``weak_check_4797.py``'s tables: one row per path, one column per beta.

* ``fig_objective_by_vote.png``: the objective at every vote, the app (weak check on) and ``end`` (off);
* ``fig_diff_by_vote.png``: the paired difference, app minus end, with a 95% band clustered on the class;
* ``fig_per_session.png``: every prompted pair's session mean over votes 1-150, app minus end, sorted.

    python weak_check_4797_figures.py --tables <dir> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ON, OFF = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, MUTED, GRID = "#3d3d3a", "#73726c", "#e5e4df"
PATHS = (("binary", "Binary"), ("region", "Region"))
BETAS = (0.25, 1.0, 4.0)
BLABEL = {0.25: "β ¼", 1.0: "β 1", 4.0: "β 4"}


def grid(title: str, sharex: bool = True):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=sharex, constrained_layout=True)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    for ax in axes.flat:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    for i, (_, pname) in enumerate(PATHS):
        for j, b in enumerate(BETAS):
            axes[i, j].set_title(f"{pname}, {BLABEL[b]}", color=INK, fontsize=10, loc="left")
    return fig, axes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cur = pd.read_csv(args.tables / "weak_4797_curves.csv")
    run = pd.read_csv(args.tables / "weak_4797_per_run.csv")

    fig, axes = grid("The objective at every vote, weak check on (the app) and off (CALIB_SPOT_CHECK=end)")
    for i, (p, _) in enumerate(PATHS):
        for j, b in enumerate(BETAS):
            ax, c = axes[i, j], cur[(cur["path"] == p) & (cur["beta"] == b)]
            if c.empty:
                continue
            ax.plot(c["vote"], c["F_app"], color=ON, linewidth=2, label="check on (app)")
            ax.plot(c["vote"], c["F_end"], color=OFF, linewidth=2, label="check off (end)")
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    for ax in axes[1]:
        ax.set_xlabel("vote", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean F-beta, withheld set", color=MUTED, fontsize=9)
    fig.savefig(args.out / "fig_objective_by_vote.png", dpi=110)
    plt.close(fig)

    fig, axes = grid("Weak check on minus off at every vote, paired (band: 95%, clustered on the class)")
    for i, (p, _) in enumerate(PATHS):
        for j, b in enumerate(BETAS):
            ax, c = axes[i, j], cur[(cur["path"] == p) & (cur["beta"] == b)]
            if c.empty:
                continue
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.fill_between(c["vote"], c["d"] - 1.96 * c["d_se"], c["d"] + 1.96 * c["d_se"], color=ON, alpha=0.18,
                            linewidth=0)  # fmt: skip
            ax.plot(c["vote"], c["d"], color=ON, linewidth=2)
    for ax in axes[1]:
        ax.set_xlabel("vote", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("F-beta, on minus off", color=MUTED, fontsize=9)
    fig.savefig(args.out / "fig_diff_by_vote.png", dpi=110)
    plt.close(fig)

    fig, axes = grid(
        "Every prompted session: mean F-beta over votes 1-150, weak check on minus off, sorted", sharex=False
    )
    for i, (p, _) in enumerate(PATHS):
        for j, b in enumerate(BETAS):
            ax = axes[i, j]
            r = run[(run["path"] == p) & (run["beta"] == b) & run["first_vote"].notna()]
            if r.empty:
                continue
            d = r["d_1_150"].sort_values().to_numpy()
            x = range(1, len(d) + 1)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.scatter(x, d, s=10, color=[ON if v > 0 else OFF for v in d], linewidths=0)
            ax.text(0.02, 0.95, f"{(d > 0).mean():.0%} gain, {(d < 0).mean():.0%} lose of {len(d)}",
                    transform=ax.transAxes, fontsize=8, color=INK, va="top")  # fmt: skip
    for ax in axes[1]:
        ax.set_xlabel("prompted session, by rank", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("on minus off, votes 1-150", color=MUTED, fontsize=9)
    for ax in axes.flat:
        ax.tick_params(labelbottom=True)
    fig.savefig(args.out / "fig_per_session.png", dpi=110)
    plt.close(fig)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
