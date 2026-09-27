#!/usr/bin/env python3
"""Figures for the #4088 / #4089 review, from measurements/.

python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"
FIGS = HERE / "figures"
DPI = 130

# Validated categorical slots 1 and 2 (CVD dE 24.7 protan, 33.6 normal).
BEFORE = "#eb6834"
AFTER = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def shortcut() -> None:
    """The mark-blind 'UCSF Tobacco first' control, before and after the banded review, per tier."""
    rows = json.loads((MEAS / "shortcut.json").read_text())
    tiers = ("s", "m", "l")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), dpi=DPI, sharey=True)
    for ax, tier in zip(axes, tiers):
        _style(ax)
        sel = [r for r in rows if r["tier"] == tier]
        for y, r in enumerate(sel):
            a, b = r["before"]["industry_prior_ap"], r["after"]["industry_prior_ap"]
            if r["after"]["positives"] == 0:
                ax.text(0.02, y, "no positives in tier", va="center", color=MUTED, fontsize=8)
                continue
            ax.plot([a, b], [y, y], color=GRID, linewidth=2, zorder=1)
            ax.plot([a], [y], "o", color=BEFORE, markersize=8, zorder=2, label="v4.1" if y == 0 else None)
            ax.plot([b], [y], "o", color=AFTER, markersize=8, zorder=3, label="v4.2" if y == 0 else None)
        ax.axvline(0.2, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(f"tier {tier}", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("AP, ignoring the mark", color=INK, fontsize=9)
    names = [r["class_id"].split("logo_")[1] for r in rows if r["tier"] == "m"]
    axes[0].set_yticks(range(len(names)), names, fontsize=9)
    axes[0].invert_yaxis()
    axes[2].legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.suptitle(
        "'Rank UCSF Tobacco pages first' after 1,610 banded pages were reviewed (dashed: #4088's 0.2 line)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "shortcut.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    shortcut()
