"""Figures for the UCSF band-class proposals (#3921, #3922), from ``measurements/summary.json``.

python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
BINS = ("80+", "40-79", "20-39", "12-19", "8-11")
# Reference palette: categorical slots 1-2; ordinal blue 700 -> 250 for the inlier bins.
FOUND, MISSED = "#2a78d6", "#eb6834"
RAMP = ("#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    summary = json.loads((HERE / "measurements" / "summary.json").read_text(encoding="utf-8"))
    names = list(summary)
    labels = [
        f"{n}\n({'extends ' + s['roster_class'].split('_')[1] if s['roster_class'] else 'new'})"
        for n, s in summary.items()
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1, 1.25]})
    y = range(len(names))

    found = [summary[n]["band_class_sm_found"] for n in names]
    missed = [summary[n]["band_class_sm"] - summary[n]["band_class_sm_found"] for n in names]
    ax1.barh(
        y, found, color=FOUND, height=0.6, label="found by the search (>= 8 inliers)", edgecolor="white", linewidth=2
    )
    ax1.barh(y, missed, left=found, color=MISSED, height=0.6, label="missed", edgecolor="white", linewidth=2)
    for i, (f, m) in enumerate(zip(found, missed)):
        ax1.text(f + m + 3, i, f"{f}/{f + m}", va="center", color=INK, fontsize=9)
    ax1.set_yticks(list(y), labels, fontsize=9, color=INK)
    ax1.invert_yaxis()
    ax1.set_xlabel("band-class members in tiers s+m", color=MUTED)
    ax1.set_title("Does the crop find its own band class?", loc="left", color=INK, fontsize=11)
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)

    # Hits outside the band class, by inlier bin: counts as an annotated grid.
    ax2.set_xlim(-0.5, len(BINS) - 0.5)
    ax2.set_ylim(len(names) - 0.5, -0.5)
    for i, n in enumerate(names):
        for j, b in enumerate(BINS):
            count = summary[n]["hits_outside_by_bin"].get(b, 0)
            ax2.add_patch(plt.Rectangle((j - 0.47, i - 0.42), 0.94, 0.84, color=RAMP[j] if count else "#f0efec"))
            ax2.text(
                j, i, f"{count:,}", ha="center", va="center", fontsize=9, color="white" if count and j < 3 else INK
            )
    ax2.set_xticks(range(len(BINS)), [f"{b}\ninliers" for b in BINS], fontsize=9, color=INK)
    ax2.set_yticks([])
    ax2.set_title(
        "Search hits outside the band class (tiers s+m; shade = inlier bin, not count)",
        loc="left",
        color=INK,
        fontsize=11,
    )
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED)
    ax1.grid(axis="x", color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "fig_search.png", dpi=110)


if __name__ == "__main__":
    main()
