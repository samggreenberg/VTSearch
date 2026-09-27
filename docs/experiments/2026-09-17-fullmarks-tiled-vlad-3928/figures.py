"""Figure for the #3928 tiled-VLAD probe, from ``measurements/`` alone.

python figures.py     # writes fig_stage1.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"
KS = (100, 500, 1000)
TIER_PAGES = {"s": "~4,900", "m": "~48,000"}


def _mean(path: Path, method: str) -> float:
    return float(
        np.mean([float(r["ap"]) for r in csv.DictReader(path.open(encoding="utf-8")) if r["method"] == method])
    )


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, tier in zip(axes, ("s", "m")):
        tiles = MEAS / f"tiles_{tier}_8192.csv"
        sift = MEAS / f"sift_rank_{tier}_8192.csv"
        arms = (
            ("tiled VLAD top K, SIFT verifies", tiles, "vlad_t4{k}_sift", "#2f6f9f"),
            ("SigLIP top K, SIFT verifies", sift, "siglip{k}_sift", "#9cc3de"),
            ("page VLAD top K, SIFT verifies", sift, "vlad{k}_sift", "#d9a3a3"),
        )
        for label, path, pattern, colour in arms:
            ax.plot(KS, [_mean(path, pattern.format(k=k)) for k in KS], marker="o", color=colour, label=label)
        ax.axhline(_mean(sift, "sift"), color="#333333", ls="--", lw=1, label="SIFT, every page")
        ax.axhline(_mean(tiles, "vlad_t4"), color="#2f6f9f", ls=":", lw=1, label="tiled VLAD alone")
        ax.set_title(f"tier {tier} ({TIER_PAGES[tier]} pages per pool)", fontsize=10)
        ax.set_xlabel("shortlist size K")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean AP, 23 classes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(HERE / "fig_stage1.png", dpi=150)


if __name__ == "__main__":
    main()
