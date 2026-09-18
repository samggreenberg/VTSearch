"""Figures for the first DocMarks roster evaluation (#3904), from the committed measurements.

python figures.py      # writes fig_ap_by_pool.png and fig_inliers_by_budget.png
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"
POOLS = ("eligible", "own_verified", "naive")
POOL_LABEL = {"eligible": "rule as written", "own_verified": "own source as\nknown negatives", "naive": "every page"}
METHODS = ("source_prior", "siglip", "vlad", "vlad_rerank")
METHOD_LABEL = {
    "source_prior": "source only (control)",
    "siglip": "SigLIP",
    "vlad": "VLAD",
    "vlad_rerank": "VLAD + re-rank",
}
COLOURS = {"source_prior": "#bdbdbd", "siglip": "#2f6f9f", "vlad": "#d08c3c", "vlad_rerank": "#8a5a1f"}


def fig_ap_by_pool() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, tier in zip(axes, ("s", "m")):
        rows = list(csv.DictReader((MEAS / f"rows_{tier}.csv").open(encoding="utf-8")))
        mean = defaultdict(list)
        for r in rows:
            mean[(r["pool"], r["method"])].append(float(r["ap"]))
        x = np.arange(len(POOLS))
        width = 0.2
        for i, method in enumerate(METHODS):
            ys = [np.mean(mean[(p, method)]) for p in POOLS]
            ax.bar(x + (i - 1.5) * width, ys, width, color=COLOURS[method], label=METHOD_LABEL[method])
        ax.set_xticks(x, [POOL_LABEL[p] for p in POOLS], fontsize=8)
        ax.set_title(f"tier {tier} ({'5,000' if tier == 's' else '50,000'} pages)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean AP over 23 classes")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "fig_ap_by_pool.png", dpi=150)


def fig_inliers_by_budget() -> None:
    rows = json.loads((MEAS / "diag_structural.json").read_text(encoding="utf-8"))
    by = defaultdict(dict)
    for r in rows:
        by[r["class_id"]][str(r["budget"])] = r
    classes = sorted(by, key=lambda c: (c.split("/")[0], c))
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), sharey=True, sharex=True)
    y = np.arange(len(classes))
    for ax, budget, title in zip(
        axes, ("stored", "16384"), ("stored features (1,024 keypoints)", "re-extracted at 16,384")
    ):
        pos = [by[c][budget]["pos_med"] for c in classes]
        neg = [by[c][budget]["neg_max"] for c in classes]
        ax.barh(y + 0.2, pos, 0.4, color="#2f6f9f", label="true positives (median)")
        ax.barh(y - 0.2, neg, 0.4, color="#d9a3a3", label="same-source negatives (max)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("RANSAC inliers against the query crop")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, classes, fontsize=7)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "fig_inliers_by_budget.png", dpi=150)


if __name__ == "__main__":
    fig_ap_by_pool()
    fig_inliers_by_budget()
