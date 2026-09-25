"""Figures for #3912, from the committed measurements.

python figures.py      # writes fig_scale_vs_inliers.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = json.loads((HERE / "measurements" / "diag_scale_floor.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    groups = (
        ("neg", "same-source negatives", "#d08080", "x"),
        ("pos", "true positives", "#2f6f9f", "o"),
        ("own", "the crop's own page", "#1b3f5f", "D"),
    )
    for role, label, colour, marker in groups:
        fits = [f for r in rows for f in r[role] if f["inliers"] > 0]
        # A collapsed fit reports scale 0; draw it at the axis floor so it stays visible.
        xs = [max(f["scale"], 1e-4) for f in fits]
        ax.scatter(
            xs, [f["inliers"] for f in fits], s=18, c=colour, marker=marker, label=label, alpha=0.75, linewidths=1
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 400)
    ax.axvline(0.1, color="#999999", ls="--", lw=1)
    ax.axvline(0.03, color="#333333", ls="-", lw=1)
    ax.axhline(8, color="#bbbbbb", ls=":", lw=1)
    ax.text(0.1, 0.02, " old floor 0.1", transform=ax.get_xaxis_transform(), va="bottom", fontsize=8, color="#666666")
    ax.text(0.03, 0.02, "new floor 0.03 ", transform=ax.get_xaxis_transform(), va="bottom", ha="right", fontsize=8)
    ax.text(1.2e-4, 8.5, "app's 8-inlier gate", fontsize=7, color="#888888")
    ax.set_xlabel("fitted similarity scale (normalised coordinates; collapsed fits drawn at 1e-4)")
    ax.set_ylabel("RANSAC inliers")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_scale_vs_inliers.png", dpi=150)


if __name__ == "__main__":
    main()
