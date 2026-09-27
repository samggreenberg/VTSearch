"""Figures for #3902 from ``measurements/band_sweep.json``.

python figures.py      # writes fig_sweep.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
MARK_AUTHORS = ("LOR, LORILLARD", "PHILIP MORRIS", "BROWN & WILLIAMSON", "RJR", "AMERICAN TOBACCO", "BATCO")
TYPESET = ("COUNCIL FOR TOBACCO RESEARCH", "TOBACCO INSTITUTE")


def main() -> None:
    sweep = json.loads((HERE / "measurements" / "band_sweep.json").read_text(encoding="utf-8"))
    panels = (
        ("largest_share", "largest component, share of bands"),
        ("usable", "classes with >= MIN_INSTANCES bands"),
        ("singleton_share", "singletons, share of bands"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, label) in zip(axes, panels):
        for author in MARK_AUTHORS + TYPESET:
            rows = sweep[author]
            style = "-" if author in MARK_AUTHORS else ":"
            ax.plot([r["threshold"] for r in rows], [r[key] for r in rows], style, marker=".", label=author.title())
        ax.set_xlabel("cosine-distance threshold")
        ax.set_title(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "fig_sweep.png", dpi=150)


if __name__ == "__main__":
    main()
