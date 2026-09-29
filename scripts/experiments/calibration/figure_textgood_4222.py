#!/usr/bin/env python3
"""Report figures for #4222, from the study directory's committed CSVs.

* ``harvest.png`` - mean Good votes by vote t, one line per Good target G,
  one panel per pool prevalence;
* ``ap_vs_today.png`` - average precision minus today's opening (G = 3),
  paired per cell, with a 2-SE band, same layout.

    python figure_textgood_4222.py [--study DIR]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
STUDY = REPO / "docs" / "experiments" / "2026-09-28-textgood-4222"
INK, SOFT, GRID = "#14181f", "#5b6472", "#e3e6ea"
#: G is ordered, so the encoding is a single blue ramp, light to dark.  G = 3 is
#: today's opening and is drawn in the ramp's own place, not as a special colour.
G_COLOR = {3: "#9dbde6", 6: "#5f95d6", 10: "#2f6db5", 20: "#173d6e"}
#: The arms' directory names, and the words the figures use for them: a pool
#: prevalence is a scenario, not a claim about users.
WORLDS = (("natural", "0.44% positive (COCO Better's default)"), ("p0.001", "0.1% positive (thinned)"))

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "axes.edgecolor": SOFT,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "savefig.dpi": 200,
    }
)


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def two_panels(study: Path, draw, ylabel: str, name: str, zero_line: bool) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (world, title) in zip(axes, WORLDS):
        draw(ax, world)
        if zero_line:
            ax.axhline(0, color=SOFT, lw=1)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("votes")
        ax.set_xticks([25, 50, 100, 150])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = study / "figures" / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def harvest(study: Path) -> Path:
    data = rows(study / "harvest.csv")

    def draw(ax, world):
        for g, color in G_COLOR.items():
            pts = sorted(
                (int(r["t"]), float(r["mean_goods"])) for r in data if r["world"] == world and int(r["G"]) == g
            )
            label = f"G = {g}" + (" (today)" if g == 3 else "")
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2, marker="o", ms=5, label=label)

    return two_panels(study, draw, "Good votes so far (mean per cell)", "harvest.png", zero_line=False)


def ap(study: Path) -> Path:
    data = [r for r in rows(study / "quality_paired.csv") if r["metric"] == "average_precision"]

    def draw(ax, world):
        for g, color in G_COLOR.items():
            if g == 3:
                continue
            pts = sorted(
                (int(r["t"]), float(r["delta_vs_G3"]), float(r["se"]))
                for r in data
                if r["world"] == world and int(r["G"]) == g
            )
            t = [p[0] for p in pts]
            d = [p[1] for p in pts]
            se = [p[2] for p in pts]
            ax.fill_between(
                t, [a - 2 * b for a, b in zip(d, se)], [a + 2 * b for a, b in zip(d, se)], color=color, alpha=0.15, lw=0
            )
            ax.plot(t, d, color=color, lw=2, marker="o", ms=5, label=f"G = {g}")

    return two_panels(study, draw, "average precision minus today's (G = 3), ±2 SE", "ap_vs_today.png", zero_line=True)


def main() -> int:
    ap_ = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap_.add_argument("--study", type=Path, default=STUDY)
    args = ap_.parse_args()
    for p in (harvest(args.study), ap(args.study)):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
