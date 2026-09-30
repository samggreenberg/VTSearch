#!/usr/bin/env python3
"""Report figures for #4184 / #4201: the seven rungs at two pool prevalences.

Reads the study directory's committed CSVs and writes, into its ``figures/``:

* ``ladder_by_prevalence.png`` - mean cost over clicks, one panel per
  prevalence, the seven rungs in the deck's order;
* ``regret_by_prevalence.png`` - each rung's regret (cost minus the best cut on
  the same model) at the natural and the 5% pool, as a dumbbell.

    python figure_prevalence_4201.py [--study DIR]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
STUDY = REPO / "docs" / "experiments" / "2026-09-25-progression-4184"

INK, SOFT, GRID = "#14181f", "#5b6472", "#e3e6ea"
#: The deck's `.cut` blue: the app as it ships.
SHIPPED = "#2F6DB5"
#: The rungs in the deck's order.  Colour is a single grey ramp that darkens as
#: the ladder climbs - the rungs are ordered, so the encoding is ordered too -
#: and the last rung, today's app, takes the shipped blue.
RUNGS = (
    ("r1_xcal", "r1 cross-calibration", "#c9ced6"),
    ("r2_gmm", "r2 mixture midpoint", "#aab1bc"),
    ("r3_blend", "r3 blend", "#8a93a1"),
    ("r4_rawmean", "r4 fused, raw average", "#6b7482"),
    ("r5_anchored", "r5 fused, rank transfer", "#4a525e"),
    ("r6_split70", "r6 70/30 split", "#2b313a"),
    ("r7_acq4", "r7 second cut (today's app)", SHIPPED),
)
#: Direct labels only where the story is; the legend carries the rest.
LABELLED = ("r4_rawmean", "r5_anchored", "r7_acq4")
ARMS = (("natural", "natural pool (0.44% positive)"), ("h0.05", "pool thinned to 5% positive"))

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
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)


def curves(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    by: dict[str, list[tuple[int, float]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            by.setdefault(row["rung"], []).append((int(row["t"]), float(row["mean"])))
    return {r: (np.array([p[0] for p in sorted(v)]), np.array([p[1] for p in sorted(v)])) for r, v in by.items()}


def spread_labels(ys: list[float], gap: float) -> list[float]:
    """Nudge end-of-line labels apart so none overlap, keeping their order."""
    order = np.argsort(ys)
    out = list(ys)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        out[b] = max(out[b], out[a] + gap)
    return out


def ladder(study: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (arm, title) in zip(axes, ARMS):
        cv = curves(study / arm / "progression_curve.csv")
        ends = []
        for rung, label, color in RUNGS:
            t, m = cv[rung]
            ax.plot(t, m, color=color, lw=2, label=label, zorder=3 if rung in LABELLED else 2)
            if rung in LABELLED:
                ends.append((m[-1], label.split(" ")[0], color))
        placed = spread_labels([e[0] for e in ends], 0.018)
        for (_, text, color), y in zip(ends, placed):
            ax.text(153, y, text, color=INK, va="center", fontsize=10)
            ax.plot([150, 152], [y, y], color=color, lw=1)
        ax.set_title(title, fontsize=12, loc="left", color=INK)
        ax.set_xlabel("votes")
        ax.set_xlim(0, 165)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("mean cost (FPR + FNR), lower is better")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out = study / "figures" / "ladder_by_prevalence.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def regret(study: Path) -> Path:
    rows = {}
    with (study / "prevalence_summary.csv").open() as f:
        for row in csv.DictReader(f):
            rows[(row["rung"], row["arm"])] = float(row["regret"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (rung, label, color) in enumerate(RUNGS):
        y = len(RUNGS) - 1 - i
        a, b = rows[(rung, "natural")], rows[(rung, "h0.05")]
        ax.plot([b, a], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter([a], [y], s=70, color="white", edgecolor=color, lw=2, zorder=3)
        ax.scatter([b], [y], s=70, color=color, edgecolor="white", lw=2, zorder=3)
    ax.set_yticks(range(len(RUNGS)))
    ax.set_yticklabels([label for _, label, _ in reversed(RUNGS)])
    ax.set_xlabel("regret: cost − best cut on the same model")
    ax.scatter([], [], s=70, color="white", edgecolor=SOFT, lw=2, label="natural pool (0.44%)")
    ax.scatter([], [], s=70, color=SOFT, edgecolor="white", lw=2, label="pool thinned to 5%")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_xlim(0, None)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = study / "figures" / "regret_by_prevalence.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--study", type=Path, default=STUDY)
    args = ap.parse_args()
    for p in (ladder(args.study), regret(args.study)):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
