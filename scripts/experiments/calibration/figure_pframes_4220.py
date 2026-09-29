#!/usr/bin/env python3
"""Report figures for #4220, from the study directory's committed CSVs.

* ``trust_follows_positives.png`` - how often a promised precision floor
  (X = 0.5) is broken, of the promises made, against the positives the
  calibration folds hold, for
  the fold-rank and in-sample lower bounds with EM, per scenario;
* ``calibration_positives.png`` - how many calibration positives today's app
  has at each checkpoint, 0.44% pool (COCO Better default) against the 5% pool, with the gate the
  report recommends.

    python figure_pframes_4220.py [--study DIR]
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
STUDY = REPO / "docs" / "experiments" / "2026-09-28-precision-frames-4220"
INK, SOFT, GRID = "#14181f", "#5b6472", "#e3e6ea"
#: Validated with the dataviz skill's checker (light surface): all checks pass.
SERIES = {
    ("natural", "same"): ("0.44% pool (COCO Better default), same corpus", "#2F6DB5"),
    ("h0.05", "same"): ("5% pool, same corpus", "#C26A1B"),
    ("h0.05", "shifted"): ("5% pool, 0.44% corpus (label shift)", "#7A5AB8"),
}
GATE = 10
BINS = ["(-1, 1]", "(1, 2]", "(2, 3]", "(3, 5]", "(5, 7]", "(7, 10]", "(10, 15]", "(15, 20]", "(20, 100]"]
BIN_LABELS = ["0-1", "2", "3", "4-5", "6-7", "8-10", "11-15", "16-20", "21+"]

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


def trust(study: Path) -> Path:
    rows = list(csv.DictReader((study / "fig_violation_by_calpos.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, est, title in zip(
        axes, ("foldrank", "insample"), ("fold-rank lower bound + EM", "in-sample lower bound + EM")
    ):
        for key, (label, color) in SERIES.items():
            pts = {
                r["bin"]: (float(r["viol"]), int(r["n"]))
                for r in rows
                if (r["arm"], r["scenario"], r["estimator"]) == (*key, est)
            }
            xs = [i for i, b in enumerate(BINS) if b in pts and pts[b][1] >= 20]
            ax.plot(xs, [pts[BINS[i]][0] for i in xs], color=color, lw=2, marker="o", ms=5, label=label)
        ax.axhline(0.05, color=SOFT, lw=1, ls=":")
        ax.text(len(BINS) - 0.5, 0.06, "5%", color=SOFT, ha="right", fontsize=9)
        ax.axvline(BINS.index("(7, 10]") + 0.5, color=SOFT, lw=1, ls="--")
        ax.text(BINS.index("(7, 10]") + 0.6, 0.93, f"gate: >{GATE - 1}", color=SOFT, fontsize=9)
        ax.set_xticks(range(len(BINS)), BIN_LABELS)
        ax.set_xlabel("positives among the calibration votes")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    # Of the promises actually made: a cell that promises nothing cannot break
    # a promise, and counting it as kept would flatter the thin bins.
    axes[0].set_ylabel("promises broken, of those made (X = 50%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = study / "figures" / "trust_follows_positives.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def positives(study: Path) -> Path:
    rows = list(csv.DictReader((study / "fig_calpos.csv").open()))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for arm, label, color in (
        ("natural", "0.44% pool (COCO Better default)", "#2F6DB5"),
        ("h0.05", "5% pool", "#C26A1B"),
    ):
        ts = sorted({int(r["t"]) for r in rows})
        med, q1, q3 = [], [], []
        for t in ts:
            v = np.array([int(r["n_cal_pos"]) for r in rows if r["arm"] == arm and int(r["t"]) == t])
            med.append(np.median(v))
            q1.append(np.percentile(v, 25))
            q3.append(np.percentile(v, 75))
        ax.fill_between(ts, q1, q3, color=color, alpha=0.15, lw=0)
        ax.plot(ts, med, color=color, lw=2, marker="o", ms=5, label=f"{label}: median, IQR shaded")
    ax.axhline(GATE, color=SOFT, lw=1, ls="--")
    ax.text(26, GATE + 0.6, f"gate: {GATE} calibration positives", color=SOFT, fontsize=9)
    ax.set_xlabel("votes")
    ax.set_ylabel("positives among the calibration votes")
    ax.set_xticks([25, 50, 100, 150])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    out = study / "figures" / "calibration_positives.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--study", type=Path, default=STUDY)
    args = ap.parse_args()
    for p in (trust(args.study), positives(args.study)):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
