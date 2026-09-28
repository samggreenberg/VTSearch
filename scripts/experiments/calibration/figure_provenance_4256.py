#!/usr/bin/env python3
"""Report figures for #4256, from the study directory's committed CSVs.

* ``broken_by_evidence.png`` - promises broken, of those made (X = 50%), per
  opening arm, for all calibration votes against learned-sort votes only, under
  the shipped reference pool and the consistent one;
* ``learned_positives.png`` - calibration positives per checkpoint, all votes
  against learned-sort votes only, per opening arm, with the 10-positive gate.

    python figure_provenance_4256.py [--study DIR]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
STUDY = REPO / "docs" / "experiments" / "2026-09-28-provenance-4256"
INK, SOFT, GRID = "#14181f", "#5b6472", "#e3e6ea"
#: The #4220 palette, validated with the dataviz skill's checker (light surface).
ARMS = {"g3": ("g3: today's opening", "#2F6DB5"), "g6": ("g6", "#C26A1B"), "g20": ("g20", "#7A5AB8")}
EVIDENCE = {"all": ("all calibration votes", "#2F6DB5"), "learned": ("learned-sort votes only", "#C26A1B")}
GATE = 10

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


def broken(study: Path) -> Path:
    rows = [r for r in csv.DictReader((study / "provenance_summary.csv").open()) if float(r["X"]) == 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, pool, title in zip(
        axes,
        ("shipped", "consistent"),
        ("shipped reference pool (voted items included)", "consistent reference pool (voted items removed)"),
    ):
        for j, (ev, (label, color)) in enumerate(EVIDENCE.items()):
            xs, ys, es, ns = [], [], [], []
            for i, arm in enumerate(ARMS):
                r = next(r for r in rows if (r["arm"], r["evidence"], r["pool"]) == (arm, ev, pool))
                xs.append(i + (j - 0.5) * 0.36)
                ys.append(float(r["broken_of_made"]))
                es.append(float(r["broken_se"]))
                ns.append(int(float(r["made_n"])))
            ax.bar(xs, ys, width=0.34, color=color, label=label, edgecolor="white", linewidth=2)
            ax.errorbar(xs, ys, yerr=[2 * e for e in es], fmt="none", ecolor=INK, lw=1, capsize=3)
            for x, y, e, n in zip(xs, ys, es, ns):
                ax.text(x, y + 2 * (e if e == e else 0) + 0.025, f"{n}", ha="center", color=SOFT, fontsize=8)
        ax.axhline(0.1, color=SOFT, lw=1, ls=":")
        ax.set_xticks(range(len(ARMS)), [ARMS[a][0] for a in ARMS])
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("promises broken, of those made (X = 50%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=2)
    fig.text(
        0.99,
        0.02,
        "number over a bar = promises made; whiskers = 2 SE; dotted line = 10%, the bound's level",
        ha="right",
        color=SOFT,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = study / "figures" / "broken_by_evidence.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def positives(study: Path) -> Path:
    rows = [
        r
        for r in csv.DictReader((study / "provenance_by_t.csv").open())
        if float(r["X"]) == 0.5 and r["pool"] == "shipped"
    ]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ts = [25, 50, 100, 150]
    for arm, (label, color) in ARMS.items():
        for ev, ls in (("all", "-"), ("learned", "--")):
            ys = [
                float(next(r for r in rows if r["arm"] == f"{arm}@t{t}" and r["evidence"] == ev)["mean_cal_pos"])
                for t in ts
            ]
            ax.plot(ts, ys, color=color, lw=2, ls=ls, marker="o", ms=5, label=f"{label}, {EVIDENCE[ev][0]}")
    ax.axhline(GATE, color=SOFT, lw=1, ls=":")
    ax.text(150, GATE + 0.3, f"gate: {GATE} calibration positives", color=SOFT, ha="right", fontsize=9)
    ax.set_xlabel("votes")
    ax.set_ylabel("calibration positives (mean over cells)")
    ax.set_xticks(ts)
    ax.set_ylim(0, 12)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    out = study / "figures" / "learned_positives.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--study", type=Path, default=STUDY)
    args = ap.parse_args()
    for p in (broken(args.study), positives(args.study)):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
