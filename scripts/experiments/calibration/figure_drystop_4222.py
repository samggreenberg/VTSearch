#!/usr/bin/env python3
"""Report figures for the #4222 dry-stop study, from the study directory's committed CSVs.

* ``ap_gain.png`` - AP gain over today's g3 opening at vote 150, per arm, in
  each pool, with 2 paired SE, beside the share of sessions left with no
  detector;
* ``ap_by_votes.png`` - the same gain at votes 25/50/100/150 for g6 and the
  floored dry stop: the long walk's early cost and where it pays back.

    python figure_drystop_4222.py [--study DIR]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
STUDY = REPO / "docs" / "experiments" / "2026-09-29-drystop-4222"
INK, SOFT, GRID = "#14181f", "#5b6472", "#e3e6ea"
#: The #4220 palette, validated with the dataviz skill's checker (light surface).
POOLS = {"0.44%": ("0.44% pool (COCO Better default)", "#2F6DB5"), "0.1%": ("0.1% pool", "#C26A1B")}
ARMS = ["g6", "g20", "g20d8", "g20d16", "g3g20d8", "g3g20d16"]
LABELS = {
    "g6": "g6",
    "g20": "g20",
    "g20d8": "g20+dry1/8",
    "g20d16": "g20+dry1/16",
    "g3g20d8": "g3, g20+dry1/8",
    "g3g20d16": "g3, g20+dry1/16",
}

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


def rows(study: Path) -> list[dict]:
    return [
        r
        for r in csv.DictReader((study / "quality_paired.csv").open())
        if r["band"] == "all" and r["metric"] == "average_precision"
    ]


def ap_gain(study: Path) -> Path:
    q = [r for r in rows(study) if r["t"] == "150"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    for j, (pool, (label, color)) in enumerate(POOLS.items()):
        xs = [i + (j - 0.5) * 0.38 for i in range(len(ARMS))]
        got = {r["arm"]: r for r in q if r["pool"] == pool}
        ys = [float(got[a]["delta_vs_g3"]) for a in ARMS]
        es = [2 * float(got[a]["se"]) for a in ARMS]
        ax.bar(xs, ys, width=0.36, color=color, label=label, edgecolor="white", linewidth=2)
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK, lw=1, capsize=3)
        no_det = [100 * float(got[a]["no_detector"]) for a in ["g3", *ARMS]]
        bx.bar(
            [i + (j - 0.5) * 0.38 for i in range(len(no_det))],
            no_det,
            width=0.36,
            color=color,
            edgecolor="white",
            linewidth=2,
        )
    ax.axhline(0, color=SOFT, lw=1)
    ax.set_xticks(range(len(ARMS)), [LABELS[a] for a in ARMS], rotation=20, ha="right")
    ax.set_ylabel("AP gain over g3 at vote 150")
    ax.set_title("detector quality (no-detector sessions score AP 0)", loc="left", fontsize=12)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    bx.set_xticks(range(len(ARMS) + 1), ["g3 (today)", *[LABELS[a] for a in ARMS]], rotation=20, ha="right")
    bx.set_ylabel("sessions with no detector at vote 150 (%)")
    bx.set_title("sessions that found no positive", loc="left", fontsize=12)
    bx.grid(axis="y", color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, loc="lower center", ncol=2)
    fig.text(0.99, 0.01, "whiskers = 2 paired SE", ha="right", color=SOFT, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = study / "figures" / "ap_gain.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def ap_by_votes(study: Path) -> Path:
    q = rows(study)
    ts = [25, 50, 100, 150]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for pool, (label, color) in POOLS.items():
        for arm, ls in (("g6", "--"), ("g3g20d16", "-")):
            got = {int(r["t"]): r for r in q if r["pool"] == pool and r["arm"] == arm}
            ys = [float(got[t]["delta_vs_g3"]) for t in ts]
            es = [2 * float(got[t]["se"]) for t in ts]
            ax.errorbar(
                ts, ys, yerr=es, color=color, ls=ls, lw=2, marker="o", ms=5, capsize=3, label=f"{label}, {LABELS[arm]}"
            )
    ax.axhline(0, color=SOFT, lw=1)
    ax.set_xlabel("votes")
    ax.set_ylabel("AP gain over g3 (2 paired SE)")
    ax.set_xticks(ts)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    out = study / "figures" / "ap_by_votes.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--study", type=Path, default=STUDY)
    args = ap.parse_args()
    for p in (ap_gain(args.study), ap_by_votes(args.study)):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
