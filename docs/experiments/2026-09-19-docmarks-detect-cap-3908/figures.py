"""Figures for the detection-cap study (#3908).

python figures.py --results /expscratch/sgreenberg/docmarks/detect-cap-3908 --out .
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Cap arms in sweep order, as (directory suffix, label, megapixels-for-x-axis).
ARMS = (
    ("025mp", "0.25", 0.25),
    ("05mp", "0.5", 0.5),
    ("1mp", "1", 1.0),
    ("2mp", "2", 2.0),
    ("4mp", "4", 4.0),
    ("uncapped", "native", 8.0),
)
BUDGETS = (8192, 1024)
INK, GRID = "#18212e", "#d9dee6"
COLOR = {8192: "#2c4f9e", 1024: "#b8621b"}


def arm_ap(base: Path, budget: int, key: str) -> float | None:
    f = base / f"b{budget}-c{key}" / "rows.csv"
    if not f.exists():
        return None
    rows = [r for r in csv.DictReader(f.open()) if r["method"] == "sift"]
    return statistics.mean(float(r["ap"]) for r in rows) if rows else None


def arm_keypoints(base: Path, budget: int, key: str) -> int | None:
    cand = [
        p
        for p in (base / "logs").glob(f"cap-b{budget}-{key}-*.out")
        if "keypoints/page" in p.read_text(errors="ignore")
    ]
    if not cand:
        return None
    t = max(cand, key=lambda p: p.stat().st_mtime).read_text(errors="ignore")
    m = re.search(r"keypoints/page median (\d+)", t)
    return int(m.group(1)) if m else None


def fig_interaction(base: Path, out: Path) -> None:
    """AP against cap, one line per keypoint budget: the interaction, not a curve."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for budget in BUDGETS:
        xs, ys, labels = [], [], []
        for key, label, mp in ARMS:
            ap = arm_ap(base, budget, key)
            if ap is not None:
                xs.append(mp), ys.append(ap), labels.append(label)
        ax.plot(xs, ys, "o-", color=COLOR[budget], label=f"budget {budget:,} keypoints", lw=2, ms=6)
        best = max(range(len(ys)), key=lambda i: ys[i])
        ax.annotate(
            f"{ys[best]:.2f}",
            (xs[best], ys[best]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color=COLOR[budget],
            fontweight="bold",
        )
    ax.axvline(2.0, color=GRID, lw=1.2, ls="--", zorder=0)
    ax.text(2.06, 0.02, "shipped cap", color="#5a6576", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([a[2] for a in ARMS]), ax.set_xticklabels([a[1] for a in ARMS])
    # A log axis relabels its minor ticks too, which overprints the cap labels.
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("detection cap (megapixels; 'native' = uncapped)"), ax.set_ylabel("mean AP over 23 classes")
    ax.set_title("The cap's optimum moves with the keypoint budget", color=INK)
    ax.set_ylim(0, 1), ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout(), fig.savefig(out / "fig_cap_budget.png", dpi=150), plt.close(fig)


def fig_cost(bench: Path, out: Path) -> None:
    """What each cap costs: detect time spread and peak RSS."""
    d = json.loads(bench.read_text())
    arms = list(reversed(d["arms"]))  # smallest cap first
    labels = [a["label"] for a in arms]
    med = [statistics.median(r["detect_ms"] for r in a["rows"]) for a in arms]
    mx = [max(r["detect_ms"] for r in a["rows"]) for a in arms]
    rss = [a["peak_rss_mb"] / 1024 for a in arms]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = range(len(arms))
    ax1.bar([i - 0.2 for i in x], med, 0.4, color="#2c4f9e", label="median page")
    ax1.bar([i + 0.2 for i in x], mx, 0.4, color="#9a6414", label="worst page")
    ax1.set_yscale("log"), ax1.set_ylabel("detect time (ms, log)"), ax1.set_title("Detection cost per page", color=INK)
    ax2.bar(x, rss, 0.55, color="#7a2d2d")
    ax2.set_ylabel("peak RSS (GB)"), ax2.set_title("Peak memory, 150-page sample", color=INK)
    for ax in (ax1, ax2):
        (
            ax.set_xticks(list(x)),
            ax.set_xticklabels(labels),
            ax.grid(axis="y", color=GRID, lw=0.8),
            ax.set_axisbelow(True),
        )
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax1.legend(frameon=False, fontsize=9)
    for i, v in enumerate(rss):
        ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    fig.tight_layout(), fig.savefig(out / "fig_cost.png", dpi=150), plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("/expscratch/sgreenberg/docmarks/detect-cap-3908"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    fig_interaction(args.results, args.out)
    fig_cost(args.results / "bench-8192.json", args.out)
    print(f"wrote {args.out}/fig_cap_budget.png and fig_cost.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
