"""Figures for the #3928 cached-Stage-1 report, from ``measurements/`` alone.

python figures.py     # writes fig_width.png and fig_cost.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"
KS = (100, 500, 1000, 2000)
WIDTHS = ("raw", "d512", "d256", "d128")
COLOUR = {"raw": "#333333", "d512": "#2f6f9f", "d256": "#5fa0cf", "d128": "#9cc3de"}
LABEL = {"raw": "8,192 (raw)", "d512": "512", "d256": "256", "d128": "128"}


def _rows(tier: str) -> list[dict[str, str]]:
    path = MEAS / f"rows_{tier}.csv"
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []


def _mean(rows: list[dict[str, str]], cell: str, method: str, field: str = "ap") -> float:
    sel = [float(r[field]) for r in rows if r["cell"] == cell and r["method"] == method and r[field] != ""]
    return float(np.mean(sel)) if sel else float("nan")


def fig_width() -> None:
    """AP against shortlist size, one line per stored width, per tier."""
    tiers = [t for t in ("s", "m") if _rows(t)]
    fig, axes = plt.subplots(1, len(tiers), figsize=(5.2 * len(tiers), 3.9), sharey=True, squeeze=False)
    for ax, tier in zip(axes[0], tiers):
        rows = _rows(tier)
        cells = [c for c in WIDTHS if any(r["cell"] == c for r in rows)]
        for cell in cells:
            ys = [_mean(rows, cell, f"{cell}_{k}_sift") for k in KS]
            ax.plot(KS, ys, marker="o", color=COLOUR[cell], label=f"{LABEL[cell]} dims")
        exhaustive = _mean(rows, cells[0], "sift_exhaustive")
        ax.axhline(exhaustive, color="#b03030", ls="--", lw=1, label="SIFT, every page")
        ax.set_title(f"tier {tier}", fontsize=10)
        ax.set_xlabel("shortlist size K")
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0][0].set_ylabel("mean AP over 23 classes")
    axes[0][-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "35x smaller than the raw tile: better than it at 5,000 pages, and at 50,000 the width starts to matter",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(HERE / "fig_width.png", dpi=150)


def fig_cost() -> None:
    """What a width costs to store and to search."""
    timings: list[dict] = []
    for tier in ("s", "m"):
        path = MEAS / f"timings_{tier}.json"
        if path.exists():
            for row in json.loads(path.read_text(encoding="utf-8")):
                timings.append({**row, "tier": tier})
    if not timings:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for tier, marker in (("s", "o"), ("m", "s")):
        sel = [t for t in timings if t["tier"] == tier]
        if not sel:
            continue
        dims = [t["dim"] for t in sel]
        ax1.plot(dims, [t["gib"] for t in sel], marker=marker, label=f"tier {tier}", color=COLOUR["d512"])
        ax2.plot(dims, [t["search_ms_mean"] for t in sel], marker=marker, label=f"tier {tier}", color=COLOUR["d256"])
    for ax, ylabel, title in (
        (ax1, "cell size (GiB)", "storage"),
        (ax2, "ms per query", "search"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("stored dimensions")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_cost.png", dpi=150)


if __name__ == "__main__":
    fig_width()
    fig_cost()
    print("wrote fig_width.png and fig_cost.png")
