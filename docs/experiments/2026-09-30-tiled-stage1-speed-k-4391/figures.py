#!/usr/bin/env python3
"""Figures for #4391 (tiled Stage 1 on the GPU; shortlist-size arms), from measurements/.

python figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAS, FIGS, DPI = HERE / "measurements", HERE / "figures", 130
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARMS = (
    ("before", "K = 1,000, Stage 1 on the CPU (before)", MUTED, (0, (3, 3))),
    ("fixed", "K = 1,000, Stage 1 on the GPU (ships)", "#2a78d6", "-"),
    ("adaptive", "adaptive K (grow while the tail verifies)", "#eb6834", "-"),
    ("cap", "K = 4,000 (reference)", "#1baf7a", "-"),
)


def _rows(tier: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with (MEAS / f"replay-{tier}.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["policy"], []).append(r)
    with (MEAS / f"before-{tier}.csv").open(encoding="utf-8") as fh:
        out["before"] = list(csv.DictReader(fh))
    return out


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), dpi=DPI)
    for row, tier in enumerate(("s", "m")):
        data = _rows(tier)
        for col, (key, ylabel) in enumerate(
            (("ap", "AP on the unlabelled remainder"), ("retrain_s", "retrain seconds, median"))
        ):
            ax = axes[row][col]
            _style(ax)
            for name, label, color, ls in ARMS:
                rows = data.get(name, [])
                vs = sorted({int(r["v"]) for r in rows})
                agg = np.nanmean if key == "ap" else np.nanmedian
                ys = [
                    agg([float(r[key]) if r[key] not in ("", "nan") else np.nan for r in rows if int(r["v"]) == v])
                    for v in vs
                ]
                ax.plot(vs, ys, color=color, linestyle=ls, linewidth=2, marker="o", markersize=3.5, label=label)
            ax.set_xticks([0, 3, 5, 10, 20])
            ax.set_xlabel("votes", color=INK, fontsize=9)
            ax.set_ylabel(ylabel, color=INK, fontsize=9)
            ax.set_title(f"tier {tier}: {ylabel} (closed loop)", loc="left", color=INK, fontsize=10)
            if key == "ap":
                ax.set_ylim(0.5, 1)
            else:
                ax.set_ylim(0, None)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "arms.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
