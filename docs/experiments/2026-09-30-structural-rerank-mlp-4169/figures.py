#!/usr/bin/env python3
"""Figures for #4169 (match-statistic MLP vs inliers in the structural re-rank), from measurements/.

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
MEAS = HERE / "measurements"
FIGS = HERE / "figures"
DPI = 130

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
CORPORA = (
    ("fullmarks-s", "FullMarks tier s (documents)"),
    ("fullmarks-m", "FullMarks tier m (documents)"),
    ("belgalogos", "BelgaLogos (photos)"),
)
#: Exhaustive arms solid, the app's VLAD-shortlist arms dashed; one colour per scorer.
ARMS = {
    "a1_max": ("inliers, every page", "#2a78d6", "-"),
    "a5_mlp": ("MLP, every page", "#1baf7a", "-"),
    "a3s_cold": ("inlier gate, app path (VLAD top 50)", "#2a78d6", (0, (4, 2))),
    "a3s_production": ("MLP, app path as shipped", "#1baf7a", (0, (4, 2))),
}


def _rows(corpus: str) -> list[dict[str, str]]:
    with (MEAS / corpus / "rows.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(x: str) -> float:
    return float("nan") if x in ("", None) else float(x)


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def curves(key: str, ylabel: str, name: str) -> None:
    """Shared-sequence *key* on the unlabelled remainder against votes, one panel per corpus."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=DPI, sharey=True)
    for ax, (corpus, title) in zip(axes, CORPORA):
        _style(ax)
        rows = [r for r in _rows(corpus) if r["readout"] == "shared"]
        for arm, (label, color, ls) in ARMS.items():
            sel = [r for r in rows if r["arm"] == arm and int(r["left"]) > 0]
            if not sel:
                continue
            vs = sorted({int(r["v"]) for r in sel})
            ys = [np.nanmean([_num(r[key]) for r in sel if int(r["v"]) == v]) for v in vs]
            ax.plot(vs, ys, color=color, linestyle=ls, linewidth=2, marker="o", markersize=3.5, label=label)
        ax.set_xticks([0, 3, 5, 10, 20, 40])
        ax.set_xlabel("votes", color=INK, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel(ylabel, color=INK, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / name)
    plt.close(fig)


def paired() -> None:
    """Per class, after 10 shared votes: MLP minus inliers, AP and F1, every page."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=DPI, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes, (("ap", "AP"), ("f1", "F1 (accept decision)"))):
        _style(ax)
        for i, (corpus, label) in enumerate(CORPORA):
            rows = [r for r in _rows(corpus) if r["readout"] == "shared" and int(r["v"]) == 10 and int(r["left"]) > 0]
            val = {(r["arm"], r["class_id"]): _num(r[key]) for r in rows}
            d = np.array(
                [
                    val[("a5_mlp", c)] - val[("a1_max", c)]
                    for (a, c) in val
                    if a == "a1_max" and ("a5_mlp", c) in val and not np.isnan(val[("a5_mlp", c)] - val[(a, c)])
                ]
            )
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=16, color="#1baf7a", alpha=0.8, linewidths=0)
            ax.plot([i - 0.28, i + 0.28], [d.mean()] * 2, color=INK, linewidth=2)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(range(len(CORPORA)), [c[1].replace(" (", "\n(") for c in CORPORA], fontsize=8, color=INK)
        ax.set_title(f"{title}: MLP minus inliers, per class", loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("difference after 10 votes (bar = mean)", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "paired_v10.png")
    plt.close(fig)


if __name__ == "__main__":
    curves("ap", "AP on the unlabelled remainder", "ap_curves.png")
    curves("f1", "F1 of the accept decision", "f1_curves.png")
    paired()
