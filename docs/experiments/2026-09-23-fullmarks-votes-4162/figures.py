#!/usr/bin/env python3
"""Figures for the #4162 vote curves, from measurements/.

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
MEAS = HERE / "measurements" / "tier-s"
FIGS = HERE / "figures"
DPI = 130

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
#: Fixed categorical slots, in order; the exemplar is the muted baseline, not a series.
ARMS = {
    "a0_exemplar": ("exemplar only", MUTED, (0, (3, 3))),
    "a1_max": ("max over templates", "#2a78d6", "-"),
    "a2_repick": ("re-pick one template", "#eb6834", "-"),
    "a5_mlp": ("match-stat MLP", "#1baf7a", "-"),
    "a6_stoplist": ("stop-list from Bads", "#eda100", "-"),
    "a4r_siglip_sift": ("SigLIP SVM → SIFT", "#e87ba4", "-"),
    "a3s_production": ("app today (VLAD SVM, top 50)", "#4a3aa7", "-"),
}


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for f in [MEAS / "rows.csv", *sorted(MEAS.glob("rows-a6-*.csv"))]:
        with f.open(encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def curves() -> None:
    """Closed-loop positives found, and shared-sequence AP on the remainder, against votes."""
    rows = _rows()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), dpi=DPI)
    panels = (
        ("closed", "found", "positives found (mean per class)", "Each arm picks its own votes"),
        ("shared", "ap", "AP on the unlabelled remainder", "Every arm trains on the same votes"),
    )
    for ax, (readout, key, ylabel, title) in zip(axes, panels):
        _style(ax)
        for arm, (label, color, ls) in ARMS.items():
            sel = [r for r in rows if r["readout"] == readout and r["arm"] == arm]
            if not sel:
                continue
            vs = sorted({int(r["v"]) for r in sel if readout == "closed" or int(r["v"]) <= 20})
            ys = []
            for v in vs:
                xs = [float(r[key]) for r in sel if int(r["v"]) == v]
                xs = [x for x in xs if not np.isnan(x)]
                ys.append(np.mean(xs))
            ax.plot(vs, ys, color=color, linestyle=ls, linewidth=2, marker="o", markersize=4, label=label)
        ax.set_xticks([0, 3, 5, 10, 20, 40] if readout == "closed" else [0, 3, 5, 10, 20])
        ax.set_xlabel("votes", color=INK, fontsize=9)
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    axes[1].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    for h, lab in zip(h2, l2):
        if lab not in labels:
            handles.append(h)
            labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "vote_curves.png")
    plt.close(fig)


def paired() -> None:
    """Per class: shared-sequence AP at 10 votes, each arm against the control."""
    rows = [r for r in _rows() if r["readout"] == "shared" and int(r["v"]) == 10]
    ap = {(r["arm"], r["class_id"]): float(r["ap"]) for r in rows}
    classes = sorted({c for (a, c) in ap if a == "a1_max" and not np.isnan(ap[(a, c)])})
    arms = [
        a for a in ("a0_exemplar", "a2_repick", "a5_mlp", "a6_stoplist", "a4r_siglip_sift") if (a, classes[0]) in ap
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=DPI)
    _style(ax)
    rng = np.random.default_rng(0)
    for i, arm in enumerate(arms):
        label, color, _ = ARMS[arm]
        d = np.array([ap[(arm, c)] - ap[("a1_max", c)] for c in classes if (arm, c) in ap])
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=16, color=color, alpha=0.8, linewidths=0)
        ax.plot([i - 0.28, i + 0.28], [d.mean()] * 2, color=INK, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(arms)), [ARMS[a][0] for a in arms], fontsize=8.5, color=INK)
    ax.set_ylabel("AP minus max-over-templates", color=INK, fontsize=9)
    ax.set_title("After 10 shared votes, per class (bar = mean)", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "paired_v10.png")
    plt.close(fig)


if __name__ == "__main__":
    curves()
    paired()
