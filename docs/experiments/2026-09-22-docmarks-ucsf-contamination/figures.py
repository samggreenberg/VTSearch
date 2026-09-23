#!/usr/bin/env python3
"""Figures for the UCSF un-banded contamination check (#3922).

    python figures.py            # regenerate every figure into figures/

Inputs are ``measurements/`` beside this script (written by ``measure.py``),
so the figures rebuild without the GRID or the corpus.
"""

from __future__ import annotations

import collections
import math
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"
FIGS = HERE / "figures"
DPI = 130

# Validated categorical slots 1 and 2 (CVD dE 24.7 protan, 33.6 normal).
V40 = "#eb6834"
V41 = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bounds() -> None:
    """Zero hits in every stratum: what each sample size rules out (3/n)."""
    rows = [json.loads(line) for line in (MEAS / "verdicts.jsonl").read_text().splitlines()]
    n = collections.Counter()
    hits = collections.Counter()
    for r in rows:
        if r["arm"] == "control":
            continue
        if r["arm"] == "ranked":
            keys = ["SigLIP top-30 · all"]
        else:
            food = "Food" if r["industry"] == "Food" else "other industries"
            keys = [f"random · {food}", f"random · {r['class_id'].split('logo_')[1]}"]
        for key in keys:
            n[key] += 1
            hits[key] += r["verdict"] == "good"
    order = [
        "random · Food",
        "random · other industries",
        "SigLIP top-30 · all",
        "random · bat_leaf",
        "random · bw_oval_emblem",
        "random · p_lorillard_crest",
        "random · rjr_script",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=DPI)
    _style(ax)
    for y, key in enumerate(order):
        ub = 100 * 3 / n[key]
        ax.plot([0, ub], [y, y], color=V41, linewidth=2, solid_capstyle="round")
        ax.plot([0], [y], "o", color=V41, markersize=8)
        shown = math.ceil(ub * 10) / 10  # an upper bound rounds up
        ax.text(ub + 0.15, y, f"0 of {n[key]} · ≤{shown:.1f}%", va="center", color=INK, fontsize=9)
        assert hits[key] == 0, key
    ax.set_yticks(range(len(order)), order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 8.5)
    ax.set_xlabel("contamination rate: observed (dot) to 95% upper bound (line end), %", color=INK, fontsize=9)
    ax.set_title("No UCSF mark on any sampled un-banded page (tier m)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "bounds.png")
    plt.close(fig)


def shortcut() -> None:
    """Mark-blind controls on each UCSF class's headline pool, v4.0 vs v4.1."""
    rows = [r for r in json.loads((MEAS / "pools.json").read_text()) if r["tier"] == "m"]
    names = [r["class_id"].split("logo_")[1] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=DPI, sharey=True)
    for ax, key, title in (
        (axes[0], "source_prior_ap", "rank UCSF pages first"),
        (axes[1], "industry_prior_ap", "rank UCSF Tobacco pages first"),
    ):
        _style(ax)
        for y, r in enumerate(rows):
            a, b = r["v4.0"][key], r["v4.1"][key]
            ax.plot([a, b], [y, y], color=GRID, linewidth=2, zorder=1)
            ax.plot([a], [y], "o", color=V40, markersize=8, zorder=2, label="v4.0" if y == 0 else None)
            ax.plot([b], [y], "o", color=V41, markersize=8, zorder=3, label="v4.1" if y == 0 else None)
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("AP of a control that ignores the mark", color=INK, fontsize=9)
    axes[0].set_yticks(range(len(names)), names, fontsize=9)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK)
    fig.suptitle(
        "v4.1 removes the source shortcut; the letterhead-style one remains", x=0.01, ha="left", color=INK, fontsize=11
    )
    fig.tight_layout()
    fig.savefig(FIGS / "shortcut.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    bounds()
    shortcut()
