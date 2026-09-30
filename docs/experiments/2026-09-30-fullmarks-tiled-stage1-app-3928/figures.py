#!/usr/bin/env python3
"""Figures for the #3928 end-to-end replay, from measurements/.

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
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = (
    ("replay", "the app with tiles (this change)", "#2a78d6", "-"),
    ("a1_max", "SIFT verifies every page", MUTED, (0, (3, 3))),
    ("a3s_production", "the app before (page VLAD top 50)", "#eb6834", "-"),
)


def _num(x: str) -> float:
    return float("nan") if x in ("", "nan", None) else float(x)


def _load(tier: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    replay = MEAS / f"replay-{tier}.csv"
    if replay.exists():
        with replay.open(encoding="utf-8") as fh:
            out["replay"] = list(csv.DictReader(fh))
    with (MEAS / f"curves-{tier}.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["readout"] == "closed":
                out.setdefault(r["arm"], []).append(r)
    return out


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def curves() -> None:
    tiers = [t for t in ("s", "m") if (MEAS / f"replay-{t}.csv").exists()]
    fig, axes = plt.subplots(len(tiers), 2, figsize=(10.4, 3.6 * len(tiers)), dpi=DPI, squeeze=False)
    for row, tier in enumerate(tiers):
        data = _load(tier)
        classes = {r["class_id"] for r in data["replay"]}
        for col, (key, ylabel) in enumerate((("found", "positives found"), ("ap", "AP on the unlabelled remainder"))):
            ax = axes[row][col]
            _style(ax)
            for name, label, color, ls in SERIES:
                rows = [r for r in data.get(name, []) if r["class_id"] in classes]
                if not rows:
                    continue
                vs = sorted({int(r["v"]) for r in rows if int(r["v"]) <= 20})
                ys = [np.nanmean([_num(r[key]) for r in rows if int(r["v"]) == v]) for v in vs]
                ax.plot(vs, ys, color=color, linestyle=ls, linewidth=2, marker="o", markersize=3.5, label=label)
            ax.set_xticks([0, 3, 5, 10, 20])
            ax.set_xlabel("votes", color=INK, fontsize=9)
            ax.set_ylabel(ylabel, color=INK, fontsize=9)
            ax.set_title(f"tier {tier}: {ylabel} (closed loop)", loc="left", color=INK, fontsize=10)
            if key == "ap":
                ax.set_ylim(0, 1)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "closed_loop.png")
    plt.close(fig)


def per_class(tier: str = "s", v: int = 10) -> None:
    """Per class at *v* votes: the app with tiles against verifying every page."""
    data = _load(tier)
    rep = {r["class_id"]: _num(r["ap"]) for r in data["replay"] if int(r["v"]) == v}
    exh = {r["class_id"]: _num(r["ap"]) for r in data.get("a1_max", []) if int(r["v"]) == v}
    pairs = sorted((c, rep[c], exh[c]) for c in rep if c in exh and not np.isnan(rep[c]) and not np.isnan(exh[c]))
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=DPI)
    _style(ax)
    ax.scatter([p[2] for p in pairs], [p[1] for p in pairs], s=20, color="#2a78d6", linewidths=0)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
    ax.set_xlabel("AP, SIFT verifies every page", color=INK, fontsize=9)
    ax.set_ylabel("AP, the app with tiles", color=INK, fontsize=9)
    ax.set_title(f"tier {tier}, {v} votes, per class", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / f"per_class_{tier}.png")
    plt.close(fig)


if __name__ == "__main__":
    curves()
    per_class("s")
    if (MEAS / "replay-m.csv").exists():
        per_class("m")
