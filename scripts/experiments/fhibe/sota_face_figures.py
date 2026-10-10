#!/usr/bin/env python
"""The face review's figures (#4762).

* ``fig_objective.png`` - what the session shows at every click: the objective (F-beta of the
  withheld set at the app's line, at the sessions' own beta), click 0 the example sort. One panel
  per starting-photo count; colour is the preset, line style the stored size.
* ``fig_find_by_vote.png`` - what a Find returns at every vote (Test, the export), where the dip
  is read; under it, the share of runs on the Goods' centroid and the share that has had a prompted
  spot check by then (#4496). Two rows rather than two axes.
* ``fig_strata.png`` - the objective after the check per stratum of the people, at beta 1, one
  photo and 1024 px, with +-1 SE; the dashed line is everyone.

Colours are the validated categorical slots 1-3 in fixed order (beta 1/4, 1, 4), every line
labelled at its end (slides/STYLE.md: no leader lines).

Usage::

    python sota_face_figures.py --analysis <dir> --tables <dir with find_by_vote.csv, strata.csv> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BETA = {0.25: ("#2a78d6", "β ¼"), 1.0: ("#eb6834", "β 1"), 4.0: ("#1baf7a", "β 4")}
SIZE = {"1024": "-", "640": "--"}
KS = (1, 4)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )


def _spread(ys: list[float], gap: float) -> list[float]:
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def _end_labels(ax, ends: list[tuple[float, float, str, str]], gap: float = 0.035) -> None:
    ys = _spread([y for _, y, _, _ in ends], gap)
    for (x, _y, text, color), y in zip(ends, ys):
        ax.text(x, y, text, color=color, va="center", fontsize=8)


def photos(k: int) -> str:
    return "1 starting photo" if k == 1 else f"{k} starting photos"


def fig_objective(analysis: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, k in zip(axes, KS):
        ends = []
        for size, ls in SIZE.items():
            path = analysis / f"k{k}-{size}-objective_by_click.csv"
            if not path.exists():
                continue
            d = pd.read_csv(path)
            for beta, (color, name) in BETA.items():
                c = d[d["beta"].astype(float) == beta].sort_values("t")
                if c.empty:
                    continue
                ax.plot(c["t"], c["fbeta"], color=color, ls=ls, lw=1.6)
                ends.append((c["t"].max() + 2, float(c["fbeta"].iloc[-1]), f"{name}, {size}", color))
        _end_labels(ax, ends)
        ax.set_title(photos(k), loc="left", fontsize=10, color=INK)
        ax.set_xlabel("click (0 = the example sort)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("objective: F-beta at the line, withheld half")
    fig.tight_layout()
    fig.savefig(out / "fig_objective.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_find_by_vote(tables: Path, out: Path, horizon: int) -> None:
    d = pd.read_csv(tables / "find_by_vote.csv")
    d = d[d["vote"] <= horizon]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for j, k in enumerate(KS):
        top, bottom = axes[0][j], axes[1][j]
        ends = []
        for size, ls in SIZE.items():
            for beta, (color, name) in BETA.items():
                c = d[(d["K"] == k) & (d["size"].astype(str) == size) & (d["beta"] == beta)].sort_values("vote")
                if c.empty:
                    continue
                top.plot(c["vote"], c["F"], color=color, ls=ls, lw=1.6)
                ends.append((horizon + 1, float(c["F"].iloc[-1]), f"{name}, {size}", color))
        _end_labels(top, ends)
        top.set_title(photos(k), loc="left", fontsize=10, color=INK)
        top.set_ylim(0, 1)
        top.grid(axis="y", color=GRID, lw=0.8)
        b1 = d[(d["K"] == k) & (d["size"].astype(str) == "1024") & (d["beta"] == 1.0)].sort_values("vote")
        bottom.plot(b1["vote"], b1["share_centroid"], color=INK2, lw=1.4)
        col = "share_prompted_by" if "share_prompted_by" in b1 else "share_prompt"
        bottom.plot(b1["vote"], b1[col], color=INK, lw=1.2, ls=":")
        if len(b1):
            _end_labels(
                bottom,
                [
                    (horizon + 1, float(b1["share_centroid"].iloc[-1]), "on the centroid", INK2),
                    (horizon + 1, float(b1[col].iloc[-1]), "had a spot check", INK),
                ],
                gap=0.18,
            )
        bottom.set_ylim(0, 1)
        bottom.set_xlabel("vote (the starting photos are the first votes)")
        bottom.grid(axis="y", color=GRID, lw=0.8)
    axes[0][0].set_ylabel("what a Find returns: F-beta")
    axes[1][0].set_ylabel("share of runs\n(β 1, 1024)")
    fig.tight_layout()
    fig.savefig(out / "fig_find_by_vote.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_strata(tables: Path, out: Path, k: int = 1, size: str = "1024", beta: float = 1.0) -> None:
    s = pd.read_csv(tables / "strata.csv", dtype={"level": str})
    s = s[(s["K"] == k) & (s["size"].astype(str) == size) & (s["beta"] == beta)]
    h = pd.read_csv(tables / "headline.csv")
    everyone = float(h[(h["K"] == k) & (h["size"].astype(str) == size) & (h["beta"] == beta)]["F_after_check"].iloc[0])
    panels = [
        ("age_bracket", "age"),
        ("pronoun", "pronoun"),
        ("skin_tone", "skin tone (FHIBE 0-5)"),
        ("ancestry", "ancestry"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        g = s[(s["stratum"] == col) & (s["people"] >= 5)].sort_values("level")
        x = range(len(g))
        ax.errorbar(list(x), g["F_after_check"], yerr=g["F_after_check_se"], fmt="o", color=BETA[1.0][0], ms=5,
                    elinewidth=1.2, capsize=0)  # fmt: skip
        ax.axhline(everyone, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{lv}\n(n={n})" for lv, n in zip(g["level"], g["people"])], fontsize=7)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel(f"objective after the check\n(β {beta:g}, {photos(k)}, {size} px)")
    fig.tight_layout()
    fig.savefig(out / "fig_strata.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--find-horizon", type=int, default=60, help="votes the Find figure shows (the dip is early)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _style()
    fig_objective(args.analysis, args.out)
    fig_find_by_vote(args.tables, args.out, args.find_horizon)
    fig_strata(args.tables, args.out)
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
