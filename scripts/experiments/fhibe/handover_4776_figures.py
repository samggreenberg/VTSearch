#!/usr/bin/env python
"""#4776's figures, from ``handover_4776.py``'s ``curves.csv``.

* ``fig_find_by_vote.png`` - what a Find returns at every vote on FHIBE's faces (1024 px), under
  today's hand-over, no hand-over, the rule priced (``--judge``) and the per-click oracle. One row per
  starting-photo count, one column per preset.
* ``fig_coco_guard.png`` - the same four on COCO Better's Binary Photo opening, one panel per preset.
* ``fig_prior_ap.png`` - AP at every vote at beta 1: the head (today, from the quota), the centroid,
  and the heads started from the centroid (``prior_ap_s3``, ``prior_ap_s10``), faces from one and
  four photos and COCO. Ranking only: these heads have no line yet.

Colours are the validated categorical slots 1-4 in fixed order; every line is labelled at its end
(slides/STYLE.md: no leader lines).

Usage::

    python handover_4776_figures.py --curves <curves.csv> [--curves <coco curves.csv>] --judge 'judge:mrr' --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
BETAS = (("025", "β ¼"), ("1", "β 1"), ("4", "β 4"))


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


def _lines(ax, c: pd.DataFrame, setname: str, series: list[tuple[str, str, str, str]], col: str, gap: float) -> None:
    """Draw *series* ``(rule, label, colour, style)`` of *setname* and label each at its end."""
    ends = []
    for rule, label, color, ls in series:
        s = c[(c["set"] == setname) & (c["rule"] == rule)].sort_values("vote")
        if s.empty:
            continue
        ax.plot(s["vote"], s[col], color=color, lw=2, ls=ls)
        ends.append((float(s[col].iloc[-1]), label, color))
    for (y, label, color), yy in zip(ends, _spread([e[0] for e in ends], gap)):
        ax.text(152, yy, label, color=color, va="center", fontsize=8)
    ax.set_xlim(0, 150)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)


def fig_find(c: pd.DataFrame, judge: str, out: Path) -> None:
    series = [
        ("today", "today", SLOTS[0], "-"),
        ("never", "centroid kept", SLOTS[1], "-"),
        (judge, judge.replace("judge:", "judge "), SLOTS[2], "-"),
        ("oracle", "oracle", SLOTS[3], "--"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for r, k in enumerate((1, 4)):
        for j, (tag, title) in enumerate(BETAS):
            ax = axes[r, j]
            _lines(ax, c, f"k{k}-b{tag}-1024", series, "F", 0.035)
            ax.set_title(f"{title}, {'1 starting photo' if k == 1 else '4 starting photos'}", fontsize=10)
            if j == 0:
                ax.set_ylabel("Find's F-beta (withheld half)")
            if r == 1:
                ax.set_xlabel("vote")
    fig.suptitle("FHIBE faces, 1024 px: what a Find returns at every vote, by hand-over rule (562 people)")
    fig.tight_layout(rect=(0, 0, 0.95, 0.96))
    fig.savefig(out / "fig_find_by_vote.png", dpi=150)
    plt.close(fig)


def fig_coco(c: pd.DataFrame, judge: str, out: Path) -> None:
    series = [
        ("today", "today", SLOTS[0], "-"),
        ("never", "centroid kept", SLOTS[1], "-"),
        (judge, judge.replace("judge:", "judge "), SLOTS[2], "-"),
        ("oracle", "oracle", SLOTS[3], "--"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (tag, title) in zip(axes, BETAS):
        _lines(ax, c, f"coco-b{tag}", series, "F", 0.025)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("vote")
    axes[0].set_ylabel("Find's F-beta (withheld half)")
    fig.suptitle("COCO Better, Binary Photo opening: what a Find returns at every vote (720 runs a preset)")
    fig.tight_layout(rect=(0, 0, 0.95, 0.94))
    fig.savefig(out / "fig_coco_guard.png", dpi=150)
    plt.close(fig)


def fig_prior(c: pd.DataFrame, out: Path) -> None:
    series = [
        ("today", "head (today)", SLOTS[0], "-"),
        ("never", "centroid", SLOTS[1], "-"),
        ("prior_ap_s3", "head from centroid, s=3", SLOTS[2], "-"),
        ("prior_ap_s10", "head from centroid, s=10", SLOTS[3], "--"),
    ]
    panels = [
        ("k1-b1-1024", "faces, 1 starting photo"),
        ("k4-b1-1024", "faces, 4 starting photos"),
        ("coco-b1", "COCO Binary"),
    ]
    panels = [p for p in panels if (c["set"] == p[0]).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8), sharey=True, squeeze=False)
    for ax, (setname, title) in zip(axes[0], panels):
        _lines(ax, c, setname, series, "AP", 0.03)
        ax.set_title(f"{title}, β 1", fontsize=10)
        ax.set_xlabel("vote")
    axes[0][0].set_ylabel("AP of the withheld half")
    fig.suptitle("Ranking at every vote: the head, the centroid, and heads started from the centroid")
    fig.tight_layout(rect=(0, 0, 0.93, 0.93))
    fig.savefig(out / "fig_prior_ap.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curves", type=Path, action="append", required=True)
    ap.add_argument("--judge", default="judge:mrr")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    _style()
    c = pd.concat([pd.read_csv(p) for p in args.curves], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if c["set"].str.startswith("k").any():
        fig_find(c, args.judge, args.out)
    if c["set"].str.startswith("coco").any():
        fig_coco(c, args.judge, args.out)
    fig_prior(c, args.out)
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
