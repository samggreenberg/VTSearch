#!/usr/bin/env python
"""#4776's figures, from ``handover_4776.py``'s ``curves.csv``.

* ``fig_find_by_vote.png`` - what a Find returns at every vote on FHIBE's faces (1024 px): today,
  the centroid kept past the quota, the per-click oracle of the two (from ``handover_4776.py``), and
  the arm whose head starts from the centroid (``prior_4776.py``). One row per starting-photo count,
  one column per preset.
* ``fig_coco_guard.png`` - the same four on COCO Better's Binary Photo opening, one panel per preset.
* ``fig_ap_by_vote.png`` - AP at every vote at beta 1: today's head, the centroid, and the arm's head.

Colours are the validated categorical slots 1-4 in fixed order; every line is labelled at its end
(slides/STYLE.md: no leader lines).

Usage::

    python handover_4776_figures.py --curves <curves.csv> [--curves ...] [--arm <prior_curves.csv> ...] --out <dir>
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
    """Push labels apart by at least *gap*, keeping their mean where it was."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    shift = (sum(out) - sum(ys)) / max(len(ys), 1)
    return [y - shift for y in out]


Series = tuple[pd.DataFrame, str, str, str, str]  # (frame with vote + column, column, label, colour, style)


def _draw(ax, series: list[Series]) -> None:
    """Draw each series and label it at its end, the labels spread by a share of the axis range."""
    ends = []
    for frame, col, label, color, ls in series:
        if frame.empty:
            continue
        f = frame.sort_values("vote")
        ax.plot(f["vote"], f[col], color=color, lw=2, ls=ls)
        ends.append((float(f[col].iloc[-1]), label, color))
    ax.set_xlim(0, 150)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    lo, hi = ax.get_ylim()
    for (_y, label, color), yy in zip(ends, _spread([e[0] for e in ends], 0.06 * (hi - lo))):
        ax.text(153, yy, label, color=color, va="center", fontsize=8, clip_on=False)


def _rule(c: pd.DataFrame, setname: str, rule: str) -> pd.DataFrame:
    return c[(c["set"] == setname) & (c["rule"] == rule)]


def _arm(p: pd.DataFrame | None, setname: str, metric: str) -> pd.DataFrame:
    if p is None:
        return pd.DataFrame()
    return p[(p["set"] == setname) & (p["metric"] == metric)]


def _find_series(c: pd.DataFrame, p: pd.DataFrame | None, setname: str) -> list[Series]:
    return [
        (_rule(c, setname, "today"), "F", "today", SLOTS[0], "-"),
        (_rule(c, setname, "never"), "F", "centroid kept", SLOTS[1], "-"),
        (_arm(p, setname, "F"), "arm", "head from centroid", SLOTS[2], "-"),
        (_rule(c, setname, "oracle"), "F", "per-click oracle", SLOTS[3], "--"),
    ]


def fig_find(c: pd.DataFrame, p: pd.DataFrame | None, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharex=True, sharey=True)
    for r, k in enumerate((1, 4)):
        for j, (tag, title) in enumerate(BETAS):
            ax = axes[r, j]
            _draw(ax, _find_series(c, p, f"k{k}-b{tag}-1024"))
            ax.set_title(f"{title}, {'1 starting photo' if k == 1 else '4 starting photos'}", fontsize=10)
            if j == 0:
                ax.set_ylabel("Find's F-beta (withheld half)")
            if r == 1:
                ax.set_xlabel("vote")
    fig.suptitle("FHIBE faces, 1024 px: what a Find returns at every vote (562 people)")
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    fig.savefig(out / "fig_find_by_vote.png", dpi=150)
    plt.close(fig)


def fig_coco(c: pd.DataFrame, p: pd.DataFrame | None, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    for ax, (tag, title) in zip(axes, BETAS):
        _draw(ax, _find_series(c, p, f"coco-b{tag}"))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("vote")
    axes[0].set_ylabel("Find's F-beta (withheld half)")
    fig.suptitle("COCO Better, Binary Photo opening: what a Find returns at every vote (720 runs a preset)")
    fig.tight_layout(rect=(0, 0, 0.93, 0.93))
    fig.savefig(out / "fig_coco_guard.png", dpi=150)
    plt.close(fig)


def fig_ap(c: pd.DataFrame, p: pd.DataFrame | None, out: Path) -> None:
    panels = [
        ("k1-b1-1024", "faces, 1 starting photo"),
        ("k4-b1-1024", "faces, 4 starting photos"),
        ("coco-b1", "COCO Binary"),
    ]
    panels = [x for x in panels if (c["set"] == x[0]).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.9), squeeze=False)
    for ax, (setname, title) in zip(axes[0], panels):
        _draw(
            ax,
            [
                (_rule(c, setname, "today"), "AP", "head (today)", SLOTS[0], "-"),
                (_rule(c, setname, "never"), "AP", "centroid", SLOTS[1], "-"),
                (_arm(p, setname, "AP"), "arm", "head from centroid", SLOTS[2], "-"),
            ],
        )
        ax.set_title(f"{title}, β 1", fontsize=10)
        ax.set_xlabel("vote")
    axes[0][0].set_ylabel("AP of the withheld half")
    fig.suptitle("Ranking at every vote: today's head, the Goods' centroid, and the head started from it")
    fig.tight_layout(rect=(0, 0, 0.92, 0.92))
    fig.savefig(out / "fig_ap_by_vote.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curves", type=Path, action="append", required=True, help="handover_4776.py curves.csv")
    ap.add_argument("--arm", type=Path, action="append", default=[], help="prior_4776.py prior_curves.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    _style()
    c = pd.concat([pd.read_csv(x) for x in args.curves], ignore_index=True)
    p = pd.concat([pd.read_csv(x) for x in args.arm], ignore_index=True) if args.arm else None
    args.out.mkdir(parents=True, exist_ok=True)
    if c["set"].str.startswith("k").any():
        fig_find(c, p, args.out)
    if c["set"].str.startswith("coco").any():
        fig_coco(c, p, args.out)
    fig_ap(c, p, args.out)
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
