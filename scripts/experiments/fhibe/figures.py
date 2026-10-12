#!/usr/bin/env python
"""The FHIBE baseline's curve figure (#4699): what Find returns at every vote, per arm and example count.

Reads ``curves.csv`` from ``analyze.py`` and writes ``fig_fbeta_by_vote.png``: one panel per example
count, one line per arm. Colour is the representation (photo blue, face orange), line style the stored
size (1024 solid, 640 dashed). The panels share the y axis so the gap between them reads directly.
Lines are labelled at their right ends, not through leader lines (slides/STYLE.md).

Usage::

    python figures.py --curves <analysis>/curves.csv --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
#: Representation -> colour (categorical slots 1 and 2, validated), size -> line style.
STYLE = {
    "fhibe_1024": ("#2a78d6", "-", "photo 1024"),
    "fhibe_640": ("#2a78d6", "--", "photo 640"),
    "fhibe_faces_1024": ("#eb6834", "-", "face 1024"),
    "fhibe_faces_640": ("#eb6834", "--", "face 640"),
}


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge end-label heights apart by at least *gap*, keeping their order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    curves = pd.read_csv(args.curves)
    labels = sorted(curves["K"].unique(), key=lambda k: int(str(k).lstrip("k")))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
        }
    )
    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 4.2), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], labels):
        g = curves[curves["K"] == k]
        ends = []
        for ds, (colour, ls, name) in STYLE.items():
            d = g[g["dataset"] == ds].sort_values("t")
            if d.empty:
                continue
            ax.plot(d["t"], d["fbeta"], color=colour, ls=ls, lw=2)
            ends.append((float(d["fbeta"].iloc[-1]), name, colour))
        for y, (_, name, colour) in zip(_spread([e[0] for e in ends], 0.05), ends):
            ax.text(152, y, name, color=INK2, va="center", fontsize=9)
            ax.plot([149.5], [y], marker="s", ms=4, color=colour, clip_on=False)
        n_people = int(g["n"].max()) if not g.empty else 0
        ax.set_title(
            f"start from {str(k).lstrip('k')} photo{'s' if str(k) != 'k1' else ''} ({n_people} people)", color=INK
        )
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 1)
        ax.set_xlabel("votes (the starting photos are the first)")
        ax.grid(True, color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("F-beta of what Find returns (withheld half)")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig_fbeta_by_vote.png", dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out / 'fig_fbeta_by_vote.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
