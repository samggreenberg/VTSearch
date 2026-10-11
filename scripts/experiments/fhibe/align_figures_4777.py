#!/usr/bin/env python
"""The alignment study's figures (#4777).

* ``fig_screen_by_roll.png`` - the screen (``align_screen_4777.py``): how well one starting photo's
  example sort ranks the person's other photos (mean AP, top row) and the share of starting photos
  that fail (AP < 0.1, bottom row), by how far the starting face is turned in the photo (the roll
  the 5-point fit removes), per arm. One column per stored size. Two rows rather than two axes.
* ``fig_session_gain.png`` - the sessions (``align_compare_4777.py``'s inputs): the paired
  difference, aligned minus plain, in the objective the session shows at every click (F-beta of the
  withheld set at the app's line, filled as ``by_click.py`` fills it), mean over the 562 people with
  a +-1 SE band. One panel per starting-photo count; colour is the preset, line style the size.

Colours are the validated categorical slots 1-3 in fixed order (slides/STYLE.md, as
``sota_face_figures.py``), every line labelled at its end.

Usage::

    python align_figures_4777.py screen --screen <screen dir> --out <dir>
    python align_figures_4777.py sessions --plain <analysis> --plain-baseline <dir> \\
        --arm roll=<analysis>,<baseline dir> --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sota_face_figures import BETA, INK2, GRID, SIZE, _end_labels, _style, photos  # noqa: E402

ARMS = {"faces": ("#2a78d6", "plain crop"), "faces_roll": ("#eb6834", "roll"), "faces_sim": ("#1baf7a", "similarity")}
BANDS = ["0-10", "10-20", "20-45", "45-90"]
BAND_LABELS = ["0–10°", "10–20°", "20–45°", "45–90°"]


def fig_screen(screen: Path, out: Path) -> None:
    df = pd.read_csv(screen / "screen_by_roll.csv")
    df = df[df["roll_band"].isin(BANDS)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex="col", sharey="row")
    x = np.arange(len(BANDS))
    for col, size in enumerate((1024, 640)):
        d = df[df["size"] == size]
        for row, (metric, label) in enumerate((("mean_ap", "mean AP of the start's ranking"),
                                               ("fail_share", "starts that fail (AP < 0.1)"))):  # fmt: skip
            ax = axes[row, col]
            ends = []
            for arm, (color, name) in ARMS.items():
                a = d[d["arm"] == arm].set_index("roll_band").reindex(BANDS)
                if a[metric].isna().all():
                    continue
                ax.plot(x, a[metric], color=color, lw=2, marker="o", ms=6, zorder=3)
                ends.append((x[-1] + 0.12, float(a[metric].iloc[-1]), name, color))
            _end_labels(ax, ends, gap=0.045 if row == 0 else 0.03)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_ylabel(label if col == 0 else "")
            if row == 0:
                ax.set_title(f"{size} px", color=INK2, fontsize=10)
            else:
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
                n = d[d["arm"] == "faces"].set_index("roll_band").reindex(BANDS)["queries"]
                ax.set_xticks(x, [f"{b}\n{int(q):,} starts" for b, q in zip(BAND_LABELS, n)])
                ax.set_xlabel("how far the starting face is turned in the photo")
            ax.set_xlim(-0.3, len(BANDS) - 0.2)
    fig.tight_layout()
    fig.savefig(out / "fig_screen_by_roll.png", dpi=150)
    plt.close(fig)


def fig_sessions(plain: Path, plain_base: Path, arm: str, arm_dir: Path, arm_base: Path, out: Path) -> None:
    from align_compare_4777 import BETAS  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "state_of_app"))
    from by_click import filled_runs, typed_query  # noqa: PLC0415

    def wide(analysis: Path, base_dir: Path, k: int, size: str, beta: float, faces: str) -> pd.DataFrame:
        d = analysis / f"k{k}-b{BETAS[beta]}-{size}"
        cells, curves = pd.read_csv(d / "cells.csv"), pd.read_csv(d / "curves.csv")
        base = pd.read_csv(base_dir / f"example_baseline_k{k}.csv")
        base = base[base["dataset"] == f"{faces}_{size}"]
        filled, _ = filled_runs(curves, cells, typed_query(base, "face", beta))
        return filled[[t for t in filled.columns if 1 <= int(t) <= 150]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, k in zip(axes, (1, 4)):
        ends = []
        for beta, (color, blabel) in BETA.items():
            for size, ls in SIZE.items():
                p = wide(plain, plain_base, k, size, beta, "fhibe_faces")
                a = wide(arm_dir, arm_base, k, size, beta, f"fhibe_faces_{arm}")
                d = (a - p.reindex(a.index)).dropna(how="all")
                m = d.mean(axis=0)
                se = d.std(axis=0, ddof=1) / np.sqrt(d.notna().sum(axis=0))
                t = m.index.astype(int)
                ax.plot(t, m, color=color, ls=ls, lw=1.6)
                if size == "1024":
                    ax.fill_between(t, m - se, m + se, color=color, alpha=0.15, lw=0)
                    ends.append((t[-1] + 3, float(m.iloc[-1]), blabel, color))
        _end_labels(ax, ends, gap=0.006)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(photos(k), color=INK2, fontsize=10)
        ax.set_xlabel("click")
        ax.set_xlim(0, 165)
    axes[0].set_ylabel(f"objective: {ARMS['faces_' + arm][1]} minus plain crop")
    fig.text(0.99, 0.01, "solid 1024 px (±1 SE band), dashed 640 px", ha="right", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig_session_gain.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("figure", choices=("screen", "sessions"))
    ap.add_argument("--screen", type=Path)
    ap.add_argument("--plain", type=Path)
    ap.add_argument("--plain-baseline", type=Path)
    ap.add_argument("--arm", help="tag=<analysis dir>,<baseline dir>")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.figure == "screen":
        fig_screen(args.screen, args.out)
    else:
        tag, dirs = args.arm.split("=", 1)
        a, b = dirs.split(",", 1)
        fig_sessions(args.plain, args.plain_baseline, tag, Path(a), Path(b), args.out)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
