#!/usr/bin/env python
"""The figures of #5330's report: buffalo_m's detector against buffalo_l's on FHIBE's face cells.

* ``fig_landmarks.png`` - the share of subject photos whose five landmarks sit within a given error of
  FHIBE's annotated points (NME, over the inter-ocular distance), per detector, at 1024 and 640 px,
  from ``buffalo_m_compare_5330.py``'s ``nme_cdf.csv``;
* ``fig_cosine.png`` - the cosine between buffalo_l's and buffalo_m's vector of the same subject photo,
  binned, from its ``cosine_hist.csv``.

    python buffalo_m_figures_5330.py --compare <compare dir> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from face_gap_4780_figures import INK2, GRID, _style  # noqa: E402

#: The detectors take the reference palette's blue (buffalo_l, det_10g) and orange (buffalo_m, det_2.5g).
DETS = (("l", "buffalo_l (det_10g)", "#2a78d6"), ("m", "buffalo_m (det_2.5g)", "#eb6834"))


def fig_landmarks(cdf: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, size in zip(axes, (1024, 640)):
        for det, label, color in DETS:
            c = cdf[(cdf["size"] == size) & (cdf["det"] == det)]
            ax.plot(c["nme"], 1 - c["share_at_or_below"], color=color, lw=2, label=label)
        ax.set_yscale("log")
        ax.set_title(f"{size} px", fontsize=9)
        ax.set_xlabel("landmark error against FHIBE's points (NME)")
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("share of subject photos with a larger error")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_landmarks.png", dpi=150)
    plt.close(fig)


def fig_cosine(hist: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, size in zip(axes, (1024, 640)):
        h = hist[hist["size"] == size]
        labels = [f"< {hi:g}" if lo < 0 else f"{lo:g}–{min(hi, 1):g}" for lo, hi in zip(h["lo"], h["hi"])]
        x = np.arange(len(h))
        ax.bar(x, h["photos"], color=INK2, width=0.8)
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_yscale("log")
        ax.set_title(f"{size} px, {h['photos'].sum():,} subject photos in both cells", fontsize=9)
        ax.set_xlabel("cosine, buffalo_l vector vs buffalo_m vector of the same photo")
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("subject photos")
    fig.tight_layout()
    fig.savefig(out / "fig_cosine.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--compare", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_landmarks(pd.read_csv(args.compare / "nme_cdf.csv"), args.out)
    fig_cosine(pd.read_csv(args.compare / "cosine_hist.csv"), args.out)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
