#!/usr/bin/env python
"""The figure of #5341's report: the three InsightFace detectors' landmarks against FHIBE's annotated points.

``fig_landmarks.png`` - the share of subject photos whose five landmarks sit beyond a given error of FHIBE's
points (NME, over the inter-ocular distance), for det_10g (buffalo_l), det_2.5g (buffalo_m) and det_500m
(buffalo_s), at 1024 and 640 px. It reads a ``nme_cdf.csv`` in ``buffalo_m_compare_5330.py``'s shape with a
``det`` of ``l``, ``m`` or ``s``.

    python buffalo_s_figures_5341.py --nme <nme_cdf.csv> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from face_gap_4780_figures import GRID, INK2, _style  # noqa: E402

#: The reference palette's first three categorical slots, in order (validated as a set).
DETS = (
    ("l", "buffalo_l (det_10g)", "#2a78d6"),
    ("m", "buffalo_m (det_2.5g)", "#eb6834"),
    ("s", "buffalo_s (det_500m)", "#1baf7a"),
)


def fig_landmarks(cdf: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, size in zip(axes, (1024, 640)):
        for det, label, color in DETS:
            c = cdf[(cdf["size"] == size) & (cdf["det"] == det)]
            ax.plot(c["nme"], 1 - c["share_at_or_below"], color=color, lw=2, label=label)
            tail = c[c["nme"] <= 0.5].iloc[-1]
            ax.annotate(
                det,
                (tail["nme"], 1 - tail["share_at_or_below"]),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK2,
            )
        ax.set_yscale("log")
        ax.set_xlim(0, 0.55)
        ax.set_title(f"{size} px", fontsize=9)
        ax.set_xlabel("landmark error against FHIBE's points (NME)")
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("share of subject photos with a larger error")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_landmarks.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nme", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_landmarks(pd.read_csv(args.nme), args.out)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
