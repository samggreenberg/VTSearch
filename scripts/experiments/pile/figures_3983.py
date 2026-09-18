#!/usr/bin/env python3
"""Figures for #3983, from `coco_only_supply.py --out`'s JSON.

python coco_only_supply.py --annotations <dir> --out supply.json
python figures_3983.py --supply supply.json --outdir docs/experiments/2026-09-18-coco-only-supply-3983
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pile_config as pc  # noqa: E402

BANDS = ("small", "medium", "large")
# The VG-bounded supply `exact_supply.py` reported, for the cells it named.
VG_BOUNDED = {"dog@small": 114, "spoon@large": 106}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--supply", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.supply.read_text())
    cells = data["cells"]
    floor = data["floor"]

    classes = list(pc.SCALE_CLASSES)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = range(len(classes))
    marks = {"small": ("o", "#4C72B0"), "medium": ("s", "#DD8452"), "large": ("^", "#55A868")}
    for band in BANDS:
        m, c = marks[band]
        ax.scatter(xs, [cells[f"{k}@{band}"] for k in classes], marker=m, color=c, s=46, label=band, zorder=3)
    ax.axhline(floor, color="#C44E52", lw=1.6, ls="--", zorder=2, label=f"SCALE_N_POS = {floor}")
    ax.axhline(300, color="#8172B3", lw=1.2, ls=":", zorder=2, label="exact_supply.py's NEED = 300")
    for j, (cell, n) in enumerate(VG_BOUNDED.items()):
        i = classes.index(cell.split("@")[0])
        ax.scatter(
            [i],
            [n],
            marker="x",
            color="#555555",
            s=80,
            zorder=4,
            label="same cell on the VG-bounded pool" if j == 0 else None,
        )
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(classes, rotation=55, ha="right", fontsize=9)
    ax.set_ylabel("candidate images for the cell (log scale)")
    ax.set_title(
        "Pure-COCO cell supply: every one of the 75 cells clears the shipped floor\n"
        "(x marks dog@small and spoon@large, the two cells exact_supply.py named as short)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    out = args.outdir / "fig_cell_supply.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
