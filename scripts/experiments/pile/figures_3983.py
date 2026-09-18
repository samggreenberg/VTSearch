#!/usr/bin/env python3
"""Figures for #3983.

`fig_cell_supply.png` comes from `coco_only_supply.py --out`'s JSON;
`fig_scatter.png` needs the COCO annotations, since it draws real boxes.

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
    ap.add_argument("--annotations", type=Path, default=None, help="for fig_scatter.png")
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

    if args.annotations:
        _scatter_figure(args.annotations, args.outdir)


#: One image per panel, chosen as the clearest instance of each verdict in
#: COCO val2017: both hold three `car` boxes and they band differently.
SCATTER_PANELS = ((424162, "car", "SCATTERED - no band"), (78823, "car", "KEPT - bands on the union"))


def _scatter_figure(ann: Path, outdir: Path) -> None:
    """Draw the scatter guard on real boxes: union vs largest single instance."""
    import collections

    from matplotlib.patches import Rectangle

    with (ann / "instances_val2017.json").open() as fh:
        data = json.load(fh)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    dims = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
    boxes: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        if w > 0 and h > 0:
            boxes[a["image_id"]][cats[a["category_id"]]].append([x, y, x + w, y + h])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, (iid, cls, title) in zip(axes, SCATTER_PANELS):
        w_img, h_img = dims[iid]
        bs = boxes[iid][cls]
        area = w_img * h_img
        ux0, uy0 = min(b[0] for b in bs), min(b[1] for b in bs)
        ux1, uy1 = max(b[2] for b in bs), max(b[3] for b in bs)
        union = (ux1 - ux0) * (uy1 - uy0) / area
        big = max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        largest = (big[2] - big[0]) * (big[3] - big[1]) / area
        ax.add_patch(Rectangle((0, 0), w_img, h_img, fc="#f2f2f2", ec="#999", lw=1.2))
        ax.add_patch(
            Rectangle((ux0, uy0), ux1 - ux0, uy1 - uy0, fc="#C44E52", alpha=0.16, ec="#C44E52", lw=2.2, ls="--")
        )
        for b in bs:
            is_big = b is big
            ax.add_patch(
                Rectangle(
                    (b[0], b[1]),
                    b[2] - b[0],
                    b[3] - b[1],
                    fill=False,
                    ec="#2F6DB5" if is_big else "#6f6f6f",
                    lw=2.6 if is_big else 1.5,
                )
            )
        ax.set_xlim(-w_img * 0.04, w_img * 1.04)
        ax.set_ylim(h_img * 1.04, -h_img * 0.04)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{title}\n`{cls}`, {len(bs)} boxes - largest {largest * 100:.1f}% of frame, "
            f"union {union * 100:.1f}%  =>  {union / largest:.1f}x",
            fontsize=10,
        )
    handles = [
        Rectangle((0, 0), 1, 1, fc="#C44E52", alpha=0.3, ec="#C44E52", ls="--", lw=2),
        Rectangle((0, 0), 1, 1, fill=False, ec="#2F6DB5", lw=2.6),
        Rectangle((0, 0), 1, 1, fill=False, ec="#6f6f6f", lw=1.5),
    ]
    fig.legend(
        handles,
        ["union of the class's boxes (what the band is computed from)", "largest single instance", "other instances"],
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "The scatter guard: reject when the union is more than 1.5x the largest single box", fontsize=12, y=0.99
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = outdir / "fig_scatter.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
