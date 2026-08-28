"""Look at what a region-voting Good vote would actually drag.

A box is the one field in a pile cell that nothing downstream reads back in a
form a person can check. Labels get adjudicated on contact sheets; geometry gets
a number. #3281 lived through three studies because a corrupted box lands its
image in `@small` and `@small` is exactly where a sub-pixel area belongs -- the
cell name and its boxes stayed consistent with each other, and no aggregate
could see it. The first thing that did see it was a picture.

So: one tile per positive, the full frame with the stored box marked, and the
box's own pixels enlarged beside it. A box that describes its object looks
obviously right; one that does not looks obviously wrong, at a glance, which is
the property no summary statistic has.

    python box_sheets.py --dataset vg_scale --category bird@small --out sheet.jpg
    python box_sheets.py --dataset vg_scale --category bird@small --order area --n 12

``--order area`` puts the smallest boxes first, which is where geometry bugs
surface: a scaling error shrinks a box, it does not grow one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

import common  # noqa: E402

common.setup_env()

import pile_config as pc  # noqa: E402

#: Where the source JPEGs live, per dataset kind. A sheet is only worth making
#: where the pixels are reachable.
IMAGE_DIRS = {
    "vg": [pc.DEMO_CACHE / "visual_genome" / "VG_100K", pc.DEMO_CACHE / "visual_genome" / "VG_100K_2"],
    "coco": [pc.COCO_IMAGES],
}


def _dirs_for(dataset: str) -> list[Path]:
    kind = pc.DATASETS.get(dataset, {}).get("kind", "")
    return IMAGE_DIRS["coco"] if kind == "coco" else IMAGE_DIRS["vg"]


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="vg_scale")
    ap.add_argument("--embedder", default="siglip", help="which cell to read labels from (the small one)")
    ap.add_argument("--category", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--order", default="area", choices=["area", "id"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from PIL import Image, ImageDraw, ImageFont

    from vtscore.datasets import loader as _loader
    from vtscore.eval.labels import media_is_positive, region_box_for_category

    from _cells_io import load_medias  # noqa: PLC0415

    medias = load_medias(_loader.EMBEDDINGS_DIR / f"{args.dataset}__{args.embedder}.pkl")
    dirs = _dirs_for(args.dataset)

    def path_of(mid, m):
        for cand in (m.get("filename"), m.get("media_string"), f"{mid}.jpg"):
            if not cand:
                continue
            for d in dirs:
                p = d / Path(str(cand)).name
                if p.exists():
                    return p
        return None

    items = []
    for mid, m in medias.items():
        if not media_is_positive(m, args.category):
            continue
        box = region_box_for_category(m, args.category)
        p = path_of(mid, m)
        if box is not None and p is not None:
            items.append((_area(box), mid, box, p))
    print(f"{args.dataset} {args.category}: {len(items)} positives with a resolvable image")
    items.sort(key=(lambda it: it[0]) if args.order == "area" else (lambda it: it[1]))
    items = items[: args.n]
    if not items:
        print("nothing to draw")
        return 1

    TILE, IMG_H, CROP, PAD, HDR = 320, 230, 230, 12, 24
    cols = 4
    rows = (len(items) + cols - 1) // cols
    tile_h = HDR + IMG_H + PAD + CROP + PAD
    sheet = Image.new("RGB", (cols * (TILE + PAD) + PAD, rows * tile_h + PAD), "#fcfcfb")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for k, (a, mid, box, path) in enumerate(items):
        r, c = divmod(k, cols)
        x0, y0 = PAD + c * (TILE + PAD), PAD + r * tile_h
        img = Image.open(path).convert("RGB")
        W, H = img.size
        px = [box[0] * W, box[1] * H, box[2] * W, box[3] * H]
        full = img.copy()
        d = ImageDraw.Draw(full)
        # A sub-pixel box is invisible at thumbnail scale, which is the case
        # this sheet exists for -- so it gets a crosshair as well as an outline.
        d.rectangle([px[0] - 1, px[1] - 1, px[2] + 1, px[3] + 1], outline="#e11d48", width=max(2, W // 200))
        cx, cy = (px[0] + px[2]) / 2, (px[1] + px[3]) / 2
        d.line([(cx, 0), (cx, H)], fill="#e11d48", width=max(1, W // 600))
        d.line([(0, cy), (W, cy)], fill="#e11d48", width=max(1, W // 600))
        full.thumbnail((TILE, IMG_H))
        sheet.paste(full, (x0 + (TILE - full.width) // 2, y0 + HDR))
        m = max(px[2] - px[0], px[3] - px[1]) * 1.6 + 10
        cb = (max(0, cx - m), max(0, cy - m), min(W, cx + m), min(H, cy + m))
        crop = img.crop([int(v) for v in cb]).resize((CROP, CROP), Image.LANCZOS)
        dc = ImageDraw.Draw(crop)
        sx, sy = CROP / max(1e-6, cb[2] - cb[0]), CROP / max(1e-6, cb[3] - cb[1])
        dc.rectangle(
            [(px[0] - cb[0]) * sx, (px[1] - cb[1]) * sy, (px[2] - cb[0]) * sx, (px[3] - cb[1]) * sy],
            outline="#e11d48",
            width=2,
        )
        sheet.paste(crop, (x0 + (TILE - CROP) // 2, y0 + HDR + IMG_H + PAD))
        # Both the fraction and the pixels: a percentage alone cannot tell you
        # the box is smaller than one pixel, and that is the failure mode.
        draw.text(
            (x0, y0 + 5),
            f"{mid}   {a * 100:.4f}% of frame   {px[2] - px[0]:.1f}x{px[3] - px[1]:.1f}px",
            fill="#292524",
            font=font,
        )

    sheet.save(args.out, quality=92)
    print(f"wrote {args.out} {sheet.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
