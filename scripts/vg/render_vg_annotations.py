#!/usr/bin/env python3
"""Render Visual Genome object (noun) annotations onto sample images.

Given an integer ``count``, pick that many random images from the staged
Visual Genome dataset and draw each image's largest object bounding boxes as
colored rectangles with the noun as a label.  A quick visual sanity check on
the annotations.

Reads the dataset staged at ``/exp/scale26/datasets/external/VisualGenome``
(see that dir's README):

* ``annotations/image_data.json``        - per-image width/height
* ``derived/objects_flat.jsonl.gz``      - one row per object, box normalized
                                           to [0,1] as (x0, y0, x1, y1)
* ``images/images.zip`` + ``images2.zip``- the JPEGs, KEPT ZIPPED; members are
                                           read in-memory (never unpacked)

Usage::

    python scripts/render_vg_annotations.py 10 --out-dir /tmp/vg_check
    python scripts/render_vg_annotations.py 25 --max-boxes 12 --seed 7

For a large ``count`` (many random reads over the 15 GB zips on NFS), run it
on a compute node, e.g.::

    srun --partition=cpu --cpus-per-task=2 --mem=8G --time=1:00:00 \\
        python scripts/render_vg_annotations.py 500 --out-dir /exp/$USER/vg_check
        
matthew Usage::
    cd /exp/mlucio/projects/VTSearch
    source .venv/bin/activate
    srun --partition=cpu --cpus-per-task=2 --mem=4G --time=1:00:00 python ./scripts/vg/render_vg_annotations.py 10 --out-dir ./data/vg/10_images_no_max_test --seed 0
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_VG_DIR = Path("/exp/scale26/datasets/external/VisualGenome")

# High-contrast palette; box color is chosen deterministically per noun so the
# same word gets a stable color within and across images.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 215, 0),  # gold
    (40, 170, 200),  # cyan
    (220, 50, 50),  # red
    (60, 200, 90),  # green
    (160, 90, 220),  # purple
    (255, 140, 0),  # orange
    (0, 160, 160),  # teal
    (230, 80, 160),  # pink
    (110, 200, 40),  # lime
    (70, 120, 240),  # blue
    (200, 170, 60),  # mustard
    (250, 100, 100),  # salmon
    (140, 140, 140),  # grey
    (180, 220, 40),  # chartreuse
)


def _color_for(name: str) -> tuple[int, int, int]:
    """Stable palette color for a noun (same word -> same color)."""
    return _PALETTE[(hash(name) & 0x7FFFFFFF) % len(_PALETTE)]


def _font() -> ImageFont.ImageFont:
    """A legible default font, robust across PIL builds (no bundled TTF)."""
    try:
        return ImageFont.load_default(size=14)  # Pillow 10+
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None  # type: ignore[return-value]


def _text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white text depending on background luminance."""
    r, g, b = bg
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luma > 140 else (255, 255, 255)


def load_image_dims(image_data_path: Path) -> dict[int, tuple[int, int]]:
    """image_id -> (width, height) from image_data.json."""
    with image_data_path.open("rb") as fh:
        data = json.load(fh)
    dims: dict[int, tuple[int, int]] = {}
    for entry in data:
        iid = entry.get("image_id")
        w = entry.get("width")
        h = entry.get("height")
        if iid is not None and w and h:
            dims[int(iid)] = (int(w), int(h))
    return dims


def build_member_index(zip_paths: list[Path]) -> dict[int, tuple[Path, str]]:
    """image_id -> (zip_path, member) over all archives.

    Keyed on the integer stem of each ``.jpg`` member so it works regardless of
    whether an archive stores flat names (``107899.jpg``) or prefixed ones
    (``VG_100K_2/2377381.jpg``).  Uses ``namelist`` (central directory only), so
    this is fast and does not read image data.
    """
    index: dict[int, tuple[Path, str]] = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            for member in zf.namelist():
                if not member.endswith(".jpg"):
                    continue
                stem = Path(member).stem
                try:
                    index[int(stem)] = (zp, member)
                except ValueError:
                    continue
    return index


def collect_boxes(extract_path: Path, chosen: set[int]) -> dict[int, list[dict]]:
    """Stream the flattened extract, keeping only rows for ``chosen`` images.

    Cheap prefix filter: every line begins ``{"image_id":<id>,`` so we test the
    line prefix as a string before paying for ``json.loads`` — parsing only the
    handful of matching lines instead of all ~2.5 M.
    """
    prefixes = {f'{{"image_id":{iid},': iid for iid in chosen}
    max_pref = max((len(p) for p in prefixes), default=0)
    by_image: dict[int, list[dict]] = {iid: [] for iid in chosen}
    with gzip.open(extract_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            head = line[:max_pref]
            matched = None
            for pref, iid in prefixes.items():
                if head.startswith(pref):
                    matched = iid
                    break
            if matched is None:
                continue
            by_image[matched].append(json.loads(line))
    return by_image


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box_px: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    img_w: int,
    img_h: int,
) -> None:
    """Draw one outlined box plus a filled label chip clamped inside the image."""
    x0, y0, x1, y1 = box_px
    draw.rectangle((x0, y0, x1, y1), outline=color, width=3)

    # Size the label chip via textbbox, then clamp it inside the frame so boxes
    # at the top/left edge keep their labels visible.
    try:
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        tw, th = 7 * len(label), 12
    pad = 2
    chip_w, chip_h = tw + 2 * pad, th + 2 * pad
    cx = min(max(0, int(x0)), max(0, img_w - chip_w))
    cy = int(y0) - chip_h
    if cy < 0:  # no room above the box; put the chip just inside the top edge
        cy = min(int(y0) + 1, max(0, img_h - chip_h))
    draw.rectangle((cx, cy, cx + chip_w, cy + chip_h), fill=color)
    draw.text((cx + pad, cy + pad), label, fill=_text_color(color), font=font)


def render_image(
    iid: int,
    zip_member: tuple[Path, str],
    dims: tuple[int, int],
    rows: list[dict],
    max_boxes: int | None,
    font: ImageFont.ImageFont,
) -> Image.Image:
    """Open the JPEG from its zip and draw its largest ``max_boxes`` boxes.

    ``max_boxes=None`` draws every box (``boxes[:None]`` is the whole list).
    """
    zp, member = zip_member
    with zipfile.ZipFile(zp) as zf:
        raw = zf.read(member)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = dims
    draw = ImageDraw.Draw(img)

    # Denormalize, rank by pixel area, keep the largest few.
    boxes = []
    for r in rows:
        x0, y0, x1, y1 = r["x0"] * w, r["y0"] * h, r["x1"] * w, r["y1"] * h
        area = (x1 - x0) * (y1 - y0)
        boxes.append((area, (x0, y0, x1, y1), r.get("name", "")))
    boxes.sort(key=lambda b: b[0], reverse=True)

    for _area, box_px, name in boxes[:max_boxes]:
        _draw_box(draw, box_px, name, _color_for(name), font, img.width, img.height)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("count", type=int, help="number of images to render")
    ap.add_argument("--vg-dir", type=Path, default=DEFAULT_VG_DIR, help="staged Visual Genome dir")
    ap.add_argument("--out-dir", type=Path, default=Path("vg_annotated"), help="where to write rendered images")
    ap.add_argument(
        "--max-boxes",
        type=int,
        default=None,
        help="cap to the N largest-area boxes per image (default: all boxes)",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible image selection")
    ap.add_argument("--format", choices=("png", "jpg"), default="png", help="output image format")
    args = ap.parse_args()

    if args.count <= 0:
        raise SystemExit("count must be a positive integer")

    image_data = args.vg_dir / "annotations" / "image_data.json"
    extract = args.vg_dir / "derived" / "objects_flat.jsonl.gz"
    zips = [args.vg_dir / "images" / "images.zip", args.vg_dir / "images" / "images2.zip"]
    for p in [image_data, extract, *zips]:
        if not p.exists():
            raise SystemExit(
                f"missing {p}\nExpected the staged Visual Genome dataset under --vg-dir "
                f"({args.vg_dir}); run fetch_visual_genome.sbatch first."
            )

    print(f"indexing images in {zips[0].name} + {zips[1].name}…", flush=True)
    dims = load_image_dims(image_data)
    members = build_member_index(zips)
    candidates = sorted(set(dims) & set(members))
    print(f"  {len(candidates):,} images with both pixels and dimensions", flush=True)

    n = min(args.count, len(candidates))
    if n < args.count:
        print(f"  only {n} images available; rendering {n} instead of {args.count}", flush=True)
    rng = random.Random(args.seed)
    chosen = set(rng.sample(candidates, n))

    print("collecting annotations…", flush=True)
    by_image = collect_boxes(extract, chosen)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = _font()
    written = 0
    skipped = 0
    for iid in sorted(chosen):
        rows = by_image.get(iid) or []
        if not rows:
            skipped += 1
            print(f"  [{iid}] no annotations — skipping", flush=True)
            continue
        try:
            img = render_image(iid, members[iid], dims[iid], rows, args.max_boxes, font)
        except Exception as exc:  # corrupt JPEG etc. — VG has a few
            skipped += 1
            print(f"  [{iid}] render failed ({exc}) — skipping", flush=True)
            continue
        out_path = args.out_dir / f"{iid}.{args.format}"
        img.save(out_path)
        written += 1
        drew = len(rows) if args.max_boxes is None else min(len(rows), args.max_boxes)
        print(f"  [{iid}] {len(rows)} objects, drew {drew} -> {out_path.name}", flush=True)

    print(f"\nwrote {written}/{n} images to {args.out_dir}" + (f" ({skipped} skipped)" if skipped else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
