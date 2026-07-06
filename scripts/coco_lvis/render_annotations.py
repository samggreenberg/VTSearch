#!/usr/bin/env python3
"""Render COCO / LVIS object annotations onto random sample images.

The COCO/LVIS analog of ``scripts/vg/render_vg_annotations.py``.  Pick
``count`` random images from the staged dataset and draw every object's box
with its category label — a quick visual sanity check on the annotations.

Reads the dataset staged under ``/exp/scale26/datasets/external`` (COCO or
LVIS; see each dir's README):

* ``derived/objects_flat_*.jsonl.gz``     - one row per object, box normalized
                                            to [0,1] as (x0,y0,x1,y1) + file_name
* ``images/{val2017,train2017}.zip``      - the JPEGs, KEPT ZIPPED; members are
                                            read in-memory (never unpacked)

Boxes are pre-normalized, so they are denormalized against each loaded image's
own pixel size — no separate width/height file is needed.

Usage::

    python scripts/coco_lvis/render_annotations.py 10 --dataset coco --out-dir /tmp/coco_check
    python scripts/coco_lvis/render_annotations.py 25 --dataset lvis --max-boxes 12 --seed 7

For a large ``count`` (many random reads over the zips on NFS, and LVIS pulls
from the 18 GB train2017.zip), run it on a compute node::

    cd /exp/mlucio/projects/VTSearch && source .venv/bin/activate
    srun --partition=cpu --cpus-per-task=2 --mem=8G --time=1:00:00 \\
        python ./scripts/coco_lvis/render_annotations.py 200 --dataset lvis \\
            --out-dir ./data/coco_lvis/lvis_200
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import _common as C


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("count", type=int, help="number of images to render")
    ap.add_argument("--dataset", choices=("coco", "lvis"), default="coco", help="which staged dataset")
    ap.add_argument("--extract", type=Path, default=None, help="override the derived .jsonl.gz path")
    ap.add_argument("--images-dir", type=Path, default=None, help="override the image-zip dir")
    ap.add_argument("--out-dir", type=Path, default=None, help="output dir (default: <dataset>_annotated)")
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

    extract, images_dir = C.resolve_paths(args.dataset, args.extract, args.images_dir)
    out_dir = args.out_dir or Path(f"{args.dataset}_annotated")

    print(f"loading annotations from {extract.name}…", flush=True)
    by_image = C.load_all(extract)
    candidates = sorted(by_image)
    print(f"  {len(candidates):,} images with annotations", flush=True)

    n = min(args.count, len(candidates))
    if n < args.count:
        print(f"  only {n} images available; rendering {n} instead of {args.count}", flush=True)
    chosen = sorted(random.Random(args.seed).sample(candidates, n))

    out_dir.mkdir(parents=True, exist_ok=True)
    ft = C.font()
    written = 0
    skipped = 0
    with C.ZipImageReader(images_dir) as reader:
        for iid in chosen:
            rows = by_image[iid]
            split, file_name = C.file_name_of(rows)
            try:
                img = reader.open_rgb(split, file_name)
                img = C.draw_rows(img, rows, ft, args.max_boxes)
            except Exception as exc:  # missing member / corrupt JPEG
                skipped += 1
                print(f"  [{iid}] render failed ({exc}) — skipping", flush=True)
                continue
            out_path = out_dir / f"{iid}.{args.format}"
            img.save(out_path)
            written += 1
            drew = len(rows) if args.max_boxes is None else min(len(rows), args.max_boxes)
            print(f"  [{iid}] {len(rows)} objects, drew {drew} -> {out_path.name}", flush=True)

    print(f"\nwrote {written}/{n} images to {out_dir}" + (f" ({skipped} skipped)" if skipped else ""), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # stdout closed by a downstream pipe (e.g. `| head`); exit quietly.
        sys.exit(0)
