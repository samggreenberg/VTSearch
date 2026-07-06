#!/usr/bin/env python3
"""Render specific COCO / LVIS image(s) by id with ALL their object annotations.

The COCO/LVIS analog of ``scripts/vg/show_vg_image.py``.  Given one or more
``image_id`` values, write each image with every object's box + category label.
Unlike ``render_annotations.py`` (random sampling), this targets exact ids — for
inspecting "what does the dataset say is in image N".

Note COCO and LVIS share the COCO 2017 image ids, but their annotation sets
differ (80 COCO categories vs 1,203 LVIS categories, and LVIS is federated /
non-exhaustive), so ``--dataset`` selects which annotations are drawn.

Reads the dataset staged under ``/exp/scale26/datasets/external`` (see READMEs):

* ``derived/objects_flat_*.jsonl.gz``     - rows {image_id, name, split, file_name, x0..y1}
* ``images/{val2017,train2017}.zip``      - JPEGs read in-memory (never unpacked)

Usage::

    python scripts/coco_lvis/show_image.py 289343 --dataset coco --out-dir /tmp/coco_show
    python scripts/coco_lvis/show_image.py 446522 128506 --dataset lvis --max-boxes 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _common as C


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_id", type=int, nargs="+", help="one or more COCO image_id values")
    ap.add_argument("--dataset", choices=("coco", "lvis"), default="coco", help="which staged dataset")
    ap.add_argument("--extract", type=Path, default=None, help="override the derived .jsonl.gz path")
    ap.add_argument("--images-dir", type=Path, default=None, help="override the image-zip dir")
    ap.add_argument("--out-dir", type=Path, default=None, help="output dir (default: <dataset>_show)")
    ap.add_argument(
        "--max-boxes",
        type=int,
        default=None,
        help="cap to the N largest-area boxes per image (default: all annotations)",
    )
    ap.add_argument("--format", choices=("png", "jpg"), default="png", help="output image format")
    args = ap.parse_args()

    extract, images_dir = C.resolve_paths(args.dataset, args.extract, args.images_dir)
    out_dir = args.out_dir or Path(f"{args.dataset}_show")

    ids = set(args.image_id)
    print(f"collecting annotations for {len(ids)} image(s) from {extract.name}…", flush=True)
    by_image = C.collect_boxes(extract, ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    ft = C.font()
    written = 0
    with C.ZipImageReader(images_dir) as reader:
        for iid in sorted(ids):
            rows = by_image.get(iid) or []
            if not rows:
                print(f"  [{iid}] no annotations in {args.dataset} — skipping", flush=True)
                continue
            split, file_name = C.file_name_of(rows)
            try:
                img = reader.open_rgb(split, file_name)
                img = C.draw_rows(img, rows, ft, args.max_boxes)
            except Exception as exc:  # missing member / corrupt JPEG
                print(f"  [{iid}] render failed ({exc}) — skipping", flush=True)
                continue
            out_path = out_dir / f"{iid}.{args.format}"
            img.save(out_path)
            written += 1
            drew = len(rows) if args.max_boxes is None else min(len(rows), args.max_boxes)
            print(f"  [{iid}] {len(rows)} objects, drew {drew} -> {out_path.name}", flush=True)

    print(f"\nwrote {written}/{len(ids)} images to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # stdout closed by a downstream pipe (e.g. `| head`); exit quietly.
        sys.exit(0)
