#!/usr/bin/env python3
"""Find every COCO / LVIS image containing a category and box each occurrence.

The COCO/LVIS analog of ``scripts/vg/annotate_vg_noun.py``.  Given a category
name, look up which images contain that object (a metadata lookup over the
annotations — *not* a pixel search) and write each matching image with a colored
rectangle around every occurrence of that category.

Category names come from a closed vocabulary (80 for COCO, 1,203 for LVIS), so
matching defaults to an exact name match that treats COCO's ``traffic light``
and LVIS's ``traffic_light`` as equal.  ``--match substring`` broadens it;
``--match synset`` matches the LVIS WordNet synset lemma (LVIS only).  Pass
``--list-categories`` to dump the available names with counts.

Reads the dataset staged under ``/exp/scale26/datasets/external`` (see READMEs):

* ``derived/objects_flat_*.jsonl.gz``     - rows {image_id, name, synset, split, file_name, x0..y1}
* ``images/{val2017,train2017}.zip``      - JPEGs read in-memory (never unpacked)

Usage::

    python scripts/coco_lvis/annotate_category.py dog --dataset coco --out-dir /tmp/coco_dog --limit 20
    python scripts/coco_lvis/annotate_category.py "traffic light" --dataset lvis --limit 50
    python scripts/coco_lvis/annotate_category.py --dataset lvis --list-categories

For a common category (full extract scan + many image reads, LVIS pulling from
the 18 GB train2017.zip) use a compute node::

    cd /exp/mlucio/projects/VTSearch && source .venv/bin/activate
    srun --partition=cpu --cpus-per-task=2 --mem=8G --time=0:30:00 \\
        python ./scripts/coco_lvis/annotate_category.py person --dataset coco \\
            --out-dir ./data/coco_lvis/coco_person --limit 200
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import re
import sys
from pathlib import Path

import _common as C


def _row_matches(row: dict, mode: str, q_name: str, q_syn: str) -> bool:
    """Whether a flattened object row matches the query under ``mode``."""
    name = C.norm_category(str(row.get("name", "")))
    if mode == "name":
        return name == q_name
    if mode == "substring":
        return q_name in name
    # synset lemma (LVIS); fall back to name when a row has no synset.
    syn = str(row.get("synset", "")).strip().lower()
    if syn:
        return syn.split(".")[0] == q_syn
    return name == q_name


def find_matches(extract_path: Path, mode: str, q_name: str, q_syn: str) -> dict[int, list[dict]]:
    """Stream the extract, returning image_id -> matching object rows.

    Cheap pre-filter: only ``json.loads`` lines containing the query's first
    token (the real match runs after parsing).
    """
    tok = q_name.split()[0] if q_name else ""
    by_image: dict[int, list[dict]] = {}
    with gzip.open(extract_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if tok and tok not in line:
                continue
            row = json.loads(line)
            if _row_matches(row, mode, q_name, q_syn):
                by_image.setdefault(int(row["image_id"]), []).append(row)
    return by_image


def list_categories(extract_path: Path) -> None:
    """Print every category name with its annotation count, most frequent first."""
    counts: collections.Counter[str] = collections.Counter()
    with gzip.open(extract_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            counts[json.loads(line).get("name", "")] += 1
    print(f"{len(counts)} categories (name: #annotations):", flush=True)
    for name, c in counts.most_common():
        print(f"  {c:>8,}  {name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("category", nargs="?", help="category to search for (e.g. 'dog', 'traffic light')")
    ap.add_argument("--dataset", choices=("coco", "lvis"), default="coco", help="which staged dataset")
    ap.add_argument("--extract", type=Path, default=None, help="override the derived .jsonl.gz path")
    ap.add_argument("--images-dir", type=Path, default=None, help="override the image-zip dir")
    ap.add_argument("--out-dir", type=Path, default=None, help="output dir (default: <dataset>_<category>)")
    ap.add_argument(
        "--match",
        choices=("name", "substring", "synset"),
        default="name",
        help="how the query matches category names (default: exact name)",
    )
    ap.add_argument("--limit", type=int, default=200, help="max images to annotate; <=0 means no cap")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed when sampling down to --limit")
    ap.add_argument("--format", choices=("png", "jpg"), default="png", help="output image format")
    ap.add_argument("--list-categories", action="store_true", help="print available categories and exit")
    args = ap.parse_args()

    extract, images_dir = C.resolve_paths(args.dataset, args.extract, args.images_dir)

    if args.list_categories:
        list_categories(extract)
        return 0
    if not args.category or not args.category.strip():
        raise SystemExit("category is required (or pass --list-categories)")

    q_name = C.norm_category(args.category)
    q_syn = q_name.replace(" ", "_")
    out_dir = args.out_dir or Path(f"{args.dataset}_{re.sub(r'[^a-z0-9]+', '_', q_name).strip('_')}")

    print(f"searching {args.dataset} annotations for '{args.category}' (match={args.match})…", flush=True)
    by_image = find_matches(extract, args.match, q_name, q_syn)
    if not by_image:
        print(
            f"no {args.dataset} images contain '{args.category}'. Run with --list-categories to see available names.",
            flush=True,
        )
        return 0
    total = len(by_image)
    print(f"  {total:,} matching images", flush=True)

    selected = sorted(by_image)
    if 0 < args.limit < total:
        selected = sorted(random.Random(args.seed).sample(selected, args.limit))
        print(f"  capping to {args.limit} (--limit); raise --limit or pass <=0 for all", flush=True)

    color = C.color_for(q_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    ft = C.font()
    written = 0
    skipped = 0
    with C.ZipImageReader(images_dir) as reader:
        for iid in selected:
            rows = by_image[iid]
            split, file_name = C.file_name_of(rows)
            try:
                img = reader.open_rgb(split, file_name)
                # highlight only the matched occurrences, in the category's color.
                img = C.draw_rows(img, rows, ft, None, forced_color=color)
            except Exception as exc:  # missing member / corrupt JPEG
                skipped += 1
                print(f"  [{iid}] render failed ({exc}) — skipping", flush=True)
                continue
            out_path = out_dir / f"{iid}.{args.format}"
            img.save(out_path)
            written += 1
            print(f"  [{iid}] {len(rows)} box(es) -> {out_path.name}", flush=True)

    print(
        f"\n'{args.category}': {total:,} matching {args.dataset} images, wrote {written} to {out_dir}"
        + (f" ({skipped} skipped)" if skipped else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # stdout closed by a downstream pipe (e.g. `--list-categories | head`);
        # exit quietly instead of dumping a traceback.
        sys.exit(0)
