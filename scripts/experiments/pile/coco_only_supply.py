#!/usr/bin/env python3
"""Does COCO 2017 alone supply `vg_scale`'s cells, with no Visual Genome at all?

`exact_supply.py` asked a neighbouring question and is often read as having
answered this one. It has not: it draws its candidates from ``vg_source()``, so
its "COCO half" is the VG-COCO overlap (~51,497 images), not COCO 2017
train+val (123,287). The ~72,000 COCO images outside VG were never candidates.

This asks the question with the pool unbounded by VG. It needs **no VG source
and no pile** -- only ``instances_train2017.json`` and ``instances_val2017.json``
-- so it runs anywhere, in about a minute, against ~490 MB of JSON.

It imports the SHIPPED band rule (:func:`pilebuild.loaders.vg_scale.band_for`)
rather than restating it, for the reason that function's own docstring gives: a
second copy of the rule would answer a supply question with its own drift. The
scatter guard and `BOX_BANDS` therefore apply exactly as the builder applies
them.

Two deliberate choices, both stricter than the builder needs to be:

* ``iscrowd`` regions are dropped. A crowd box is a region, not an instance, so
  admitting it would let one annotation band an image by an extent no user
  would drag. VG has no equivalent concept, so this has no counterpart upstream.
* Nothing else is filtered. `scale_study_exclusion` exists to keep VG's free
  text (parts, places, polysemous bare names) out of a band; COCO's vocabulary
  is curated and disjoint, so there is nothing for it to reject.

Written for #3983. Results: `docs/experiments/2026-09-18-coco-only-supply-3983/`.

    python coco_only_supply.py --annotations <dir holding instances_*2017.json>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pile_config as pc  # noqa: E402
from pilebuild.loaders.vg_scale import OVERSIZE, SCATTERED, band_for  # noqa: E402

BANDS = ("small", "medium", "large")


def read_coco(ann: Path) -> tuple[dict[int, tuple[int, int]], dict[int, dict[str, list]], set[str]]:
    """``(dims, boxes, class_names)`` over both 2017 splits."""
    dims: dict[int, tuple[int, int]] = {}
    boxes: dict[int, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    names: set[str] = set()
    for split in ("val2017", "train2017"):
        path = ann / f"instances_{split}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}; fetch annotations_trainval2017.zip")
        with path.open() as fh:
            data = json.load(fh)
        cats = {c["id"]: c["name"] for c in data["categories"]}
        names |= set(cats.values())
        for im in data["images"]:
            dims[im["id"]] = (int(im["width"]), int(im["height"]))
        for a in data["annotations"]:
            if a.get("iscrowd"):
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes[a["image_id"]][cats[a["category_id"]]].append([x, y, x + w, y + h])
        del data
    return dims, boxes, names


def band_counts(dims, boxes) -> tuple[collections.Counter, collections.Counter, dict[str, set]]:
    """Per-``(class, band)`` and band-free candidate counts, by the shipped rule."""
    cell: collections.Counter = collections.Counter()
    bandfree: collections.Counter = collections.Counter()
    holders: dict[str, set] = collections.defaultdict(set)
    for iid, per_class in boxes.items():
        w, h = dims[iid]
        for cls, bs in per_class.items():
            holders[cls].add(iid)
            band = band_for(bs, w, h)
            if band in (SCATTERED, OVERSIZE):
                continue
            cell[(cls, band)] += 1
            bandfree[cls] += 1
    return cell, bandfree, holders


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, required=True, help="dir holding instances_{train,val}2017.json")
    ap.add_argument(
        "--floor", type=int, default=pc.SCALE_N_POS, help=f"per-cell floor (default SCALE_N_POS={pc.SCALE_N_POS})"
    )
    ap.add_argument("--out", default="", help="write the table as JSON")
    args = ap.parse_args()

    dims, boxes, names = read_coco(args.annotations)
    cell, bandfree, holders = band_counts(dims, boxes)

    classes = list(pc.SCALE_CLASSES)
    clean = set(dims) - set().union(*(holders[c] for c in classes))
    print(f"COCO 2017 train+val images: {len(dims):,}")
    print(
        f"shared-pool candidates (hold none of the {len(classes)}): {len(clean):,} "
        f"(need SCALE_N_NEG={pc.SCALE_N_NEG:,})\n"
    )

    print(f"{'class':<14}" + "".join(f"{b:>9}" for b in BANDS) + f"{'band-free':>11}   short")
    print("-" * 66)
    short = 0
    for c in classes:
        row = [cell[(c, b)] for b in BANDS]
        miss = [b for b, n in zip(BANDS, row) if n < args.floor]
        short += len(miss)
        print(f"{c:<14}" + "".join(f"{n:>9,}" for n in row) + f"{bandfree[c]:>11,}   {','.join(miss) or '-'}")
    print("-" * 66)
    print(f"cells short of {args.floor}: {short} of {len(classes) * 3}")

    wide = [c for c in sorted(names) if min(cell[(c, b)] for b in BANDS) >= args.floor]
    print(
        f"\nCOCO classes clearing {args.floor} in all three bands: {len(wide)} of {len(names)} "
        f"({len([c for c in wide if c not in set(classes)])} beyond the current {len(classes)})"
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "images": len(dims),
                    "clean_pool": len(clean),
                    "floor": args.floor,
                    "cells": {f"{c}@{b}": cell[(c, b)] for c in classes for b in BANDS},
                    "band_free": {c: bandfree[c] for c in classes},
                    "all_classes_clearing_floor": wide,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
