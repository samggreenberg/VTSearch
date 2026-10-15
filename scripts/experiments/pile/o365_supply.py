#!/usr/bin/env python3
"""Can Objects365 v2 val fill `coco_better`'s cells? The gate before `o365_better` is built (#4815).

#3983 asked this of COCO before `coco_better` existed; this asks it of a second
source. It needs **no images, no pile and no GPU** -- only the 269 MB
``zhiyuan_objv2_val.json`` -- so it runs anywhere, in about a minute.

It measures the supply the BUILD will draw from, not a restatement of it: the
reader is the loader's own :func:`~pilebuild.loaders.o365_better.read_o365_labels`
(class map, flags, ambiguity and all) and the banding is the loader's own
:func:`~pilebuild.loaders.o365_better.banded`, which is the shipped
:func:`~pilebuild.scale_core.band_candidates` on the largest instance.

Four answers:

* **Supply per cell**, against ``SCALE_N_POS``. The cells under it are the
  proposal for :data:`pile_config.O365_DROPPED_CELLS`.
* **The clean pool** -- images holding none of *C*, not even doubtfully --
  against the ``SCALE_N_NEG + SCALE_N_NEG_SPARE`` the draw takes.
* **What makes a pair doubtful**, by reason: the release's three flags, a box
  with no area, or an ambiguous name. A doubtful pair is never a positive.
* **The price of each ambiguous name** (:data:`pile_config.O365_AMBIGUOUS`): the
  clean-pool images it alone withholds, and the (image, class) pairs it alone
  keeps out of the cross-class negatives. A ruling that drops the name buys
  those back; one that moves it to :data:`pile_config.O365_NAMES` turns its
  pairs into positive candidates instead.

Written for #4815. Results: ``docs/experiments/2026-10-13-o365-supply-4815/``.

    python o365_supply.py --annotations <zhiyuan_objv2_val.json> --out supply.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pile_config as pc  # noqa: E402
from pilebuild.loaders.o365_better import banded, label_fields, plan_from, read_o365_labels  # noqa: E402

BANDS = tuple(pc.BOX_BANDS)


def census(path: Path, floor: int) -> dict:
    """Every number the report quotes, as one JSON-able dict."""
    src = read_o365_labels(path, pc.SCALE_CLASSES)
    supply, _, clean = banded(src)
    classes = list(pc.SCALE_CLASSES)

    cells = {pc.scale_cell(c, b): len(supply[c][b]) for c in classes for b in BANDS}
    holders = collections.Counter(c for by_cls in src.labels.values() for c in by_cls)
    plain_holders = collections.Counter(
        c for iid, by_cls in src.labels.items() for c in by_cls if (iid, c) not in src.doubtful
    )
    short = sorted(cell for cell, n in cells.items() if n < floor)

    by_reason: collections.Counter = collections.Counter()
    for why in src.reasons.values():
        for r in why:
            by_reason[r] += 1

    # The price of an ambiguous name: what it ALONE withholds. A pair whose only
    # reason is this name would be a cross-class negative without it; an image
    # all of whose held pairs are such pairs would be clean.
    alone = {pair: next(iter(why)) for pair, why in src.reasons.items() if len(why) == 1}
    pairs_alone: collections.Counter = collections.Counter()
    clean_alone: collections.Counter = collections.Counter()
    for pair, why in alone.items():
        if why.startswith("name:"):
            pairs_alone[why] += 1
    # The ceiling on what rulings can do for the pool: images held ONLY through
    # ambiguous names, which ruling every name out would make clean.
    names_only = sum(
        1
        for iid, by_cls in src.labels.items()
        if by_cls
        and all((iid, c) in src.doubtful and all(r.startswith("name:") for r in src.reasons[(iid, c)]) for c in by_cls)
    )
    for iid, by_cls in src.labels.items():
        if not by_cls:
            continue
        reasons = {alone.get((iid, c)) for c in by_cls}
        if len(reasons) == 1:
            (only,) = reasons
            if only is not None and only.startswith("name:"):
                clean_alone[only] += 1

    # The haystack: how many negatives each BUILT cell is scored against -- the
    # drawn pool plus the other cells' positives that do not hold its class
    # (#3667) -- read off the build's own label fields, so it is what a study
    # will see. A `coco_better` cell carries 23,891.
    plan = plan_from(src)
    haystack: collections.Counter = collections.Counter()
    for iid in plan.emit_ids:
        fields = label_fields(plan, iid)
        for cell in set(fields["evaluable_categories"]) - set(fields["categories"]):
            haystack[cell] += 1

    need = pc.SCALE_N_NEG + pc.SCALE_N_NEG_SPARE
    return {
        "source": path.name,
        "images": len(src.labels),
        "floor": floor,
        "cells": cells,
        "cells_short": short,
        "cells_built": len(plan.cells),
        "images_emitted": len(plan.emit_ids),
        "pool_drawn": len(plan.neg_set),
        "haystack": {cell: haystack[cell] for cell in plan.cells},
        "holders": {c: holders[c] for c in classes},
        "plain_holders": {c: plain_holders[c] for c in classes},
        "clean_pool": len(clean),
        "clean_pool_need": need,
        "clean_if_no_ambiguous": len(clean) + names_only,
        "doubtful_pairs": len(src.doubtful),
        "doubtful_by_reason": dict(by_reason.most_common()),
        "ambiguous_name_price": {
            name: {"pairs_alone": pairs_alone[f"name:{name}"], "clean_alone": clean_alone[f"name:{name}"]}
            for name in sorted({n for names in pc.O365_AMBIGUOUS.values() for n in names})
        },
    }


def report(res: dict) -> None:
    print(f"Objects365 {res['source']}: {res['images']:,} images")
    print(
        f"clean pool (holds none of the {len(pc.SCALE_CLASSES)}): {res['clean_pool']:,} (draw takes {res['clean_pool_need']:,})\n"
    )
    print(f"{'class':<32}" + "".join(f"{b:>9}" for b in BANDS) + f"{'holders':>10}{'plain':>8}")
    print("-" * 87)
    for c in pc.SCALE_CLASSES:
        row = [res["cells"][pc.scale_cell(c, b)] for b in BANDS]
        mark = "".join(f"{n:>8,}{'*' if n < res['floor'] else ' '}" for n in row)
        print(f"{c:<32}{mark}{res['holders'][c]:>10,}{res['plain_holders'][c]:>8,}")
    print("-" * 87)
    print(f"* under SCALE_N_POS={res['floor']}: {len(res['cells_short'])} of {len(res['cells'])} cells")
    hay = sorted(res["haystack"].values())
    print(
        f"\nas built (O365_DROPPED_CELLS applied): {res['cells_built']} cells, {res['images_emitted']:,} images, "
        f"pool drawn {res['pool_drawn']:,}"
    )
    if hay:
        print(f"negatives per cell: min {hay[0]:,}, median {hay[len(hay) // 2]:,}, max {hay[-1]:,}")
    print(f"\ndoubtful (image, class) pairs: {res['doubtful_pairs']:,}")
    for why, n in res["doubtful_by_reason"].items():
        print(f"  {why:<40}{n:>8,}")
    print(f"\nclean pool if every ambiguous name were ruled out: {res['clean_if_no_ambiguous']:,}")
    print("ambiguous names -- what each ALONE withholds")
    print(f"  {'name':<24}{'pairs':>8}{'clean':>8}")
    for name, price in res["ambiguous_name_price"].items():
        print(f"  {name!r:<24}{price['pairs_alone']:>8,}{price['clean_alone']:>8,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=pc.O365_VAL_ANNOTATIONS, help="zhiyuan_objv2_val.json")
    ap.add_argument("--floor", type=int, default=pc.SCALE_N_POS, help=f"per-cell floor (default {pc.SCALE_N_POS})")
    ap.add_argument("--out", type=Path, help="write the census as JSON")
    args = ap.parse_args()
    res = census(args.annotations, args.floor)
    report(res)
    if args.out:
        args.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
