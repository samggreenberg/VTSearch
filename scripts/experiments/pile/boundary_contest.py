#!/usr/bin/env python3
"""Can COCO carry a boundary between classes in C? Ask LVIS, never a detector.

LVIS re-annotated COCO's own images with 1,203 defined synsets, by different
people. So for one LVIS object type -- `duffel_bag`, `flowerpot`, `ski` -- ask
what COCO's annotators called the same box. A type COCO labels consistently is a
boundary COCO can carry; a type COCO splits is one it cannot, whatever the right
answer is philosophically. Human against human, no model (owner ruling on
`truck`/`car`, 2026-09-20: "we're not going to define our ground truth based on
our detector").

**The statistic is per object type, and the unit is the BOX.** coco_quarry's
region arm drags a box, and a box has no co-occurring instance to redeem it, so
the rate that binds is the share of boxes carrying the MINORITY COCO label for
their own LVIS type. That is what merged `car`+`truck` (9.6%) and `cup`+`wine
glass` (10.7%) in #4056. Conditioning the other way -- what fraction of class A
is really B -- made `truck`/`car` look twice as bad as it is (#3618).

This is `mush.py` from the #4056 ruling, moved into the repo, generalised past a
pair, and reading BOTH LVIS splits (`pile_config.LVIS_DIR`): the #4056 numbers
used val alone, 16% of COCO.

    python boundary_contest.py backpack handbag suitcase
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pile_config as pc  # noqa: E402

IOU_MIN = 0.5
MIN_TYPE = 20
SPLIT = (0.2, 0.8)


def _boxes(path: Path, keep: set[str] | None = None) -> dict[int, list[tuple[list[float], str]]]:
    with path.open() as fh:
        d = json.load(fh)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    out: dict = collections.defaultdict(list)
    for a in d["annotations"]:
        name = cats[a["category_id"]]
        if a.get("iscrowd") or (keep is not None and name not in keep):
            continue
        out[int(a["image_id"])].append((a["bbox"], name))
    return out


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / (aw * ah + bw * bh - i)


def split_by_type(
    coco: dict[int, list[tuple[list[float], str]]],
    lvis: dict[int, list[tuple[list[float], str]]],
    classes: tuple[str, ...],
) -> dict[str, collections.Counter]:
    """``{lvis_type: Counter(coco_class)}`` over mutual-best-match box pairs at IoU >= 0.5."""
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for iid, cbs in coco.items():
        lbs = lvis.get(iid)
        if not lbs:
            continue
        grid = [[iou(cb, lb) for lb, _ in lbs] for cb, _ in cbs]
        for i, (_, ccls) in enumerate(cbs):
            if ccls not in classes:
                continue
            j = max(range(len(lbs)), key=lambda k: grid[i][k])
            # Mutual best match only, so one LVIS box never votes twice.
            if grid[i][j] < IOU_MIN or max(range(len(cbs)), key=lambda k: grid[k][j]) != i:
                continue
            out[lbs[j][1]][ccls] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("classes", nargs="+", help="COCO class names, as COCO spells them")
    ap.add_argument("--annotations", type=Path, default=pc.COCO_ANCHOR_DIR)
    ap.add_argument("--lvis", type=Path, nargs="+", default=[pc.LVIS_DIR / f"lvis_v1_{s}.json" for s in pc.LVIS_SPLITS])
    ap.add_argument("--out", type=Path, help="write the per-type table as JSON")
    args = ap.parse_args()
    classes = tuple(args.classes)

    coco: dict = collections.defaultdict(list)
    for split in ("val2017", "train2017"):
        for k, v in _boxes(args.annotations / f"instances_{split}.json", set(classes)).items():
            coco[k] += v
    lvis: dict = collections.defaultdict(list)
    for path in args.lvis:
        for k, v in _boxes(path).items():
            lvis[k] += v

    split_of = split_by_type(coco, lvis, classes)
    rows = sorted(
        ((name, sum(c.values()), c) for name, c in split_of.items() if sum(c.values()) >= MIN_TYPE),
        key=lambda r: -r[1],
    )
    total = sum(n for _, n, _ in rows)
    if not total:
        print(f"no LVIS type reaches n>={MIN_TYPE} for {classes}")
        return 0

    head = "".join(f"{c[:12]:>13}" for c in classes)
    print(f"LVIS types matched to COCO {' / '.join(classes)}, n>={MIN_TYPE}: {len(rows)} types, {total:,} boxes\n")
    print(f"{'LVIS type':<30}{'n':>6}{head}   minority")
    print("-" * (48 + 13 * len(classes)))
    contested = minority = 0
    table = []
    for name, n, c in rows:
        top = max(c.values())
        mino = n - top
        share = top / n
        flag = "  <- SPLIT" if SPLIT[0] <= 1 - share and share <= SPLIT[1] else ""
        contested += n if flag else 0
        minority += mino
        cells = "".join(f"{100 * c[k] / n:>12.0f}%" for k in classes)
        print(f"{name:<30}{n:>6}{cells}   {mino:>5}{flag}")
        table.append({"lvis_type": name, "n": n, "by_class": dict(c), "minority": mino})

    # The rate per COCO class too: a merge helps a class exactly as much as its
    # own boxes sit in types another class also claims.
    print()
    for k in classes:
        own = sum(c[k] for _, _, c in rows)
        wrong = sum(c[k] for _, _, c in rows if max(c, key=c.get) != k)
        print(f"  `{k}`: {own:,} matched boxes, {wrong:,} ({100 * wrong / max(1, own):.1f}%) in a type another class wins")
    print(f"\n{contested:,} of {total:,} boxes ({100 * contested / total:.0f}%) sit in a type COCO splits 20-80%.")
    print(f"MINORITY-label boxes: {minority:,} of {total:,} = {100 * minority / total:.1f}%   <- the region-voting rate")
    print("(reference, #4056 on LVIS val: car/truck 9.6%, cup/wine glass 10.7% -- both merged)")
    if args.out:
        args.out.write_text(json.dumps({"classes": classes, "types": table, "minority": minority, "total": total}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
