#!/usr/bin/env python3
"""What else is inside a COCO class, read off a finer vocabulary that has definitions.

**A one-off class-selection instrument, not part of any build.** It answers the
question that costs a reviewer the most time -- *what does this class quietly
include?* -- without a review pass, and it needs LVIS only to ask it. Nothing
downstream depends on LVIS; once the note exists, the instrument is done.

COCO's annotation is a single forced choice with no disagreement recorded, so
definitional ambiguity is invisible from inside COCO: two COCO classes almost
never box the same pixels, because the vocabulary forbids it. LVIS re-annotated
COCO's images with 1,203 WordNet synsets that each carry a written definition, so
the boxes COCO calls `cup` split into `glass_(drink_container)` / `cup` / `mug` --
and that split is the fault line a reviewer would otherwise find by hand.

**Read the output, do not threshold it** (#3983 follow-up). Purity ranks class
*heterogeneity*, and heterogeneity is not what made review painful: `book` is 85%
pure with `magazine` at 6%, and `book` is the class whose review split 21 verdicts
against 49. What splits reviewers is an unwritten rule meeting a 4% minority, not
a 40% one. So the useful output is the **name list**, which tells you what to
write in `SCALE_CLASS_RULES`; the percentage only says how often it will come up.

Matching is **mutual best** at IoU >= 0.7. A one-way IoU >= 0.5 match pairs a COCO
`person` box with an LVIS `jacket` box of similar extent and reads it as impurity;
requiring each box to be the other's best candidate means they are the same object
under two vocabularies. Morphology is left in and NOT scored away -- `donut` ->
`doughnut`, `skis` -> `ski`, `tie` -> `necktie` are spellings, and a regex that
tried to fold them mis-ranked the classes worse than raw purity did.

    # annotations dir: instances_{train,val}2017.json
    # lvis: lvis_v1_val.json from https://www.lvisdataset.org/dataset
    python coco_class_purity.py --annotations <dir> --lvis lvis_v1_val.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

IOU_MIN = 0.7
MIN_MATCHES = 60


def _iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _read_boxes(path: Path, cat_key: str = "categories") -> tuple[dict[int, list], dict[str, str]]:
    """``({image_id: [(box, name)]}, {name: definition})`` from a COCO-shaped file."""
    with path.open() as fh:
        data = json.load(fh)
    cats = {c["id"]: c["name"] for c in data[cat_key]}
    defs = {c["name"]: c.get("def", "") for c in data[cat_key]}
    out: dict[int, list] = collections.defaultdict(list)
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        if w > 0 and h > 0:
            out[a["image_id"]].append(([x, y, x + w, y + h], cats[a["category_id"]]))
    return out, defs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--lvis", type=Path, required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    coco: dict[int, list] = collections.defaultdict(list)
    for split in ("val2017", "train2017"):
        part, _ = _read_boxes(args.annotations / f"instances_{split}.json")
        for iid, bs in part.items():
            coco[iid] += bs
    lvis, defs = _read_boxes(args.lvis)

    names: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for iid, cbs in coco.items():
        lbs = lvis.get(iid)
        if not lbs:
            continue
        grid = [[_iou(cb, lb) for lb, _ in lbs] for cb, _ in cbs]
        for i, (_, ccls) in enumerate(cbs):
            if not grid[i]:
                continue
            j = max(range(len(lbs)), key=lambda k: grid[i][k])
            # Mutual: the COCO box must also be the best COCO box for that LVIS box.
            if grid[i][j] < IOU_MIN or max(range(len(cbs)), key=lambda k: grid[k][j]) != i:
                continue
            names[ccls][lbs[j][1]] += 1

    rows = []
    for cls, ctr in names.items():
        total = sum(ctr.values())
        if total < MIN_MATCHES:
            continue
        dom, dom_n = ctr.most_common(1)[0]
        rows.append((cls, total, 100 * dom_n / total, dom, ctr))
    rows.sort(key=lambda r: r[2])

    print(f"mutual best match at IoU>={IOU_MIN}; {len(rows)} classes with >={MIN_MATCHES} matches\n")
    print(f"{'coco class':<15}{'n':>7}{'purity':>8}  {'LVIS dominant':<26} what else is in the class")
    print("-" * 112)
    for cls, total, pur, dom, ctr in rows:
        rest = ", ".join(f"{n} {100 * v / total:.0f}%" for n, v in ctr.most_common(4)[1:])
        print(f"{cls:<15}{total:>7,}{pur:>7.0f}%  {dom:<26} {rest}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    cls: {
                        "total": t,
                        "purity": p,
                        "dominant": d,
                        "names": dict(c),
                        "definitions": {n: defs.get(n, "") for n in c},
                    }
                    for cls, t, p, d, c in rows
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
