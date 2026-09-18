#!/usr/bin/env python3
"""Does COCO box a PILE as one object? Ask a finer annotation of the same images.

The scatter guard (:func:`pilebuild.loaders.vg_scale.band_for`) catches instances
that are *spread out*: the union outgrows the largest box, so the image is
dropped. It cannot catch the opposite failure, and that failure is the harder
one. **A stack is compact.** Five books on a shelf boxed as one lump have a union
equal to the lump, an inflation ratio of 1.0, and they sail through the guard --
after which the image is banded by the size of the *pile*, and nothing downstream
can tell.

COCO's explicit mechanism for a group is ``iscrowd=1``, and it is not the vector:
only 2.6% of `book` annotations carry it. The lumping happens inside ordinary
``iscrowd=0`` boxes, so it has to be measured against a second opinion.

**Instance counts alone cannot answer it**, because two different faults both
raise LVIS's count above COCO's:

* COCO drew one box around a pile -- **lumping**, and the band is then wrong;
* COCO annotated fewer instances of the same size -- **incompleteness**, which
  costs supply but leaves each box honest.

Box *area* separates them. Under lumping COCO's mean box is much larger than
LVIS's for the same object; under incompleteness the two match. So this reports
both ratios and only calls lumping when the area ratio moves.

Only classes with an unambiguous 1:1 LVIS name are compared -- :data:`SAME` is
hand-written for that reason, and a class whose LVIS counterpart is a set of
subtypes (`bottle` -> wine/water/beer) is left out rather than guessed at.

Written for #3983's follow-up. Results:
`docs/experiments/2026-09-18-coco-only-supply-3983/`.

    python coco_box_granularity.py --annotations <dir> --lvis lvis_v1_val.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

#: COCO class -> the LVIS category that means the same thing, where exactly one
#: does. Classes whose LVIS counterpart splits into subtypes are absent on
#: purpose: summing a subtype family would mix this measurement with the
#: composition question `coco_class_purity.py` answers.
SAME: dict[str, str] = {
    "banana": "banana",
    "apple": "apple",
    "orange": "orange_(fruit)",
    "book": "book",
    "donut": "doughnut",
    "sheep": "sheep",
    "suitcase": "suitcase",
    "bird": "bird",
    "chair": "chair",
    "umbrella": "umbrella",
    "car": "car_(automobile)",
    "kite": "kite",
    "bottle": "bottle",
    "boat": "boat",
    "cup": "cup",
    "bowl": "bowl",
    "knife": "knife",
    "dog": "dog",
    "truck": "truck",
    "bus": "bus_(vehicle)",
    "clock": "clock",
}

#: Area ratio at which COCO's box is judged to be around a group rather than an
#: object. Set from the gap in the measured data, not from theory: the classes
#: that agree land at 1.04-1.32 and the lumped ones at 1.76-8.15, with nothing
#: in between.
LUMP_AREA_RATIO = 1.6
MIN_IMAGES = 40


def _load(path: Path, areas: bool) -> tuple[dict, dict]:
    with path.open() as fh:
        data = json.load(fh)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    dims = {im["id"]: im["width"] * im["height"] for im in data["images"]}
    per: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        per[a["image_id"]][cats[a["category_id"]]].append(a["bbox"][2] * a["bbox"][3])
    return per, dims


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--lvis", type=Path, required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    coco: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    dims: dict = {}
    for split in ("val2017", "train2017"):
        part, d = _load(args.annotations / f"instances_{split}.json", areas=True)
        dims.update(d)
        for iid, per in part.items():
            for cls, bs in per.items():
                coco[iid][cls] += bs
    lvis, _ = _load(args.lvis, areas=True)

    rows = []
    for cls, lname in SAME.items():
        counts, ratios, ca, la = [], [], [], []
        for iid, per in lvis.items():
            cb, lb = coco.get(iid, {}).get(cls), per.get(lname)
            if not cb or not lb:
                continue
            counts.append(len(lb) / len(cb))
            ca.append(statistics.mean(cb) / dims[iid])
            la.append(statistics.mean(lb) / dims[iid])
            ratios.append(ca[-1] / la[-1])
        if len(counts) < MIN_IMAGES:
            continue
        rows.append(
            (
                cls,
                len(counts),
                statistics.mean(counts),
                statistics.mean(ratios),
                100 * statistics.mean(ca),
                100 * statistics.mean(la),
            )
        )
    rows.sort(key=lambda r: -r[3])

    print(f"{'class':<11}{'imgs':>7}{'cnt ratio':>11}{'COCO box':>10}{'LVIS box':>10}{'area':>8}   verdict")
    print("-" * 78)
    for cls, n, cnt, area, cbox, lbox in rows:
        if area >= LUMP_AREA_RATIO:
            verdict = "COCO LUMPS the pile"
        elif cnt >= 1.5:
            verdict = "mostly incompleteness"
        else:
            verdict = "agree"
        print(f"{cls:<11}{n:>7,}{cnt:>11.2f}{cbox:>9.1f}%{lbox:>9.1f}%{area:>8.2f}   {verdict}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    r[0]: {
                        "images": r[1],
                        "count_ratio": r[2],
                        "area_ratio": r[3],
                        "coco_box_pct": r[4],
                        "lvis_box_pct": r[5],
                    }
                    for r in rows
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
