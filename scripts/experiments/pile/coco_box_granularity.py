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

**The class mean is a tail statistic, so it is not the headline (#3985).** Every
class in *C* has a median area ratio of ~1.00 -- `boat` 1.01, `car` 0.99, `kite`
1.01, and only `book` moves at 1.22. A class-level verdict of "lumps" therefore
says a *minority* of its images carry a pile, not that its boxes are typically
around groups: `boat`'s 1.76 is a handful of marinas. The band error is per
image, so ``share_over`` -- the fraction of images at or above
:data:`LUMP_AREA_RATIO` -- is the number a guard would have to act on, and it is
4-12% for classes this script calls clean. Both columns are printed; read them
together, and never quote the mean alone.

**A fragmented LVIS counterpart makes the reading a FLOOR, not an estimate.**
:data:`SAME` pairs a COCO class with the one LVIS category that means the same
thing. Where LVIS splits the same objects across subtype names, the instances
under those names are invisible here: LVIS spells 676 of the birds on COCO's
`bird` images `pigeon`/`gull`/`flamingo`/`duck` against 1,883 plain `bird`, and
on COCO's `cup` images it writes `glass_(drink_container)` 1,094 and `wineglass`
415 beside `cup` 760. Those classes are kept -- dropping them would leave *C*
with no reading at all -- but their ratios are lower bounds, and
:data:`SUBTYPE_FLOOR` marks them in the output. (An earlier version of this
docstring claimed such classes were "left out rather than guessed at" while
`bottle` sat in :data:`SAME`; the table, not the prose, was right.)

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
#: does. A class whose LVIS counterpart fragments into subtypes is still listed,
#: because a floor on a class in *C* beats no reading at all -- see
#: :data:`SUBTYPE_FLOOR`, which is what the output warns on.
#:
#: Every class in *C* (:data:`pile_config.SCALE_CLASSES`) is here. The first run
#: (#3985) carried only 21 entries and left ten of *C* unmeasured, which read as
#: "clean" in the issue when it meant "never asked"; `sink` came back at 1.42
#: with 18% of its images over the threshold once it was.
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
    # The ten classes in *C* the first run never asked about.
    "backpack": "backpack",
    "bench": "bench",
    "bicycle": "bicycle",
    "cell phone": "cellular_telephone",
    "fire hydrant": "fire_hydrant",
    "fork": "fork",
    "sink": "sink",
    "spoon": "spoon",
    "stop sign": "stop_sign",
    "vase": "vase",
}

#: Classes whose LVIS counterpart is one name among several for the same objects,
#: so the instances LVIS files under a sibling name are invisible to the 1:1
#: match and the measured ratio is a LOWER BOUND. Value is the evidence, counted
#: on the images where COCO names the class.
SUBTYPE_FLOOR: dict[str, str] = {
    "bird": "pigeon 233, gull 169, flamingo 161, duck 113 vs bird 1,883",
    "cup": "glass_(drink_container) 1,094, wineglass 415 vs cup 760",
    "bottle": "LVIS splits by contents (water/wine/beer)",
}

#: Area ratio at which COCO's box is judged to be around a group rather than an
#: object. Set from the gap in the measured data, not from theory: over the 24
#: comparable classes the means run 1.03-1.44 and then 1.76-8.15, with `sink`
#: 1.42 and `bench` 1.36 the closest below. The gap is in the MEANS only -- the
#: per-image distribution is continuous, which is why ``share_over`` is printed
#: beside the verdict.
LUMP_AREA_RATIO = 1.6
MIN_IMAGES = 40


def _load(path: Path) -> tuple[dict, dict]:
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
        part, d = _load(args.annotations / f"instances_{split}.json")
        dims.update(d)
        for iid, per in part.items():
            for cls, bs in per.items():
                coco[iid][cls] += bs
    lvis, _ = _load(args.lvis)
    shared = set(coco) & set(lvis)

    rows, thin = [], []
    for cls, lname in SAME.items():
        counts, ratios, ca, la = [], [], [], []
        seen = 0
        for iid in shared:
            if coco[iid].get(cls):
                seen += 1
            cb, lb = coco[iid].get(cls), lvis[iid].get(lname)
            if not cb or not lb:
                continue
            counts.append(len(lb) / len(cb))
            ca.append(statistics.mean(cb) / dims[iid])
            la.append(statistics.mean(lb) / dims[iid])
            ratios.append(ca[-1] / la[-1])
        # A class LVIS barely overlaps is reported as uncompared, never dropped in
        # silence: "no row" and "a clean row" are not the same claim (#3985).
        if len(counts) < MIN_IMAGES:
            thin.append((cls, lname, seen, len(counts)))
            continue
        rows.append(
            (
                cls,
                len(counts),
                seen,
                statistics.mean(counts),
                statistics.mean(ratios),
                statistics.median(ratios),
                sum(r >= LUMP_AREA_RATIO for r in ratios) / len(ratios),
                100 * statistics.mean(ca),
                100 * statistics.mean(la),
            )
        )
    rows.sort(key=lambda r: -r[4])

    head = ("class", "imgs", "cover", "cnt ratio", "COCO box", "LVIS box", "area", "med", "over")
    print(
        f"{head[0]:<12}{head[1]:>7}{head[2]:>7}{head[3]:>11}{head[4]:>10}{head[5]:>10}"
        f"{head[6]:>7}{head[7]:>7}{head[8]:>7}   verdict"
    )
    print("-" * 98)
    for cls, n, seen, cnt, area, med, over, cbox, lbox in rows:
        if area >= LUMP_AREA_RATIO:
            verdict = "COCO LUMPS the pile"
        elif cnt >= 1.5:
            verdict = "mostly incompleteness"
        else:
            verdict = "agree"
        if cls in SUBTYPE_FLOOR:
            verdict += " (floor)"
        print(
            f"{cls:<12}{n:>7,}{100 * n / seen:>6.0f}%{cnt:>11.2f}{cbox:>9.1f}%{lbox:>9.1f}%"
            f"{area:>7.2f}{med:>7.2f}{100 * over:>6.0f}%   {verdict}"
        )

    for cls, lname, seen, n in thin:
        print(f"{cls:<12}{n:>7,}   NOT COMPARED: {seen:,} COCO images, under {MIN_IMAGES} carry LVIS `{lname}`")
    if SUBTYPE_FLOOR:
        print("\n(floor) LVIS files the same objects under sibling names, so the ratio is a lower bound:")
        for cls, why in SUBTYPE_FLOOR.items():
            print(f"    {cls:<10}{why}")
    print(
        "\n`area` is a mean over per-image ratios and rides on a tail; `med` is the median "
        "and\n`over` the share of images at or above "
        f"{LUMP_AREA_RATIO}, which is what a per-image guard would act on."
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "compared": {
                        r[0]: {
                            "images": r[1],
                            "coverage": r[1] / r[2],
                            "count_ratio": r[3],
                            "area_ratio": r[4],
                            "median_area_ratio": r[5],
                            "share_over_threshold": r[6],
                            "coco_box_pct": r[7],
                            "lvis_box_pct": r[8],
                            "floor": r[0] in SUBTYPE_FLOOR,
                        }
                        for r in rows
                    },
                    "not_compared": {
                        cls: {"lvis_name": lname, "coco_images": seen, "matched": n} for cls, lname, seen, n in thin
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
