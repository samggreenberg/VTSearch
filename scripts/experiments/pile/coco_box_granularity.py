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
#: because a floor on a class in *C* beats no reading at all. Since #3992 the
#: sibling names are listed in :data:`SAME` itself, so there is no floor.
#:
#: Every class in *C* (:data:`pile_config.SCALE_CLASSES`) is here. The first run
#: (#3985) carried only 21 entries and left ten of *C* unmeasured, which read as
#: "clean" in the issue when it meant "never asked"; `sink` came back at 1.42
#: with 18% of its images over the threshold once it was.
#: A COCO class, and the LVIS categories that name the SAME OBJECTS at finer
#: resolution. Derived from `coco_class_purity.py`'s mutual-best-match table
#: (every name below coincided with a COCO box of that class at IoU >= 0.5),
#: then curated, because the derivation alone is not safe.
#:
#: **Include** a sibling that names the same kind of thing more precisely -- a
#: species (`bird` / `duck`, `gull`), a variant (`bottle` / `wine_bottle`), a
#: model (`car_(automobile)` / `minivan`). Those are exactly the split that makes
#: LVIS's instance count the right second opinion.
#:
#: **Exclude** two shapes, both of which would manufacture the very signal this
#: script looks for -- more LVIS instances than COCO ones:
#:
#: * a PART or COVERING of the object, boxed in the same place. `person` /
#:   `jacket`, `wet_suit`: LVIS boxes the garment where COCO boxes the wearer, so
#:   counting both reads one person as two. `dining table` / `tablecloth` is the
#:   same relation.
#: * a DIFFERENT object COCO mislabelled. `apple` / `pear`, `orange` / `lemon`,
#:   `stop sign` / `street_sign`. Those are label errors (and are already
#:   recorded in :data:`pile_config.SCALE_CLASS_CONTENTS`); folding them in here
#:   would count another object as a finer view of this one.
#:
#: Sets rather than single names also retires the "(floor)" caveat the first
#: version carried for `bird`, `cup` and `bottle`: LVIS filed those objects under
#: sibling names, so a one-name comparison understated LVIS and INFLATED the
#: ratio. The floors were an artefact of the table, not a property of the data.
SAME: dict[str, tuple[str, ...]] = {
    # --- the original twenty-five's COCO categories ---------------------------
    "backpack": ("backpack",),
    "bench": ("bench",),
    "bicycle": ("bicycle",),
    "bird": ("bird", "duck", "gull", "pigeon", "goose", "flamingo"),
    "boat": ("boat",),
    "book": ("book", "magazine"),
    "bottle": ("bottle", "wine_bottle", "water_bottle", "beer_bottle"),
    "bowl": ("bowl",),
    "bus": ("bus_(vehicle)", "school_bus"),
    "car": ("car_(automobile)", "minivan", "cab_(taxi)"),
    "cell phone": ("cellular_telephone",),
    "chair": ("chair", "armchair", "deck_chair"),
    "clock": ("clock", "wall_clock", "alarm_clock"),
    "cup": ("cup", "glass_(drink_container)", "mug", "teacup"),
    "dog": ("dog",),
    "fire hydrant": ("fireplug",),
    "fork": ("fork",),
    "kite": ("kite",),
    "knife": ("knife",),
    "sink": ("sink", "kitchen_sink"),
    "spoon": ("spoon", "ladle"),
    "stop sign": ("stop_sign",),  # NOT street_sign: a different sign
    "truck": ("truck", "pickup_truck", "trailer_truck", "fire_engine", "garbage_truck"),
    "umbrella": ("umbrella",),
    "vase": ("vase", "flowerpot"),
    # --- added to C by #4056's widening --------------------------------------
    "airplane": ("airplane", "fighter_jet", "jet_plane"),
    "apple": ("apple",),  # NOT pear: a different fruit COCO mislabelled
    "banana": ("banana",),
    "baseball bat": ("baseball_bat",),
    "dining table": ("dining_table", "table"),  # NOT tablecloth: the covering
    "frisbee": ("frisbee",),
    "handbag": ("handbag", "shoulder_bag", "tote_bag"),
    "keyboard": ("computer_keyboard",),
    "laptop": ("laptop_computer",),
    "microwave": ("microwave_oven",),
    "motorcycle": ("motorcycle", "motor_scooter", "dirt_bike"),
    "mouse": ("mouse_(computer_equipment)",),
    "orange": ("orange_(fruit)", "mandarin_orange"),  # NOT lemon
    "parking meter": ("parking_meter",),
    "person": ("person",),  # NOT jacket/wet_suit: the garment, not the wearer
    "potted plant": ("flower_arrangement", "flowerpot"),
    "remote": ("remote_control", "control"),
    "scissors": ("scissors",),
    "skateboard": ("skateboard",),
    "skis": ("ski",),
    "snowboard": ("snowboard",),
    "suitcase": ("suitcase",),
    "surfboard": ("surfboard",),
    "tennis racket": ("tennis_racket",),
    "tie": ("necktie", "bow-tie"),
    "toothbrush": ("toothbrush",),
    "traffic light": ("traffic_light",),
    "tv": ("television_set", "monitor_(computer_equipment) computer_monitor"),
    "wine glass": ("wineglass",),
    # --- context: measured but NOT in C, kept because #3985 quotes them -------
    "donut": ("doughnut",),
    "sheep": ("sheep",),
}

#: Classes whose LVIS counterpart is one name among several for the same objects,
#: so the instances LVIS files under a sibling name are invisible to the 1:1
#: match and the measured ratio is a LOWER BOUND. Value is the evidence, counted
#: on the images where COCO names the class.
#: RETIRED. `bird`, `cup` and `bottle` used to be reported as lower bounds,
#: because LVIS filed their objects under sibling names that :data:`SAME` did not
#: list -- `pigeon`, `gull`, `flamingo`; `glass_(drink_container)`, `mug`;
#: `water_bottle`, `wine_bottle`. A one-name comparison undercounted LVIS and so
#: INFLATED the ratio, and the caveat described the table rather than the data.
#: :data:`SAME` now carries the sibling sets, so there is no floor left to warn
#: about; the entry is kept as a comment because the output used to print it and
#: a reader coming back to an older run needs to know why it stopped.
SUBTYPE_FLOOR: dict[str, str] = {}

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


#: A count ratio at or above this is COCO merging INSTANCES, not merely drawing
#: wider. Below it, LVIS finds about as many objects as COCO does and the extra
#: area is extent rather than lumping.
LUMP_COUNT_RATIO = 1.5

#: A count ratio in this window is suspiciously close to exactly two, which is
#: what a class of PAIRED objects looks like -- `skis` measures 2.09. A pair of
#: skis is arguably ONE object in use, so this is a definitional difference
#: rather than an annotation error, and a guard that rejected it would be
#: throwing away good data. The instrument cannot tell the two apart, so it
#: flags rather than rules.
PAIR_WINDOW = (1.8, 2.4)


def _verdict(area: float, cnt: float) -> str:
    """Which fault, if any. Three hide under a single area ratio.

    The area ratio alone cannot separate them, and they want different fixes:

    * **lumping** -- one box round several instances. LVIS finds many more
      objects AND COCO's box is much bigger: `banana` 7.58 / 8.15.
    * **wider extent** -- the same objects, drawn around more. Count ratio near
      one, area ratio large: `potted plant` 1.40 / 5.92, which is plausibly
      pot-plus-plant against LVIS's foliage. No guard looking for repetition
      will ever see this one.
    * **structural pairing** -- count ratio near exactly two, because the class
      comes in pairs and COCO boxes the pair: `skis` 2.09.
    """
    if area < LUMP_AREA_RATIO:
        return "agree" if cnt < 1.5 else "mostly incompleteness"
    if cnt < LUMP_COUNT_RATIO:
        return f"WIDER EXTENT, same count (area/count {area / cnt:.1f}x)"
    if PAIR_WINDOW[0] <= cnt <= PAIR_WINDOW[1]:
        return "LUMPS -- or a PAIRED class, check"
    return "COCO LUMPS the pile"


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
    for cls, lnames in SAME.items():
        counts, ratios, ca, la = [], [], [], []
        seen = 0
        for iid in shared:
            if coco[iid].get(cls):
                seen += 1
            cb = coco[iid].get(cls)
            lb = [a for n in lnames for a in (lvis[iid].get(n) or [])] or None
            if not cb or not lb:
                continue
            counts.append(len(lb) / len(cb))
            ca.append(statistics.mean(cb) / dims[iid])
            la.append(statistics.mean(lb) / dims[iid])
            ratios.append(ca[-1] / la[-1])
        # A class LVIS barely overlaps is reported as uncompared, never dropped in
        # silence: "no row" and "a clean row" are not the same claim (#3985).
        if len(counts) < MIN_IMAGES:
            thin.append((cls, lnames, seen, len(counts)))
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
        verdict = _verdict(area, cnt)
        if cls in SUBTYPE_FLOOR:
            verdict += " (floor)"
        print(
            f"{cls:<12}{n:>7,}{100 * n / seen:>6.0f}%{cnt:>11.2f}{cbox:>9.1f}%{lbox:>9.1f}%"
            f"{area:>7.2f}{med:>7.2f}{100 * over:>6.0f}%   {verdict}"
        )

    for cls, lnames, seen, n in thin:
        print(f"{cls:<12}{n:>7,}   NOT COMPARED: {seen:,} COCO images, under {MIN_IMAGES} carry LVIS {list(lnames)}")
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
                        cls: {"lvis_names": list(lnames), "coco_images": seen, "matched": n}
                        for cls, lnames, seen, n in thin
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
