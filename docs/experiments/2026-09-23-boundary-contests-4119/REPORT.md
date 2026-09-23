# Three boundary contests inside C, ruled

**Issue:** #4119. **Dataset:** `coco_quarry`, *C* = 52 → **49**. **Date:** 2026-09-23.

## Verdict

**Two merges and one boundary kept.** `backpack`, `handbag` and `suitcase`
become `bag or luggage`. `vase` and `potted plant` become `vase or potted
plant`. `skis` and `snowboard` stay separate. These are owner rulings, made
on the measurement below.

## The measurement

This is the #4056 test that merged `car`+`truck` and `cup`+`wine glass`. For
each LVIS object type, it asks what COCO's annotators called the same box, and
counts the share of boxes that carry the **minority** COCO label for their own
type. It measures boxes, not images, because region voting drags a box and
nothing else in the image rescues it. It is human annotation against human
annotation, with no detector. `boundary_contest.py` implements it: mutual best
match at IoU ≥ 0.5, on LVIS train and val together.

| contest | matched boxes | minority-label boxes | boxes in a type split 20–80% |
|---|---:|---:|---:|
| backpack / handbag / suitcase | 12,557 | **10.6%** | 5% |
| vase / potted plant | 5,798 | **6.1%** | 14% |
| skis / snowboard | 4,582 | **3.4%** | 0% |
| *car / truck (merged in #4056), same instrument* | *12,522* | *10.7%* | *15%* |

**The carriers.** The confusion is broad, not one bad type. LVIS suitcases are
filed as a backpack 6% of the time and as a handbag 6%, duffel bags split
41/31/28, and briefcases 61/39. `suitcase` on its own is clean: 2.9% of its
boxes sit in a type another class wins. `backpack` and `handbag` each lose
about 15%. Merging only backpack and handbag would leave 6.3%, most of it real
suitcases filed as a bag.

**Vase and potted plant.** Nearly all the confusion is one type: an LVIS
`flowerpot` is `vase` 39% of the time. `potted plant` itself is clean (1.1%).
The retired `vase not planters` rule asked reviewers to draw exactly the line
COCO did not draw.

**Skis and snowboard.** Every object type is at least 92% one label. The
minority is scattered labelling noise, and COCO carries this boundary.

## Why not an LVIS veto

With LVIS train, LVIS now covers 99% of these classes' images, so dropping a
box LVIS assigns to the other class became a real option. It was not in #4056,
when LVIS covered 16%. But LVIS matches only **37–48% of the boxes**, because
it annotates only some categories on each image. A veto would therefore fix
roughly half the confusion, while a merge fixes all of it and needs no LVIS at
build time.

## What a merge costs, and what it does not

- **No annotation is lost.** coco_quarry takes COCO's labels with no human
  corrections (`corrections={}`). The VG-era backpack and vase verdicts in
  `human_record/` stay readable, because the five retired rules are frozen in
  full in `SCALE_CLASS_RULES_FROZEN`. `apply_recheck.py` replays the vase
  recheck and stamps each row with the rule's digest, and the frozen rule
  reproduces the digest the record was written with, `29e5d90e768c`.
- **No annotation is needed.** A merged class is the union of COCO's labels.
- **C loses distinctions users do search for.** The opening queries name the
  halves: "a bag or suitcase" and "a vase or potted plant".

Raw per-type tables: `measurements/*.json`.

## Rebuilt, 2026-09-23

Job 688902 (v100), commit `cbdb8f207`, built from a clean tree.

- **144 cells, all full, 14,400 positives.** That is 49 classes × 3 bands,
  less the three fruit `@small` cells that were already dropped.
- **Against the previous 52-class build:** the 15 cells of the five member
  classes are replaced by 6 merged cells. **All 138 other cells have identical
  membership**, and all 10,900 negatives and spares are the same images. That
  is expected: the union covers exactly the same COCO classes, so no image
  changes whether it holds a class in *C*.
- **The merged cells are fresh draws.** Membership is a hash of
  `(cell, image)`, and the cell names are new, so only 6–18 of each merged
  cell's 100 were positives of a member class before.
- **Vectors are unchanged:** 18,952 of the 24,028 shared images match
  bit-for-bit, and the rest differ by at most 5e-7 (batch composition).
