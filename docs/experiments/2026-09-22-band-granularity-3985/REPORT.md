# COCO boxes a pile as one object — and it is three faults, not one

**Issues:** #3985, #3992. **Dataset:** `coco_quarry`, *C* = 52.
**Date:** 2026-09-22. **Unit rulings and 120 owner votes, 2026-09-22.** Fruit remedy pending.

## Verdict

**The measurement is extended and corrected; the remedy is not chosen.** Two of
the worst-affected classes had never been measured, the single "area ratio"
turns out to conflate three different faults that want different fixes, and
three of the four classes #3985 named as lumpers are not lumping.

**The owner's votes (120) confirm lumping for the three fruit classes only.**
An estimated ~56% of `banana`'s images, ~39% of `apple`'s and ~23% of
`orange`'s have a box around a pile. For `skis` and `potted plant` the high
ratio flags a legitimate unit, so any guard must exempt them. No COCO-only
signal separates the piles; the LVIS ratio does.

| fault | classes in *C* | signature |
|---|---|---|
| **lumping** | `banana` 7.58/8.15, `apple` 4.48/5.69, `orange` 4.69/6.67 | count **and** area high — one box round many |
| **wider extent** | `potted plant` 1.40/5.92, `kite` 1.20/2.08, `car` 1.27/1.88, `boat` 1.29/1.76 | count ≈ 1, area large — same objects, bigger box |
| **structural pairing** | `skis` 2.09/2.74 | count ≈ 2 — a pair boxed as one, arguably correct |

(count ratio / area ratio, both LVIS against COCO on the same images.)

46 of 56 comparable classes agree.

## What was wrong with the previous reading

**`SAME` covered 21 classes when *C* was 25.** It now covers all 54 COCO
categories *C* spans. The two worst cases had no row at all:

| class | count | area | median | % images ≥1.6 |
|---|---:|---:|---:|---:|
| `potted plant` | 1.40 | 5.92 | **2.95** | **67%** |
| `skis` | 2.09 | 2.74 | **1.81** | **62%** |

**#3992's central claim does not survive.** It argued *"every median is ~1.00
except `book`"*, therefore the defect is a minority tail, therefore the guard
should be per-image. Five classes in *C* now sit above 1.4, and on
`potted plant` two images in three are affected.

**There is a baseline offset, and #3985's "clean gap" is partly an artefact of
ignoring it.** Across 56 classes the median area ratio is **1.13** and the
median count ratio **1.15**, with only 3 of 56 below 1.0. LVIS is a finer
vocabulary: it systematically finds ~15% more instances and boxes ~13% tighter.
So agreement is ~1.15, not 1.0, and `LUMP_AREA_RATIO = 1.6` is **1.4x baseline**
rather than a clean separation. `banana` at 8.15 and `potted plant` at 5.92 are
5–7x baseline and unaffected by this; the marginal classes — `sink` 1.33,
`bench` 1.36, `dining table` 1.36 — are much closer to noise than they look.

**`SAME` now maps to SETS of LVIS names**, derived from `coco_class_purity.py`'s
mutual-best-match table and then curated. The curation is the part that matters:
a blanket rule would have folded in `person`/`jacket`, `dining table`/
`tablecloth` and `apple`/`pear`, and each manufactures the very signal this
script looks for — more LVIS instances than COCO ones — out of a garment, a
covering, or a mislabel. Sets also retire `SUBTYPE_FLOOR`: `bird`, `cup` and
`bottle` were reported as lower bounds because LVIS filed their objects under
sibling names the table did not list, so the caveat described the table rather
than the data. `car` moves 2.09 → 1.88 once `minivan` and `cab_(taxi)` join it.

## The widening put the lumpers in

All three genuine lumpers — `banana`, `apple`, `orange` — entered *C* through
#4056's widening. None was in *C* when either issue was written, and the count
rule that admitted them asks whether a class has enough images, never whether
COCO boxes it as **individual objects**.

## A remedy that does not work

#3985 floats banding on the **largest** box rather than the union, noting that
`region_box_for_category`'s docstring only argued against picking one
*arbitrarily* and never against picking the largest. It does not help here: for
`banana` COCO draws **one** box round the bunch, so the largest box *is* the
lump. The change may still be right on its own merits; it is not a fix for this.

Nor is a COCO-only guard obviously reachable. #3992 asks for one and the
candidates it lists are all repetition signals — which by construction cannot
see the **wider extent** fault, where COCO's box count is already correct.

## What one object is — owner rulings, 2026-09-22

The first issue of the queues asked *"is the red box around ONE object?"* and
could not be answered: nothing said whether a pair of skis is one object. That
is a definition, not something an image can show, so it is ruled once per class
and recorded as `ClassRule.unit` in `pile_config`. The unit now appears in each
queue's name:

| class | one object is | so the fault is |
|---|---|---|
| `skis` | the pair one skier uses, or one loose ski | **not an error**: COCO's plural name, like `scissors` |
| `potted plant` | one pot or vase **with** its plant or flowers | **not an error**: LVIS's pot-only `flowerpot` box is the odd one out |
| `banana` / `apple` / `orange` | one fruit; a bunch, hand, pile or bowl is Bad | **an error**, joined at the stem or not |
| `scissors` | one pair, i.e. one tool | no fault: COCO and LVIS agree, count ratio 1.00 |

`unit` is separate from `test` on purpose. A bunch of bananas is all `banana`,
so it belongs to the class, but it is still more than one banana.

## The votes — 120 owner answers, 2026-09-22

Five queues of 24, half **lumped** (per-image ratio ≥ 1.6) and half **clean**
(< 1.2), shuffled. The arm could not be read from the filename. Raw answers:
[`votes-20260922.jsonl`](votes-20260922.jsonl), one row per question, joined
to the image, annotation, box and ratio.

| class | clean arm: Bad | lumped arm: Bad | images ≥ 1.6 | est. share of images with a pile box |
|---|---:|---:|---:|---:|
| `banana` | 1 / 12 | **11 / 12** | 61% | ~56% |
| `apple` | 0 / 12 | **10 / 12** | 46% | ~39% |
| `orange` | 0 / 12 | **7 / 12** | 40% | ~23% |
| `potted plant` | 0 / 12 | 1 / 12 | 67% | — |
| `skis` | 0 / 12 | **0 / 12** | 62% | — |

(The estimate multiplies the share of images at ≥ 1.6 by the lumped arm's Bad
rate. It covers LVIS-annotated images only, and with 12 votes per arm its
interval is wide: read it as "about half", "about a third", "about a fifth".)

**The attention control passed.** The clean arms are 59 Good out of 60, so the
Bad votes are real answers and not a button pressed on every question.

**For `skis` and `potted plant`, the high ratio is not a fault.** 23 of the 24
lumped-arm answers are Good: under the unit rulings, COCO's box is the object
and LVIS's finer box is a part of it (one ski, the pot). A guard built on the
ratio must **exempt these two classes**, or it would delete about two thirds of
their images for nothing. The one Bad is a single `potted plant` box around
several pots (ratio 14.9; COCO drew 2 boxes where LVIS drew 13).

**For the fruit, lumping is real and common.** At the 1.6 cut, the ratio flags
36 of the 72 fruit questions and catches 28 of the 29 Bads (precision 0.78).
Raising the cut trades recall for precision:

| cut | flagged | Bad caught | precision | recall |
|---:|---:|---:|---:|---:|
| 1.6 | 36 | 28 | 0.78 | 0.97 |
| 3.0 | 25 | 23 | 0.92 | 0.79 |
| 4.5 | 22 | 21 | 0.95 | 0.72 |

For example, `apple` at 18.1 is one COCO box where LVIS drew 7 separate
apples, and it was voted Bad. `orange` at 6.6 is a single large orange that
LVIS boxed tighter, and it was voted Good.

**No COCO-only signal separates the piles in this sample.** COCO's box count
for the class runs from 1 to 13 in both groups, and so does the raw box area.
Every signal that works needs LVIS, which covers 90% of `banana`'s images, 62%
of `apple`'s and 57% of `orange`'s. #3992's COCO-only guard is still not
found, and on this evidence it may not exist.

## What is still not settled

**The remedy for the three fruit classes.** The candidates are: keep only
LVIS-covered images below a ratio cut; use LVIS's per-fruit boxes wherever
LVIS covers the image; or drop the three classes from *C*.

## Limits

- **LVIS covers 16% of COCO** (19,626 of 122,218 images), and per class the
  coverage runs 14% (`dining table`) to 95% (`frisbee`). Every ratio here is
  measured on that subset.
- **The pair window is a heuristic and misfires.** `book` at 2.38 lands in it
  and books are not structurally paired — a shelf of books is plain lumping. The
  verdict says "check" rather than ruling, but it is not free of false flags.
- **Area ratio treats LVIS's box extent as ground truth**, and there is no
  strong reason to. Count ratio is the safer signal: *how many objects are
  there* has a right answer; *how tightly should the box hug* does not.
