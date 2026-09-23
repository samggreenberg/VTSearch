# COCO boxes a pile as one object — and it is three faults, not one

**Issues:** #3985, #3992, #4096. **Dataset:** `coco_quarry`, *C* = 52.
**Date:** 2026-09-22. **Unit rulings, 176 owner votes, the fruit and book remedy, and largest-instance banding, 2026-09-22.**

## Verdict

**The measurement is extended and corrected, and the remedy is built for fruit and `book`, on largest-instance banding (#4096).** Two of
the worst-affected classes had never been measured, the single "area ratio"
turns out to conflate three different faults that want different fixes, and
three of the four classes #3985 named as lumpers are not lumping.

**The owner's votes (120) confirm lumping for the three fruit classes only.**
An estimated ~56% of `banana`'s images, ~39% of `apple`'s and ~23% of
`orange`'s have a box around a pile. For `skis` and `potted plant` the high
ratio flags a legitimate unit, so any guard must exempt them. No COCO-only
signal separates the piles; the LVIS ratio does.
The rebuild uses that ratio: a fruit image is a positive only when LVIS boxed
the class and COCO's box is under 1.8× LVIS's. The three fruit `@small` cells
are dropped because nothing fills them, leaving **153 cells**.

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
Every signal that works needs LVIS boxes for the image. #3992's COCO-only guard is still not
found, and on this evidence it may not exist.

## The remedy, as built — owner rulings, 2026-09-22

> **Superseded in part by the second rebuild below.** The 1.8 ratio cut
> described here was replaced by a count of LVIS instances inside the picked
> box, and banding moved to the largest instance (#4096). The fruit rulings
> themselves stand.

**A fruit box is a positive only if LVIS says it is ONE fruit.** For `banana`,
`apple` and `orange`, an image stays a positive only if LVIS boxed the class
there **and** COCO's mean box area is under **1.8×** LVIS's. An excluded image
keeps its COCO label, so it is never scored as a negative for its own class
either. `skis` and `potted plant` are not filtered.

**LVIS train had to be fetched.** Only val (16% of COCO) was on disk, and with
val alone every fruit cell fell to 3–54 of its 100 positives. Train and val
together cover 119,979 of COCO's 123,287 images. They now live in
`$VTS_PILE/lvis/` (`pile_config.LVIS_DIR`).

**Why 1.8, on the owner's votes:**

| cut | piles caught (of 29) | Good images lost (of 43) | `apple@large` supply |
|---:|---:|---:|---:|
| 1.6 | 28 | 8 | 93 |
| **1.8** | **28** | **6** | **98** |
| 2.0 | 25 | 5 | 101 |

**The three fruit `@small` cells are dropped** (`SCALE_DROPPED_CELLS`). LVIS
rarely boxes a small fruit: only 81 of 166 small bananas have LVIS boxes, and
the cells end at 45 / 32 / 22 positives under any cut. Built that short, their
prevalence and therefore their AP would not be comparable with any other
cell. **C keeps 52 classes, now in 153 cells.** `apple@large` runs at 98 of
100, which the build logs as a warning.

**What the build reported:**

| class | COCO images | not positives (pile, or no LVIS box) |
|---|---:|---:|
| `banana` | 2,346 | 1,373 |
| `apple` | 1,662 | 955 |
| `orange` | 1,784 | 1,144 |

The build has 153 cells, 15,298 positives and 9,900 negatives, with the same
+1,000 spares as before.

**Rebuilt and verified 2026-09-22** (job 677654, v100, commit `316e078a8`).
`--verify` passes for all five columns. Against the previous build, kept at
`keep/coco-quarry-52-20260922/`:

- **147 of 153 cells have identical membership**, and all 10,900 negatives and
  spares are the same images. Selection is a per-image hash, so the filter
  moved only the six fruit cells it touches.
- **The six fruit cells kept 31–51 of their 100 images**. For example,
  `orange@medium` kept 31 and `banana@medium` kept 51. The rest were piles, or
  boxes LVIS never drew.
- **Vectors are unchanged**: 24,936 of the 25,012 shared images match
  bit-for-bit. The other 76 differ by at most 1e-6, which is batch-composition
  noise from a changed media set (#3683), not a change of input.

## Second rebuild — the largest instance (#4096), and `book`

### Band on the most obvious instance

The build used to band each image on the **union** of every instance of the
class, and the evaluator's simulated user dragged that same union. When the
instances were spread out (union > 1.5× the largest), the image was
*scattered* and never a positive. That was **111,324 of 287,303 image-class
pairs in C (39%)**, led by `traffic light` 70%, `book` 58% and
`enclosed road vehicle` 55%. So positives leaned toward images holding one
isolated object, and busy scenes were mostly missing.

The owner ruled to use the most obvious instance, the largest box. The build
now bands on it, and the media records **only that box** as the cell's region,
so the simulated user drags one object (`SCALE_BAND_ON_LARGEST`,
`scale_core.largest_box`, where ties go top-left, never to annotation order).
In a dry run over all of C, total banded supply rose from **166,845 to
270,921**. The largest gains were busy scenes: `person@medium` 6,634 → 19,518,
`traffic light@small` 745 → 2,555, `chair@small` 430 → 1,149.

### The pile test moves to the same box

The per-image mean-area ratio no longer describes the box that is dragged. The
test is now a count: **how many LVIS instances lie at least half inside the
picked box**. For the fruit, keep the box only when exactly one does. On the 72
fruit votes, which were cast on this very box, that catches 28 of 29 piles and
loses 5 of 43 Good boxes. The ratio at 1.8 caught the same 28 and lost 6.

### `book` needs its own rule — 56 owner votes over two rounds

| LVIS books inside the picked box | Good | Bad |
|---|---:|---:|
| 0 | 8 | 1 |
| 2–4 | **15** | **0** |
| 5 | 3 | 0 |
| 6–9 | 2 | 3 |
| 10+ | 1 | **9** |

Two votes of round 1 are left out of the table: a puzzle box and a toy book,
which were Bad because they are not books at all. Round 1 sampled the old
population with the ratio arms. Round 2 sampled the new population, 8 per
stratum.

- **LVIS sees neighbouring volumes inside one COCO book box.** Two to four LVIS
  books inside is still one book, 15 of 15 times. The fruit rule would have
  thrown those away.
- **Absence proves nothing for books.** 63% of picked book boxes have no LVIS
  book inside at all, so books do not require LVIS confirmation.
- **A stack at 6 or more inside** catches 12 of the 13 stacks and loses 3 Good
  books. The one stack it misses has no LVIS book inside, so no LVIS rule could
  catch it. The edge between 5 and 6 rests on few votes, and `pile_at` is one
  constant to move.
- **Largest-box banding brought the shelves back.** Five or more LVIS books
  inside is 1% of the population the old rule banded and 10% of the returning
  one, which is why `book` needed a rule now.

Each class carries a `LumpRule(lvis_names, pile_at, require_lvis)`: the fruit
use `(…, 2, True)` and `book` uses `(("book", "magazine"), 6, False)`. The fruit
`@small` cells stay dropped. Even on the largest box they reach only 35 / 83 / 60
positives (banana / apple / orange).

**Numbers from before this rebuild are not comparable with numbers after it.**
Membership changes in every multi-instance cell.

**Rebuilt and verified 2026-09-22** (job 679144, v100, commit `30de11db9`;
the only uncommitted file at build time was this report).

- **153 cells, 15,300 positives: every cell is full.** `apple@large`, 98 in
  the first rebuild, now fills from images the union rule had called
  scattered.
- **Excluded as positives** (not ONE by LVIS): `banana` 1,714 of 2,346,
  `apple` 1,058 of 1,662, `orange` 1,268 of 1,784, `book` 290 of 5,562.
- **`--verify` passes on all five columns**, 25,255 medias each.
- **Every positive (media, cell) pair now carries exactly one region**, so the
  simulated drag is one object in every case.
- **Against the previous 153-cell build:** all 10,900 negatives and spares are
  the same images. Membership moved in 152 of 153 cells, with a median of 70 of
  100 images kept. The busy-scene cells moved most: `traffic light@medium`
  kept 20, `person@small` 29, `book@small` 34. Vectors of the 21,218 shared
  images are unchanged: 16,683 match bit-for-bit, and the rest differ by at
  most 4e-7, which is batch-composition noise.


## Limits

- **Only LVIS v1 *val* was read: 16% of COCO** (19,626 of 122,218 images).
  Every ratio here is measured on that subset. The `cover` column
  (`dining table` 14% to `frisbee` 95%) is a share **within LVIS val**: of the
  LVIS-val images where COCO has the class, the fraction where LVIS also boxed
  it (LVIS annotates each class on only some images). It is **not** a share of
  the class's COCO images. An earlier draft of this report, and a message to the
  owner, read it that way: "LVIS covers 90% of banana's images" was really about
  16% × 90%.
- **The pair window is a heuristic and misfires.** `book` at 2.38 lands in it
  and books are not structurally paired — a shelf of books is plain lumping. The
  verdict says "check" rather than ruling, but it is not free of false flags.
- **Area ratio treats LVIS's box extent as ground truth**, and there is no
  strong reason to. Count ratio is the safer signal: *how many objects are
  there* has a right answer; *how tightly should the box hug* does not.
