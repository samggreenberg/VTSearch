# COCO boxes a pile as one object — and it is three faults, not one

**Issues:** #3985, #3992. **Dataset:** `coco_quarry`, *C* = 52.
**Date:** 2026-09-22. **Rulings pending** — see *What is not settled*.

## Verdict

**The measurement is extended and corrected; the remedy is not chosen.** Two of
the worst-affected classes had never been measured, the single "area ratio"
turns out to conflate three different faults that want different fixes, and
three of the four classes #3985 named as lumpers are not lumping.

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

## What is not settled

**Which of the three faults is an error.** The measurement says what COCO did;
it cannot say whether a pair of skis or a boxed flower arrangement is a
legitimate single object for this benchmark. Five VTSearch queues put that to
the owner — 24 questions each, half images the measurement calls lumped and half
it calls clean, shuffled, with the arm unreadable from the filename so the split
in the answers is itself the attention control:

```
coco_quarry <class> - is the red box around ONE object?
    banana · apple · orange · potted plant · skis
```

`queues/*.json` maps each crop back to its image, annotation and per-image
ratio, so the votes can be read against the exact boxes that produced them.

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
