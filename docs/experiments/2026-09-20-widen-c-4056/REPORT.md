# Widening *C* to 53 costs nothing in supply, and almost nothing in accuracy

**Issue:** #4056. **Dataset:** `coco_quarry`, COCO 2017 train+val, 123,287
images. **Date:** 2026-09-20. **Corrected 2026-09-20** — see *The correction*.

## Verdict

**The count rule admits 53 classes, the shared pool supports all of them, and
rebuilding costs a published cell −0.03 AP — almost all of it prevalence.**

*C* goes from 25 to 53 (`wine glass` held out; see below). Rebuilt and measured
against the preserved 25-class build, on the 75 cells that existed before:

| | `siglip` | `siglip2_l` |
|---|---:|---:|
| paired ΔAP, **as shipped** | **−0.0301 ± 0.0039** | **−0.0281 ± 0.0037** |
| paired ΔAP, **size-matched** | −0.0034 ± 0.0031 | **+0.0008 ± 0.0033** |
| paired ΔAUC | +0.0020 ± 0.0015 | +0.0024 ± 0.0014 |
| mean AP | 0.411 → 0.381 | 0.472 → 0.444 |

Three statistics agree on the same reading, and they are not redundant:

- **Size-matched ΔAP straddles zero** — −0.003 and +0.001, opposite signs, both
  inside one standard error. Hold the negative count still and widening *C*
  changes nothing about how hard a cell is.
- **ΔAUC is +0.002**, and AUC is prevalence-invariant by construction. Same
  answer from a statistic that cannot see the haystack's size.
- **So the −0.03 as shipped is prevalence**, and only prevalence: a rebuilt cell
  carries 23,891 negatives against 16,535, 44% more, and AP falls mechanically
  when the haystack grows.

![The pool really does empty, and it reaches the benchmark anyway](figures/widening-empties-the-pool.png)

**What a reader should do with a published `coco_quarry` number:** nothing, for
any conclusion that survives 0.03 AP, and re-read the cell against the current
build for anything finer. Rankings are untouched.

## The correction

**An earlier version of this report headlined +0.24 AP and said no published
number survived the change. That was wrong in sign and an order of magnitude
too large, and it was merged before it was caught (PR #4063).**

The first instrument drew both arms from each roster's **clean set** — the
barren component alone — and found that widening empties it: the mean count of
*C*-classes held by a clean pool image falls **1.163 → 0.000**, which made cells
read **+0.2447 ± 0.0230** (`siglip`) easier. Every number in that measurement is
reproducible and none of it is retracted.

**What was wrong was treating it as what a rebuild does.** A real
`coco_quarry` cell's negatives are barren **plus** #3667's cross-class
negatives, and the rebuild moves both:

| | *C* = 25 | *C* = 53 |
|---|---:|---:|
| clean candidates the barren pool is drawn from | 49,503 | **16,091** |
| classes of *C* held per clean pool image | 1.163 | **0.000** |
| barren negatives per cell | 9,900 | **9,900** |
| #3667 cross-class negatives per cell | 6,635 | **13,991** |
| total negatives per cell (median) | 16,535 | **23,891** |
| co-occurring share | 40% | **58%** |

**The barren pool never shrinks.** It is capped at `SCALE_N_NEG` = 9,900 in both
builds, so the emptiness the first instrument found is real but reaches a
fixed-size slice — and that slice's share of the cell falls from 60% to 42% as
the cross-class negatives more than double. The configuration the simulation
measured, where the barren component is the whole pool *and* varies with the
roster, is not a state `coco_quarry` ever occupies.

**The lesson is the one #3986 learned the hard way two hours earlier and this
report did not apply to itself**: a probe that changes one component of a
composite pool is measuring a benchmark nobody runs unless the other components
are held as they ship. The first instrument's `Limits` said it did "not build
the 84 new cells"; what it never said is that building them is exactly what
doubles the cross-class negatives and cancels the effect.

## The roster

All 25 shipped classes survive; 28 join. `coco_only_supply.py` reproduces #3983:
**159 cells, 0 short of `SCALE_N_POS` = 100**, pool headroom **1.48x**.

```
airplane, apple, banana, baseball bat, dining table, frisbee, handbag,
keyboard, laptop, microwave, motorcycle, mouse, orange, parking meter,
person, potted plant, remote, scissors, skateboard, skis, snowboard,
suitcase, surfboard, tennis racket, tie, toothbrush, traffic light, tv
```

**The rule is the count and nothing else.** Selection deliberately ignores
scatter and purity: both correlate with how hard a class is to detect, so
screening on either would make the benchmark easier and bias every result
optimistically. Scatter is recorded as a covariate.

### One exception, and it is about identity

`wine glass` clears the count rule and is **held out** (owner ruling,
2026-09-20). `SCALE_CLASS_MERGES` folds it into `cup`, so admitting it would
redefine `cup` and cost every published `cup` number its meaning. The purity
measurement agrees from the other side: `cup` is 39% `glass_(drink_container)`
and `wine glass` is 24% of that *same* synset, so these are not two disjoint
classes waiting to be split. Dropping it returns 33 images to the clean pool
(16,058 → 16,091). It is the only exception, and it is about class **identity**,
not difficulty.

## The rebuild

`--force`, from a clean checkout at `573c1908a`, 57 minutes on a V100S:

| | |
|---|---:|
| cells | 75 → **159** |
| designated positives | 7,500 → **15,900** |
| medias in the cell | 18,135 → **25,760** |
| columns rebuilt | 5 (`siglip`, `siglip2_l`, `clip`, `clip_l`, `dinov3_patch`) |

**Positives are identical in all 75 pre-existing cells**, which is what makes
the comparison paired rather than two different benchmarks; the instrument
asserts it and refuses to report a difference if it fails. `build_pile.py
--verify` passes.

**The 25-class build is preserved** at
`/expscratch/sgreenberg/keep/coco-quarry-25-20260920/`, byte-for-byte, because a
rebuild here is a *replace*: every old cell's negatives were drawn as *holds
none of the 25* and cannot survive a wider *C*. `SCALE_CLASSES_25` freezes the
roster; the copy is what makes that promise keepable rather than rhetorical.

**A first rebuild was discarded.** It stamped `WARNING: that checkout had
uncommitted tracked changes -- this cell is not reproducible` on all five
columns, because the loader docstring was edited and not committed before
launch. The diff was docstring-only and the cells were behaviourally identical,
but a reader following the provenance cannot tell a docstring edit from a logic
edit, so it was rebuilt from a committed tree rather than shipped with a
permanent asterisk.

## What COCO actually put in the 28

From `coco_class_purity.py` against LVIS's defined synsets. Four names misread
badly enough that a reader who trusts the name will misreport the result:

| class | what COCO boxed |
|---|---|
| `dining table` | **10.4%** `dining_table` — 34.2% `tablecloth`, 30.5% `table`. The object is the covered *surface*. |
| `tv` | 50.4% `television_set`, **46.9%** computer monitor. A screen class, not a television class. |
| `potted plant` | **75.1%** `flower_arrangement`, 20.7% `flowerpot`. Mostly cut flowers. |
| `remote` | 56.2% `remote_control`, 41.2% `control` — one object under two LVIS names, so effectively ~97% pure. |

`mouse` is **99.3%** `mouse_(computer_equipment)` and not once an animal, so an
animal there is an annotation error rather than a boundary case. `person` reads
64% pure, but that is an **artefact**: LVIS boxes the garment where COCO boxes
the wearer, and mutual best match pairs the two.

**Five of the 28 are cross-class pairs inside *C*** — the `truck`/`car`
situation, where each side looks right alone: `skis`/`snowboard`,
`backpack`/`handbag`/`suitcase`, `apple`/`orange`, `tv`/`laptop`,
`potted plant`/`vase`.

## The briefs, and the instrument that no longer exists

Every class in *C* needs a `SCALE_CLASS_RULES` brief —
`test_contents_and_rules_cover_the_same_classes` enforces parity with the
composition notes — so 28 were written.

They could not be written the way the first 25 were. `pile_config.py` said each
of those was "measured with `coco_folds.py` before it was written", and
**`coco_folds.py` was deleted by the VG retirement** (`e466e3a19`, #4038). That
was correct — fold-in asked which *VG name* lands on a COCO box, and there are
no VG names now — but three live documents still instruct a reader to run it,
filed as **#4060**.

So the briefs are written against `coco_class_purity.py`, which is the right
replacement rather than a fallback: under a pure-COCO build nobody reviews
COCO's labels, so the question changed to "what did COCO's annotators actually
put in this class".

## What this does and does not settle

**Settled:** the roster (53 classes, 159 cells, none short, 1.48x headroom), and
the cost of the rebuild (−0.03 AP, prevalence; composition null at −0.003 to
+0.001 and ΔAUC +0.002).

**Settled, and worth keeping separate from the above:** the emptiness mechanism
is real. The clean candidate set does collapse 49,503 → 16,091 and does empty
out 1.163 → 0.000. It fails to reach the benchmark only because `SCALE_N_NEG`
caps the barren draw. **A design that let the barren pool scale with its
candidate set would inherit the +0.24**, so this is a live constraint on any
future change to how the pool is sized, not a curiosity.

**Not settled:** whether per-class pools behave the same way. A per-class pool
is drawn as *{i : i does not hold A}*, which does not depend on *C* at all, so
it should be immune by construction; #4034 already made the pool a query at
export. Worth confirming rather than assuming.

**Unmeasured:** the 84 new cells' own numbers. This report measures what the
rebuild did to the cells that already existed, not how hard the new ones are.

## Limits

- **Two columns**, `siglip` and `siglip2_l`. They agree to 0.002 AP on the
  as-shipped arm and straddle zero on the size-matched one; `clip` and `clip_l`
  are not measured here.
- **One head family** — a linear head on whole-image embeddings, the production
  head, as in #3986. Patch and region arms unmeasured (#4043).
- **AP is per cell** (100 positives against the cell's negatives), matching
  #3986's convention, not the three-band 300-positive figure a shipped cell
  quotes.
- **The superseded simulation's numbers are kept** in `measurements/` and in the
  correction above rather than deleted, because the mechanism they measure is
  real and a future pool-sizing change will need them.
