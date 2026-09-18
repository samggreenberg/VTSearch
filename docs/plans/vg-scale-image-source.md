# Migrate `vg_scale` off Visual Genome onto COCO

**Decided 2026-09-18.** `vg_scale` moves to COCO 2017 train+val as its image
pool, and Visual Genome leaves the construction entirely. This plan holds the
migration and the work it retires.

The decision rests on one measurement, not on preference:
[`docs/experiments/2026-09-18-coco-only-supply-3983/`](../experiments/2026-09-18-coco-only-supply-3983/REPORT.md)
(#3983). COCO alone supplies **all 75 cells** at the shipped `SCALE_N_POS` —
thinnest `bus@small` at 177, 1.8x the floor — with 49,503 shared-pool candidates
against the 9,900 needed. The class axis roughly doubles: 54 of COCO's 80 classes
clear 100 in all three bands, 29 beyond the current 25.

## Background: why VG was carrying the cost and not the load

Recorded because the migration only makes sense against it.

The construction had already migrated off VG's *annotation*, one well-argued
issue at a time, without anyone re-asking about the *source*: all 25
`SCALE_CLASSES` are COCO-2017 classes, `anchor_to_coco` replaces VG's labels
with COCO's wherever COCO annotates, and since #3670 the negative pool is 100%
COCO-scored with VG-silence contamination zero by construction. What remained
VG's was ~43% of the positives — and the entire apparatus built to guess at
labels VG cannot give: `SCALE_VG_NAMES`, `SCALE_VG_AMBIGUOUS`,
`name_evidence.py`, `coco_folds.py`, `vg_name_families.py`,
`scan_name_overlap.py`, `pool_contamination.py`, `withheld_difficulty.py`.

The price of that half: **4,709 correction rows** and **5,904 human judgements**,
against a source whose recall over *C* is 0.61 and whose boxes sit on a smaller
instance than the frame's main one 8.3% of the time (#3924, #3925). There is no
cheaper read available either — every one of VG's 2,516,939 objects carries a
`names` list of length one (#3618).

**What the switch does not buy is difficulty.** Supply is not hardness, and
`anchor_to_coco`'s claim that dropping VG's non-COCO half loses "VG's non-COCO
diversity for nothing" is still unmeasured. It is the one live argument for VG
and it is cheap to settle — see the first item below.

## Open work

<!-- item-sep -->

- **Settle the diversity claim before the rebuild, not after.** The only
  remaining argument for VG is that its non-COCO half is *differently
  distributed*, not merely additional. Measure it: are off-COCO positives harder,
  or differently composed, than COCO-anchored ones at matched band and class?
  `provenance_probe.py` and `provenance_shortcut.py` already read provenance off
  the vectors (AUC 0.53–0.56) and are the instrument. A null result closes the
  question and the docstring claim comes out; a positive one is a reason to keep
  a VG arm as a named contrast rather than as the pool. This is the one item that
  could still change the decision, so it runs first and it is cheap. (Sonnet 5)

<!-- item-sep -->

- **Build the COCO loader, and keep the band rule identical.** A `kind: "coco"`
  sibling of `pilebuild/loaders/vg_scale.py` reading `instances_*2017.json`
  directly: no `anchor_to_coco`, no `canonicalise`, no `lift_ambiguous`, no
  corrections file. `band_for` is imported unchanged — that is what makes the old
  and new sets comparable at all, and
  [`coco_only_supply.py`](../../scripts/experiments/pile/coco_only_supply.py)
  already demonstrates the read end to end. Decide `iscrowd` explicitly (the
  supply census drops crowd regions as not-an-instance; the builder should do the
  same and say so). (Opus 4.8)

<!-- item-sep -->

- **Decide what the new set is called, and do not overwrite the old one.**
  `pile_config` already warns that rebuilding a cell silently changes what it is,
  and five-plus studies (#3115, #3196, #3287, #3290, #3318, #3319) are conditioned
  on the VG-built cells. Reproducing those is no longer a reason to keep cleaning
  VG — they can be re-run — but it is a reason not to reuse the *name*. A new
  dataset key leaves the old cells readable while nothing new is built on them.
  (human)

<!-- item-sep -->

- **Re-validate the band rule against COCO's boxes at full scale.** #3637 scored
  the band statistic against COCO's exhaustive boxes on the VG∩COCO overlap and
  found COCO right where VG disagreed. Under pure COCO that overlap becomes the
  whole dataset, so the scatter guard and the oversize cut now act on adjudicated
  boxes only. Re-measure what they reject and at what rate — `BAND_MAX_INFLATION`
  was tuned against VG's multiply-annotated objects (21 bench boxes on one image),
  which COCO does not produce. (Sonnet 5)

<!-- item-sep -->

- **Retire the inference machinery, and delete it.** The name tables, the
  two-search candidate hunt, pooled adjudication, `pool_contamination.py`,
  `withheld_difficulty.py`, the `provable`/`matched` composition switch (#3702),
  #3655's global ambiguous exclusion and #3659's silent un-banding all exist to
  substitute inference for an answer VG cannot give. Under COCO there is no
  question for them to answer. They are experiment-tier code, so the
  backwards-compatibility rules do not protect them — delete rather than
  deprecate, and prune the README sections and plan pointers that describe them.
  Retire or re-caption the published numbers conditioned on the old construction
  at the same time. The risk is deleting something still load-bearing for a
  published result, so do it after the rebuild, not before. (Opus 4.8)

<!-- item-sep -->

- **Keep `SCALE_CLASS_RULES`, and re-aim it.** Under VG it recorded where a
  reviewer's reading had to be pinned down. Under COCO the definitions become
  *inherited* — COCO annotates magazines as `book`, which is what split #3612's
  review — so the table's job changes from adjudicating to **documenting where
  COCO's class boundary differs from plain English**. That is still load-bearing
  for anyone reading a result, and it is the one piece of the review programme
  that survives the switch. (Sonnet 5)

<!-- item-sep -->

- **Move prevalence, band and class list to eval-time queries.** The point the
  old exhaustive-annotation plan was aiming at, and the switch delivers its
  precondition for free: COCO is exhaustively annotated, so a cell becomes a
  filter over a fixed set rather than a build-time designation. `SCALE_N_POS` /
  `SCALE_N_NEG` / `SCALE_PREVALENCE` stop being re-embeds, and a class-list change
  stops being a rebuild. This is the item that turns the benchmark into a dataset.
  (Opus 4.8)

<!-- item-sep -->

- **Widen *C* to every class meeting the count requirement — 54 of COCO's 80.**
  The selection rule is the count and nothing else: a class is in if it clears
  `SCALE_N_POS` in all three bands. Measured, that is **54 classes, 29 beyond the
  current 25**, and the shared negative pool survives all of them — 16,058 clean
  images against the 10,900 the pool and its spares need (**1.5x**, against 4.5x
  at 25). Per-class supply is therefore not what caps the list; the pool is, and
  it does not bind.

  **Do not select on scatter, purity, or anything else correlated with
  difficulty.** Scatter is a proxy for how hard a class is to detect, so building
  *C* out of low-scatter classes would make the benchmark systematically easier
  and bias every result optimistically — a selection rule that correlates with
  the quantity being measured is a confound, not a convenience. Record scatter as
  a **covariate** beside a result instead; scatter-diversity across a wider *C*
  is a property worth having rather than a cost to manage. The same logic retires
  the purity idea: #3983 showed it does not predict review pain anyway, and even
  had it, it would have been the wrong axis to filter on.

  Two facts the census settles in advance. The classes VG blocked for
  *vocabulary* reasons all return (`motorcycle`, `surfboard`, `skateboard`,
  `snowboard` stop being aliases of `bike`/`board`; `potted plant` goes from 1
  small-band image to 299). And #3603's structural finding is independently
  confirmed and still binds — `giraffe` has 3 small-band images in all of COCO
  against 14 in VG — so the scene-exclusive easy end cannot be widened whatever
  the source; everything else can. (Sonnet 5)

<!-- item-sep -->

- **Decide `person` separately from the other 28.** It qualifies on counts, and
  it is the only candidate whose inclusion is a genuine trade: it holds 66,808
  images and costs the shared negative pool **26,132** on its own, more than the
  other 28 candidates combined. With it the pool sits at 1.5x over-subscription,
  without it 2.1x. That is a decision about pool headroom, not about `person`
  being awkward, and it wants a measured negative-retirement rate rather than a
  feel — at 1.6x, 68% of the clean pool is spoken for and there is little room to
  retire a contaminated negative later, which is what spares are for (#3670). (human)

<!-- item-sep -->

- **Decide the `truck`/`car` pair on the measurement.** 17% of COCO `truck` boxes
  are objects LVIS calls `car_(automobile)`, against 2% the other way. #3588 added
  `truck` beside `car` as a same-scene partner; the asymmetry says the pair is
  partly one population relabelled rather than two. Either keep both and say so
  where results are read, or drop `truck` — it is the one genuine boundary
  contest the census surfaced among the current 25. (human)

<!-- item-sep -->

- **Write the class notes from the purity table, once, and ship them.**
  [`coco_class_purity.py`](../../scripts/experiments/pile/coco_class_purity.py)'s
  name list is the text `SCALE_CLASS_RULES` needs — `cup` is
  glass/cup/mug, `book` includes magazines, `stop sign` includes 20% generic
  `street_sign`. A minute per class against a review pass, and it is what would
  have prevented #3612's split. This is the whole of the definitional work the
  migration still owes. (Sonnet 5)

<!-- item-sep -->

- **Ask the questions the set was built for.** Co-occurrence (does a class get
  harder when a same-scene partner is present?), natural-composition negatives
  instead of a designed ratio, multi-label arms, and calibration at a prevalence
  chosen per question. None are reachable under a designation; all are one query
  away under an annotated set. File them as they become concrete rather than
  listing them here. (Sonnet 5)

<!-- item-sep -->

## If the class list ever needs to leave COCO's 80

Not owed, and recorded so the next person feeling the pull toward a free-text
source finds the comparison already made.

**LVIS** is the first choice by some distance: built on the *same* COCO images,
so `COCO_ROOT` and the coordinate space carry over unchanged; 1,203 WordNet
synsets **with written definitions**, which is exactly what `SCALE_CLASS_RULES`
hand-writes; and **federated** annotation, giving every category an explicit
*negative image set* — images a human verified do not contain it. That is
`labels_exhaustive` as a delivered artifact. Caveats: federated is not per-image
exhaustive, so #3667's cross-class negative rule needs restating per category,
and tail categories are thin, so at 100 positives per band the usable vocabulary
is the head few hundred.

Then **Objects365 v2** (365 dense classes, ~2M images — best raw supply, no
explicit negatives, no COCO overlap so a new pool to embed), **Open Images V7**
(600 boxable classes with human-verified *negative* image-level labels at scale,
but `group-of` complicates a box→size rule and the framing differs from COCO's),
and **ADE20K** — far too thin to build cells from at ~27k images, but the right
**gold audit set** for silence and box-quality rates, which #3696 measured against
no reference at all.

## Related

- [`vg-scale-bands-and-corrections.md`](vg-scale-bands-and-corrections.md) — the
  band construction the migration preserves unchanged.
- [`docs/experiments/2026-09-17-vg-scale-corrections/REPORT.md`](../experiments/2026-09-17-vg-scale-corrections/REPORT.md)
  — the log of what VG got wrong, and the source of the rates quoted above.
- [`docs/experiments/2026-09-03-vg-scale-classes/REPORT.md`](../experiments/2026-09-03-vg-scale-classes/REPORT.md)
  — VG's class-list ceiling, and the definition-risk instrument.
