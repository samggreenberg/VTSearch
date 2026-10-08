# `coco_better`: migrate `vg_scale` off Visual Genome onto COCO

**Decided 2026-09-18.** `vg_scale` moves to COCO 2017 train+val as its image
pool, Visual Genome leaves the construction entirely, and the result is named
**`coco_better`**. This plan holds the migration and the work it retires.

**Why that name.** It is a fixed body of material you cut blocks out of to spec,
which is what the thing actually is once band, prevalence and class list become
export-time queries (#3987). Nothing parameter-like is baked in, and that is the
point: `vg_scale_any` and `vg_scale_deep` exist *because* band and depth were
baked into a name, so every new question needed a new dataset. A `coco_better`
version is described by its parameters — `bus@small`, 200:20k, π = 0.99% — and
the dataset keeps one name however many versions are cut from it. "Banded-COCO"
was the proposal it replaces; it names one queryable axis as though it were the
defining property, which is the shape that produced the `_any` / `_deep`
siblings. ("Pile" was unavailable: it already means the shared pre-embedded
cache.)

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

**What the switch does not buy is difficulty — and it does not cost any
either.** Supply is not hardness, and `anchor_to_coco` (retired with VG) claimed
that dropping VG's non-COCO half loses "VG's non-COCO diversity for nothing".
#3997 measured it: off-COCO is differently distributed (AUC 0.53–0.54 matched)
but **not harder** on the shipped head, so there is no difficulty argument for
keeping VG.

## Open work

**State, 2026-09-23: ready for studies.** Both builds are current and verified: the designated `coco_better`, and `coco_better_full` with every image in COCO 2017. The definitional work is finished. What remains is the studies the set exists to support.

**What the set is now:**

- **49 classes in 144 cells.** Each cell holds 100 positives against a shared pool of 9,900 negatives (+1,000 spares), about 1% prevalence. The three fruit `@small` cells are dropped (`SCALE_DROPPED_CELLS`) because no honest supply fills them.
- **The owner's rulings are built in:**
  - Two merges: `enclosed road vehicle`, `single serving drinking vessel` (#4056), `bag or luggage` and `vase or potted plant` (#4119).
  - One unit per class (`ClassRule.unit`).
  - LVIS pile filters for fruit and `book` (#3985).
  - Banding and the simulated drag use the **largest instance**, not the union (#4096).
- **Each positive carries exactly one region.** That is the box the simulated user drags.

**Using it in a study:**

- **Environment:** `CALIB_DATASETS=coco_better`.
- **Arms:** the whole-image arm (`siglip`) and the region arm (`siglip+dinov3_patch`, which opens in SigLIP and learns in DINOv3 space). `CALIB_COCO_BETTER_EMBEDDERS` overrides them.
- **Other prevalences or pool shapes:** cut a manifest from `coco_better_full` with `coco_better_export.py`. Nothing is re-embedded.
- **Vectors are unit-norm** in the pile (`build_pile.py --verify` enforces it), and the harness renormalises on load (#4095, #4099).
- **Comparability boundary:** numbers measured on `coco_better` before #4106 (2026-09-22) and #4124 (2026-09-23) are **not comparable** with later ones. Membership changed in almost every cell.
- **After a rules change:** `build_pile.py --datasets coco_better_full,... --relabel` rewrites the full corpus's labels in about 8 minutes. It never re-embeds, and it refuses to write if a vector moved (#4091). The designated build is a `--force` rebuild, about 1 hour on a v100.

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- **Use it as a bench for METHOD comparisons (owner, 2026-09-23).** The data is
  never ours to choose in the field, so the value is A vs B, or A across settings,
  with `coco_better`'s classes and bands as strata to report ACROSS rather than as
  the variable under study. Data-property studies (#4051, #4043, #3589, #3807)
  were closed on that ruling. The queued method studies that want this bench
  include the acquisition offset's environment dependence (#3546) and region
  styles (#2895).

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

- [`scripts/experiments/pile/README.md`](../../scripts/experiments/pile/README.md) —
  the band construction the migration preserves unchanged.
- [`docs/experiments/2026-09-17-vg-scale-corrections/REPORT.md`](../experiments/2026-09-17-vg-scale-corrections/REPORT.md)
  — the log of what VG got wrong, and the source of the rates quoted above.
- [`docs/experiments/2026-09-03-vg-scale-classes/REPORT.md`](../experiments/2026-09-03-vg-scale-classes/REPORT.md)
  — VG's class-list ceiling, and the definition-risk instrument.
