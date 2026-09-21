# `coco_quarry`: migrate `vg_scale` off Visual Genome onto COCO

**Decided 2026-09-18.** `vg_scale` moves to COCO 2017 train+val as its image
pool, Visual Genome leaves the construction entirely, and the result is named
**`coco_quarry`**. This plan holds the migration and the work it retires.

**Why that name.** It is a fixed body of material you cut blocks out of to spec,
which is what the thing actually is once band, prevalence and class list become
export-time queries (#3987). Nothing parameter-like is baked in, and that is the
point: `vg_scale_any` and `vg_scale_deep` exist *because* band and depth were
baked into a name, so every new question needed a new dataset. A `coco_quarry`
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
`name_evidence.py`, `coco_folds.py` (retired with Visual Genome — see `docs/plans/coco-quarry.md`), `vg_name_families.py`,
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

**State, 2026-09-19.** The build is finished: `coco_quarry` and `coco_quarry_full` are built and verified, the export layer is in, and the VG machinery is retired. What is left is three things — the band re-validation, the `truck`/`car` ruling, and the studies the set exists to support. Widening *C* landed 2026-09-20 (#4056) and is marked DONE below. Items below marked **DONE** keep their original argument, because a plan whose history is deleted reads as though every decision was obvious.

<!-- item-sep -->

- **DONE — #3997, merged in #3998.** Off-COCO is *differently distributed* (AUC 0.53–0.54 matched within `class@band`, five embedders) and **not harder**: on the shipped head every statistic leans the other way, 34–37 of 58 cells agreeing. The difficulty argument for keeping VG is gone.

  **Settle the diversity claim before the rebuild, not after.** The only
  remaining argument for VG is that its non-COCO half is *differently
  distributed*, not merely additional. Measure it: are off-COCO positives harder,
  or differently composed, than COCO-anchored ones at matched band and class?
  `provenance_probe.py` and `provenance_shortcut.py` already read provenance off
  the vectors (AUC 0.53–0.56) and are the instrument. A null result closes the
  question and the docstring claim comes out; a positive one is a reason to keep
  a VG arm as a named contrast rather than as the pool. This is the one item that
  could still change the decision, so it runs first and it is cheap. (Sonnet 5)

<!-- item-sep -->

- **DONE — #4008.** `pilebuild/loaders/coco_quarry.py`, 200 lines against `vg_scale`'s 1,064, importing `band_for` unchanged. Against the real corpus: 75 cells, 7,500 positives, 0 short of `SCALE_N_POS`. `iscrowd` dropped explicitly, as the census did.

  **Build the COCO loader, and keep the band rule identical.** A `kind: "coco"`
  sibling of `vg_scale.py` reading `instances_*2017.json`
  directly: no `anchor_to_coco`, no `canonicalise`, no `lift_ambiguous`, no
  corrections file. `band_for` is imported unchanged — that is what makes the old
  and new sets comparable at all, and
  [`coco_only_supply.py`](../../scripts/experiments/pile/coco_only_supply.py)
  already demonstrates the read end to end. Decide `iscrowd` explicitly (the
  supply census drops crowd regions as not-an-instance; the builder should do the
  same and say so). (Opus 4.8)

<!-- item-sep -->

- **DONE — #4038 made this the settled position.** `vg_scale*` cells stay on disk and load by path; nothing can rebuild one. Studies conditioned on them keep working, and nothing new is built on them.

  **Keep the old cells readable under their own name.** `pile_config` warns that
  rebuilding a cell silently changes what it is, and five-plus studies (#3115,
  #3196, #3287, #3290, #3318, #3319) are conditioned on the VG-built ones.
  Reproducing those is no longer a reason to keep cleaning VG — they can be
  re-run — but it is a reason not to reuse the key. `coco_quarry` is a new
  dataset; `vg_scale*` stays readable and nothing new is built on it. (Sonnet 5)

<!-- item-sep -->

- **Re-validate the band rule against COCO's boxes at full scale.** #3637 scored
  the band statistic against COCO's exhaustive boxes on the VG∩COCO overlap and
  found COCO right where VG disagreed. Under pure COCO that overlap becomes the
  whole dataset, so the scatter guard and the oversize cut now act on adjudicated
  boxes only. Re-measure what they reject and at what rate — `BAND_MAX_INFLATION`
  was tuned against VG's multiply-annotated objects (21 bench boxes on one image),
  which COCO does not produce. (Sonnet 5)

<!-- item-sep -->

- **DONE — #4038.** 38 files, 11,188 deletions: four loaders, the seven scripts named below, thirteen VG-bound analysis scripts, and their tests. The source-agnostic core was lifted to `pilebuild/scale_core.py` first, because `coco_quarry` imported all of it from the VG loader. Published reports keep their citations, re-captioned rather than unlinked.

  **Retire the inference machinery, and delete it.** The name tables, the
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

- **DONE.** `SCALE_CLASS_CONTENTS` carries all 25 classes and says what COCO's annotators counted rather than what a reviewer should. Re-measured 2026-09-19 and it reproduces exactly (`cup` 39% glass, `truck` 38% with 17% car, `stop sign` 79%, `book` 85%).

  **Keep `SCALE_CLASS_RULES`, and re-aim it.** Under VG it recorded where a
  reviewer's reading had to be pinned down. Under COCO the definitions become
  *inherited* — COCO annotates magazines as `book`, which is what split #3612's
  review — so the table's job changes from adjudicating to **documenting where
  COCO's class boundary differs from plain English**. That is still load-bearing
  for anyone reading a result, and it is the one piece of the review programme
  that survives the switch. (Sonnet 5)

<!-- item-sep -->

- [x] #3987 — prevalence is an axis (#4034). π = 0.001 is reachable for **every** cell at 100 positives; it is reachable at full positives **nowhere** (`car@large` would need 1.29M negatives against 96,458), so the envelope is two-dimensional and the exporter refuses an infeasible ask with the shortfall named.

<!-- item-sep -->

- **DONE — #4034.** `quarry_export.py` selects positives and negatives from the embedded corpus at any prevalence the data supports, so `SCALE_N_POS`, `SCALE_N_NEG` and π are filters rather than rebuilds. All 75 cells reproduce the designated membership exactly at `SCALE_N_POS`.

  **Move band and class list to eval-time queries too.** #3987 does prevalence;
  the other two are the same change and should land with it. COCO is exhaustively
  annotated, so a cell is a filter over a fixed set rather than a build-time
  designation: `SCALE_N_POS` / `SCALE_N_NEG` stop being re-embeds and a class-list
  change stops being a rebuild. Together with #3986 this is what turns the
  benchmark into a dataset. (Opus 4.8)

<!-- item-sep -->

- [x] #3988 — COCO is embedded (#4017, #4022). **Five columns, not six**: `dinov2_patch` dropped by the owner. The four single-vector columns cover all 123,287 images (1.8 GB, 3 h 41 m); `dinov3_patch` is 8 shards of 4.7 GB, because one full-corpus patch cell is ~37 GB — unwritable in a single `pickle.dump` and unreadable through `load_medias`.

<!-- item-sep -->

- **The embedder roster, decided 2026-09-18.** Six columns:
  `siglip`, `siglip2_l`, `clip`, `clip_l`, `dinov2_patch`, `dinov3_patch`. Sized
  in #3988 from the repo's own measured `cuda+cuml` fit
  (`vtscore/datasets/stages/_load_cost_model.py`) over 123,287 images.

  Two rulings are worth keeping here rather than only in the issue, because both
  are easy to undo by accident. **DINOv2 is a patch column, not
  `dinov2_single`** — DINOv3 is already a patch column, so pairing it with a
  single-vector DINOv2 would confound generation with embedder *kind*, the same
  trap `pile_config` documents for `siglip` → `siglip2_l`. And **storage is the
  binding cost, not GPU time**: the two patch columns are 98% of the 81 GB while
  the four single-vector ones total 1.5 GB, so a future column is cheap if it is
  single-vector and a real decision if it is not. (Sonnet 5)

<!-- item-sep -->

- [x] #4056 — ***C* is 53** (PR pending). The count rule admitted 54; `wine glass` is
  held out by owner ruling, because `SCALE_CLASS_MERGES` folds it into `cup` and
  admitting it would redefine `cup`. 159 cells, **0 short** of `SCALE_N_POS`, pool
  headroom **1.48x**. **Rebuilt 2026-09-20 and measured against the preserved
  25-class build: −0.030 ± 0.004 AP** (`siglip`; `siglip2_l` −0.028 ± 0.004) on
  the 75 cells that existed before — and **−0.003 / +0.001 once the negative
  count is held still**, with ΔAUC **+0.002**. The cost is prevalence: a rebuilt
  cell carries 23,891 negatives against 16,535. Rankings are untouched.
  **An earlier reading of +0.24 AP was wrong in sign and is corrected in the
  report**; it varied the barren component alone, and the barren draw is capped
  at `SCALE_N_NEG` in both builds while #3667's cross-class negatives more than
  double. The emptiness mechanism is real (clean candidates 49,503 → 16,091,
  classes held per clean pool image **1.163 → 0.000**) and constrains any future
  change to how the pool is SIZED, but it does not reach a capped draw. Owner
  ruled widen-and-renumber; `SCALE_CLASSES_25` freezes the old roster and the
  25-class build is preserved at `keep/coco-quarry-25-20260920/`. Per-class pools
  should be immune by construction and are unmeasured. The original argument
  follows.

  **Widen *C* to every class meeting the count requirement — 54 of COCO's 80.**
  The selection rule is the count and nothing else: a class is in if it clears
  `SCALE_N_POS` in all three bands. Measured, that is **54 classes, 29 beyond the
  current 25**. The shared negative pool survives all of them at 1.5x (16,058
  clean images against the 10,900 the pool and its spares need, versus 4.5x at
  25) — and under #3986 that constraint goes away entirely, since a per-class
  pool leaves 5x to 11x. Either way, per-class supply is not what caps the list.

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

- [x] #3986 — per-class negatives are the default (#4034). ~100,000 per cell against the designated 9,900, about half holding another class in *C* where the shared pool is 0% by construction. `--pool shared` reproduces the old shape for arms compared against a published number. **Measured 2026-09-20 (#3986 report): −0.002 AP, 0 of 75 cells changing winning column** — the shortcut is undiminished at 2.0-2.3x co-occurring over barren, but #3667's fix already put 40% of every cell's negatives into the hard kind, so the rest of the way costs nothing. Make the change for supply, not accuracy.

<!-- item-sep -->

- **`person` sizes the pool; it is no longer a trade.** Under a shared pool it
  was the one costly addition — 66,808 images held, 26,132 off the pool, more
  than the other 28 candidates combined. Under #3986's per-class pools that cost
  disappears and `person` simply sets |P|: it is in 54% of COCO, so a uniform
  pool needs 10,900 / 0.458 ≈ 24,000 images to serve it. Nothing to decide, just
  a number to use. (Sonnet 5)

<!-- item-sep -->

- **Decide the `truck`/`car` pair on the measurement.** 17% of COCO `truck` boxes
  are objects LVIS calls `car_(automobile)`, against 2% the other way. #3588 added
  `truck` beside `car` as a same-scene partner; the asymmetry says the pair is
  partly one population relabelled rather than two. Either keep both and say so
  where results are read, or drop `truck` — it is the one genuine boundary
  contest the census surfaced among the current 25. (human)

<!-- item-sep -->

- **DONE — the table already existed** and was verified by re-running the instrument. Worth recording what it establishes: **share does not rank difficulty**. Every class whose review actually split is homogeneous — `book` 85%, `cell phone` 89%, `knife` 93% — so a 4% minority meeting an unwritten rule splits reviewers as surely as a 40% one.

  **Write the class notes from the purity table, once, and ship them.**
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
