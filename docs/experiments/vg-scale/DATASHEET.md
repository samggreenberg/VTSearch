# `vg_scale` — one class list across three box-size bands

A dataset for asking **"how well can we find buses in the middleground?"** — the
same twelve classes at three object scales, so a small-vs-large difference is
about size rather than about which words happen to live at which size.

It exists because the published `vg_box_small/medium/large` sets cannot answer
that question: each category is banded by its *median* box area, so the three
sets carry disjoint vocabularies and their gap confounds box size with class
identity (`nose`/`glasses`/`watch` against `fence`/`hill`/`lady`). Those sets
remain valid for what they measured and are **not** comparable to this one.

## What it is

| | |
|---|---|
| medias | ~7,700 VG images, no pixels stored (vectors + `patch_grid` only) |
| cells | 36 = 12 classes × {small, medium, large} |
| positives | exactly **100 per cell**, each carrying its ground-truth box |
| negatives | one shared pool of **3,900**, identical for every cell |
| prevalence | **0.0250 in all 36 cells**, by construction |
| embedders | `siglip`, `siglip2_l`, `dinov3_patch` (the last carries patch grids, so region voting is real) |

Classes: `backpack` `bicycle` `bird` `boat` `book` `bus` `clock` `dog` `kite`
`knife` `stop sign` `umbrella` — every one also a COCO-2017 class, which is what
made the correction pass affordable.

**Bands** are anchored to the patch embedder's geometry, as fractions of image
area: `small` < 1/196 (below one DINOv3 patch), `medium` 1/196–1/12 (patch to
smallest HAC leaf), `large` 1/12–0.80. Size means the **union box** over a
class's instances — what one Good vote actually drags — and an image whose union
is scattered far wider than its largest instance is excluded rather than banded,
because there the box describes the scatter and not the object.

**Cells are designated, not inferred.** Each is exactly its 100 positives plus
the shared negatives; everything else is *excluded* from it. That third value is
the point: an image holding a large bus is not a `bus@small` positive, and
calling it a negative would penalise a detector for finding a real bus. Consumers
must honour it via `vtscore.eval.labels.evaluable_pool` — the harness does this
once per cell, and a pool built without it silently scores excluded images as
negatives.

## Where the labels come from, and what they are worth

VG's own annotation cannot support this construction: measured against COCO, its
recall over these twelve classes is **0.61**, and **1.35%** of the images it
treats as negatives actually hold the object. So labels are VG's, repaired:

- **48% of images are COCO-sourced** (`image_data.json`'s `coco_id`) and take
  COCO's exhaustive annotation, which replaces VG's for that image. Copies whose
  aspect ratio disagrees with the COCO original by >1% are left unrepaired: 49 of
  51,497 are re-cropped or rotated, and there a COCO box describes the wrong
  pixels.
- **The rest were reviewed by hand**, in VTSearch, across three passes: ranked
  negatives, a uniform random stratum, and every positive re-issued with its box
  drawn and a magnified inset.

### The numbers a reader should hold

| measured on | result |
|---|---|
| residual contamination of the negative pool, after review | **2.0%** (4/200), 95% CI **0.8–5.0%** |
| — concentrated in | `book` (3/20) and `bus` (1/20); zero in the other eight |
| small-band positives confirmable *with the box drawn* | **~2/3** |
| reviewer vs COCO on pairs COCO had settled | 9.0% disagreement — reviewer error, COCO error and definition drift, **not attributable without adjudication** |

**COCO is not a gold standard here.** The review found images COCO annotates as
empty that plainly hold the object, and four adjudicated COCO errors among the
twelve classes (two prohibition circles and a school-crossing paddle labelled
`stop sign`; a box on a hedge labelled `umbrella`).

**The small band is at the limit of verification, and this is a property of the
data, not a defect to hide.** A sub-patch object is under 1/196 of the frame;
reviewing bare thumbnails rejected 43% of small-band positives against 10% of
large, and drawing the box cut that to 18% vs 3%. Both a human and a
vision-language model confirm only ~2/3 of them even with the box. So a
small-band "not confirmed" is recorded as *unconfirmed* and the label stands —
**any small-band result should be read beside that fact.**

## Reproducing and extending it

```bash
source scripts/experiments/pile/pile_env.sh
python build_pile.py --datasets vg_scale --force   # rebuild all three cells
python build_pile.py --verify --datasets vg_scale  # structure + review coverage
```

Corrections live in `corrections.json` as `(image_id, class, present, boxes)` and
are merged over `objects.json` **before** banding, so a corrected box can move an
image between bands. A correction with no box excludes the pair from every cell
of that class rather than promoting it: a band is a claim about size, and no size
was measured.

Membership is pinned by `vg_scale_roster.json` and selection is hash-stable, so a
rebuild does not reshuffle cells. **Run `check_review_coverage.py` after any
rebuild.** Coverage is the one property no structural check implies — cells can
be full, prevalence exact and boxes valid while the dataset no longer contains
the images its review was performed on, which happened here and cost a day.
