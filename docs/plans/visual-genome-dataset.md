# Visual Genome demo dataset (multi-label + region annotations)

Status: **Phase 1 in progress.** Adds the Visual Genome (VG) dataset as a demo
dataset for hands-on browsing *and* automated eval, introducing two new
capabilities the existing demo datasets don't have: per-image **multi-label**
ground truth and stored **bounding-box region** annotations.

## Why VG is different

Every existing demo dataset is **single-label and pretend-disjoint**: each media
carries exactly one `media["category"]` string, and the eval harness
(`vtscore/eval/runner.py`, `vtscore/eval/voting_iterations.py`) treats
"category == target" as positive and *everything else* as negative.

VG annotations don't fit that model:

1. **Multi-label.** `img123 = "man eating an apple"` puts the image in `man`
   **and** `apple` simultaneously, and (because nobody annotated a banana)
   implicitly *not* in `banana`. Inclusive category lists become per-image
   booleans.
2. **Bounding boxes.** Every annotated object carries a pixel box. We want to
   keep those for future **region voting** / region-level evals rather than
   throwing them away.

## Decisions (locked)

These were chosen explicitly with the user:

- **Negatives — closed-world.** Membership is strictly binary: a category is
  either in an image's annotated object set (**positive, +1**) or it isn't
  (**negative, −1**). There is no third "unknown" value — if one of our
  categories is not among an image's annotated objects, the image counts as a
  negative for it, full stop. We accept that VG's incompleteness turns a few
  real-but-unannotated objects (the unlooked-for banana) into false negatives —
  the same accidental-overlap noise the existing datasets already tolerate. So a
  VG image just needs a **set of positive categories**; everything outside that
  set is negative.
- **Scope — additive, VG-only.** Existing single-label datasets are left exactly
  as they are. The eval positive/negative test branches: if a media carries a
  `categories` list it uses set membership, otherwise it falls back to the
  legacy `category ==` string compare. No migration of the other demos.
- **Regions — store-only.** Ground-truth boxes are persisted on the media dict
  (in the dataset pickle, the one place derived per-item data is allowed) as
  `media["regions"]`. **No consumer is built in this phase** — this is raw
  material for the existing region-voting plumbing
  (`LabeledElement.region_box`, `DetectorContext.vote_region_boxes`, and
  `docs/plans/patch-embedder.md`) to draw on later.
- **Categories — top-100 by frequency.** The vocabulary is the 100 most frequent
  VG object names (the well-known VG scene-graph object vocab), kept as one
  **static hardcoded list** (`VISUAL_GENOME_CATEGORIES`) — *not* recomputed from
  each slice's frequencies. All four `visual_genome_{s,m,l,a}` variants share
  that exact same list, so the category space is identical across slices; only
  which images fall in the slice changes. A category with zero images in a small
  slice is still one of the 100 (its queries just have no positives there). The
  hardcoded list also lets the UI show categories before the (large) download.
  Object→category matching normalizes case/whitespace and folds simple plurals.

## Data model (the additive bits)

A VG image's media dict gains two keys on top of the usual image media fields:

```python
media["categories"] = ["man", "apple"]          # positive categories (⊆ the 100)
media["category"]    = "man"                      # primary = first positive, kept
                                                  #   so legacy readers (UI filter,
                                                  #   label export) still work
media["regions"] = [                              # store-only ground-truth boxes
    {"box": [0.12, 0.30, 0.44, 0.88], "label": "man"},    # normalized x0,y0,x1,y1
    {"box": [0.40, 0.55, 0.52, 0.69], "label": "apple"},
]
```

`categories` and `regions` live only in RAM and the dataset pickle — never in
detector JSON or settings (consistent with the "No Persisted Vectors or MLPs"
rule, whose single exception is the dataset pickle snapshot).

## Eval semantics

`media_is_positive(media, category)` (new, `vtscore/eval/labels.py`) is the one
place that decides membership:

```python
cats = media.get("categories")
if cats is not None:          # VG-style multi-label
    return category in cats
return media.get("category") == category   # legacy single-label
```

Closed-world means **negative = not positive**, so the four eval selection sites
(text-sort relevant set, learned-sort target/other split, voting-iterations vote
sequence and test scoring) all route through this helper. Existing datasets have
no `categories` key, so their behavior is byte-for-byte unchanged.

## Ingestion path

Mirrors the existing image demo-source machinery
(`vtscore/media/image/_demo_sources.py`):

- `download_visual_genome()` (`vtscore/datasets/downloader/images.py`) fetches
  the two VG image zips + `objects.json` into `data/visual_genome/`.
- `_collect_visual_genome_files()` parses `objects.json`, maps each object name
  to the vocab, and yields per-image `(path, positive_categories, pixel_regions)`
  for images with ≥1 in-vocab object. VG is **not** folder-per-class, so slicing
  is a flat fractional slice over the image list (sorted by image id) for the
  S/M/L/A variants — not the per-category slice the other sources use.
- `_embed_vg_images()` embeds each image and stamps `categories` + normalized
  `regions` (pixel boxes ÷ image dims, clamped to [0,1]).

## What shipped

Phase 1:

- **Multi-label eval.** `vtscore/eval/labels.py::media_is_positive` routes the
  four eval selection sites (text-sort relevant set, learned-sort target/other
  split, voting-iterations vote sequence + test scoring). Datasets with a
  `categories` list use set membership; everything else is unchanged.
- **VG ingestion.** `download_visual_genome()` (two image zips + objects.json),
  `_collect_visual_genome_files()` (objects.json → per-image positives + pixel
  regions, flat-sliced), `_embed_vg_images()` (stamps `categories` + normalized
  store-only `regions`), and `visual_genome_{s,m,l,a}` demo datasets.
- **Vocab.** `VISUAL_GENOME_CATEGORIES` (top-100 VG object names) +
  `_vg_category_for()` case/plural-folding matcher.
- **Eval registration.** `_VISUAL_GENOME_QUERIES` + `visual_genome_{s,m}`
  entries in `EVAL_DATASETS`.
- **Tests.** `tests_lib/downloads/test_visual_genome_download.py` (fixture-based
  download/collect/load + region normalization) and multi-label eval tests in
  `tests_lib/detectors/test_eval.py`.

## Open follow-ups

- **Region consumers.** Region-level eval and seeding region votes from
  ground-truth boxes are deferred. The boxes are stored now; wiring them into
  the patch-embedder region-voting flow is the natural Phase 2.
- **Vocab matching quality.** Object→category matching is a case/plural-folding
  heuristic. VG synonyms/synsets (`names` has multiple aliases; `synsets` exists)
  are only partially exploited; a richer synonym map would recover more positives.
- **Attributes & relationships.** VG also ships `attributes.json` and
  `relationships.json` (e.g. "red apple", "man holding apple"). Out of scope for
  Phase 1; potential future eval axes.
- **Real download verification.** The VG archives are ~15 GB; CI exercises the
  ingestion path against small fixtures only. The hardcoded URLs/sizes should be
  smoke-checked against a real download before relying on them.
