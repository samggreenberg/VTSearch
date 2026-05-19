# C901 Refactor Triage (≥20 complexity)

**Status:** Triage complete; refactor of `_apply_clip_and_embed` in flight.

## Why this file exists

[brainstorm.md §20.7.1](brainstorm.md) tracks the C901 noqa burn-down
as a long-running, opportunistic effort. This file is the *next-step*
triage: for each remaining function with cyclomatic complexity ≥20, we
record whether refactoring it is high payoff or whether the complexity
is honest dispatch that shouldn't be touched. Without this, every new
contributor to the burn-down has to re-evaluate the same list of
functions from scratch.

When a "refactor" entry below ships, move it to **Done** with a one-line
summary of what landed, and delete the noqa marker from the source.
When all the "refactor" rows are done, this file can be deleted and
brainstorm.md §20.7.1 updated to reflect the new "worst offenders
remaining" set.

## Method

Measured with `radon cc vtsearch/ app.py -s -n C` against `origin/dev`
on 2026-05-19. Anything ≥20 was opened and inspected. Each was
classified as either:

- **Refactor** — complexity reflects multiple intertwined responsibilities
  (or repeated near-identical branches) that can be cleanly split into
  named helpers.
- **Skip** — complexity reflects a long chain of distinct guards /
  validations / numeric steps where the obvious extracted helpers would
  need ~6 args each and the call site would still be the same length.

## Refactor (high payoff)

| File:line | Func | CC | Plan |
|---|---|---|---|
| `vtsearch/detectors/resolver.py:436` | `_apply_clip_and_embed` | 25 | Three near-identical try/clip/tempfile/embed/cleanup branches for audio/image/text. Extract `_clip_audio_to_bytes`, `_clip_image_to_bytes`, `_clip_text_to_bytes` (each returns `(bytes, suffix)`), plus one `_embed_via_tempfile(data, suffix, …)` helper. Collapses ~110 lines into ~40. |
| `vtsearch/routes/labels/vote.py:42` | `export_labels` | 29 | Three concerns interleaved: labelset selection, corrections annotation, enrichment. Each is a clean subroutine — extract `_select_labelset(query)`, `_annotate_corrections(result, find_initial, all_medias)`, `_enrich_with_metadata(result, all_medias)`. |
| `vtsearch/converters/audio2image.py:108` | `convert` | 25 | Linear pipeline: param coerce → load audio → compute spec → render PNG. Extract `_coerce_params`, `_compute_spectrogram`, `_render_to_png`. |
| `vtsearch/converters/image2text.py:83` | `convert` | 22 | Same shape as audio2image — split by stage. |
| `vtsearch/routes/sorting.py:789` | `label_file_sort` | 21 | Route handler doing parse + path-validate + embed-loop + train + score. Pull out `_embed_external_labels(file, emb) → (X, y, loaded, skipped)`; route becomes ~30 lines. |
| `vtsearch/detectors/resolver.py:560` | `resolve_label_embeddings` | 25 | Per-entry resolution + three logging-by-failure-reason branches + final summary. Extract `_resolve_one(entry, media_type) → ResolvedEntry` (a small dataclass) so the loop becomes accumulate-and-log-summary. |
| `vtsearch/detectors/labelset_training.py:83` | `populate_label_embeddings` | 21 | Three resolution strategies per element in one loop. Extract `_resolve_one_element_embedding(elem, cache, snap, media_type, embedder_name) → np.ndarray | None`. |
| `vtsearch/cli.py:421` | `_run_pipeline` | 31 | The dry-run branch (~50 lines of validation + emission) is essentially its own function inside an early return. Lift it to `_run_dry_run(...)` and the main path is straightforward. |

## Skip (complexity is honest)

| File:line | Func | CC | Why |
|---|---|---|---|
| `vtsearch/cli_pipeline.py:31` | `load_pipeline_file` | 26 | Already structured as section validators; splitting just shuffles the YAML-key-parser code without reducing what a reader has to hold in their head. |
| `vtsearch/routes/detectors/crud.py:314` | `combine_detectors` | 24 | Every branch is a distinct abort/guard with a unique error message + a small merge step. Hoisting helpers makes the call site less direct, not more. |
| `vtsearch/routes/detectors/scoring.py:47` | `_resolve_or_train_detector` | 25 | Progress emission interleaved with logic, but the logic itself is linear (try-cached → resolve unresolved → train). Mild win at best. |
| `vtsearch/datasets/loader_demo.py:43` | `load_demo_dataset` | 23 | Heavily nested numeric pipeline; extracted helpers would each need ~6 args. |
| `vtsearch/detectors/labeling_progress.py:386` | `_eval_cached_models` | 22 | Same shape as `load_demo_dataset` — numeric pipeline, no clean split point. |

## Sequencing

Ship in this order — earlier rows are more mechanical / lower risk:

1. `_apply_clip_and_embed` *(in flight)*
2. `label_file_sort`
3. `export_labels`
4. `audio2image.convert` / `image2text.convert` (one PR)
5. `resolve_label_embeddings`
6. `populate_label_embeddings`
7. `_run_pipeline`

## Done

*(none yet)*

## Open follow-ups

- Once all rows in **Refactor** ship, re-run `radon cc -n C` and update
  brainstorm.md §20.7.1 with the new worst-offenders list.
- The remaining C-grade (11-19) noqa entries are the long tail; they
  do not need a new triage file — burn them down as code in the area
  is touched.
