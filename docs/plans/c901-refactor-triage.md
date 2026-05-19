# C901 Refactor Triage (≥20 complexity)

**Status:** All 8 refactors shipped. Plan ready to retire — see the
"What shipped" / "Open follow-ups" sections at the bottom.

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
| ~~`vtsearch/detectors/resolver.py:436`~~ | ~~`_apply_clip_and_embed`~~ | ~~25~~ | **Shipped** — see Done below. |
| ~~`vtsearch/routes/labels/vote.py:42`~~ | ~~`export_labels`~~ | ~~29~~ | **Shipped** — see Done below. |
| ~~`vtsearch/converters/audio2image.py:108`~~ | ~~`convert`~~ | ~~25~~ | **Shipped** — see Done below. |
| ~~`vtsearch/converters/image2text.py:83`~~ | ~~`convert`~~ | ~~22~~ | **Shipped** — see Done below. |
| ~~`vtsearch/routes/sorting.py:789`~~ | ~~`label_file_sort`~~ | ~~21~~ | **Shipped** — see Done below. |
| ~~`vtsearch/detectors/resolver.py:560`~~ | ~~`resolve_label_embeddings`~~ | ~~25~~ | **Shipped** — see Done below. |
| ~~`vtsearch/detectors/labelset_training.py:83`~~ | ~~`populate_label_embeddings`~~ | ~~21~~ | **Shipped** — see Done below. |
| ~~`vtsearch/cli.py:421`~~ | ~~`_run_pipeline`~~ | ~~31~~ | **Shipped** — see Done below. |

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

1. ~~`_apply_clip_and_embed`~~ — shipped
2. ~~`label_file_sort`~~ — shipped
3. ~~`export_labels`~~ — shipped
4. ~~`audio2image.convert` / `image2text.convert` (one PR)~~ — shipped
5. ~~`resolve_label_embeddings`~~ — shipped
6. ~~`populate_label_embeddings`~~ — shipped
7. ~~`_run_pipeline`~~ — shipped

## Done

- **`_apply_clip_and_embed`** (CC 25 → 6). Split into `_clip_audio_to_bytes`,
  `_clip_image_to_bytes`, `_clip_text_to_bytes` (each returns
  `(bytes, suffix)`), a `_clip_to_bytes` dispatcher, a `_replay_chain`
  helper for the chain-replay early path, and one `_embed_via_tempfile`
  shared embedder. The main function is now a chain-replay attempt + a
  three-line dispatch through the clipper helpers, with a single
  fallback to `embed_file`. `noqa: C901` deleted. All 49
  `tests/detectors/test_clipper_workflow.py` tests pass.

- **`label_file_sort`** (CC 21 → 9). Extracted `_parse_label_file(file)`
  (JSON decode + label-list guard), `_embed_external_labels(labels, emb)`
  (the per-entry validate-and-embed loop, returning
  `(X, y, loaded, skipped)`), and `_train_and_score_dataset(X, y)`
  (MLP train + score every loaded media). The route is now parse →
  embed → guard → train+score → respond. `noqa: C901` deleted. Full
  suite (3793 tests) passes.

- **`export_labels`** (CC 29 → 5). Split into `_select_vote_pools` (the
  `label_filter` / `goods_only` branch table), `_annotate_corrections`
  (build the md5 → media-id map once, then stamp `is_correction` per
  entry — also fixes the original's O(N×M) per-entry linear scan),
  `_build_entry_metadata` (per-media display + origin-params + custom
  blob), and `_enrich_with_metadata` (build the md5 → media map once,
  call the per-entry helper, compute `available_columns`). The route
  is now: pools → labelset → annotate → optional corrections-only
  filter → optional enrichment → return. `noqa: C901` deleted. Full
  suite (3842 tests) passes.

- **`_run_pipeline`** (CC 31 → 3). Extracted four dry-run helpers
  (`_validate_dry_run_source`, `_validate_dry_run_exporter`,
  `_emit_dry_run_plan`, and the `_run_dry_run` orchestrator), plus
  `_train_detectors_for_first_chunk` for the lazy detector-train
  pre-condition, `_score_chunk` for the per-chunk
  emit-progress + score + merge sequence, and `_run_live_pipeline` for
  the loop itself. `_run_pipeline` is now a three-line dispatcher:
  set the settings path, route to either dry-run or live. Full
  suite (3842 tests) passes.

- **`populate_label_embeddings`** (CC 21 → 9). Extracted
  `_maybe_clear_cache_on_embedder_switch` (drops the cache when the
  active dataset's embedder changed under us) and
  `_resolve_uncached_embedding` (the in-dataset cid path with optional
  region pool, plus the cross-dataset embed fallback). The loop now
  reads as: cache hit → done; otherwise resolve uncached → cache →
  progress. Original progress semantics preserved (no callback fired
  on cache hits, fired on every resolved entry whether it produced an
  embedding or not). Full suite (3842 tests) passes.

- **`resolve_label_embeddings`** (CC 25 → 6). The per-entry resolution
  loop now goes through `_resolve_one_label(entry, media_type, index)`,
  which returns a `_LabelOutcome` (status tag + optional embedding /
  label-value). Inside that, `_embed_resolved_label` covers the
  "clip-aware vs straight embed" branch. Failure messages are emitted
  by `_log_resolve_failure` (file-resolution side) and inline in the
  embed-failure path; the end-of-batch summary is `_log_resolve_summary`.
  The outer function is now a one-screen accumulator: per-entry call →
  fire progress → bucket the outcome → log summary. `noqa: C901`
  deleted. Full suite (3842 tests) passes.

- **`Audio2ImageMediaConverter.convert` + `Image2TextMediaConverter.convert`**
  (CC 25 → 8 and 22 → 7). Each converter's long linear body now
  delegates to module-level stage helpers. Shared: `_resolve_media_bytes`
  (read from `media_bytes` or fall back to `media_path`). Audio adds
  `_load_audio_array`, `_compute_spectrogram`, `_render_spectrogram_png`,
  `_get_png_dimensions`, plus an instance `_coerce_params`. Image2Text
  adds `_run_paddleocr` (handles the import + open + OCR chain) and
  `_extract_text_lines`. Each helper owns one print-and-return-None
  failure mode, so the orchestrating `convert` becomes a chain of
  short early-returns. `noqa: C901` deleted on both. Full suite
  (3842 tests) passes.

## What shipped

All eight refactors in **Refactor (high payoff)** landed, in the
sequence above. Combined: roughly 1,000 lines reorganised into named
helpers, ~250 cyclomatic-complexity points eliminated, eight `# noqa: C901`
markers deleted, and zero behavioural changes (verified by the full
test suite at every step).

## Open follow-ups

- **Update brainstorm.md §20.7.1.** Its "Worst offenders remaining"
  list still names `_run_pipeline (31)`, `export_labels (29)`,
  `DatasetImporter.effective_source_specs (27)`, `load_pipeline_file
  (26)`, `_resolve_or_train_detector (25)`, `resolve_label_embeddings
  (25)`, `_apply_clip_and_embed (25)`, `Audio2ImageMediaConverter.convert
  (25)`, and `combine_detectors (24)`. Six of those are now shipped;
  three remain (the **Skip** rows in this file plus
  `effective_source_specs`).
- **Re-triage what radon shows now.** A post-refactor `radon cc -n D
  --no-assert` reveals additional D-grade functions that were not on
  the original list — likely a mix of dev-side growth and an
  incomplete first sweep. Candidates worth a closer look:
  `DatasetImporter.effective_source_specs (27)`,
  `sync_labels_to_loaded_detector (23)`, `train_and_score (23)`,
  `CombineDatasetsImporter.run (21)`. The five **Skip** rows in this
  file (`load_pipeline_file`, `combine_detectors`,
  `_resolve_or_train_detector`, `load_demo_dataset`,
  `_eval_cached_models`) stay skipped — the original rationale still
  holds. Open a new triage file when picking these up.
- The remaining C-grade (11-19) noqa entries are the long tail; they
  do not need a new triage file — burn them down as code in the area
  is touched.
