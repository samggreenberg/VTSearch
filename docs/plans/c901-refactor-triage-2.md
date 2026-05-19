# C901 Refactor Triage — second wave (≥20 complexity)

**Status:** All four refactors shipped. Plan ready to retire — see the
"What shipped" / "Open follow-ups" sections at the bottom.

## Why this file exists

The first triage shipped eight refactors of ≥20-CC functions in 2026-05.
Its "Open follow-ups" section flagged a fresh batch of D-grade functions
revealed by a post-sweep `radon cc -n D --no-assert` — a mix of dev-side
growth and items the original sweep didn't have time to triage — and
explicitly asked the next contributor to "open a new triage file when
picking these up". This is that file.

Same shape as the predecessor: each remaining function is classified as
either **Refactor** (clean split available, worth the churn) or **Skip**
(complexity is honest dispatch). When a refactor lands, move its row to
**Done** with a one-line summary and delete the `# noqa: C901` marker.
When all the refactor rows are done, delete this file and update
[brainstorm.md §20.7.1](brainstorm.md) to reflect the new "worst
offenders remaining" set.

## Method

Measured with `radon cc vtsearch/ app.py -s -n D --no-assert` on
2026-05-19 against `origin/dev` (which does **not** yet include the
PR #1497 refactors — the eight functions from the first sweep are
expected to drop out once that PR merges). The list below is the
expected post-merge state: nine D-grade functions, of which five are
the **Skip** rows carried forward from the first triage and four are
new **Refactor** candidates flagged in its Open follow-ups.

## Refactor (high payoff)

| File:line | Func | CC | Plan |
|---|---|---|---|
| ~~`vtsearch/datasets/importers/base.py:583`~~ | ~~`DatasetImporter.effective_source_specs`~~ | ~~27~~ | **Shipped via PR #1504** — see Done below. |
| ~~`vtsearch/detectors/training.py:142`~~ | ~~`train_and_score`~~ | ~~23~~ | **Shipped** — see Done below. |
| ~~`vtsearch/detectors/label_sync.py:17`~~ | ~~`sync_labels_to_loaded_detector`~~ | ~~23~~ | **Shipped** — see Done below. |
| ~~`vtsearch/datasets/importers/combine_datasets/__init__.py:72`~~ | ~~`CombineDatasetsImporter.run`~~ | ~~21~~ | **Shipped** — see Done below. |

## Skip (complexity is honest)

Carried forward from [c901-refactor-triage.md](c901-refactor-triage.md)
— rationale unchanged. Re-verified against current `origin/dev` to
confirm the functions still look the way they did then.

| File:line | Func | CC | Why |
|---|---|---|---|
| `vtsearch/cli_pipeline.py:31` | `load_pipeline_file` | 26 | Already structured as section validators; splitting just shuffles the YAML-key-parser code without reducing what a reader has to hold in their head. |
| `vtsearch/routes/detectors/crud.py:314` | `combine_detectors` | 24 | Every branch is a distinct abort/guard with a unique error message + a small merge step. Hoisting helpers makes the call site less direct, not more. |
| `vtsearch/routes/detectors/scoring.py:47` | `_resolve_or_train_detector` | 25 | Progress emission interleaved with logic, but the logic itself is linear (try-cached → resolve unresolved → train). Mild win at best. |
| `vtsearch/datasets/loader_demo.py:43` | `load_demo_dataset` | 23 | Heavily nested numeric pipeline; extracted helpers would each need ~6 args. |
| `vtsearch/detectors/labeling_progress.py:386` | `_eval_cached_models` | 22 | Same shape as `load_demo_dataset` — numeric pipeline, no clean split point. |

## Sequencing

Order picked to maximise the "one PR retires two noqa markers" payoff
and to keep mechanical risk monotonically increasing:

1. ~~`CombineDatasetsImporter.run` + `run_chunked`~~ — shipped
2. ~~`sync_labels_to_loaded_detector`~~ — shipped
3. ~~`train_and_score`~~ — shipped
4. ~~`effective_source_specs`~~ — shipped (via PR #1504, in parallel)

## Done

- **`CombineDatasetsImporter.run` + `run_chunked`** (CC 21+21 → 7+4).
  Extracted `_parse_dataset_paths(raw)` (comma/list parsing + `<2`
  + missing-file guards) and `_iter_unique_source_clips(paths, thin,
  seen_md5s, mtype_state, progress)` (per-pickle load with progress,
  media-type latch, MD5 dedup; yields `(pkl_path, deduped_list,
  dupe_count)`). `run` becomes parse → iterate → accumulate → fresh
  sequential IDs + summary; `run_chunked` becomes parse → iterate →
  emit each non-empty source as its own chunk. Both `noqa: C901`
  markers retired. Full suite (3849 tests) passes.

- **`sync_labels_to_loaded_detector`** (CC 23 → 6). Split into
  `_get_loaded_detector_state()` (collapses the find-mode / no-
  detector / no-entry / no-on-disk-data guards into one early-return
  returning `(entry, path, data, det_ctx)` or `None`),
  `_merge_labelsets_across_datasets(existing_ls, current_ls,
  current_dataset_keys)` (pure merge: drop existing entries owned by
  the active dataset, append current entries, dedupe by
  `element_key`), and `_refresh_detector_caches(det_ctx, merged,
  path, media_type)` (good/bad counts + cached labelset +
  media-type fallback + mtime-with-OSError-fallback). The
  orchestrator reads as: guard → snapshot+build → merge → write →
  refresh → update entry timestamp. `noqa: C901` deleted. Full
  suite passes.

- **`train_and_score`** (CC 23 → 5). Three helpers take the bulk:
  `_build_vote_tensors(clips_dict, good_votes, bad_votes,
  region_boxes)` filters votes against the active clips, applies the
  region-aware training-vector pick for goods + plain embedding for
  bads, performs the ≤1-sample / single-class guard, and returns
  `(X, y, X_list, y_list, input_dim, hidden_dim)` or `None`;
  `_score_all_media(model, clips_dict)` decides the region-aware vs
  plain path internally and returns `(all_ids, scores, best_region)`;
  `_format_results(all_ids, scores, best_region, clips_dict)` sorts
  desc by raw score, rounds for output, and attaches `best_region.box`
  for region-aware media. Outer body becomes: build → guard →
  calibrate → train → score → safe-threshold blend → format.
  `noqa: C901` deleted. Full suite passes.

- **`DatasetImporter.effective_source_specs`** (CC 27 → low). Landed
  via PR #1504 ("Refactor highest-complexity functions",
  commit `392c6f6d`) before this branch could rebase. That PR split
  the method into three module-level helpers:
  `_parse_multi_media_specs`, `_parse_legacy_specs`, and
  `_validate_spec_converter`. Equivalent goal, slightly different
  helper boundaries — the duplicate refactor on this branch was
  dropped during rebase. `noqa: C901` deleted by the PR.

## Open follow-ups

- Re-run `radon cc vtsearch/ app.py -s -n D --no-assert` and update
  the "Worst offenders remaining" list in
  [brainstorm.md §20.7.1](brainstorm.md) to reflect just the five
  **Skip** rows above plus whatever new D-grade functions PR #1504's
  unrelated refactors (load_pipeline, embedder, scoring) reshaped on
  dev. At that point this file can be deleted.
- The remaining C-grade (11-19) `# noqa: C901` entries are the long
  tail; they don't need a new triage file — burn them down as code in
  the area is touched.
