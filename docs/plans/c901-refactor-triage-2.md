# C901 Refactor Triage — second wave (≥20 complexity)

**Status:** Triaged; refactors not yet started. Picks up where
[c901-refactor-triage.md](c901-refactor-triage.md) left off (eight 2026-05
sweep refactors shipped).

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
| `vtsearch/datasets/importers/base.py:583` | `DatasetImporter.effective_source_specs` | 27 | Two distinct branches (multi-media vs legacy) joined by a shared output-type resolution preamble. Extract `_resolve_output_type(field_values)`, `_parse_specs_raw(raw)` (None/empty/str/list → list[dict] with the JSON-decode error path), `_validate_spec(spec, output_type)` (the per-entry converter / source-type / target-type guard table — each branch keeps its unique ValueError message), and `_legacy_source_specs(field_values, output_type)` (the `converters` CSV synthesis). The method then reduces to: resolve output type → branch on `self.multi_media` → for multi: parse + per-item validate, for legacy: synthesise. Expected CC ≈ 7. |
| `vtsearch/detectors/training.py:142` | `train_and_score` | 23 | Five clearly-staged concerns: (a) build training tensors from votes (region-aware for goods, plain embedding for bads); (b) guard ≤1 sample / single-class; (c) calibrate threshold (skip when <6 labels); (d) train final MLP; (e) score every media (region-aware max-pool vs plain). Extract `_build_vote_tensors(clips_dict, good_votes, bad_votes, region_boxes)` → `(X, y, X_list, y_list, hidden_dim)` (or `None` to early-return), `_score_all_media(model, clips_dict, has_regions)` → `(all_ids, scores, best_region)`, and `_format_results(all_ids, scores, best_region, clips_dict)` → result-dict list. Outer body becomes: build → guard → calibrate → train → score → safe-threshold blend → format. Expected CC ≈ 8. |
| `vtsearch/detectors/label_sync.py:17` | `sync_labels_to_loaded_detector` | 23 | Three concerns interleaved: (a) the guard chain (find-mode / no-detector / no-entry / no-on-disk-data); (b) the cross-dataset LabelSet merge (drop existing entries owned by the current dataset, append current entries, dedup); (c) the post-write cache + counter + mtime refresh. Extract `_get_loaded_detector_state()` returning `(entry, path, data, det_ctx)` or `None` so the guards become one early-return, `_merge_labelsets_across_datasets(existing_ls, current_ls, current_dataset_keys)` → `LabelSet` (pure merge logic, easy to unit-test), and `_refresh_detector_caches(det_ctx, merged, path, media_type)` for the post-write side effects. Outer body becomes: guard → snapshot+build current ls → merge → write → refresh → update entry timestamp. Expected CC ≈ 6. |
| `vtsearch/datasets/importers/combine_datasets/__init__.py:72` | `CombineDatasetsImporter.run` | 21 | Pairs with `run_chunked` (CC 21 — also `# noqa: C901`): both parse the comma-or-list `datasets` field, validate the same `<2`/`!exists` guards, and walk each source pickle with identical media-type consistency + MD5 dedup. Extract `_parse_dataset_paths(raw)` → `list[Path]` (with both guards) and `_iter_unique_source_clips(paths, thin, seen_md5s, mtype_state, progress)` — a generator that yields `(path, source_clips_dict_with_dedup)` for each non-empty pickle, raising `ValueError` on media-type mismatch. `run` accumulates into `all_clips` (with `MemoryError` handling and the final ID assignment + progress); `run_chunked` yields per-pickle chunks with fresh IDs. Two markers retired in one PR. Expected CC ≈ 7 each. |

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

1. `CombineDatasetsImporter.run` + `run_chunked` (one PR, shared
   helper — both markers retire together).
2. `sync_labels_to_loaded_detector` — pure refactor, easy to verify
   with the existing `tests/detectors/test_label_sync*.py` suite.
3. `train_and_score` — exercised by virtually every detector test
   in the suite; the safety net is dense.
4. `effective_source_specs` — last because the validation branches
   carry user-facing error messages that have to round-trip exactly;
   covered by `tests/datasets/test_effective_source_specs.py` and
   the multi-media import suites.

## Done

(empty — refactors not yet started)

## Open follow-ups

- After all four refactors land, re-run `radon cc vtsearch/ app.py -s
  -n D --no-assert` and update the "Worst offenders remaining" list in
  [brainstorm.md §20.7.1](brainstorm.md) to reflect just the five
  **Skip** rows above. At that point this file can be deleted.
- The remaining C-grade (11-19) `# noqa: C901` entries are the long
  tail; they don't need a new triage file — burn them down as code in
  the area is touched.
