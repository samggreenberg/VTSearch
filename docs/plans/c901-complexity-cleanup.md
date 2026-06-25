# C901 complexity-marker cleanup

**Status:** First pass shipped (stale-marker removal + the McCabe ≥16 tier refactored). Remaining refactor candidates deferred; see Open follow-ups.

## Background

Every legacy hot-spot over the McCabe threshold (`max-complexity = 10`) carries a
per-function `# noqa: C901`. This pass audited all of them to answer: *does each
function genuinely need the exception, or is the marker hiding avoidable complexity?*

Method: `ruff check --select C901 --ignore-noqa` reports the true McCabe number for
each suppressed function. Each was then classified **stale** (already ≤10, marker
inert), **justified** (intrinsic complexity — flat dispatch, coupled validation
cascade, retry state machine, or nested-closure factory), or **refactorable** (a
cohesive sub-block extracts cleanly into a helper, dropping the function under 10
*and* reading better).

## What shipped

**Inert markers removed (5):** the four functions below were already ≤10 (simplified
since the marker was added), and `scripts/**` is already C901-exempt via
`per-file-ignores`, so its inline token was redundant.

- `vtsearch/achievements.py` `_credit_vote`
- `vtsearch/routes/detectors/scoring.py` `find_label`
- `vtscore/datasets/archive.py` `extract_archive`
- `vtscore/datasets/clipper_chain.py` `_select_chain_output`
- `scripts/spike_structural_roxford.py` `main` (dropped the redundant `C901`, kept `PLR0915`)

**Refactored below threshold (9), highest-complexity tier (McCabe ≥16):**

| Function | Was | Helpers extracted |
|---|---|---|
| `stages/embedding.embed_missing` | 33 | `_resolve_embedder`, `_run_embed_pass`, `_run_backfill_pass`, `_missing_for_embedder`, `_needs_side_channel`, `_ensure_model_loaded`, `_noop_progress` |
| `media/list.add_media_to_pile` | 20 | `_read_pile_upload`, `_resolve_embedder`, `_embed_upload`, `_make_pile_thumbnail`, `_insert_or_collide` |
| `file_browser.browse` | 17 | `_entry_to_listing`, `_parse_allowed_exts` |
| `detectors/registry.load_detector_route` | 17 | `_run_detector_load_task`, `_unload_active_detector` |
| `loader_demo.load_demo_dataset` | 17 | `_try_load_cached`, `_resolve_demo_embedder`, `_write_demo_cache` |
| `downloader/core._download_and_extract` | 17 | `_extract_archive` |
| `detectors/labels.import_labels_into_detector` | 16 | `_merge_entries_into_labelset` |
| `datasets/ingest.ingest_missing_medias` | 16 | `_ingest_via_importer` (mirrors existing `_ingest_via_source`/`_ingest_via_resolver`) |
| `downloader/text.download_bbc_news` | 16 | `_ensure_bbc_extracted` |

All behavior-preserving; full suite (5168 tests) green.

## Justified — left as-is (noqa kept)

These stay flagged because the complexity is intrinsic and extraction would *hurt*
readability: flat one-branch-per-case dispatch (`_demo_sources.load_demo_source`,
`replay_chain_on_file`, `resolver._resolve_with_stack`, `_export_autodetect`),
coupled validation/guard cascades (`combine_detectors`, `load_pipeline_file`,
`sync_from_labelset_source`, `resolve_or_train_detector`, `register_detector_from_labelset`,
`load_registered_dataset`, `resolve_file`, `run_label_import`,
`detect_media_types_in_folder`, `_stamp_origin`, `download_ucsf_documents`,
`download_arxiv_abstracts`, `apply_chain_to_clips`, `_fixup_clip_md5_and_embeddings`),
retry/resume state machines (`download_file_with_progress`), irreducible loop nests
(`_build_node`, `run_label_curve_eval`), the train-first/rollback ordering in
`apply_and_retrain`, and nested-closure factories whose McCabe is inflated by
aggregated `def`s sharing captured state (`_make_per_side_setting`,
`intercept_tqdm_progress`, `stream_progress_events`, `_resolve_context`,
`_transcode_to_mp4`, `media_type.load_demo_source`).

## Open follow-ups

Refactorable but deferred (the audit found a clean extraction for each; not done in
this pass). Each drops under 10 with the named helper:

| Function | McCabe | Suggested extraction |
|---|---|---|
| `media/list._resolve_display_image` | 11 | `_MIMETYPE_BY_EXT` dict lookup |
| `converters/video2image.convert` | 14 | `_compute_n_frames(...)` |
| `concurrency/progress.list_tasks` | 14 | `_build_snapshot(task_id, entry)` (de-dup branches) |
| `detectors/labeling_progress._ensure_cache` | 11 | `_resolve_step_model(...)` |
| `detectors/labeling_progress._eval_cached_models` | 14 | `_score_step(...)` |
| `detectors/label_restoration.restore_labels_from_detector` | 15 | `_resolve_unmatched(...)` |
| `detectors/media_seeding.seed_good_votes_from_examples` | 14 | `_ensure_embedder(...)` |
| `routes/labels/vote.fill_labels_from_sort` | 15 | `_partition_candidates(...)` |
| `labels/importers/server_csv_file._parse_csv_bytes` | 13 | `_row_to_entry(...)` |
| `datasets/stages/clipper._apply_clipper` | 12 | `_stamp_default_clipper(...)` |
| `datasets/stages/clipper._regenerate_clip_thumbnails` | 14 | `_thumb_for(clip, media_type)` over one loop |
| `downloader/images.download_caltech101` | 14 | `_extract_members(...)` |
| `media/image/clipper.clip` | 14 | `_box_from_detection(det, w, h)` |
