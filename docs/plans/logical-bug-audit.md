# Logical-Bug Audit — 2026-05

**Status:** In progress — most findings still open; resolved items
are listed under [Open follow-ups → Shipped](#shipped) at the
bottom and marked inline as struck-through headings.

**Scope:** Multi-agent audit (10 per-subsystem + 5 cross-section
interaction passes) of the entire VTSearch codebase, focused on
**logical** bugs — missed expectations between modules, race conditions,
silent miscompute, data corruption, and broken invariants. Syntax,
typing, and lint issues were explicitly out of scope (those are covered
by `ruff` / `pyright` / `tsc`).

**Total distinct findings (after dedup):** ~95.

## How to read this doc

- Findings are grouped first by **severity**, then by subsystem within
  each tier.
- Each open finding has a stable ID (`C#` Critical, `H#` High,
  `M#` Medium, `L#` Low) so it can be referenced from other docs / PRs.
- Shipped findings are reduced to a struck-through heading; the full
  fix summary lives in [Open follow-ups → Shipped](#shipped).
- The "Recurring patterns" section at the bottom is the actionable
  starting point — most findings collapse into a small number of root
  causes; fixing the pattern usually fixes many findings at once.
- The "Suggested fix order" section recommends which root-cause PRs to
  land first for maximum leverage.
- File:line references are approximate; line numbers may shift slightly
  as the codebase evolves.

## Audit coverage

Per-subsystem agents:

1. State management & concurrency (`vtsearch/state/`, `concurrency/`)
2. Detector training pipeline (`vtsearch/detectors/`, `training/`)
3. Datasets & importers (`vtsearch/datasets/`)
4. Routes & API contracts (`vtsearch/routes/`)
5. Settings & sync sources (`vtsearch/settings.py`, `settings_io/`,
   `sync/`)
6. Embedding & training infrastructure (`vtsearch/embedding/`,
   `training/`)
7. Security validation (`vtsearch/security/`)
8. Plugins, converters, eval, exporters
9. Auth, CLI, app entry (`vtsearch/auth/`, `app.py`, `cli.py`)
10. Frontend Angular logic (`frontend/src/`)

Cross-section interaction agents:

11. Dataset ↔ Detector ↔ Embedder triple
12. Settings ↔ everything (user vs server tier, sync sources)
13. Background-jobs / context propagation
14. Error / exception flow across layers
15. Frontend ↔ backend contract seams

---

## Critical — data corruption / loss / hangs / silent miscompute

### ~~C1. Download gate is never released if importer skips the `"embedding"` status~~

### ~~C2. JobManager never sets dataset/detector thread-local context~~

### ~~C3. Dataset background load tasks don't set thread-local dataset context~~

### ~~C4. Embedding-matrix cache is not invalidated after clip/dedup~~

### ~~C5. `find_label` allows body field to override the request's dataset context~~

### ~~C6. Zip-slip in HTTP archive importer (zip AND tar)~~

### ~~C7. NaN/Infinity threshold leaks through safe-threshold blending~~

### ~~C8. Bulk label paths skip achievement recording entirely~~

### ~~C9. Path-template substitution missing post-resolution validation~~

### ~~C10. MD5 / metadata cache collision returns wrong-dataset media on switch~~

### ~~C11. `fill_labels_from_sort` silently swallows sync failures~~

### C12. Orphaned dataset registry entry on activation failure

- **File:** `vtsearch/datasets/load_pipeline.py` ~L596–610, ~L825–842
- **Bug:** If the importer/pickle save succeeds (registry entry
  created) but a later stage (diversity tree, post-embed clip fix-up)
  throws, the registry entry remains while the `DatasetContext` is
  unregistered. The dataset shows up in the UI list but can't be
  loaded — and the next "load" hits the half-built path again.

---

## High — likely-encountered correctness or security bugs

### State / concurrency

- ~~**`record_vote()` called after releasing `_state_lock`**~~
- **H1. Vote-progress invalidation / `record_vote` race** when the same
  media is rapid-toggled across two tabs — counter inflates while
  in-memory shows one vote.

### Detector / training

- **H2. Cross-dataset region-box loss** —
  `vtsearch/detectors/labelset_training.py` L122–124. When an element
  has `region_box` but the file isn't in the active dataset, fallback
  `_embed_one` embeds the full file. Region-level training intent
  silently downgrades to image-level.
- **H3. Embedder drift on save → reload** —
  `vtsearch/detectors/training.py` L449, L463.
  `train_detector_from_origins()` calls
  `embed_file(file_path, media_type)` with no `embedder_name`, so a
  CLAP-trained detector re-trains with whatever the default audio
  embedder is. CLAUDE.md's "re-derive with active embedder"
  invariant becomes "re-derive with **wrong** embedder."
- **H4. Asymmetric region pooling between good/bad votes** —
  `vtsearch/detectors/training.py` L171. Good votes use
  `_training_vec_for_vote(..., region_boxes.get(cid))`; bad votes use
  the raw embedding unconditionally. Mixed-modality MLP training on
  the same detector when patch_grid is in play.
- **H5. Detector embedder not revalidated on dataset switch** —
  `vtsearch/routes/detectors/scoring.py` + `dataset_sync.py`.
  `ensure_votes_match_active_dataset()` rehydrates votes but doesn't
  check the new dataset's embedder against the detector's training
  embedder. An MLP trained on CLAP can score SigLIP vectors with no
  warning.
- **H6. `train_model` produces degenerate single-class model** —
  `vtsearch/training/mlp.py` L180–189. If all `y` are 0 or all 1,
  weights default to 1.0 each and the model trains on an
  uninformative loss → returns ~0.5 for everything instead of
  raising.
- **H7. Vote applied before retrain; retrain failure leaves vote live** —
  `vtsearch/detectors/workflow.py` L40–105. User sees the vote, model
  is stale or absent.

### Datasets

- **H8. Origin dict shared by reference across medias** —
  `vtsearch/datasets/load_pipeline.py` L699–705. `_tag_origins()`
  stamps the same dict object; later mutations of `origin.params`
  propagate to siblings. Each media must hold its own copy.
- **H9. Single-item split has 0 test samples** —
  `vtsearch/datasets/split.py` L67–72. When a category has exactly 1
  item, the docstring promises 1 test sample but the function leaves
  test empty. Eval over that category silently skips.
- **H10. Clipped media re-ingest reads whole file for MD5** —
  `vtsearch/datasets/ingest.py` L275. MD5 of the full parent
  ≠ MD5 of the clip, so dedup never re-matches the original clip.
- **H11. Multi-media import with empty form yields empty dataset** —
  `vtsearch/datasets/importers/base.py` `effective_source_specs()`
  returns `[]`; downstream loops silently do nothing.

### Routes / API

- **H12. `add_media_to_pile` race** — `vtsearch/routes/media/list.py`
  L611–615. Snapshot in dataset A, `apply_label` in
  (concurrently-switched) dataset B. Labels applied to wrong
  dataset.
- **H13. `vote_media` silent-mistarget on dropped header** —
  `vtsearch/routes/media/list.py` L524–563. Missing `X-Dataset-Id`
  falls back to whatever the context proxy resolves to; vote applies
  to wrong media.
- **H14. `export_labels` leaks votes across datasets** —
  `vtsearch/routes/labels/vote.py` L146–162. `good_votes` / `bad_votes`
  are detector-scoped, not dataset-scoped. Re-using a detector across
  datasets bleeds votes into the export of the wrong dataset.
- **H15. File-browser symlink-traversal** — `vtsearch/routes/file_browser.py`
  L84–92. `target.resolve().relative_to(root.resolve())` succeeds for
  `/data/link → /etc`. If the configured root contains a symlink,
  listings escape.
- **H16. Header refers to unloaded dataset → silent fallback** —
  `app.py` `before_request`. `ctx = get_context("nonexistent_id")`
  returns None, `g._dataset_context` is never set, the proxy silently
  resolves to the empty context. Client sees stale data with no
  error.

### Plugins / converters / exporters

- **H17. Plugin scanner silently shadows duplicate names** —
  `vtsearch/plugins/__init__.py` ~L466. Second plugin with the same
  `name` overwrites the first; no warning, no error.
- **H18. CSV exporter doesn't escape embedded newlines** —
  `vtsearch/exporters/server_csv_file/__init__.py` L129–141. Label
  text containing `\n` breaks row structure; re-import fails.
- **H19. Required select fields silently accept empty string** —
  `vtsearch/plugins/schema.py` L119. `required` + `load_default=""`
  lets empty value through marshmallow.
- **H20. `audio2image` has no upper bound on `n_mels`** —
  `vtsearch/converters/audio2image.py` L201. `max(8, n_mels)` only
  sets the floor; absurdly large user input → OOM/DoS.

### CLI / app entry

- **H21. Successful `--autodetect` falls through to Flask startup** —
  `app.py` ~L636, ~L792. `_run_pipeline()` returns `None`, no
  `sys.exit(0)`, the `elif args.local or not args.autodetect:`
  branch evaluates true and starts the server. Breaks any CI / cron /
  one-shot orchestration.

### Embedding

- **H22. `predict_embedders_to_preload` mismatched media type** —
  `vtsearch/embedding/loader.py` L162. Registry filters by
  `media_type_id`, but dataset metadata may carry a different
  `media_type` string for a custom embedder; preloads the wrong
  model.

### Security / sync

- **H23. Settings-source filepath template not path-validated** —
  `vtsearch/settings.py` L939–946 + the source plugins above. Even
  with `sanitize_template_value()`, the template itself can contain
  `../`, and the substituted path is never re-validated.

### Frontend

- **H24. Vote-state polling chain dies on a single error** —
  `frontend/src/app/services/vote-state.service.ts` L153–169.
  `switchMap(() => sortingApi.getVotes())` terminates the whole
  observable on error; `polling` flag never resets; votes freeze
  indefinitely.
- **H25. Active dataset pair is set before load completes** —
  `frontend/src/app/services/context-switch.service.ts` L132–134.
  `setActivePair()` runs before the load is verified; interceptor
  immediately tags subsequent requests with a context that may not
  be loaded → cascade of 404s.
- **H26. `recordVote` runs before the HTTP vote returns** —
  `frontend/src/app/components/center-panel/center-panel.component.ts`
  L249–251. Undo stack contains a vote that may have never reached
  the server; Cmd-Z then posts a "reversal" of nothing.
- **H27. Binary media endpoints bypass the `activeContextInterceptor`** —
  `frontend/src/app/services/medias-api.service.ts` L41–60. Raw
  `HttpClient.get()` for `/api/medias/.../audio|video|image` lacks
  `X-Dataset-Id`; correct dataset is inferred by interceptor only for
  typed-client calls.

### Multi-process / settings

- **H28. Per-user cache is process-local; concurrent worker writes lose
  updates** — `vtsearch/settings.py` `_user_caches` + `_synced_users`.
  Two gunicorn workers each sync independently, then race the
  per-user write.
- **H29. `_save_user` holds `_settings_lock` across sync I/O** —
  `vtsearch/settings.py` ~L331–338. Slow source (NFS, webhook) blocks
  all other settings reads/writes globally.

### Error flow

- **H30. Detector save's `os.replace` failure leaves in-memory state
  "saved"** — `vtsearch/detectors/store.py` L52–59. Next save doesn't
  realize prior persistence failed.
- **H31. Partial label-import has no rollback** —
  `vtsearch/routes/labels/importers.py` L143. Failure mid-loop leaves
  a half-applied labelset; user has to manually figure out which
  ones landed.

---

## Medium — real bugs but lower frequency or non-corrupting impact

### State / concurrency

- **M1.** Lock held during cross-lock callbacks in `toggle_vote` — state ↔
  progress lock ordering creates a narrow but real deadlock window.
- **M2.** `combine_datasets.run_chunked` re-issues IDs starting at 1 on every
  call → cid collision when consumed twice.
- **M3.** `importers/base.py` L407–410 — skipped records leave `next_id`
  unincremented, ID collisions on first-record-skip.

### Detector

- **M4.** `populate_label_embeddings` cache not invalidated when a
  `region_box` is removed from an element; stale pooled vector
  continues to be used.
- **M5.** `labelset_elements.resolve_current_dataset_cid` can return a
  colliding-MD5 cid in cross-dataset labelsets → clicks vote the
  wrong media.
- **M6.** `restore_labels_from_detector` resolves by MD5 only on the second
  pass; dedup-collapsed cids can land votes on the wrong cid after
  reload.
- **M7.** `safe_thresholds` is read at *training* time and baked into the
  detector JSON; per-user threshold preference cannot change after
  save.

### Datasets / loaders

- **M8.** Thin-mode pickle loader treats `embedding: None` as present, then
  `np.array(None)` produces an object-dtype row.
- **M9.** `loader_folder._has_override` doesn't warn when both `rel_path`
  and `file_name` override entries exist with different embeddings.
- **M10.** `clipper_chain._run_clipper_step` assumes deterministic output count
  across calls — no validation.
- **M11.** Stale media in `cli._score_medias_with_detectors` when some
  embeddings are `None` (zip truncates silently).
- **M12.** `loader_pickle._build_pickle_full_media` has no null-check before
  `np.array(media_info["embedding"])`.

### Routes / API

- **M13.** `learned_scores` in `/api/votes` can serialize as JSON
  `NaN`/`Infinity` if the MLP destabilizes — invalid JSON to strict
  clients.
- **M14.** `diversity_tree_next_sample` references stale media IDs after
  `/api/dataset/clear`.

### Settings / sync

- **M15.** Pending labelset sync stores `dataset_ctx = None` without checking;
  later `_run_pending_sync` triggers `AttributeError`.
- **M16.** Sync to source on first read after `_synced_users` marker can still
  return stale local config if source changes silently.
- **M17.** Legacy migration `_maybe_migrate_legacy_settings_locked` pops keys
  from in-memory cache before per-user disk write; per-user-write
  failure leaves cache and disk diverged.

### Embedding / training

- **M18.** `_PeekUnpickler` doesn't override `FLOAT` / `SETITEMS` opcodes →
  falls back to slow real unpickle for older protocols.
- **M19.** `embed_text_enriched` crashes (`np.mean` on empty) when text encoder
  fails and all wrappers return None.
- **M20.** XCLIP single-frame video: `linspace(0, 0, 1)` + padding gives 8
  identical frames → degenerate embedding.
- **M21.** Empty paragraph clip survives to dataset with `None` embedding;
  embedding-matrix builder later misbehaves.

### Auth / context

- **M22.** `set_thread_user()` cleanup relies on every caller's `finally`; a
  future `ThreadPoolExecutor` reuse would leak user identity across
  requests.
- **M23.** `setup_logging` re-running can leave duplicate handlers on
  non-root loggers.
- **M24.** `CoreConfig.from_settings()` raises if a blueprint's module-level
  code runs before `vtsearch/shim/__init__.py` registers the
  builder.

### Frontend

- **M25.** `LeftPanelComponent` lacks `OnDestroy` / `takeUntil` on init
  subscriptions; subscriptions leak across dataset switches.
- **M26.** `labelset-state.service` `startPolling()` is not tied to
  `destroy$`; rapid switches leak polls.
- **M27.** `progress-events.service` doesn't reconcile stale `task_id`s after
  backend restart.
- **M28.** Audio waveform fetch's `catch {}` silently shows "Unable to load
  waveform" with no UI state propagation.
- **M29.** `AudioContext` not cleaned up on rapid navigation → resource
  exhaustion.
- **M30.** Autopilot phase transitions can oscillate
  (`hard → new → hard → new`) when smart/stable status flickers.
- **M31.** `settings-importer-modal` auto-closes after 1.5s timeout regardless
  of operation duration.

### Security

- **M32.** `sanitize_template_value` allows `...` (and worse — any non-`.` /
  `..` token of dots).
- **M33.** `rglob_follow_symlinks` doesn't detect cycles → CPU/RAM DoS on
  circular link layouts.
- **M34.** Email exporter validates "@" only; `"@example.com"` passes and
  fails at SMTP time.

### Eval / exporters

- **M35.** `voting_iterations.py` reports F1/FPR with one good + one bad vote
  as if reliable.

---

## Low — latent / cosmetic / hypothetical

- **L1.** `_run_pipeline` returns `None` silently in text mode (no "done"
  signal).
- **L2.** Pipeline-vs-server logic in `app.py` L792 is masked by `sys.exit()`
  but breaks if the function is ever refactored to return.
- **L3.** Frontend cross-pane settings: `getViewMode()` falls back to
  hard-coded defaults when the per-media-type entry isn't loaded.
- **L4.** `get_settings_source_config` reads cache without re-checking sync
  state.
- **L5.** Frontend `clip_box` exports: list-of-floats CSV-joined without
  quote-protection (edge case).
- **L6.** Empty `Origin.params` not always dropped consistently across
  importers.
- **L7.** `eval/metrics.py` returns F1=0 for empty test set without flagging
  the degenerate case.
- **L8.** Logging of vote-related events truncates achievement traceback for
  UI display.
- **L9.** `get_param` returns `""` for unknown keys; silently swallows
  renamed parameters.

---

## Recurring patterns (fix-in-batches root causes)

These themes show up across many findings. Addressing each pattern in
one PR is far more effective than one-off fixes.

1. **Background-thread context propagation.** Every `Thread(target=…)`
   spawn in the repo should set user + dataset + detector
   thread-locals (where applicable) before invoking the target, in a
   `try/finally`. Candidates:
   - `JobManager._run`
   - `_run_importer_in_background`
   - `_stage_importer_in_background`
   - `_warmup_embedder_async`
   - `smart_preload_in_background`
   - `preload_embedder_for_dataset`
   - `timed_progress` ticker.

   A small helper
   `with run_with_context(user, dataset_id, detector_id): …` removes
   the repetition.

2. **Template-path substitution → re-validate.** Every source /
   exporter that interpolates `{username}`, `{detector_id}`,
   `{detector_name}` must call `validate_server_filepath()` on the
   **resolved** path before opening a file. Affected:
   `labels/sources/server_json_file`,
   `settings_io/sources/server_json_file`, and any future plugin in
   the same family.

3. **Achievement recording at one site.** Move `record_vote()` (and
   all achievement hooks) into `apply_label_with_click_time` so bulk
   paths (`fill-from-sort`, label import, find-label) credit the user
   uniformly. Also: call it *inside* the state lock, not after.

4. **Embedding-matrix cache invalidation.** Bump a `media_revision`
   counter on every `medias` mutation; the matrix accessor checks the
   counter. Same fix neutralizes both clip/dedup invalidation (C4)
   and dynamic seeding races.

5. **Embedder identity is its own first-class attribute.** Track the
   detector's *training* embedder separately from the active
   dataset's embedder. Every train/score path must compare them and
   refuse / clear caches when they differ — don't only compare
   against the dataset.

6. **`X-Dataset-Id` / `X-Detector-Id` headers must be required, not
   silently defaulted.** When the header is missing or refers to an
   unloaded context, the `before_request` middleware should return
   400 / 410, not fall back to the empty proxy. Catches C5, the
   silent-mistarget routes, and the binary-streaming bypass.

7. **Frontend cache keys must be dataset-qualified.**
   `MediaMetadataCacheService` and any other id-keyed cache should
   key on `${datasetId}:${mediaId}`.

8. **Vote endpoints should be transactional.**
   apply-then-retrain-then-sync should either all-succeed or
   all-rollback; surface failures to the frontend instead of
   returning 200 and logging server-side.

9. **Background jobs need a clear error state.** Every failure path
   on a load / embed / train job should call
   `update_progress(task_id, error="…")` so the frontend can stop
   spinning and show what went wrong.

---

## Suggested fix order (highest leverage first)

1. **C1, C2, C3** — gate hand-off + thread-local context
   propagation. One small helper (pattern #1) unblocks 3+ bug
   classes. (C1 verified safe and C2 shipped on 2026-05-19; C3
   still open.)
2. **C4** + pattern #4 — embedding-matrix cache invalidation via a
   `media_revision` counter.
3. **C7** — clamp `xcal_threshold` to a finite sentinel and assert
   no `NaN` in `safe_threshold`.
4. **C9** + pattern #2 — re-validate every resolved template path
   in `SyncSource._resolve_filepath()`.
5. **C5, C10, C11, C12** — straightforward request-handling and
   frontend-cache fixes once the above patterns are in place.
6. Pattern #6 — make header presence + context-loaded a hard
   precondition.
7. Detector-embedder identity (pattern #5) + C8 (achievement
   uniformity).

---

## Open follow-ups

This plan started as discovery-only. As individual fixes land, the
corresponding finding above is collapsed to a struck-through heading
and a one-line summary is recorded here.

### Shipped

- **C1 — Download gate safety net verified** (2026-05-19, branch
  `claude/fix-logical-bug-audit-c1-DV5Lc`). After closer inspection
  the bug as described is not real: the audit looked at
  `_LoadGateController` in isolation and missed two safety nets in
  the caller (`_run_origin_load_in_background.task()`) — an
  unconditional `controller.swap_to_embed()` after the importer
  returns, plus a `try/finally` that calls `controller.release()` on
  every exit path. Together they cover minimalist importers exactly
  as the audit's "fix sketch" recommended. Inline comment at the
  unconditional swap site expanded to call out the invariant, and a
  regression test
  (`tests/datasets/test_parallel_loading.py::TestLoadingGates::test_minimalist_importer_releases_both_gates`)
  was added so a future refactor can't silently remove either safety
  net.

- **C2 — JobManager thread-local context propagation** (2026-05-19).
  `JobManager._run` in `vtscore/concurrency/async_jobs.py` now sets
  `set_thread_dataset_context` / `set_thread_detector_context` from
  `job.dataset_id` / `job.detector_id` before invoking the target, and
  clears all three thread-locals (user + dataset + detector) in a
  `finally`. Covered by `tests_lib/integration/test_async_jobs.py::TestThreadContextPropagation`.
- **C3 — Dataset background load thread-local context** (2026-05-19).
  `_run_origin_load_in_background.task()` in
  `vtscore/datasets/load_pipeline.py` now pins the freshly-created
  `DatasetContext` to its worker thread via
  `set_thread_dataset_context(ctx)` immediately after creation, and
  clears the thread-local in the task's `finally` (before
  `loading_tasks.mark_finished`, so callers waiting on
  `has_active_tasks() == False` see fully-cleaned-up worker state).
  Regression in
  `tests/datasets/test_parallel_loading.py::TestBackgroundLoadThreadContext`.
- **C4 — embedding-matrix cache after clip/dedup** (2026-05-19).
  `_apply_clipper_stage`, `_collapse_duplicates_stage`, and the
  registry's reload-from-pickle dedup now all call
  `invalidate_embedding_matrix(ctx)` after mutating the medias dict.
  Regression coverage in `tests/datasets/test_load_stage_matrix_cache.py`.
- **C5 — `find_label` body-field dataset override** (2026-05-19).
  Removed the body override in `find_label`, dropped `dataset_id` from
  `FindLabelRequestSchema`, and removed the redundant body field from
  the frontend caller. The `X-Dataset-Id` header is now the only
  dataset selector for `/api/find-label`.
- **C6 — zip-slip in HTTP archive importer (zip, tar, rar)**
  (2026-05-19). Added `_reject_traversal(extract_dir_resolved,
  member_name)` validating member names before extract on all three
  archive code paths; regression tests in
  `tests_lib/io/test_importers.py::TestExtractArchive`.
- **C7 — NaN/Infinity threshold sentinel** (2026-05-19). Replaced the
  `inf` sentinel with finite `NO_GOOD_THRESHOLD = 2.0`, hardened
  `calculate_safe_threshold` to detect non-finite inputs and fall back
  to the finite side (or `0.5` if both non-finite), and asserted
  finite output. Regression tests cover `inf`/`NaN` xcal and the new
  sentinel.
- **C8 — Bulk label paths skip achievement recording entirely**
  (2026-05-19). Centralised achievement credit inside `_state_lock`
  in `vtscore/state/votes.py`; also fixed the related High finding
  ("`record_vote()` called after releasing `_state_lock`") in the
  same change.
- **C9 — Path-template post-resolution validation** (2026-05-19).
  `_resolve_filepath()` in both
  `vtscore/labels/sources/server_json_file/__init__.py` and
  `vtsearch/settings_io/sources/server_json_file/__init__.py` (plus the
  labels source's `resolve_filepath_for()` used by the rename flow) now
  ends in `validate_server_filepath(resolved, base_dir=
  get_file_access_base_dir())`. Regression tests in
  `tests/io/test_sync_sources.py`.
- **C10 — MediaMetadataCacheService dataset-qualified keys** (2026-05-19).
  Cache entries in
  `frontend/src/app/services/media-metadata-cache.service.ts` now key on
  `${datasetId}:${mediaId}` (snapshotted from `ActiveContextService`),
  and pending IDs are bucketed by the dataset they were queued under so
  each batch fetch dispatches only while its dataset is active and the
  `X-Dataset-Id` interceptor header matches what the response will be
  cached against.
- **C11 — `fill_labels_from_sort` sync-failure surfacing** (2026-05-19).
  Disk sync now runs before the response and is wrapped in
  `try/except`; failures from `sync_labels_to_loaded_detector()`
  surface as 500. Known limitation: in-memory labels are not rolled
  back (full transactional rollback deferred to pattern #8).
  Regression: `tests/io/test_export_options.py::TestFillFromSortConfirm::test_disk_sync_failure_surfaces_as_500`.

### Still open

Every other finding (C12 and the H / M / L tiers) remains as written.
When the next fix lands, collapse the finding above to a struck-through
heading and add a line here. When every critical and high is addressed,
this doc can be retired into the relevant subsystem docs (or deleted,
per the `docs/plans/` lifecycle).

Specific open items called out by previously-shipped fixes:

- **Pattern #4 (media_revision counter)** is still unimplemented.
  The C4 stage-level invalidation closes the known clip/dedup hole,
  but any future mutation site that changes embeddings without
  changing the id set will reintroduce the same class of bug. A
  `media_revision` counter on `DatasetContext` bumped from every
  `medias` mutation (or a `MediasDict` subclass that does so
  transparently) would neutralise the whole category and let the
  matrix accessor compare a single int instead of two id lists.
