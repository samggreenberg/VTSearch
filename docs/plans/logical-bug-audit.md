# Logical-Bug Audit — 2026-05

**Status:** In progress — most findings still open; resolved items
are listed under [Open follow-ups → Resolved](#resolved) at the
bottom and marked inline.

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

### C1. Download gate is never released if importer skips the `"embedding"` status — **shipped** (verified not a bug)

- **File:** `vtscore/datasets/load_pipeline.py` (was
  `vtsearch/datasets/load_pipeline.py` before the
  extract-library rename) — controller class at L619–661,
  task body at L897–945.
- **Original claim:** `_LoadGateController` only swaps from
  `download_gate` → `embed_gate` when the importer's progress
  callback fires `status="embedding"`. A minimalist importer that
  completes without that status keeps the download gate held forever,
  blocking every subsequent dataset load.
- **What's actually there:**
  `_run_origin_load_in_background.task()` already covers minimalist
  importers two ways: (1) `controller.swap_to_embed()` is called
  **unconditionally** at L918 after `_run_importer` returns, which
  releases the download gate even if the importer never emitted an
  `"embedding"` status; and (2) `controller.release()` lives in a
  `try/finally` at L939–940, releasing whichever gate is held on
  every exit path including exceptions. The audit's "fix sketch"
  (release in `finally` or guaranteed swap) describes the existing
  protection. The original audit pass appears to have analyzed
  `_LoadGateController` in isolation and missed both safety nets in
  the caller.
- **Fix:** Inline comment at the unconditional swap site expanded to
  call out the minimalist-importer invariant so a future refactor
  can't accidentally drop it. Regression test
  `tests/datasets/test_parallel_loading.py::TestLoadingGates::test_minimalist_importer_releases_both_gates`
  drives a load whose importer never fires `"embedding"` and asserts
  both `_download_gate.active == 0` and `_embed_gate.active == 0`
  after the task completes.

### C2. JobManager never sets dataset/detector thread-local context — **shipped**

- **File:** `vtscore/concurrency/async_jobs.py` (`JobManager._run`)
- **Bug:** `_run()` set `set_thread_user(job.user)` but never called
  `set_thread_dataset_context(get_context(job.dataset_id))` or the
  detector equivalent. Every learned-sort / eval / training job target
  that called `get_active_context()` could resolve to the empty fallback
  context — empty votes, missing media, silent miscompute.
- **Fix:** `JobManager._run` now resolves `job.dataset_id` /
  `job.detector_id` through `get_context()` / `get_detector_context()`
  and sets the thread-local contexts before invoking the target, with a
  `finally` that clears all three thread-locals (user + dataset +
  detector). Route closures in `vtsearch/routes/sorting.py` and
  `vtsearch/routes/eval.py` continue to set the contexts via captured
  refs; those calls are now redundant but kept as a no-op safety net in
  case the registry is mutated between `start()` and `_run()` (the
  closure holds a direct reference to the live context).
- **Severity note:** Highest-impact context-propagation bug; flagged by
  two independent agents.

### C3. Dataset background load tasks don't set thread-local dataset context

- **Status:** Shipped 2026-05-19 — `_run_origin_load_in_background.task()`
  now pins the freshly-created `DatasetContext` to its worker thread via
  `set_thread_dataset_context(ctx)` immediately after creation, and
  clears the thread-local in the task's `finally` (before
  `loading_tasks.mark_finished`, so callers waiting on
  `has_active_tasks() == False` see fully-cleaned-up worker state).
  Regression covered by `TestBackgroundLoadThreadContext` in
  `tests/datasets/test_parallel_loading.py`.
  The warmup and staging spawn sites originally listed alongside the
  importer site do not register a dataset context (warmup uses the
  passed `media_dict` directly; staging writes to a temp dict via the
  combine flow), so no `set_thread_dataset_context` call belongs there.
- **File:** `vtsearch/datasets/load_pipeline.py` ~L822, ~L947, ~L1141
  (warmup, importer, staging spawn sites)
- **Bug:** `task()` closures set thread user but never call
  `set_thread_dataset_context(ctx)`. Any importer / clipper /
  diversity-tree code that resolves `get_active_context()` lands on
  `_empty_dataset_context` instead of the in-flight context. Mutations
  meant for the new dataset are silently lost.

### C4. Embedding-matrix cache is not invalidated after clip/dedup — **SHIPPED 2026-05-19**

- **File:** `vtscore/datasets/load_pipeline.py`
  (`_collapse_duplicates_stage`, `_apply_clipper_stage`) +
  `vtsearch/routes/datasets/registry.py` (registry reload's dedup) +
  `vtscore/embedding/matrix.py`
- **Bug:** After media items are removed/renumbered,
  `DatasetContext._emb_matrix_ids` is left stale. The next
  `train_and_score()` reads a matrix whose row order no longer matches
  the live `medias` dict — training vectors and scored results are
  mapped to the wrong media IDs. Silent ranking corruption.
- **Fix:** Both load-pipeline stages now call
  `invalidate_embedding_matrix(ctx)` after mutating `ctx.medias`, and
  the registry's reload-from-pickle path does the same after its own
  `collapse_duplicates` call. Pattern #4 (a `media_revision` counter
  hooked into every `medias` mutation) remains the durable fix for the
  broader category — see the "Recurring patterns" section.

### C5. `find_label` allows body field to override the request's dataset context — **SHIPPED**

- **File:** `vtsearch/routes/detectors/scoring.py` ~L207–214, ~L336
- **Bug:** The endpoint mutates `g._dataset_context` from a `dataset_id`
  field in the request body, *after* `before_request` resolved it from
  headers. Combined with `replace_all=True` further down, a confused or
  malicious client can wipe one detector's votes while the UI thinks
  it's labeling a different one.
- **Fix shipped:** Removed the body override in `find_label`, dropped
  `dataset_id` from `FindLabelRequestSchema`, and removed the redundant
  body field from the frontend caller. The `X-Dataset-Id` header (set
  by `activeContextInterceptor`) is now the only dataset selector for
  `/api/find-label`, matching every other route.

### C6. Zip-slip in HTTP archive importer (zip AND tar) — SHIPPED 2026-05-19

- **File:** `vtscore/datasets/importers/http_archive/__init__.py`
- **What landed:** Added `_reject_traversal(extract_dir_resolved, member_name)`
  helper that validates member names **before** any extract call.  It
  rejects absolute paths (POSIX `/`, Windows `\\` and drive-letter form)
  and any name that, once joined and `os.path.normpath`-normalised,
  escapes the resolved extract dir (checked with `Path.is_relative_to`,
  not the old prefix-buggy `str.startswith`).  Applied uniformly to the
  zip, tar, and rar code paths — the tar branch previously had **no**
  pre-extraction member validation and relied solely on
  `filter="data"`; it now validates explicitly *in addition to* the
  `filter="data"` defense.  Regression tests added in
  `tests_lib/io/test_importers.py::TestExtractArchive` cover `../`
  traversal, absolute paths, and the prefix-collision case that the old
  `startswith` check accepted.

### C7. NaN/Infinity threshold leaks through safe-threshold blending — **SHIPPED**

- **File:** `vtscore/training/thresholds.py` (and matching sentinel in
  `vtscore/detectors/training.py`).
- **Bug:** `calculate_cross_calibration_threshold()` returned
  `float("inf")` when a valid calibration split was impossible (n_cal
  too aggressive on tiny label sets). When `label_weight == 0.0` and
  `xcal_threshold == inf`, the blend `0.0 * inf + 1.0 * gmm` evaluated
  to `NaN`. NaN then landed on `DetectorContext.threshold`, broke every
  `score >= threshold` comparison, and corrupted every result that
  touched that detector.
- **Fix:** Replaced the `inf` sentinel with a finite
  `NO_GOOD_THRESHOLD = 2.0` (above the [0, 1] sigmoid range, so
  `score >= threshold` is still always False) and updated the matching
  `safe and len(X_list) < 6` short-circuit in `train_and_threshold` to
  use the same constant. Hardened `calculate_safe_threshold` to detect
  non-finite inputs on either side (xcal or gmm), fall back to the
  finite side (or `0.5` if both are non-finite), and assert the final
  blended value is finite before returning it. Regression tests cover
  `inf`/`NaN` xcal and the new sentinel.

### C8. Bulk label paths skip achievement recording entirely — **SHIPPED 2026-05-19**

- **File:** `vtscore/state/votes.py` — `apply_label_with_click_time()`
  (no `record_vote()` call) vs `toggle_vote()` (calls `record_vote()`).
- **Bug:** `/api/labels/fill-from-sort`, label importers, and bulk
  find-label apply votes without crediting achievements. `votes_cast`,
  `days_active`, `vote_streak` are inert for any user who uses
  search-then-bulk-label flows.
- **Fix:** Achievement recording is now centralised in
  `vtscore/state/votes.py` via `_record_vote_locked()`. `toggle_vote`,
  `apply_label`, `apply_label_with_click_time`, and
  `apply_labels_bulk_with_click_time` all credit a vote inside
  `_state_lock` (which also fixes the cross-section "record_vote after
  releasing the lock" race noted in the High tier). `apply_label` gains
  a `record_achievement: bool = True` opt-out, used by
  `sync_from_labelset_source` and `seed_good_votes_from_examples` —
  system-driven paths that aren't user vote actions.  Idempotent
  re-applies (the media was already at that label) don't credit, so
  re-importing the same labelset doesn't inflate `votes_cast`.

### C9. Path-template substitution missing post-resolution validation

- **Status:** **Shipped 2026-05-19.**  Both
  `vtscore/labels/sources/server_json_file/_resolve_filepath()` /
  `resolve_filepath_for()` and
  `vtsearch/settings_io/sources/server_json_file/_resolve_filepath()`
  now call `validate_server_filepath(resolved, base_dir=
  get_file_access_base_dir())` after expanding templates, so a
  traversal template like `../labels/{detector_name}.json` is
  rejected at `load` / `load_full` / `save` time before any file
  handle is opened.  Regression tests live in
  `tests/io/test_sync_sources.py`.  Pattern #2 below is now satisfied
  for every source listed there; no other plugin in the family was
  found to need the same fix (file exporters route through
  `validate_filepath_field()` *before* template expansion, and the
  substituted values cannot reintroduce traversal because
  `sanitize_template_value()` replaces separators).
- **Files:**
  - `vtsearch/labels/sources/server_json_file/__init__.py`
    (`load`, `load_full`, `save`)
  - `vtsearch/settings_io/sources/server_json_file/__init__.py`
    (`load`, `save`)
- **Bug:** Per-value `sanitize_template_value()` is called, but the
  resolved final path is never run through
  `validate_server_filepath()`. A template like
  `../labels/{detector_name}.json` plus a detector name that survives
  sanitization yields a path that escapes the intended directory.
- **Fix sketch:** Architectural — every source/exporter must validate
  the resolved path before opening it. Make
  `validate_server_filepath()` a required step in
  `SyncSource._resolve_filepath()`.

### C10. MD5 / metadata cache collision returns wrong-dataset media on switch

- **File:** `frontend/src/app/services/media-metadata-cache.service.ts`
  ~L31
- **Bug:** Cache key is `media_id` only. Media IDs are per-dataset, so
  after the user switches datasets the cache returns dataset A's
  filename / md5 / custom_metadata for dataset B's id=1.
- **Fix sketch:** Key the cache by `${datasetId}:${mediaId}`.

### C11. `fill_labels_from_sort` silently swallows sync failures — **SHIPPED 2026-05-19**

- **File:** `vtsearch/routes/labels/vote.py` ~L259–301
- **Bug:** After applying labels, the endpoint calls
  `sync_labels_to_loaded_detector()` and `sync_to_labelset_source()`
  outside any try-block. The HTTP response is built before those
  calls' results are known; a failure logs but returns "success" —
  labels appear committed in the UI but never reach disk.
- **Fix:** Both sync calls now run **before** the response is built
  and are wrapped in `try/except`. A failure from
  `sync_labels_to_loaded_detector()` (disk write, detector-registry
  update, etc.) is logged and re-raised as a 500 with the underlying
  error message so the frontend can react instead of treating the
  labels as persisted. `sync_to_labelset_source()` is fire-and-forget
  by design (debounced background timer) — its synchronous portion is
  also wrapped defensively, but a failure there does **not** fail the
  request because disk state is already consistent. Regression test:
  `tests/io/test_export_options.py::TestFillFromSortConfirm::test_disk_sync_failure_surfaces_as_500`.
- **Known limitation / follow-up:** When the disk sync fails, the
  in-memory labels are *not* rolled back — they remain applied until
  the next successful sync writes them through (any subsequent vote
  or fill-from-sort run will reconcile). Full transactional rollback
  (pattern #8) is deferred; this fix only stops the silent-success
  failure mode that lets the UI diverge from disk indefinitely. The
  same untrapped sync pattern still exists in
  `vtsearch/routes/labels/vote.py::import_labels`,
  `vtsearch/routes/sorting.py`, `vtsearch/routes/media/list.py`,
  `vtsearch/routes/labels/importers.py`, and
  `vtsearch/routes/detectors/{registry,scoring}.py`; each is its own
  audit item under pattern #8 and out of scope for C11.

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

- **`record_vote()` called after releasing `_state_lock`** —
  `vtsearch/state/votes.py` ~L156–206. Detector context can change
  between unlock and credit; achievement is credited to the wrong
  detector. Cross-section with C8. **SHIPPED 2026-05-19 as part of
  C8** — `_record_vote_locked()` runs inside the state lock now.
- **Vote-progress invalidation / `record_vote` race** when the same
  media is rapid-toggled across two tabs — counter inflates while
  in-memory shows one vote.

### Detector / training

- **Cross-dataset region-box loss** —
  `vtsearch/detectors/labelset_training.py` L122–124. When an element
  has `region_box` but the file isn't in the active dataset, fallback
  `_embed_one` embeds the full file. Region-level training intent
  silently downgrades to image-level.
- **Embedder drift on save → reload** —
  `vtsearch/detectors/training.py` L449, L463.
  `train_detector_from_origins()` calls
  `embed_file(file_path, media_type)` with no `embedder_name`, so a
  CLAP-trained detector re-trains with whatever the default audio
  embedder is. CLAUDE.md's "re-derive with active embedder"
  invariant becomes "re-derive with **wrong** embedder."
- **Asymmetric region pooling between good/bad votes** —
  `vtsearch/detectors/training.py` L171. Good votes use
  `_training_vec_for_vote(..., region_boxes.get(cid))`; bad votes use
  the raw embedding unconditionally. Mixed-modality MLP training on
  the same detector when patch_grid is in play.
- **Detector embedder not revalidated on dataset switch** —
  `vtsearch/routes/detectors/scoring.py` + `dataset_sync.py`.
  `ensure_votes_match_active_dataset()` rehydrates votes but doesn't
  check the new dataset's embedder against the detector's training
  embedder. An MLP trained on CLAP can score SigLIP vectors with no
  warning.
- **`train_model` produces degenerate single-class model** —
  `vtsearch/training/mlp.py` L180–189. If all `y` are 0 or all 1,
  weights default to 1.0 each and the model trains on an
  uninformative loss → returns ~0.5 for everything instead of
  raising.
- **Vote applied before retrain; retrain failure leaves vote live** —
  `vtsearch/detectors/workflow.py` L40–105. User sees the vote, model
  is stale or absent.

### Datasets

- **Origin dict shared by reference across medias** —
  `vtsearch/datasets/load_pipeline.py` L699–705. `_tag_origins()`
  stamps the same dict object; later mutations of `origin.params`
  propagate to siblings. Each media must hold its own copy.
- **Single-item split has 0 test samples** —
  `vtsearch/datasets/split.py` L67–72. When a category has exactly 1
  item, the docstring promises 1 test sample but the function leaves
  test empty. Eval over that category silently skips.
- **Clipped media re-ingest reads whole file for MD5** —
  `vtsearch/datasets/ingest.py` L275. MD5 of the full parent
  ≠ MD5 of the clip, so dedup never re-matches the original clip.
- **Multi-media import with empty form yields empty dataset** —
  `vtsearch/datasets/importers/base.py` `effective_source_specs()`
  returns `[]`; downstream loops silently do nothing.

### Routes / API

- **`add_media_to_pile` race** — `vtsearch/routes/media/list.py`
  L611–615. Snapshot in dataset A, `apply_label` in
  (concurrently-switched) dataset B. Labels applied to wrong
  dataset.
- **`vote_media` silent-mistarget on dropped header** —
  `vtsearch/routes/media/list.py` L524–563. Missing `X-Dataset-Id`
  falls back to whatever the context proxy resolves to; vote applies
  to wrong media.
- **`export_labels` leaks votes across datasets** —
  `vtsearch/routes/labels/vote.py` L146–162. `good_votes` / `bad_votes`
  are detector-scoped, not dataset-scoped. Re-using a detector across
  datasets bleeds votes into the export of the wrong dataset.
- **File-browser symlink-traversal** — `vtsearch/routes/file_browser.py`
  L84–92. `target.resolve().relative_to(root.resolve())` succeeds for
  `/data/link → /etc`. If the configured root contains a symlink,
  listings escape.
- **Header refers to unloaded dataset → silent fallback** —
  `app.py` `before_request`. `ctx = get_context("nonexistent_id")`
  returns None, `g._dataset_context` is never set, the proxy silently
  resolves to the empty context. Client sees stale data with no
  error.

### Plugins / converters / exporters

- **Plugin scanner silently shadows duplicate names** —
  `vtsearch/plugins/__init__.py` ~L466. Second plugin with the same
  `name` overwrites the first; no warning, no error.
- **CSV exporter doesn't escape embedded newlines** —
  `vtsearch/exporters/server_csv_file/__init__.py` L129–141. Label
  text containing `\n` breaks row structure; re-import fails.
- **Required select fields silently accept empty string** —
  `vtsearch/plugins/schema.py` L119. `required` + `load_default=""`
  lets empty value through marshmallow.
- **`audio2image` has no upper bound on `n_mels`** —
  `vtsearch/converters/audio2image.py` L201. `max(8, n_mels)` only
  sets the floor; absurdly large user input → OOM/DoS.

### CLI / app entry

- **Successful `--autodetect` falls through to Flask startup** —
  `app.py` ~L636, ~L792. `_run_pipeline()` returns `None`, no
  `sys.exit(0)`, the `elif args.local or not args.autodetect:`
  branch evaluates true and starts the server. Breaks any CI / cron /
  one-shot orchestration.

### Embedding

- **`predict_embedders_to_preload` mismatched media type** —
  `vtsearch/embedding/loader.py` L162. Registry filters by
  `media_type_id`, but dataset metadata may carry a different
  `media_type` string for a custom embedder; preloads the wrong
  model.

### Security / sync

- **Settings-source filepath template not path-validated** —
  `vtsearch/settings.py` L939–946 + the source plugins above. Even
  with `sanitize_template_value()`, the template itself can contain
  `../`, and the substituted path is never re-validated.

### Frontend

- **Vote-state polling chain dies on a single error** —
  `frontend/src/app/services/vote-state.service.ts` L153–169.
  `switchMap(() => sortingApi.getVotes())` terminates the whole
  observable on error; `polling` flag never resets; votes freeze
  indefinitely.
- **Active dataset pair is set before load completes** —
  `frontend/src/app/services/context-switch.service.ts` L132–134.
  `setActivePair()` runs before the load is verified; interceptor
  immediately tags subsequent requests with a context that may not
  be loaded → cascade of 404s.
- **`recordVote` runs before the HTTP vote returns** —
  `frontend/src/app/components/center-panel/center-panel.component.ts`
  L249–251. Undo stack contains a vote that may have never reached
  the server; Cmd-Z then posts a "reversal" of nothing.
- **Binary media endpoints bypass the `activeContextInterceptor`** —
  `frontend/src/app/services/medias-api.service.ts` L41–60. Raw
  `HttpClient.get()` for `/api/medias/.../audio|video|image` lacks
  `X-Dataset-Id`; correct dataset is inferred by interceptor only for
  typed-client calls.

### Multi-process / settings

- **Per-user cache is process-local; concurrent worker writes lose
  updates** — `vtsearch/settings.py` `_user_caches` + `_synced_users`.
  Two gunicorn workers each sync independently, then race the
  per-user write.
- **`_save_user` holds `_settings_lock` across sync I/O** —
  `vtsearch/settings.py` ~L331–338. Slow source (NFS, webhook) blocks
  all other settings reads/writes globally.

### Error flow

- **Detector save's `os.replace` failure leaves in-memory state
  "saved"** — `vtsearch/detectors/store.py` L52–59. Next save doesn't
  realize prior persistence failed.
- **Partial label-import has no rollback** —
  `vtsearch/routes/labels/importers.py` L143. Failure mid-loop leaves
  a half-applied labelset; user has to manually figure out which
  ones landed.

---

## Medium — real bugs but lower frequency or non-corrupting impact

### State / concurrency

- Lock held during cross-lock callbacks in `toggle_vote` — state ↔
  progress lock ordering creates a narrow but real deadlock window.
- `combine_datasets.run_chunked` re-issues IDs starting at 1 on every
  call → cid collision when consumed twice.
- `importers/base.py` L407–410 — skipped records leave `next_id`
  unincremented, ID collisions on first-record-skip.

### Detector

- `populate_label_embeddings` cache not invalidated when a
  `region_box` is removed from an element; stale pooled vector
  continues to be used.
- `labelset_elements.resolve_current_dataset_cid` can return a
  colliding-MD5 cid in cross-dataset labelsets → clicks vote the
  wrong media.
- `restore_labels_from_detector` resolves by MD5 only on the second
  pass; dedup-collapsed cids can land votes on the wrong cid after
  reload.
- `safe_thresholds` is read at *training* time and baked into the
  detector JSON; per-user threshold preference cannot change after
  save.

### Datasets / loaders

- Thin-mode pickle loader treats `embedding: None` as present, then
  `np.array(None)` produces an object-dtype row.
- `loader_folder._has_override` doesn't warn when both `rel_path`
  and `file_name` override entries exist with different embeddings.
- `clipper_chain._run_clipper_step` assumes deterministic output count
  across calls — no validation.
- Stale media in `cli._score_medias_with_detectors` when some
  embeddings are `None` (zip truncates silently).
- `loader_pickle._build_pickle_full_media` has no null-check before
  `np.array(media_info["embedding"])`.

### Routes / API

- `learned_scores` in `/api/votes` can serialize as JSON
  `NaN`/`Infinity` if the MLP destabilizes — invalid JSON to strict
  clients.
- `diversity_tree_next_sample` references stale media IDs after
  `/api/dataset/clear`.

### Settings / sync

- Pending labelset sync stores `dataset_ctx = None` without checking;
  later `_run_pending_sync` triggers `AttributeError`.
- Sync to source on first read after `_synced_users` marker can still
  return stale local config if source changes silently.
- Legacy migration `_maybe_migrate_legacy_settings_locked` pops keys
  from in-memory cache before per-user disk write; per-user-write
  failure leaves cache and disk diverged.

### Embedding / training

- `_PeekUnpickler` doesn't override `FLOAT` / `SETITEMS` opcodes →
  falls back to slow real unpickle for older protocols.
- `embed_text_enriched` crashes (`np.mean` on empty) when text encoder
  fails and all wrappers return None.
- XCLIP single-frame video: `linspace(0, 0, 1)` + padding gives 8
  identical frames → degenerate embedding.
- Empty paragraph clip survives to dataset with `None` embedding;
  embedding-matrix builder later misbehaves.

### Auth / context

- `set_thread_user()` cleanup relies on every caller's `finally`; a
  future `ThreadPoolExecutor` reuse would leak user identity across
  requests.
- `setup_logging` re-running can leave duplicate handlers on
  non-root loggers.
- `CoreConfig.from_settings()` raises if a blueprint's module-level
  code runs before `vtsearch/shim/__init__.py` registers the
  builder.

### Frontend

- `LeftPanelComponent` lacks `OnDestroy` / `takeUntil` on init
  subscriptions; subscriptions leak across dataset switches.
- `labelset-state.service` `startPolling()` is not tied to
  `destroy$`; rapid switches leak polls.
- `progress-events.service` doesn't reconcile stale `task_id`s after
  backend restart.
- Audio waveform fetch's `catch {}` silently shows "Unable to load
  waveform" with no UI state propagation.
- `AudioContext` not cleaned up on rapid navigation → resource
  exhaustion.
- Autopilot phase transitions can oscillate
  (`hard → new → hard → new`) when smart/stable status flickers.
- `settings-importer-modal` auto-closes after 1.5s timeout regardless
  of operation duration.

### Security

- `sanitize_template_value` allows `...` (and worse — any non-`.` /
  `..` token of dots).
- `rglob_follow_symlinks` doesn't detect cycles → CPU/RAM DoS on
  circular link layouts.
- Email exporter validates "@" only; `"@example.com"` passes and
  fails at SMTP time.

### Eval / exporters

- `voting_iterations.py` reports F1/FPR with one good + one bad vote
  as if reliable.

---

## Low — latent / cosmetic / hypothetical

- `_run_pipeline` returns `None` silently in text mode (no "done"
  signal).
- Pipeline-vs-server logic in `app.py` L792 is masked by `sys.exit()`
  but breaks if the function is ever refactored to return.
- Frontend cross-pane settings: `getViewMode()` falls back to
  hard-coded defaults when the per-media-type entry isn't loaded.
- `get_settings_source_config` reads cache without re-checking sync
  state.
- Frontend `clip_box` exports: list-of-floats CSV-joined without
  quote-protection (edge case).
- Empty `Origin.params` not always dropped consistently across
  importers.
- `eval/metrics.py` returns F1=0 for empty test set without flagging
  the degenerate case.
- Logging of vote-related events truncates achievement traceback for
  UI display.
- `get_param` returns `""` for unknown keys; silently swallows
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
corresponding finding above is annotated **— shipped** with a short fix
summary, and is also recorded here.

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
- **C6 — zip-slip in HTTP archive importer (zip, tar, rar)**
  (2026-05-19). See the finding above for the full landed shape.
- **C8 — Bulk label paths skip achievement recording entirely**
  (2026-05-19). Centralised achievement credit inside `_state_lock`
  in `vtscore/state/votes.py`; also fixed the related High finding
  ("`record_vote()` called after releasing `_state_lock`") in the
  same change. See the C8 entry above for details.
- **C9 — Path-template post-resolution validation** (2026-05-19).
  `_resolve_filepath()` in both
  `vtscore/labels/sources/server_json_file/__init__.py` and
  `vtsearch/settings_io/sources/server_json_file/__init__.py` (plus the
  labels source's `resolve_filepath_for()` used by the rename flow) now
  ends in `validate_server_filepath(resolved, base_dir=
  get_file_access_base_dir())`, so a template like
  `../labels/{detector_name}.json` is rejected at `load` / `load_full` /
  `save` time before any file handle is opened. Regression tests live in
  `tests/io/test_sync_sources.py`.
- **C11 — `fill_labels_from_sort` sync-failure surfacing** (2026-05-19).
  See the finding above for the full landed shape.

### Still open

Every other finding (C5, C7, C10, C12 and the High / Medium /
Low tiers) remains as written. When the next fix lands, edit the
finding above with **— shipped** and add a line here. When every
critical and high is addressed, this doc can be retired into the
relevant subsystem docs (or deleted, per the `docs/plans/` lifecycle).

Specific open items called out by previously-shipped fixes:

- **Pattern #4 (media_revision counter)** is still unimplemented.
  The C4 stage-level invalidation closes the known clip/dedup hole,
  but any future mutation site that changes embeddings without
  changing the id set will reintroduce the same class of bug. A
  `media_revision` counter on `DatasetContext` bumped from every
  `medias` mutation (or a `MediasDict` subclass that does so
  transparently) would neutralise the whole category and let the
  matrix accessor compare a single int instead of two id lists.
