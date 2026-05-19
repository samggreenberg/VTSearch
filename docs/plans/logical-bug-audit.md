# Logical-Bug Audit — 2026-05

**Status:** Proposed — findings only, no fixes applied. Branch
`claude/audit-logical-errors-tcszV` is clean against `origin/dev`.

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

### C1. Download gate is never released if importer skips the `"embedding"` status

- **File:** `vtsearch/datasets/load_pipeline.py` ~L648–705
- **Bug:** `_LoadGateController` only swaps from `download_gate` →
  `embed_gate` when the importer's progress callback fires
  `status="embedding"`. A minimalist importer that completes without
  that status keeps the download gate held forever, blocking every
  subsequent dataset load.
- **Fix sketch:** Release whichever gate is held at task exit in a
  `finally` block, or add a guaranteed swap when the importer signals
  completion regardless of progress status.

### C2. JobManager never sets dataset/detector thread-local context

- **File:** `vtsearch/concurrency/async_jobs.py` ~L209–217
- **Bug:** `_run()` sets `set_thread_user(job.user)` but never calls
  `set_thread_dataset_context(get_context(job.dataset_id))` or the
  detector equivalent. Every learned-sort / eval / training job target
  that calls `get_active_context()` resolves to the empty fallback
  context — empty votes, missing media, silent miscompute.
- **Severity note:** Highest-impact context-propagation bug; flagged by
  two independent agents.

### C3. Dataset background load tasks don't set thread-local dataset context

- **File:** `vtsearch/datasets/load_pipeline.py` ~L822, ~L947, ~L1141
  (warmup, importer, staging spawn sites)
- **Bug:** `task()` closures set thread user but never call
  `set_thread_dataset_context(ctx)`. Any importer / clipper /
  diversity-tree code that resolves `get_active_context()` lands on
  `_empty_dataset_context` instead of the in-flight context. Mutations
  meant for the new dataset are silently lost.

### C4. Embedding-matrix cache is not invalidated after clip/dedup

- **File:** `vtsearch/datasets/load_pipeline.py`
  (`_collapse_duplicates_stage`, `_apply_clipper_stage`) +
  `vtsearch/embedding/matrix.py`
- **Bug:** After media items are removed/renumbered,
  `DatasetContext._emb_matrix_ids` is left stale. The next
  `train_and_score()` reads a matrix whose row order no longer matches
  the live `medias` dict — training vectors and scored results are
  mapped to the wrong media IDs. Silent ranking corruption.

### C5. `find_label` allows body field to override the request's dataset context

- **File:** `vtsearch/routes/detectors/scoring.py` ~L207–214, ~L336
- **Bug:** The endpoint mutates `g._dataset_context` from a `dataset_id`
  field in the request body, *after* `before_request` resolved it from
  headers. Combined with `replace_all=True` further down, a confused or
  malicious client can wipe one detector's votes while the UI thinks
  it's labeling a different one.

### C6. Zip-slip in HTTP archive importer (zip AND tar)

- **File:** `vtsearch/datasets/importers/http_archive/__init__.py`
  ~L70–89
- **Bug (zip):** `zf.extract(member, extract_dir)` is called *before*
  the post-extraction path-traversal check, so a malicious member is
  already written outside the directory by the time we reject it.
- **Bug (tar):** `tf.extract(member, extract_dir, filter="data")` only
  filters tar-specific attacks (symlink/hardlink), not traversal in
  member names like `../../etc/x`.
- **Fix sketch:** Validate every member name before extraction; use
  `extractall(filter=...)` with explicit name validation.

### C7. NaN/Infinity threshold leaks through safe-threshold blending

- **File:** `vtsearch/training/thresholds.py` ~L282–283, ~L351
- **Bug:** `calculate_cross_calibration_threshold()` returns
  `float("inf")` when a valid calibration split is impossible (n_cal
  too aggressive on tiny label sets). When `label_weight == 0.0` and
  `xcal_threshold == inf`, the blend `0.0 * inf + 1.0 * gmm` evaluates
  to `NaN`. NaN is then stored on `DetectorContext.threshold`, breaks
  all `score >= threshold` comparisons, and corrupts every result that
  touches that detector.

### C8. Bulk label paths skip achievement recording entirely

- **File:** `vtsearch/state/votes.py` — `apply_label_with_click_time()`
  (no `record_vote()` call) vs `toggle_vote()` (calls `record_vote()`).
- **Bug:** `/api/labels/fill-from-sort`, label importers, and bulk
  find-label apply votes without crediting achievements. `votes_cast`,
  `days_active`, `vote_streak` are inert for any user who uses
  search-then-bulk-label flows.

### C9. Path-template substitution missing post-resolution validation

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

### C11. `fill_labels_from_sort` silently swallows sync failures

- **File:** `vtsearch/routes/labels/vote.py` ~L259–301
- **Bug:** After applying labels, the endpoint calls
  `sync_labels_to_loaded_detector()` and `sync_to_labelset_source()`
  outside any try-block. The HTTP response is built before those
  calls' results are known; a failure logs but returns "success" —
  labels appear committed in the UI but never reach disk.

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
  detector. Cross-section with C8.
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
   classes.
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

This plan is **discovery-only** — no fixes have landed. When
individual fixes are pulled out of this list, mark them here as
shipped (date + commit / PR link) and edit the corresponding finding
above. When every critical and high is addressed, this doc can be
retired into the relevant subsystem docs (or deleted, per the
`docs/plans/` lifecycle).
