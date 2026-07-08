# Comprehensive interface audit — July 2026

**Status: initial fix pass shipped; second follow-up pass shipped
(#2, #6, #10, #14 of the original open list); third follow-up pass shipped
(JobManager cancellation now stops running jobs); fourth follow-up pass
shipped (per-task progress isolation — staging imports + eval jobs); fifth
follow-up pass shipped (#5, #8 of the renumbered open list — ML threshold
correctness); sixth follow-up pass shipped (#1 of the then-current open
list — SSE connection cap); seventh follow-up pass shipped (http_archive
cache staleness); eighth follow-up pass shipped (#1 of the then-current
open list — region-matrix fallback space mixing); remaining open
follow-ups below.**

A full-codebase audit focused on interface boundaries: frontend ↔ backend
API contract, route layer ↔ state system, app tier ↔ `vtscore` library,
IO subsystems, concurrency, and frontend TypeScript logic.  Six parallel
audit passes produced ~40 findings; the confirmed, well-scoped ones were
fixed with regression tests in the same pass.  This document records what
shipped and what is deliberately deferred, so the next contributor picking
up any of these areas sees the known-weak spots.

## What shipped

### Backend — state / settings / jobs

- **Detector-state wipe via the request-missing sentinel** (high).
  `ensure_votes_match_active_dataset` guarded on `if not ds_ctx.dataset_id`,
  but inside a request with no `X-Dataset-Id` the dataset context is the
  request-missing sentinel whose id is the *truthy* string
  `"__request_missing__"`.  Any request naming a detector but no dataset
  wiped the detector's in-memory session (votes, label history, the whole
  Find-verification session) against the sentinel's frozen-empty medias.
  Fixed in `vtscore/detectors/dataset_sync.py`; regression test in
  `tests/detectors/test_detectors.py`.
- **Settings-import code injection surface** (high in multi-user).
  `_get_setter_map()` scanned the whole `vtsearch.settings` namespace for
  `set_*` callables, so an imported or synced settings dict could invoke
  process-level setters (`set_settings_path`, `set_user_data_dir_override`,
  CLI knobs) and repoint storage for the whole process.  The map is now
  restricted to real schema keys.  Tests in `tests/io/test_sync_sources.py`.
- **JobManager pending-slot coalescing across requesters** (high).
  The single pending slot coalesced *any* new `start()` into the parked
  job, so a poller of that job id could receive another user's / another
  (dataset, detector) pair's results.  Coalescing now requires the same
  requester context; anything else supersedes the parked pending with a
  visible `cancelled` status.  A pending job cancelled while parked is no
  longer promoted and run.  Tests in `tests_lib/integration/test_async_jobs.py`.
- **Cross-detector label application in labelset sync** (medium).
  `sync_from_labelset_source(detector_id)` wrote `detector_meta` to the
  *named* detector but applied the imported votes to the request's *active*
  detector.  The apply pass now runs under `override_detector_context(ctx)`.
  Test in `tests/io/test_sync_sources.py`.
- **Detector-JSON lost update** (low-medium). `sync_labels_to_loaded_detector`
  did an unlocked read→merge→write of the detector JSON; concurrent syncs
  from different dataset contexts dropped each other's cross-dataset
  entries.  Now serialised under a module lock.

### Backend — ML / thresholds

- **`find_optimal_threshold` search-space bugs** (medium).  The candidate
  set was only the observed scores, so "predict nothing" (threshold above
  max; FPR=0, FNR=1) was unreachable even when it minimised the weighted
  cost — under a precision-biased inclusion the calibrated cutoff came out
  far too permissive.  Tied scores also produced infeasible mid-tie cut
  positions.  Both fixed; the function can now return `NO_GOOD_THRESHOLD`
  when abstaining is strictly cheaper.  Tests in
  `tests_lib/sorting/test_find_optimal_threshold.py`.
- **Stale calibration cache below the ramp floor** (medium).  The <6-label
  training path left `det_ctx.calibration_cache` stale, so an inclusion
  slide re-thresholded against fold orderings computed for the old label
  set/model.  The cache is now cleared on that path.  Tests in
  `tests_lib/detectors/test_calibration_cache_invalidation.py`.
- **NaN-threshold guard** (low).  Fold calibration scores are now sanitised
  (`sigmoid_to_finite_scores`) so a destabilised fold model can't put NaN
  on `DetectorContext.threshold` when safe-thresholds is off.
- **DiversityTree `inertia=None` crash path** (low).  The k-means backend
  contract allows `inertia=None`; `_build_node` now keeps such labels as a
  fallback candidate instead of crashing on `labels == ci` with
  `best_labels is None`.
- Docstring drift fixes: GMM <2-scores fallback wording, `n_labels <= 6`
  pure-GMM boundary, stale `"embedding"`-key wording in `train_and_score`
  and the eval harness docs.

### Backend — IO / downloads

- **Whole-archive RAM read** (high).  `_validate_archive` read the entire
  (multi-GB) archive into memory to inspect 4 magic bytes; now reads 4 bytes.
- **Duplicate-media re-ingest** (medium).  `_ingest_via_source` inserted
  entries into the live `medias` dict as it went, then returned -1 mid-loop
  on a later failure; the fallback re-ingested the whole group, duplicating
  the already-inserted entries.  Partial inserts are now rolled back before
  the fallback signal.  Test in `tests/datasets/test_media_sources.py`.
- **HTTP-archive cache collision** (medium).  The resolve cache was keyed on
  the URL *basename*, so two archives sharing a final path segment silently
  served each other's bytes.  Now keyed on a hash of the full URL.
  *Note: existing caches under the old naming are orphaned and will be
  re-downloaded once.*
- **Demo-download cache poisoning** (medium).  Oxford Flowers labels,
  ROxford ground truth, and UCSF PDFs were downloaded straight to their
  final paths; a failed run left a truncated file that the `exists()` gate
  treated as cached forever.  Added `download_file_atomic` (temp + rename)
  and used it at all three sites.  Tests in
  `tests_lib/downloads/test_download_and_extract.py`.
- **Zip-slip prefix check** (low).  The downloader's inline
  `startswith` traversal check lacked a trailing separator; it now shares
  `archive.py`'s strict `_reject_traversal`.  Tests added.
- Small hardening: dataset-container writes fsync before the publishing
  rename; `video2image` no longer leaks its temp file on a failed write;
  the settings per-path lock is keyed on the resolved path for the file's
  whole lifetime.

### Frontend

- **Stuck optimistic vote on failed POST** (high).  A failed vote POST left
  `pendingOptimistic`/`pendingVerified`/`pendingRegionBoxes` entries that
  re-imposed the phantom vote over every `/api/votes` poll forever — silent
  label loss with a lying UI.  The error path now rolls the entries back and
  re-reads server state; undo/redo error paths fixed the same way.
- **Dropped `onComplete` in dashboard loading-task polling** (medium-high).
  `startProgressPolling`/`startDetectorProgressPolling` discarded the
  completion callback when the polling loop was already active (common:
  the loop auto-starts on any non-idle SSE snapshot), so
  `loadDataset`/`loadDetector` never promoted the pair via `setActivePair`.
  Callbacks are now registered on the service and fired when the loop
  settles, gated per-task so an unrelated task's failure doesn't suppress
  promotion.  New spec: `dashboard-loading-tasks.service.spec.ts`.
- **Context-switch failure handling** (medium).  A failed load kick-off or
  an errored background load task was treated as success and promoted the
  unloaded pair to active (re-opening the H25 409 cascade).  Failed
  switches now roll intent back and complete without emitting; the route
  guard maps the empty completion to a clean navigation denial via
  `defaultIfEmpty` instead of the router's `EmptyError`.  Cancel-and-replace
  now cancels only the superseded switch's own loading tasks instead of
  every non-idle task (it used to kill unrelated imports).
- **Inline error-message parsing** (low, systematic).  ~15 inline handlers
  read only `err.error?.error`, but many routes emit the flask_smorest
  `{message}` envelope; they degraded to generic fallbacks.  Added
  `utils/api-error.ts` (`apiErrorMessage`) reading both envelopes and
  converted every site.
- `uploadServerMediaFile` no longer drops `media_type` when `cropParams`
  is absent.

### Second follow-up pass (drained from the open list)

- **Registry dataset/detector load double-start + half-loaded visibility**
  (was open #2).  Added an atomic reserve-under-lock step
  (`begin_load`/`end_load` in `vtscore/datasets/registry.py`,
  `begin_detector_load`/`end_detector_load` in
  `vtscore/detectors/registry.py`): only one loader runs per id, so two
  concurrent `.../load` requests no longer spawn twins.  The deterministic
  task id is now intentional poll-coalescing (a second caller attaches to
  the in-flight load).  `register_context` / `register_detector_context`
  moved to *after* the context is fully populated, so no torn,
  empty-then-growing context is ever published and the loaded flag is never
  set ahead of the context store.  Tests in
  `tests_lib/datasets/test_registry_load_reservation.py` and
  `tests_lib/detectors/test_detector_load_reservation.py`.
- **Learned-sort signature/data decorrelation** (was open #6).  The route
  now freezes the votes at request time (`good_snapshot`/`bad_snapshot`),
  exactly as region boxes already were, and passes those snapshots to both
  the signature builder and the background job.  Previously the job copied
  the live vote proxy at *run* time, so an intervening vote change (or an
  `ensure_votes_match_active_dataset` rehydrate) cached a result under the
  wrong signature.  Test in `tests/sorting/test_sorting.py`.
- **`update_cache_for_cid` dead API removed** (was open #10).  It wrote the
  media's *primary* vector into the detector-space label-embedding cache and
  stamped `region=None`; no runtime callers.  Removed the function, its
  `labelset_ops` re-export, the vulture-whitelist entry, and two stale doc
  references.
- **Streaming exporter temp-file collision** (was open #14).  Both
  `server_json_file` and `server_csv_file` used a fixed `<name>.tmp`
  sibling; two concurrent exports to the same path clobbered each other.
  Switched to the house-style `<name>.<pid>.<uuid>.tmp` and added an fsync
  before the atomic replace.  Tests in `tests/io/test_exporters.py`.

### Third follow-up pass (drained from the open list)

- **JobManager cancellation now stops running jobs** (was renumbered open
  #2).  Cancelling a *running* learned-sort / eval job previously only set
  `AsyncJob.is_cancelled`, which no job target read, so the GIL-bound
  training kept running to completion (the cancelled-*pending* half was
  already fixed in the initial pass; both cancel routes' docstrings already
  *claimed* the loop "polls it cooperatively", but nothing did).  Added a
  thread-local job binding in `vtscore/concurrency/async_jobs.py`
  (`bind_job_cancellation` / `check_job_cancelled`, entered for the
  worker-thread lifetime in `JobManager._run`).  The deep compute loops now
  poll it at their natural boundaries — the MLP epoch boundary in
  `vtscore/training/mlp.py:train_model` and the per-step retrain boundary in
  `vtscore/detectors/labeling_progress.py:_ensure_cache`.  A poll raises the
  existing `CancelledError`, which unwinds the training/eval stack and is
  caught in `_run_inner` to mark the job `cancelled` (never caching its
  half-built result via `_last_done`) and hand off to any parked pending.
  The check is a no-op outside a bound job, so the synchronous Find flow,
  tests, and CLI that share `train_model` are unaffected.  The eval `_run`
  now clears the progress bar to `"Cancelled"` rather than `"Error"`.
  Tests in `tests_lib/integration/test_async_jobs.py`
  (`TestCheckJobCancelledPrimitive`, `TestRunningJobCancellation`) and
  `tests_lib/detectors/test_mlp_training.py` (`TestJobCancellation`).

### Fourth follow-up pass (per-task progress isolation)

- **Staging imports on the global `dataset_progress` singleton** (was open
  #3, renumbered #2 after the third pass).  `_stage_importer_in_background`
  reported through the single global `dataset_progress` tracker, so two
  concurrent staging imports interleaved one channel and the terminal
  `staging_result` was last-writer-wins — orphaning the loser's staged pkl.
  Staging now runs through a dedicated per-task `ProgressTracker` created
  via `loading_tasks` (an `extra_fields` hook on
  `LoadingTasksTracker.create_task` carries the `staging_result` key), keyed
  by a returned `task_id`; the importer's own progress and the embed pass
  route into that tracker via `set_thread_progress`.  The `stage-import` /
  `stage-demo` routes now return the `task_id` so a poller can pick up its
  own result off the `loading-tasks` SSE channel.  Test in
  `tests/datasets/test_combine_datasets.py::TestStagingPerTaskIsolation`.
- **Eval progress decorrelated from job identity** (was open #4, renumbered
  #3 after the third pass).  The eval train-and-score route wrote progress
  to the global `eval_progress` singleton and the `/result` poll read it
  back, so overlapping evals (one running, one parked pending behind the
  single-runner `eval_jobs` manager) reported each other's numbers.
  Progress now lives on the `AsyncJob` (`job.update_progress`) and the poll
  reads `job.current` / `job.total`; the singleton is written only from
  inside the actually-running job's `_run` (not from the request handler for
  a possibly-pending job), so the live `eval` SSE bar reflects the running
  job.  Test in
  `tests/sorting/test_sorting.py::TestEvalTrainAndScoreAsync::test_result_reports_job_progress_not_singleton`.

### Fifth follow-up pass — ML threshold correctness (drained from the open list)

- **Training entry points now agree at 4-5 labels** (was renumbered open
  #5).  With safe-thresholds off, `train_and_threshold` (Find / detector-
  load) ran real cross-calibration below 6 labels while `_train_and_score_xy`
  (vote-driven Train / labelset) and `train_detector_from_origins`
  (detector-load-from-origins) hard-coded 0.5 — the same user state produced
  a different cutoff depending on which route trained.  Unified on real
  cross-calibration: the `< 6 → 0.5` short-circuit is now gated on
  `safe_thresholds` (still skipped only when the pure-GMM blend would
  discard the result), so every route cross-calibrates below 6 when
  safe-thresholds is off, and an inclusion slide can move the line below 6
  too.  `train_and_threshold`'s safe-on-<6 branch now also clears the fold
  cache (parity with `_train_and_score_xy`), closing a stale-cache read on
  that path.  Tests in `tests_lib/detectors/test_calibration_cache_invalidation.py`
  and `tests/sorting/test_sorting.py`.
- **Abstain sentinel is a vote, not a number** (was renumbered open #8).
  `threshold_from_fold_orderings` numerically averaged the `NO_GOOD_THRESHOLD`
  (2.0) sentinel, so a single abstaining fold dragged the mean above the
  sigmoid range (forcing overall abstain at the default `calibrate_count=2`,
  yet often *not* at 3+ folds — a fold-count-dependent artifact that stored an
  ill-defined ~1.3 as "the threshold").  A fold that abstains is now tallied
  as a vote: the ensemble abstains only under a **strict majority** of
  abstaining folds and otherwise means the folds that produced a real cut.
  Tests in `tests_lib/sorting/test_find_optimal_threshold.py`.

### Sixth follow-up pass — SSE connection cap (drained from the open list)

- **SSE `/api/events` no longer starves the gthread pool** (was open #1).
  Each open connection pins a worker thread for its lifetime (the generator
  blocks in `queue.get()` between updates), so with the documented
  `workers = 1, threads = 8` deployment, enough open tabs could exhaust the
  pool and make ordinary requests stall while the stream's own heartbeat
  kept the connection looking alive. Chose the "bound connections" option
  from the three named in the original finding (bound / resize / move off
  the pool): a hard cap, `MAX_SSE_CONNECTIONS` in
  `vtscore/concurrency/events.py`, derived from `VTSEARCH_THREADS` minus a
  fixed reserve of 2 (override directly with
  `VTSEARCH_SSE_MAX_CONNECTIONS`). `acquire_sse_slot()` /
  `release_sse_slot()` gate `/api/events` in `vtsearch/routes/events.py`;
  once saturated, new connects get an immediate `503` with `Retry-After: 5`
  instead of starting a stream, so the rejection itself never pins a
  thread. The frontend's `EventSource` already schedules a manual
  reconnect when the server closes the connection non-2xx (see
  `progress-events.service.ts`'s `onerror`/`readyState === CLOSED` path),
  so no frontend change was needed. Tests in
  `tests/api/test_events_sse.py` (`TestSseConnectionCapPrimitive`,
  `TestSseConnectionCapRoute`).

### Seventh follow-up pass (drained from the open list)

- **`http_archive` cache staleness** (was renumbered open #6).
  `HttpArchiveSource`'s resolve cache keyed only on the URL, so once an
  extraction was published under that key it was served forever, even after
  the remote archive changed. Added `fetch_remote_signature()` in
  `vtscore/datasets/downloader/core.py` — a single-byte ranged GET (more
  universally honoured than HEAD by the flaky CDNs these archives live on)
  that reads only `ETag` / `Last-Modified` / size from the response headers.
  `_ensure_extracted()` now records that signature in a sidecar file next to
  the cached extraction (kept as a *sibling*, never inside the extraction
  tree, so it can't surface as a bogus media item to an extensionless
  `list_items(None)`); a later access whose freshly-probed signature
  disagrees with the recorded one busts the cache and re-downloads. A cache
  with no recorded signature (predates this check, or its own probe failed)
  is trusted as-is, and a probe failure (offline, flaky CDN) fails open onto
  the existing cache rather than blocking. Tests in
  `tests/datasets/test_media_sources.py`
  (`test_stale_cache_invalidated_on_signature_mismatch`,
  `test_signature_probe_failure_trusts_existing_cache`).

### Eighth follow-up pass (drained from the open list)

- **Region-matrix fallback space mixing** (was open #1).
  `embedding/matrix.py:_build_region_arrays` flattens every media's
  `patch_regions` into one `(R, D)` matrix, giving a region-less media a
  single fallback row so the segmented max-pool never sees an empty group.
  That fallback unconditionally read the media's *primary* vector - fine
  when every media in the dataset carries `patch_regions` (the only
  reachable case before the v3 trio-embedder work), but on a dataset that
  mixes patch-capable and patch-less media (a combined dataset, or a media
  type the patch embedder can't process) the primary can be a different
  embedder than the one that produced the region rows, silently stacking
  vectors from two different embedding spaces into one matrix and scoring
  the region-less media's dot-products meaninglessly.  Added
  `_patch_embedder_for_region_snap`, which derives the patch-slot embedder
  name from a media that actually carries `patch_regions` (role-typing its
  bound embedder names via `derive_binding_from_names`, not just reading the
  *first* media in the dict, which may itself be the region-less one and
  never had a patch-embedder vector to role-type from); the fallback row now
  reads that embedder's vector instead of the primary.  A region-less media
  with no vector at all under the patch embedder now raises `ValueError`
  naming the media and embedder (matching this module's existing
  loud-failure philosophy for missing embeddings) rather than silently
  substituting the wrong space.  Single-embedder datasets are unaffected:
  the derived patch embedder there always equals the primary, so the
  fallback read is byte-for-byte the same as before.  Tests in
  `tests_lib/detectors/test_region_score_pool.py`
  (`TestRegionMatrixFallbackSpace`).

## Open follow-ups

Ordered roughly by severity.  Each was found and verified during the audit
but deferred as needing a design decision or a larger change.  (Original
open items #2, #6, #10, #14 were drained in the second follow-up pass, the
renumbered #2 "JobManager cancellation advisory" in the third pass, the
staging + eval progress isolation (renumbered #2, #3) in the fourth, the
re-renumbered #5/#8 "ML threshold correctness" items in the fifth, the
re-renumbered #1 "SSE connection cap" in the sixth, the re-renumbered #6
"http_archive cache staleness" in the seventh, and the re-renumbered #1
"region-matrix fallback space mixing" in the eighth — see the
follow-up-pass subsections under What shipped.  The list below is
renumbered again after those drains.)

1. **Eval harness fidelity** (`eval/voting_iterations.py`): fold models
   don't force the full-data `hidden_dim` and don't thread the split RNG
   into calibration - so reported eval costs measure a pipeline that differs
   from production in the small-label regime.  (The "production uses 0.5
   below 6 labels" mismatch is now narrower: with safe-thresholds off,
   production cross-calibrates below 6 too — see Fifth follow-up pass.)
2. **Inclusion slide drops the safe-threshold GMM blend**
   (`state/core.py:recompute_detector_thresholds_for_inclusion` vs
   `training.py`): with safe-thresholds on and 6 ≤ n < 20 labels, sliding
   inclusion to a value and back changes the threshold semantics relative
   to a fresh retrain.  Decide whether the blend applies on slides.
3. **Dataset registry is not multi-process safe**
   (`vtscore/datasets/registry.py`): in-process lock plus a fixed `.tmp`
   name; a concurrent CLI autodetect run against the same data dir can
   silently erase the server's registrations.  Fine under the documented
   single-worker model; needs flock if that changes.
4. **Audit coverage gaps** (rate-limit casualties, treat as *not cleared*
   rather than clean): browse-canvas/minimap coordinate math and rAF
   lifecycles, modal/player component lifecycle sweep, left/right-panel
   virtualization, and a full route-blueprint sweep of
   `routes/datasets|media|detectors|labels|processors|projection`.

## Behaviour changes shipped (backwards compatibility notes)

- `find_optimal_threshold` can now return `NO_GOOD_THRESHOLD` (2.0).
- `threshold_from_fold_orderings` no longer numerically averages the abstain
  sentinel: the fold ensemble abstains only under a strict majority of
  abstaining folds and otherwise means the non-abstaining folds, so the
  stored threshold never lands at an out-of-range ~1.3 and the abstain
  outcome no longer depends on `calibrate_count`.
- With safe-thresholds **off**, all training entry points cross-calibrate at
  4-5 labels (previously the vote/labelset and origin-load paths returned
  0.5).  The Find/Train cutoff for the same small-label state now matches
  regardless of route, and an inclusion slide can move the threshold below 6
  labels too.  Safe-thresholds **on** below 6 is unchanged (pure GMM).
- `_apply_settings` no longer applies non-schema keys (previously reachable:
  `settings_path`, `user_data_dir_override`, `settings_source_config`,
  CLI knobs).
- `http_archive` resolve caches move to hash-keyed directory names; old
  cache dirs are orphaned and re-downloaded once.
- `JobManager.start()` from a different (user, dataset, detector) than the
  parked pending now supersedes it (pollers of the old job id see
  `cancelled`) instead of silently rebinding their job to the new request.
- Registry `.../load` while a load of the same id is already in flight now
  returns `{"message": "… load already in progress"}` with the shared task
  id (attach + poll) instead of spawning a second loader.  The load task id
  is the full id (`_regload_<id>` / `_detload_<id>`) rather than an 8-char
  prefix.  A registered dataset/detector context is published only once
  fully loaded, so it is no longer briefly visible half-populated.
- Streaming `server_json_file` / `server_csv_file` exports now write to a
  `<name>.<pid>.<uuid>.tmp` temp sibling instead of a fixed `<name>.tmp`.
- `update_cache_for_cid` (unused) was removed from
  `vtscore.detectors.labelset_training` and the `labelset_ops` re-export.
- Cancelling a *running* learned-sort / eval job now actually stops it
  (transitions to `cancelled` within one epoch / eval step and never caches
  the partial result) instead of running to completion with the cancel
  ignored.  The eval progress bar reports `"Cancelled"` on a running-job
  cancel rather than `"Error"`.
- Staging (`POST /api/dataset/stage-import/<importer>`, `stage-demo/<name>`)
  now returns a `task_id` and reports progress on the per-task
  `loading-tasks` SSE channel (with its `staging_result`) instead of the
  global `dataset` channel / `dataset_progress` singleton.
  `_stage_importer_in_background` returns the task id.
- Eval `GET /api/eval/train-and-score/result` reports the polled job's own
  `current` / `total` (read from the `AsyncJob`) rather than the shared
  `eval_progress` singleton, so overlapping evals no longer read each
  other's progress.
- `GET /api/events` now returns `503` (with `Retry-After: 5`) once
  `MAX_SSE_CONNECTIONS` concurrent streams are already open, instead of
  accepting an unbounded number of connections that could starve the
  gthread pool. Cap defaults to `VTSEARCH_THREADS - 2`; override with
  `VTSEARCH_SSE_MAX_CONNECTIONS`.
- `HttpArchiveSource`'s resolve cache now checks a recorded remote signature
  (ETag/Last-Modified/size) on each access to a cached extraction and
  re-downloads on a mismatch, instead of trusting the cache forever once
  published. A cache built before this change (no recorded signature) or a
  failed probe still trusts the existing cache, so this is additive: no
  previously-cached extraction is invalidated by upgrading alone.
- A region-less media in a mixed patch/non-patch dataset now has its
  region-matrix fallback row read from the dataset's patch-slot embedder
  instead of its primary embedder, and raises `ValueError` if it has no
  vector under the patch embedder at all.  No effect on any dataset where
  every media shares one embedder (the only case reachable before the v3
  trio-embedder work).
