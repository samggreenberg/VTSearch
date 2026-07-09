# Comprehensive interface audit — July 2026

**Status: initial fix pass shipped; second follow-up pass shipped
(#2, #6, #10, #14 of the original open list); third follow-up pass shipped
(JobManager cancellation now stops running jobs); fourth follow-up pass
shipped (per-task progress isolation — staging imports + eval jobs); fifth
follow-up pass shipped (#5, #8 of the renumbered open list — ML threshold
correctness); sixth follow-up pass shipped (#1 of the then-current open
list — SSE connection cap); seventh follow-up pass shipped (http_archive
cache staleness); eighth follow-up pass shipped (#1 of the then-current
open list — region-matrix fallback space mixing); ninth follow-up pass
shipped (#1 of the then-current open list — eval harness calibration
fidelity); tenth follow-up pass shipped (#1 of the then-current open list —
inclusion-slide safe-blend semantics, resolved "skip blend on slides");
eleventh follow-up pass shipped (the "audit coverage gaps" item — all four
never-cleared areas audited, confirmed findings fixed); remaining open
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

### Ninth follow-up pass — eval harness calibration fidelity (drained from the open list)

- **Eval folds now calibrate exactly as production does** (was open #1).
  `eval/voting_iterations.py:simulate_voting_iterations` computed its per-step
  threshold via `calculate_cross_calibration_threshold` **without** forcing the
  full-data `hidden_dim` (so each calibration fold auto-sized to its own
  smaller train split) and threaded the **shared per-seed simulation RNG** into
  the fold splits.  Production's `_train_and_score_xy` / `train_and_threshold`
  do the opposite: they size `hidden_dim` from the full label count, force that
  width onto the folds (via `cross_calibration_threshold_cached(...,
  hidden_dim=...)`), and always calibrate with a fresh `RandomState(42)`.  The
  eval therefore measured a threshold the live detector never computes in the
  small-label regime — narrower fold nets, calibrated against splits that
  varied with the eval seed instead of the pinned production seed.  The per-step
  call now passes `hidden_dim=_auto_hidden_dim(n_labels)` and
  `rng=np.random.RandomState(42)` (a fresh one per step, mirroring how
  `cross_calibration_threshold_cached` mints one per call), and the final
  `train_model` is handed the same `hidden_dim`.  The eval seed still varies the
  *data* (which media are voted, in what order, and the held-out test split);
  only the calibration folds are now pinned, exactly as in production.  Tests in
  `tests_lib/detectors/test_eval_voting_iterations.py`
  (`TestProductionCalibrationFidelity`).

### Tenth follow-up pass — inclusion-slide safe-blend semantics (drained from the open list)

- **Inclusion slide drops the safe-threshold GMM blend** (was open #1).
  With safe-thresholds on and 6 ≤ n < 20 labels a *fresh* retrain stores the
  cross-calibration cutoff blended with a GMM cutoff
  (`calculate_safe_threshold`'s linear ramp), but the fold-ordering cache on
  the detector context holds only the raw cross-calibration orderings — not the
  GMM component — so an inclusion slide
  (`recompute_detector_thresholds_for_inclusion`) re-derived the *raw*
  cross-calibration aggregate and silently dropped the blend, changing the
  threshold semantics relative to a fresh retrain.  **Decision (user): skip
  the blend on slides.**  A slide is a cheap re-threshold over cached orderings,
  not a re-blend; making the recompute reproduce the blend would require it to
  carry cached GMM-threshold + label-count state, not worth it for the
  6..19-label window (below 6 the cache is cleared and the slide is a no-op at
  the inclusion-independent pure-GMM value; at ≥20 the blend is already pure
  cross-calibration so a slide matches a fresh retrain exactly).  No behaviour
  change from the pre-audit code — the divergence was already present — but it
  was accidental and undocumented, and is now deliberate.  Documented the
  intent on `recompute_detector_thresholds_for_inclusion` and at both blend
  sites in `vtscore/detectors/training.py`, and pinned the semantics with tests
  in `tests_lib/detectors/test_inclusion_slide_safe_blend.py`
  (`TestInclusionSlideDropsSafeBlend` — a slide re-derives the raw aggregate,
  not the blend; a below-floor slide is a no-op).

### Eleventh follow-up pass — audit coverage gaps cleared (was open #2)

The four areas the original audit never cleared (rate-limit casualties) were
swept by five parallel audit passes: browse-canvas/minimap coordinate math +
rAF lifecycles, modal/player component lifecycles, left/right-panel
virtualization, and the full route-blueprint sweep of
`routes/datasets|media|detectors|labels|processors|projection`.  Confirmed,
well-scoped findings were fixed with regression tests; the rest are recorded
under Open follow-ups.

Route-blueprint sweep fixes:

- **Find / auto-detect cancellation was a silent no-op** (high).
  `POST /api/find/cancel` set `find_progress`'s cancel flag and every scoring
  route cleared it on entry, but **no scoring loop ever read it** — the
  cancel docstring's "long-running loops poll the flag" was false, and an
  expensive Find always ran to completion.  The loops now poll at their
  stage boundaries (`multi_find`'s dataset loop, `_score_dataset`'s detector
  loop, `find_label`'s train/score/apply boundaries, and the entry of each
  `auto-detect` worker — *before* its catch-all `except Exception`, which
  would otherwise swallow the `CancelledError`) and the routes unwind with a
  409 + idle progress.  Tests in `tests/detectors/test_find_cancel.py`.
- **Unlocked detector-JSON read-modify-write in four routes** (medium).
  `sync_labels_to_loaded_detector` serialises its RMW of
  `detectors/<slug>.json` under `_label_sync_write_lock` (added in the
  initial pass), but `save_detector_labels`, `vote_detector_label`,
  `import_labels_into_detector`, and `find_corrections_to_detector` did the
  same file's RMW unlocked — a concurrent locked sync (any media vote) and a
  route write dropped each other's entries.  All four now take the same
  lock (each acquiring it before `_state_lock`, preserving the documented
  ordering).  Tests in `tests/detectors/test_detector_json_lock.py`.
- **`save_detector_labels` sentinel empty-labelset wipe** (medium).  A
  header-less `POST /api/detectors/<name>/labels` resolved the
  request-missing sentinels, whose `validated_vote_snapshot` reports
  `safe=True` over frozen-empty votes/medias — passing the 409 guard and
  full-replacing the named detector's labelset with an empty one.  Now
  guarded with `require_dataset_header` / `require_detector_header` (H34
  defence-in-depth pattern).  Test in the same file.
- **`fill_labels_from_sort` partial state on persistence failure**
  (low-medium).  The route applied votes then aborted 500 if the disk sync
  failed, leaving live in-memory votes that the next vote-triggered sync
  silently persisted — contradicting the error.  Now snapshots and rolls
  back the vote state on failure (the `apply_and_retrain` / H30 pattern).
  Test in `tests/api/test_error_recovery.py`.
- **Header-less `/api/dataset/clear` sentinel no-op** (low).  The route read
  the sentinel's truthy `"__request_missing__"` id and "cleared" a dataset
  that doesn't exist; the intended global-clear fallback was unreachable
  in-request (and would raise on the sentinel's frozen containers anyway).
  The header is now required (400 without it); no frontend caller exists.
  Test in `tests/datasets/test_datasets.py`.
- **`example_sort_origin` bypassed multi-user file confinement** (medium in
  multi-user).  The route built a media source from the request body's
  verbatim origin dict with none of `_load_from_origin`'s
  `validate_server_filepath` checks, so a `server_folder` origin could point
  at any server-readable directory (arbitrary-file embedding + existence
  disclosure outside `data/<user>/`).  Path-like origin params are now
  validated the same way.  Tests in `tests/api/test_path_validation.py`.
- **`stage_file` whole-member RAM read** (low).  The pickle peek read the
  entire decompressed `medias.pkl` into memory via `zf.read()` before
  peeking (zip-bomb amplification, compounded by `MAX_UPLOAD_MB=0` =
  unlimited by default); it now streams via `zf.open()`.

Frontend fixes (modal/player lifecycles, virtualization, canvas/minimap):

- **One Escape closed every modal in a stack** (high).  Every
  `ModalComponent` listens for Escape on `document` guarded only by its own
  `open()`, so nested flows (New Detector → crop modal, Settings → importer,
  any modal → dialog-host confirm) lost the whole stack — and the outer
  form's state — to a single keypress meant for the inner view.  A
  module-level open-modal stack now routes Escape to the topmost open modal
  only.  Specs in `modal.component.spec.ts`.
- **Escape/backdrop on a `prompt()` resolved `false`** (high).  Dialog-host's
  `onClosed()` hard-coded `resolve(false)` for every dialog kind; `prompt()`
  callers are typed `string | null` and crashed (`false.trim()` in
  find-view's promote flow, `false.split()` in the dashboard access-list
  editors).  Dismissal now resolves the dialog kind's own cancel value, and
  a second `show()` while one dialog is pending settles the superseded
  dialog as cancelled instead of stranding its promise forever.  Specs in
  `dialog-host.component.spec.ts`.
- **Image-viewer's window-level Escape acted under open modals**
  (medium-low).  Closing a modal with Esc also cleared the user's drawn
  region box underneath; the handler now suppresses on `.modal-backdrop`
  presence, matching `KeyboardService`.  Specs in
  `image-viewer.component.spec.ts`.
- **Virtual-scroll prefetch died on viewport recreation** (medium, two
  sites).  Both the media list (`scrollSubscribed` one-shot flag) and the
  browse-bin popup (`ngAfterViewInit`-only wiring) subscribed
  `scrolledIndexChange` once per component, but their CDK viewports live
  behind `@if` branches and are destroyed/recreated on dataset switches,
  virtualization-threshold crossings, and popup re-summons
  (singleton↔multi) — after which scroll-driven metadata/thumbnail
  hydration silently never fired again.  Subscriptions are now keyed to the
  viewport *instance* (the existing `observedViewportEl` re-observe
  pattern).  Specs in `media-list.component.spec.ts` and
  `browse-bin-popup.zoneless.spec.ts`.
- **Minimap mouse math ignored the app-wide `html { zoom: 1.1 }`** (high).
  `recenterFromEvent` mixed visual-px cursor offsets with layout-px map
  transforms, so every minimap click/drag recentred ~10% off target (the
  main canvas corrects for exactly this); the corner-resize drag grew 1.1×
  faster than the cursor; and the bitmap was undersampled 1.1×.  The cursor
  offset is now scaled by the rendered-size ratio, the resize delta divides
  out the root zoom, and the backing store bakes the zoom in.  (No spec:
  jsdom has no canvas 2D context; verified against the browse-canvas
  transform conventions.)
- **Post-destroy rAF scheduling in browse-canvas** (medium).  Late
  thumbnail `img.onload` callbacks called `requestRedraw()` after
  `ngOnDestroy` (which only cancels the *current* rAF handle), running
  `draw()` on the destroyed component — emitting on destroyed outputs,
  republishing a non-null viewport over teardown's `null`, re-arming the
  idle thumb-prefetch loop, and issuing spurious tile fetches against a
  newer browse view's projection.  A `destroyed` flag now gates
  `requestRedraw()` and `scheduleThumbPrefetch()`.

## Open follow-ups

Ordered roughly by severity.  Each was found and verified during the audit
but deferred as needing a design decision or a larger change.  (Original
open items #2, #6, #10, #14 were drained in the second follow-up pass, the
renumbered #2 "JobManager cancellation advisory" in the third pass, the
staging + eval progress isolation (renumbered #2, #3) in the fourth, the
re-renumbered #5/#8 "ML threshold correctness" items in the fifth, the
re-renumbered #1 "SSE connection cap" in the sixth, the re-renumbered #6
"http_archive cache staleness" in the seventh, the re-renumbered #1
"region-matrix fallback space mixing" in the eighth, the re-renumbered
#1 "eval harness calibration fidelity" in the ninth, the re-renumbered #1
"inclusion-slide safe-blend semantics" in the tenth, and the "audit
coverage gaps" item in the eleventh — whose sweep also *added* items 2-8
below (findings confirmed during that pass but deferred as design
decisions or larger changes) — see the follow-up-pass subsections under
What shipped.  The list below is renumbered again after those drains.)

1. **Dataset registry is not multi-process safe**
   (`vtscore/datasets/registry.py`): in-process lock plus a fixed `.tmp`
   name; a concurrent CLI autodetect run against the same data dir can
   silently erase the server's registrations.  Fine under the documented
   single-worker model; needs flock if that changes.
2. **Progress-modal analysis path is broken three ways**
   (`progress-modal.component.ts`, `runAnalysis()`): Escape/X/backdrop skip
   the eval-job cancel the in-body Cancel button performs; the
   `trainAndScore` subscribe has no `takeUntil`, so a destroy while the POST
   is in flight arms `pollEvalJob()` against an already-completed `destroy$`
   (RxJS `takeUntil` never fires on a pre-completed notifier → orphan
   500 ms poller until the job ends); and the backend emits the eval
   `idle/Done` SSE frame *inside* `_run` before the job flips to `done`, so
   the SSE watcher usually kills the result poller before it observes
   completion — `analyzing` hangs forever.  All currently *latent*: the only
   in-app instantiation passes `[useCachedHistory]="true"`, but
   `runAnalysis()` is the component default and is pinned by its spec.
3. **Browse canvas gesture overlaps** (found in the eleventh pass, deferred
   as UX design decisions): wheel-zoom during an active drag-pan/marquee
   discards the zoom's cursor anchoring on the next mousemove and freezes
   painting for the 220 ms transition (gate the wheel while a drag is
   active, or make the pan math zoom-aware); the deferred single-click
   toggle resolves its captured screen coords against whatever transform
   exists 250 ms later, so a wheel notch / arrow-key glide inside the
   double-click window toggles the wrong bin (cancel or flush the pending
   toggle on view moves); the boundary-settle rAF loop can start while the
   zoom-transition loop still owns the canvas (both write
   `displayedTransform`; visible damage is a truncated transition).
4. **Root-zoom px mixing in panel-divider drags**
   (`browse-view.component.ts:onDividerMove`,
   `label-view/panel-resize.directive.ts`): visual-px cursor deltas applied
   as layout-px widths under `html { zoom: 1.1 }`, so the divider rides
   ~10% away from the pointer.  Shared app-wide wart; fix both sites
   together (the eleventh pass fixed the same class of bug in the minimap).
5. **Pure devicePixelRatio change never re-runs canvas `resize()`**
   (browse-canvas + minimap): dragging the window to a different-density
   monitor leaves `dpr` (and the thumbnail-resolution tier) stale until the
   next CSS resize; rendering-quality only (hit-testing is CSS-px based).
   Needs a `matchMedia('(resolution: …)')` listener.
6. **`save_detector_labels` full-replace drops cross-dataset labels**
   (`routes/detectors/labels.py`): the route rebuilds the labelset from the
   *active dataset's* votes only, while `sync_labels_to_loaded_detector`
   deliberately merges cross-dataset entries — saving while dataset B is
   active discards the entries accumulated under dataset C.  Decide whether
   the explicit save should merge like the sync does (probably yes, via
   `_merge_labelsets_across_datasets`) or full-replace is the intended
   "save exactly what I see" semantic.
7. **`MAX_UPLOAD_MB` defaults to 0 = unlimited** (`vtscore/config.py`):
   staging uploads are unbounded by default; decide a sane default cap.
   (The eleventh pass removed the *decompressed-copy* amplification in
   `stage_file`, but the upload itself is still unbounded.)
8. **`AutoDetectProgressModalComponent` is dead code** (modal sweep note):
   referenced nowhere; delete it or wire it up.

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
- `simulate_voting_iterations` / `run_voting_iterations_eval` now calibrate
  each step's threshold with the production `hidden_dim` (full-label width,
  forced onto the folds) and a fixed `RandomState(42)` fold split, instead of
  auto-sizing the folds and threading the per-seed simulation RNG.  Reported
  `cost` / `fpr` / `fnr` values shift in the small-label regime because the
  eval now measures the exact pipeline the live detector runs; the eval seed
  still varies the data (votes, order, test split), only the calibration folds
  are pinned.
- **No runtime change**: the inclusion-slide safe-threshold divergence
  (`recompute_detector_thresholds_for_inclusion` re-derives the raw
  cross-calibration cutoff and does not reapply the GMM blend, so with
  safe-thresholds on and 6 ≤ n < 20 labels a slide yields a slightly
  different threshold than a fresh retrain) is now documented as intentional
  ("skip blend on slides") rather than accidental — the behaviour itself is
  unchanged from before the audit.
- `POST /api/find/cancel` now actually stops an in-flight `/api/find`,
  `/api/find-label`, or `/api/auto-detect`: the cancelled request unwinds
  with **409** `{"message": "Find cancelled"}` (progress resets to idle)
  instead of running to completion and returning 200.
- `POST /api/detectors/<name>/labels` now **requires** the `X-Dataset-Id`
  and `X-Detector-Id` headers (400 without them); previously a header-less
  request silently overwrote the named detector's labelset with an empty
  one.
- `POST /api/dataset/clear` now **requires** `X-Dataset-Id` (400 without
  it); previously a header-less clear silently no-oped against the
  request-missing sentinel.  No frontend caller sends a header-less clear.
- `POST /api/example-sort-origin` rejects (400) origins whose path-like
  params escape the user's confinement dir in multi-user mode; single-user
  mode is unchanged (unrestricted).
- A failed label-persistence pass in `POST /api/labels/fill-from-sort` now
  rolls the applied in-memory votes back before returning 500, instead of
  leaving them live to be silently persisted later.
- Frontend: Escape now closes only the topmost modal of a stack; Escape or
  a backdrop click on a `prompt()` dialog resolves `null` (previously
  `false`, which crashed callers); Escape while a modal is open no longer
  clears the image viewer's drawn region box underneath.
