# Comprehensive interface audit — July 2026

**Status: fix pass shipped; open follow-ups below.**

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

## Open follow-ups

Ordered roughly by severity.  Each was found and verified during the audit
but deferred as needing a design decision or a larger change.

1. **SSE `/api/events` pins a gthread worker thread per connection.**
   With `workers = 1, threads = 8`, eight open tabs starve the whole app;
   stalled requests plus a live heartbeat read as an app hang.  Needs a
   design decision: bound SSE connections, size the pool against them, or
   move the stream off the request-thread pool.
2. **Registry dataset/detector load: check-then-act double-start and
   half-loaded context visibility.**  Two concurrent `POST .../load` both
   pass the `is_loaded` check and spawn twin loads sharing a truncated
   `task_id`; `register_context(ctx)` runs before the loader populates
   `ctx.medias`, so concurrent requests can see a torn, partially-loaded
   dataset (or hit `RuntimeError: dictionary changed size during
   iteration`).  Same pattern for detector loads
   (`routes/detectors/registry.py`).
3. **JobManager cancellation is advisory only.**  No job target reads
   `AsyncJob.is_cancelled`, so cancelling a *running* learned-sort/eval
   job never stops the GIL-bound training (the cancelled-pending half was
   fixed in this pass).  Wiring `is_cancelled` into the training loop's
   epoch boundary would make cancel real.
4. **Staging imports publish through the global `dataset_progress`
   singleton** (`load_pipeline.py:_stage_importer_in_background`): two
   concurrent staging imports interleave one channel and the terminal
   `staging_result` is last-writer-wins, orphaning the loser's staged pkl.
   Needs per-task trackers like `_run_origin_load_in_background`.
5. **Eval progress is a global singleton keyed to no job**
   (`routes/eval.py`): overlapping evals decorrelate the progress bar from
   job identity.
6. **Learned-sort signature/data decorrelation** (`routes/sorting.py`):
   the background job copies the live vote dicts at *run* time but caches
   the result under the *request-time* signature; a vote change (or an
   `ensure_votes_match_active_dataset` rehydrate) in between poisons the
   `_last_done` cache for later identical-signature requests.
7. **Region-matrix fallback can mix embedding spaces** on a multi-embedder
   dataset where only some media carry `patch_regions`
   (`embedding/matrix.py:_build_region_arrays`): region rows are in the
   patch embedder's space, fallback rows in the primary's.  Needs either an
   ingest guarantee (all-or-nothing patch_regions per dataset) or space
   checking here.
8. **Eval harness fidelity** (`eval/voting_iterations.py`): fold models
   don't force the full-data `hidden_dim`, always calibrate below 6 labels
   (production uses 0.5), and thread the split RNG into calibration - so
   reported eval costs measure a pipeline production doesn't run in exactly
   the small-label regime the eval characterises.
9. **Inclusion slide drops the safe-threshold GMM blend**
   (`state/core.py:recompute_detector_thresholds_for_inclusion` vs
   `training.py`): with safe-thresholds on and 6 ≤ n < 20 labels, sliding
   inclusion to a value and back changes the threshold semantics relative
   to a fresh retrain.  Decide whether the blend applies on slides.
10. **`update_cache_for_cid` is a dead API with a latent bug**
    (`detectors/labelset_training.py`): it writes the media's *primary*
    vector into the detector-space label-embedding cache and stamps
    `region=None` even when the element has a region box.  No runtime
    callers today - fix or remove before wiring it up.
11. **Two training entry points disagree at 4-5 labels** with
    safe-thresholds off: `train_and_threshold` runs real cross-calibration,
    `_train_and_score_xy` hard-codes 0.5.  Same user state, different
    cutoff depending on route.
12. **Dataset registry is not multi-process safe**
    (`vtscore/datasets/registry.py`): in-process lock plus a fixed `.tmp`
    name; a concurrent CLI autodetect run against the same data dir can
    silently erase the server's registrations.  Fine under the documented
    single-worker model; needs flock if that changes.
13. **`http_archive` cache never invalidates** when the remote archive
    changes (fixed collision, not staleness); `extract_archive_cached`'s
    mtime/size keying is the model.
14. **Streaming exporters use a fixed `.tmp` sibling**, so two concurrent
    exports to the same path clobber each other's temp file.
15. **Threshold averaging with the abstain sentinel**: a fold returning
    `NO_GOOD_THRESHOLD` (2.0) now raises the fold mean, possibly above 1.0
    (= abstain overall).  This is the intended lean but the blend weight is
    crude; revisit if precision-biased small-label behaviour looks off.
16. **Audit coverage gaps** (rate-limit casualties, treat as *not cleared*
    rather than clean): browse-canvas/minimap coordinate math and rAF
    lifecycles, modal/player component lifecycle sweep, left/right-panel
    virtualization, and a full route-blueprint sweep of
    `routes/datasets|media|detectors|labels|processors|projection`.

## Behaviour changes shipped (backwards compatibility notes)

- `find_optimal_threshold` can now return `NO_GOOD_THRESHOLD` (2.0); fold
  means can exceed 1.0 where abstaining is strictly cheaper.
- `_apply_settings` no longer applies non-schema keys (previously reachable:
  `settings_path`, `user_data_dir_override`, `settings_source_config`,
  CLI knobs).
- `http_archive` resolve caches move to hash-keyed directory names; old
  cache dirs are orphaned and re-downloaded once.
- `JobManager.start()` from a different (user, dataset, detector) than the
  parked pending now supersedes it (pollers of the old job id see
  `cancelled`) instead of silently rebinding their job to the new request.
