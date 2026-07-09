# Comprehensive interface audit — July 2026

**Status: eleven fix passes shipped; follow-ups #1 (registry multi-process
safety) and #2 (progress-modal analysis path) shipped; `MAX_UPLOAD_MB` default
cap fixed; 5 open follow-ups below.**

A full-codebase audit of interface boundaries (frontend ↔ API, route layer ↔
state system, app tier ↔ `vtscore`, IO, concurrency, frontend TS). Six parallel
passes produced ~40 findings; the confirmed, well-scoped ones were fixed with
regression tests across eleven passes. Open items were found and verified during
the audit but deferred as needing a design decision or a larger change.

## Open follow-ups

Ordered roughly by severity.

1. **Browse canvas gesture overlaps** (UX design decisions): wheel-zoom
   during an active drag-pan/marquee discards the zoom's cursor anchoring on
   the next mousemove and freezes painting for the 220 ms transition (gate
   the wheel while a drag is active, or make the pan math zoom-aware); the
   deferred single-click toggle resolves its captured screen coords against
   whatever transform exists 250 ms later, so a wheel notch / arrow-key glide
   inside the double-click window toggles the wrong bin (cancel or flush the
   pending toggle on view moves); the boundary-settle rAF loop can start while
   the zoom-transition loop still owns the canvas (both write
   `displayedTransform`; visible damage is a truncated transition).
2. **Root-zoom px mixing in panel-divider drags**
   (`browse-view.component.ts:onDividerMove`,
   `label-view/panel-resize.directive.ts`): visual-px cursor deltas applied
   as layout-px widths under `html { zoom: 1.1 }`, so the divider rides
   ~10% away from the pointer. Shared app-wide wart; fix both sites together
   (the eleventh pass fixed the same class of bug in the minimap).
3. **Pure devicePixelRatio change never re-runs canvas `resize()`**
   (browse-canvas + minimap): dragging the window to a different-density
   monitor leaves `dpr` (and the thumbnail-resolution tier) stale until the
   next CSS resize; rendering-quality only (hit-testing is CSS-px based).
   Needs a `matchMedia('(resolution: …)')` listener.
4. **`save_detector_labels` full-replace drops cross-dataset labels**
   (`routes/detectors/labels.py`): the route rebuilds the labelset from the
   *active dataset's* votes only, while `sync_labels_to_loaded_detector`
   deliberately merges cross-dataset entries — saving while dataset B is
   active discards the entries accumulated under dataset C. Decide whether
   the explicit save should merge like the sync does (probably yes, via
   `_merge_labelsets_across_datasets`) or full-replace is the intended
   "save exactly what I see" semantic.
5. ~~**`MAX_UPLOAD_MB` defaults to 0 = unlimited** (`vtscore/config.py`):
   staging uploads are unbounded by default; decide a sane default cap.~~
   **Fixed:** default changed to `2048` (2 GiB) — generous enough for typical
   media archives and dataset pickles, oversized requests get HTTP 413.
   `VTSEARCH_MAX_UPLOAD_MB=0` still opts back into unlimited for genuinely
   large-archive uploads.
6. **`AutoDetectProgressModalComponent` is dead code** (modal sweep note):
   referenced nowhere; delete it or wire it up.

## What shipped

Backend — state / settings / jobs:
- Dataset + detector registries made multi-process safe (was open follow-up #1):
  every mutation re-reads the manifest fresh under a cross-process `file_lock`
  and writes it back via `atomic_write_json` (pid+uuid temp name), so a
  concurrent CLI autodetect can no longer clobber the server's registrations. A
  library-tier `file_lock` was added to `vtscore.io`
  (`test_registry_multiprocess_safety.py` in both `tests_lib/datasets` and
  `tests_lib/detectors`).
- Detector-state wipe via the `"__request_missing__"` sentinel — a detector-only
  request wiped the session against the sentinel's empty medias
  (`vtscore/detectors/dataset_sync.py`).
- Settings-import setter-map restricted to real schema keys — an imported dict
  could no longer invoke process-level setters (`test_sync_sources.py`).
- JobManager pending-slot coalescing now requires the same requester context;
  a mismatch supersedes the parked pending with `cancelled` (`test_async_jobs.py`).
- Cross-detector labelset sync applies votes under `override_detector_context`
  (`test_sync_sources.py`).
- Detector-JSON RMW in `sync_labels_to_loaded_detector` serialised under a
  module lock.
- Registry load double-start fixed via atomic reserve-under-lock
  (`begin_load`/`end_load`, `begin_detector_load`/`end_detector_load`); contexts
  published only once fully populated (`test_registry_load_reservation.py`,
  `test_detector_load_reservation.py`).
- Learned-sort route freezes votes at request time so an intervening change
  can't cache under the wrong signature (`test_sorting.py`).
- Removed dead `update_cache_for_cid`, its re-export, whitelist entry, doc refs.
- Streaming exporters use `<name>.<pid>.<uuid>.tmp` + fsync instead of a fixed
  `<name>.tmp` (`test_exporters.py`).
- JobManager cancellation now stops *running* jobs via a thread-local binding
  (`bind_job_cancellation`/`check_job_cancelled`), polled at the MLP epoch and
  per-step retrain boundaries (`test_async_jobs.py`, `test_mlp_training.py`).
- Staging imports + eval jobs moved off global progress singletons onto per-task
  trackers / `AsyncJob` progress (`test_combine_datasets.py`, `test_sorting.py`).

Backend — ML / thresholds:
- `find_optimal_threshold` search-space fixed: can return `NO_GOOD_THRESHOLD`
  (predict-nothing) and handles tied scores (`test_find_optimal_threshold.py`).
- Stale calibration cache cleared on the <6-label path
  (`test_calibration_cache_invalidation.py`).
- NaN-threshold guard via `sigmoid_to_finite_scores`; DiversityTree
  `inertia=None` no longer crashes; assorted docstring drift fixes.
- All training entry points cross-calibrate at 4–5 labels (safe-thresholds off);
  abstain sentinel now tallied as a majority vote, not numerically averaged
  (`test_calibration_cache_invalidation.py`, `test_find_optimal_threshold.py`).
- Eval folds calibrate exactly as production (forced `hidden_dim`,
  `RandomState(42)`) (`test_eval_voting_iterations.py`).
- Inclusion-slide safe-blend divergence documented as intentional (skip blend on
  slides) (`test_inclusion_slide_safe_blend.py`).

Backend — IO / downloads:
- `_validate_archive` reads 4 magic bytes, not the whole archive.
- Partial media inserts rolled back before the fallback re-ingest
  (`test_media_sources.py`).
- HTTP-archive resolve cache keyed on a full-URL hash, and now busts on a remote
  signature (ETag/Last-Modified/size) mismatch via a sibling sidecar
  (`test_media_sources.py`).
- Demo downloads (Oxford Flowers, ROxford, UCSF) use `download_file_atomic`
  (`test_download_and_extract.py`); zip-slip check shares strict
  `_reject_traversal`.
- Region-matrix fallback row reads the patch-slot embedder, raising `ValueError`
  if absent, instead of silently mixing spaces
  (`test_region_score_pool.py`).
- Small hardening: fsync before publishing rename; `video2image` temp cleanup;
  per-path settings lock keyed on the resolved path.

Backend — routes (eleventh-pass sweep):
- `POST /api/find/cancel` now actually stops in-flight Find / find-label /
  auto-detect (loops poll at stage boundaries; 409 on cancel)
  (`test_find_cancel.py`).
- Unlocked detector-JSON RMW in four more routes now takes
  `_label_sync_write_lock` (`test_detector_json_lock.py`).
- `save_detector_labels` / `dataset/clear` now require the dataset/detector
  headers (sentinel wipe / no-op closed) (`test_detector_json_lock.py`,
  `test_datasets.py`).
- `fill_labels_from_sort` rolls votes back on a persistence failure
  (`test_error_recovery.py`).
- `example_sort_origin` validates path-like origin params
  (`test_path_validation.py`); `stage_file` streams the pickle peek instead of
  a whole-member RAM read.

Frontend:
- Failed vote POST rolls back optimistic/verified/region entries and re-reads
  server state (undo/redo error paths too).
- Dashboard loading-task `onComplete` callbacks now fire when the polling loop
  settles, gated per-task (`dashboard-loading-tasks.service.spec.ts`).
- Context-switch failures roll intent back; cancel-and-replace cancels only the
  superseded switch's own tasks.
- `utils/api-error.ts` (`apiErrorMessage`) reads both error envelopes; every
  inline handler converted. `uploadServerMediaFile` keeps `media_type` when
  `cropParams` is absent.
- Escape routes to the topmost open modal only (`modal.component.spec.ts`);
  dialog-host dismissal resolves the kind's own cancel value
  (`dialog-host.component.spec.ts`); image-viewer Escape suppresses under a
  modal backdrop (`image-viewer.component.spec.ts`).
- Virtual-scroll prefetch keyed to the viewport instance so it survives viewport
  recreation (`media-list.component.spec.ts`, `browse-bin-popup.zoneless.spec.ts`).
- Minimap mouse math corrects for `html { zoom: 1.1 }` (cursor offset, resize
  delta, backing store). Post-destroy rAF gated by a `destroyed` flag in
  browse-canvas.
- Progress-modal analysis path (was follow-up #2): the modal's `(closed)` now
  routes Escape/X/backdrop through `onCancel()` so every dismissal cancels the
  running eval job; the `trainAndScore` POST is guarded with
  `takeUntil(destroy$)` (no orphan poller when destroyed mid-flight); and the
  eval SSE progress watcher stops via a dedicated subject instead of
  `destroy$.next()`, so the backend's early `idle/Done` frame no longer tears
  down the result poller and hangs `analyzing`
  (`progress-modal.component.spec.ts`).

## Behaviour changes (backwards-compat notes)

- `find_optimal_threshold` can return `NO_GOOD_THRESHOLD` (2.0);
  `threshold_from_fold_orderings` abstains only under a strict majority of
  abstaining folds (no longer `calibrate_count`-dependent).
- Safe-thresholds **off**: all training entry points cross-calibrate at 4–5
  labels (was 0.5 on vote/labelset/origin-load paths); slides can move the line
  below 6. Safe-thresholds **on** below 6 unchanged (pure GMM).
- `_apply_settings` no longer applies non-schema keys.
- `http_archive` caches move to hash-keyed dirs (old dirs re-downloaded once) and
  re-download on a remote-signature mismatch (pre-existing caches trusted).
- `JobManager.start()` from a different requester supersedes the parked pending
  (pollers see `cancelled`); registry `.../load` while a load is in flight
  attaches to the shared task id (full `_regload_<id>` / `_detload_<id>`).
- Streaming JSON/CSV exports write to `<name>.<pid>.<uuid>.tmp`.
- `update_cache_for_cid` removed.
- Cancelling a running learned-sort / eval job now actually stops it (bar reports
  `"Cancelled"`).
- Staging returns a `task_id` and reports on the `loading-tasks` SSE channel;
  eval `train-and-score/result` reports the polled job's own progress.
- `GET /api/events` returns `503` + `Retry-After: 5` past `MAX_SSE_CONNECTIONS`
  (default `VTSEARCH_THREADS - 2`, override `VTSEARCH_SSE_MAX_CONNECTIONS`).
- Region-less media in a mixed patch/non-patch dataset reads its fallback row
  from the patch-slot embedder (raises if absent).
- `simulate_voting_iterations` / `run_voting_iterations_eval` calibrate with the
  production `hidden_dim` + `RandomState(42)`; small-label cost/fpr/fnr shift.
- Inclusion-slide safe-threshold divergence is now documented as intentional (no
  runtime change).
- `POST /api/find/cancel` returns 409 on a stopped in-flight request.
- `POST /api/detectors/<name>/labels` and `POST /api/dataset/clear` now require
  `X-Dataset-Id` (+ `X-Detector-Id` for the former); 400 without.
- `POST /api/example-sort-origin` rejects (400) origins escaping the user's
  confinement dir in multi-user mode.
- `fill-from-sort` rolls votes back on a 500.
- Frontend: Escape closes only the topmost modal; Escape/backdrop on a `prompt()`
  resolves `null`; Escape under a modal no longer clears the drawn region box.
