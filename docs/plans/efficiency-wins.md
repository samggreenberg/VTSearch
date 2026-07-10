# Efficiency Wins

**What this is:** The result of a systemic efficiency review (2026-07-10) of the
places a user actually waits: dataset save/load/import, demo downloads, the
vote→retrain loop, text/example sort, and the frontend voting loop. Each item
below is an **independently shippable** fix with file/line pointers, a concrete
approach, and the gotchas a fresh session needs. Items are named (stable
labels, never renumbered) and grouped so parallel efforts don't collide — the
"Files touched" line is the conflict map: two items sharing no files can ship
in parallel branches safely. Items are separated by `<!-- item-sep -->`
sentinels; when you ship a slice, delete only your item's own lines and leave
the sentinels intact (see the plan-file policy in `CLAUDE.md` for why — a
never-deleted line between items keeps two parallel deletions from conflicting).

**Relationship to [`scalability.md`](scalability.md):** that plan tracks
*scale-limit* work (what breaks at 100k–10M items) under stable `S#` IDs. This
plan tracks *latency at current target scale* (GUI Train ~50k, GUI Find ~250k).
One overlap: serial label resolution is already tracked there as **S11/S20** —
it is *not* duplicated here; when implementing S11/S20, note that the whole-file
common case should route through the existing batched `embed_media_bulk` path
(`vtscore/media/image/_image_bulk.py`), not just a thread pool around
single-item `embed_media` calls: the batched forward is the bigger win on GPU.

**Verified non-findings (do not "fix" these):** the width/height `Image.open`
in `vtscore/media/image/media_type.py:148` is a lazy header parse, not a second
full decode; `_cuda_can_run` in `vtscore/config.py` memoizes its smoke test via
`_cuda_runnable`. Both were flagged by an initial sweep and confirmed fine.

**Already well-optimized (don't re-do):** embedding-matrix revision cache with
three-phase locking (`vtscore/embedding/matrix.py`), `np.maximum.reduceat`
segmented max-pool (`vtscore/detectors/training.py:_segmented_max_pool`),
vectorized threshold search + GMM subsampling (`vtscore/training/thresholds.py`),
calibration-ordering cache across inclusion slides, bulk image embedding,
streaming/resuming downloads, ZIP_STORED containers, SSE progress (no polling),
CDK virtual scroll + OnPush + lazy thumbnails in the left grid, the browse
canvas renderer, LSH near-dup detection, async jobs with signature caches.

---

## Tier 1 — small, low-risk, immediately felt

<!-- item-sep -->

## Tier 2 — high impact, needs care

<!-- item-sep -->

- **Parallel folder ingest** — full-mode folder import processes files
  strictly serially (`vtscore/datasets/loader_folder.py:454-470` inside
  `_build_per_file_media`, driven by plain `for` loops in
  `load_dataset_from_folder` ~line 567 and `load_dataset_from_folder_chunked`
  ~line 699). Per file: disk read + `mt.load_media_data` (for images: PIL
  decode + LANCZOS thumbnail + JPEG re-encode in `make_image_thumbnail`; for
  audio: decode + waveform PNG). PIL decode/resize/encode release the GIL, so
  this parallelizes near-linearly — and the codebase already proves the
  pattern twice: `vtscore/state/near_dupes.py:300-313` (ThreadPoolExecutor,
  with a comment saying exactly this) and
  `vtscore/datasets/importers/recaller/__init__.py:254`
  (`ThreadPoolExecutor(max_workers=16)`).
  **Fix:** inside each chunk, submit `_build_per_file_media(...)` for the
  chunk's files to a `ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))`,
  then collect results **in submission order** (`executor.map`, or a list of
  futures iterated in order — not `as_completed`) so media IDs and origin
  ordering stay deterministic. Assign `media_id` before submission (it's an
  input, not derived inside). Gate on a small-N threshold (e.g. skip the pool
  under ~64 files, mirroring near_dupes' `_THREAD_MIN_IMAGES`) so tiny imports
  don't pay pool startup. Progress: call `on_progress` from the collection
  loop (main thread), not from workers, to keep ProgressTracker single-writer.
  **Gotchas:** (1) cancellation — the serial loop's `check_cancelled()` calls
  must survive: check in the collection loop each iteration, and on
  `CancelledError` call `executor.shutdown(cancel_futures=True)`. (2) Memory —
  a chunk's worth of `media_bytes` in flight is the same as today (the chunk
  already bounds it); don't widen the pool beyond the chunk. (3) Exceptions in
  a worker must propagate with the failing file's path in the message, same as
  today. (4) `thin=True` mode does no decode — keep it on the serial path or
  let it flow through (harmless either way).
  **Files touched:** `vtscore/datasets/loader_folder.py` only.
  Tests: `./run-tests.sh datasets io` — plus add one test asserting IDs/order
  match the serial path on a fixture folder.
  Impact: high (bulk image/audio import ≈ cores× faster on the decode phase).
  Complexity: medium.

<!-- item-sep -->

- **Vectorized region cosine sort** — text/example sort on a patch dataset
  (DINOv2/DINOv3/EUPE) loops per media and per region in Python:
  `score_against_query` (`vtscore/training/region_similarity.py:60-70`) does
  `np.asarray(r.vec)` + a scalar dot **per region**, called per media from
  `cosine_sort_with_boxes` (~line 139). At Find scale (~250k media × ~23
  regions) that's millions of interpreter round-trips on a user-facing sort.
  The MLP scoring path already solved the identical problem:
  `get_region_matrix_for_snap` (`vtscore/embedding/matrix.py:285`) caches a
  flattened `(R, D)` region matrix + a per-row media index, and
  `_segmented_max_pool` (`vtscore/detectors/training.py:475`) reduces per-media
  maxima with `np.searchsorted` + `np.maximum.reduceat`.
  **Fix:** the argmax machinery already exists complete — `_segmented_max_pool`
  returns both the per-media max **and** the winning region index with the
  same first-wins tie-break as today's scalar loop (`>=` + first-candidate
  `searchsorted`; see its docstring). So the region branch of
  `cosine_sort_with_boxes` becomes a mirror of `_score_all_media`
  (`vtscore/detectors/training.py:431-472`) with the MLP forward replaced by
  a matvec: call `get_region_matrix_for_snap(snap)` for
  `(all_ids, region_matrix, media_index_per_row, region_index_per_row)`
  (check its exact return signature), compute
  `sims = region_matrix.numpy() @ query_vec` (or torch matvec — match
  whatever dtype/layout `_score_all_media` feeds the model), then
  `scores, best_region = _segmented_max_pool(sims, media_index_per_row,
  region_index_per_row, len(all_ids))` and format entries with
  `{"id", "similarity": round(s, 4), "best_region": list(regions[bri].box)}`
  exactly like `_format_results` does (only emit `best_region` when the media
  has `patch_regions` and the index is valid). Import direction: `training.py`
  is in `vtscore/detectors/` and `region_similarity.py` in
  `vtscore/training/` — if importing `_segmented_max_pool` from
  `detectors.training` creates an import cycle or layering violation, move
  the helper into `vtscore/embedding/matrix.py` or a small shared module and
  have both call sites import it.
  **Gotchas:** `_score_all_media` already handles mixed snapshots (media with
  and without regions) via the same matrix helpers — copy its dispatch rather
  than inventing one. Preserve `round(sim, 4)`, the `sims` return list in
  original snapshot order (GMM thresholding consumes it), and the
  `"best_region"` shape. `tests/sorting/` has region-sort tests — run
  `./run-tests.sh sorting detectors`.
  **Files touched:** `vtscore/training/region_similarity.py`, possibly a shared
  helper in `vtscore/detectors/training.py` or `vtscore/embedding/matrix.py`.
  Impact: medium-high on patch datasets; zero on single-vector (fast path
  already vectorized). Complexity: medium — the argmax/tie/mixed-snapshot
  details are the whole job.

<!-- item-sep -->

- **Single-fetch audio player** — selecting an audio clip downloads the file
  **twice** and decodes it every time: the `<audio>` element streams
  `/api/medias/{id}/audio`
  (`frontend/src/app/components/center-panel/audio-player/audio-player.component.ts:54`
  + `.html:16`) while `drawWaveform()` independently `fetch()`es the same URL
  (line 216) and runs `decodeAudioData`. The audio route sends no cache
  headers (only the thumbnail route sets an ETag — see
  `vtsearch/routes/media/list.py:683`), so the browser can't dedupe, and
  re-selecting a previously viewed clip re-pays everything.
  **Fix (frontend):** fetch once — `drawWaveform`'s fetch already gets the full
  bytes; hand them to the `<audio>` element via
  `URL.createObjectURL(new Blob([arrayBuffer], {type}))` (revoke the previous
  object URL on media change / `ngOnDestroy`). Add a small LRU `Map` (say 20
  entries) of decoded waveform peaks keyed by
  `datasetId:mediaId:clipStart:clipEnd` so re-selection skips fetch+decode
  entirely — cache the downsampled min/max arrays, not the `AudioBuffer`
  (peaks are ~KB, buffers are ~MB).
  **Fix (backend, complementary):** add an `ETag` (media md5 is already on the
  media dict) + `Cache-Control` to the audio media response so even cold
  object-URL misses turn into 304s. Mirror the thumbnail route's conditional
  logic in `vtsearch/routes/media/list.py`.
  **Gotchas:** preserve the existing `AbortController` handling around the
  fetch (line ~216) and the clip-window drawing logic; the `<audio>` `[src]`
  swap must not break the `loadedmetadata` gate (component tracks whether
  metadata has fired for the current src — see the comment at line ~34).
  `audio-player.component.spec.ts:34` asserts `audioSrc` equals the API URL —
  update it deliberately. Frontend gate: `./run-tests.sh frontend`.
  **Files touched:** `audio-player.component.{ts,html,spec.ts}`,
  `vtsearch/routes/media/list.py` (ETag), maybe a tiny peaks-cache service.
  Impact: high for audio-heavy voting. Complexity: medium.

## Tier 3 — worthwhile, more design judgment needed

<!-- item-sep -->

- **Non-blocking labeling status** — `/api/labeling-status` (a GET the
  frontend polls every 2 s during labeling — `label-view.component.ts:554`)
  synchronously advances the per-step cache in
  `vtscore/detectors/labeling_progress.py` (`compute_labeling_status` ~line
  712 → `_ensure_cache` → `_train_step` ~line 274 trains an MLP via
  `train_model`, and `_compute_step_stability` ~line 224 runs a forward pass
  over **all unlabeled** media), all under `_progress_lock`, in the request
  thread (`vtsearch/routes/eval.py:71-90`). Steady-state polls hit the cache
  and are cheap; the stall lands on the first poll after each new vote —
  exactly when the user is active — and a polarity flip truncates the cache
  and retrains multiple steps. The sibling `/api/eval/train-and-score` was
  moved to a background job for this precise reason (its docstring says so,
  `eval.py:144-156`).
  **Fix (recommended shape):** return the last-computed status immediately and
  kick cache advancement to a background worker: keep a per-detector
  "status snapshot" plus a dirty flag; the GET returns the snapshot and, if
  `label_history` has advanced past the cached step, schedules (or joins) a
  single in-flight refresh via the existing async-job/daemon-thread pattern
  (see `learned_sort_jobs` in `vtscore/concurrency/async_jobs.py` for the
  in-flight-dedup shape). Add a `"stale": true` field to the response while a
  refresh is pending so the frontend can render the previous colors slightly
  dimmed (or just ignore staleness — the colors converge one poll later).
  Cheaper alternative if the job wiring feels heavy: subsample the stability
  forward pass to a bounded random sample of unlabeled items (mirror
  `_GMM_MAX_SAMPLES = 50_000` in `vtscore/training/thresholds.py`), seeded
  per step for determinism. Both are legitimate; the snapshot approach fixes
  the worst case (polarity-flip retrain burst), the subsample only caps the
  per-step cost.
  **Gotchas:** `_progress_lock` ordering — the background refresh must take
  the same lock the POST `/api/labeling-progress` route takes; keep the lock
  scope inside the worker, never across the HTTP response. The response
  schema is `LabelingStatusResponseSchema` — adding `stale` means updating the
  schema + OpenAPI snapshot (run-tests catches drift). Tests:
  `./run-tests.sh api detectors` and the eval group.
  **Files touched:** `vtsearch/routes/eval.py`,
  `vtscore/detectors/labeling_progress.py`, schema file under
  `vtsearch/schemas/`.
  Impact: medium (removes post-vote request-thread stalls). Complexity: medium.

<!-- item-sep -->

- **Micro-fix grab bag** — small independent cleanups; shippable as one PR or
  folded into neighboring work. Each is verified real but individually minor:
  - `add_media_to_pile` builds the full 3-way media lookup (per-item
    `json.dumps` of origins) twice — once outside and once **inside**
    `_state_lock` — to resolve a single md5
    (`vtsearch/routes/media/list.py:1070` and `:1026-1034` via
    `vtscore/state/media_lookup.py:21 build_media_lookup`). Fix: an
    md5-only helper (skip origin/name dicts), or hang a maintained md5 index
    off `DatasetContext` (that bigger shape is scalability plan **S14**; the
    md5-only helper is the cheap standalone win).
  - Sort handlers call `snapshot_medias()` (full dict copy under
    `_state_lock`) multiple times per request — `vtsearch/routes/sorting.py`:
    `sort_clips` then `_cosine_sort` again (~:117/:191), and the example-sort
    path three times (~:622, :683, :128). Fix: snapshot once at handler top,
    thread the dict through (callees already accept a snap or trivially can).
  - `train_model` reads `weighted_loss.item()` every epoch
    (`vtscore/training/mlp.py:258`) — a per-epoch host-device sync on CUDA,
    ×200 epochs ×3 trainings per calibrated retrain. Fix: sync on the
    patience-check cadence (every 5-10 epochs). GPU-only win; guard the
    change with a determinism check on early-stop behavior (seeded test).
  - `before_request` stats the detector JSON on every non-exempt request
    (`app.py:338-346` → `vtscore/detectors/dataset_sync.py:88`). Fix: cache
    mtime with a short TTL (mirror `_FRESHNESS_CHECK_INTERVAL` in
    `settings_store.py`).
  - Audio/video clip enforcement runs a 100 ms `setInterval` while playing
    (`audio-player.component.ts:124`, `video-player.component.ts:144`). Fix:
    drive from the element's `timeupdate` event.
  - `ChartsService.themeColor()` calls `getComputedStyle` ~8-10× per chart
    render (`frontend/src/app/services/charts.service.ts:19-21`). Fix:
    resolve the needed colors once per `render*` call.
