# Scalability Implementation Plan

**Status:** Open; none of the phases below shipped yet.  Separately, the
CLI-specific streaming work (lazy folder enumeration, per-chunk embed, and
streaming export — partial fixes for S20/S15/S13 on the `--autodetect` target
side) **has** shipped; see
[`cli-stream-massive-images.md`](cli-stream-massive-images.md).

**Parent:** [`scalability.md`](scalability.md); the brainstorm that defines
all S# IDs referenced here.

**Goal:** Make VTSearch usable at 100 k items (near-term) and survivable at
1 M items (medium-term) without a full architectural rewrite.  10 M items
requires Phase 4 work and is deferred.

Phases are ordered by leverage-to-effort ratio.  Each phase is shippable
independently.

---

## Phase 1: Trivial wins (no architecture change)

These are 1–3 line fixes that eliminate growing hot paths.  Land them
together in a single PR.

### 1.1  Hash-based threshold cache key (S5)

**File:** `vtscore/training/thresholds.py:71–97` (`_calibration_cache_key`)

**Problem:** `np.stack(X_list).tobytes()` produces an `N × D × 4`-byte
string stored as part of a tuple in `DetectorContext._calibration_threshold_cache`.
At 10 k training vectors it is already ~15 MB; at 100 k it is 150 MB.

**Fix:** Replace the raw bytes with a `blake2b` digest.

```python
import hashlib

def _calibration_cache_key(X_list, y_list, inclusion_value, ...):
    X_bytes = np.stack(X_list).astype(np.float32, copy=False).tobytes()
    y_bytes = np.asarray(y_list, dtype=np.float32).tobytes()
    X_hash = hashlib.blake2b(X_bytes, digest_size=16).digest()
    y_hash = hashlib.blake2b(y_bytes, digest_size=16).digest()
    return (X_hash, y_hash, int(inclusion_value), ...)
```

The hash is computed from the same bytes (so correctness is identical), but
only 32 bytes are stored in the key tuple instead of the full array.
`blake2b` at this size is ~1 GB/s on modern hardware; far cheaper than
the current path.

**Risk:** negligible (birthday collision at 2^64; unreachable).

---

### 1.2  Subsample GMM threshold (S9)

**File:** `vtscore/training/thresholds.py:24–68` (`calculate_gmm_threshold`)

**Problem:** Takes all N scores and fits a 2-component GMM.  At 1 M items
this is an 8 MB array and many EM iterations.

**Fix:** Subsample before fitting.

```python
_GMM_MAX_SAMPLES = 50_000

def calculate_gmm_threshold(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.5
    arr = np.array(scores, dtype=np.float32)
    if len(arr) > _GMM_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, size=_GMM_MAX_SAMPLES, replace=False)
    X = arr.reshape(-1, 1)
    ...  # rest unchanged
```

At 50 k samples the GMM threshold is statistically indistinguishable from
fitting on 1 M samples (two Gaussians only need their means and variances).
The same subsampling should be applied anywhere `calculate_gmm_threshold`
is called from the safe-threshold blender
(`vtscore/training/thresholds.py:378`).

**Risk:** negligible.

---

### 1.3  Epoch counter for embedding matrix invalidation (S6)

**File:** `vtscore/state/core.py` (`DatasetContext`),
`vtscore/embedding/matrix.py:58–79`

**Problem:** `get_embedding_matrix` calls `sorted(ctx.medias.keys())` on
every invocation; O(N log N); to detect whether the cached matrix is
stale.

**Fix:** Add an integer `_medias_epoch` field to `DatasetContext`.
Increment it whenever `medias` is structurally mutated (adds or deletes).
The matrix cache stores the epoch at build time and validates with O(1)
integer compare.

`DatasetContext` changes:

```python
__slots__ = (
    ...
    "_medias_epoch",      # int, incremented on add/remove
    "_emb_matrix_epoch",  # epoch at which the cached matrix was built
    "_emb_matrix",        # np.ndarray | None
)

def __init__(self, dataset_id: str = "") -> None:
    ...
    self._medias_epoch: int = 0
    self._emb_matrix_epoch: int = -1
    self._emb_matrix: Any = None
```

`_emb_matrix_ids` is dropped; it was only needed for the sorted-key
comparison and is no longer required.

All sites that structurally modify `ctx.medias` must increment the epoch:

| Site | File |
|------|------|
| `ctx.medias[cid] = …` (importer write) | `load_pipeline.py` |
| `del ctx.medias[cid]` (None-embedding drop) | `load_pipeline.py:932` |
| `ctx.medias.clear()` | `vtscore/state/__init__.py:134` |
| `collapse_duplicates` (may delete) | `vtscore/state/media_lookup.py` |

Since `medias` is a plain dict and Python does not support dict hooks,
the epoch must be incremented manually at each mutation site.  A helper
`def _bump_epoch(ctx)` in `core.py` keeps this one line per site.

`get_embedding_matrix` becomes:

```python
def get_embedding_matrix(ctx):
    with _state_lock:
        if ctx._emb_matrix is not None and ctx._emb_matrix_epoch == ctx._medias_epoch:
            sorted_ids = sorted(ctx.medias.keys())  # still needed for the return value
            return list(sorted_ids), ctx._emb_matrix
        ...  # rebuild as before; store ctx._emb_matrix_epoch = ctx._medias_epoch
```

Wait; the sorted_ids return value still needs `sorted(...)` on the cache
hit path.  Callers that need the id list on every call will still pay O(N
log N) for that.  Fix those callers to not re-sort on cache-hit:

- Store `_emb_matrix_sorted_ids: list[int]` alongside the matrix (same as
  today's `_emb_matrix_ids`, but now the epoch is the validity signal, not
  the list comparison).  Return the stored list on cache hit.

Net result: the epoch check is O(1); `sorted(...)` only runs on cache miss
(i.e. when the dataset actually changed).

**`get_embedding_matrix_for_snap`** (line 89): also performs
`sorted(snap.keys())` (line 102) and `sorted(ctx.medias.keys())` (line 115).
After this change, the ctx branch reuses the epoch path (shared cached
matrix); the temp-dict/cross-dataset branch still sorts once (unavoidable
since there is no epoch for an ad-hoc dict).

**Risk:** medium; touches `DatasetContext.__slots__` and several mutation
sites.  Must be covered by tests that verify the matrix is rebuilt exactly
when medias change.

---

### 1.4  Epoch-based learned-sort signature (S7)

**File:** `vtsearch/routes/sorting.py:340–358` (`_build_learned_sort_signature`)

**Problem:**

```python
tuple(sorted(snap.keys())),                         # O(N log N)
("regions", tuple(sorted(region_boxes_snapshot.items()))),  # O(R log R)
```

Both appear inside the signature that is checked before every learned-sort
job.

**Fix:** Replace `tuple(sorted(snap.keys()))` with
`("epoch", ctx._medias_epoch)` fetched from the active `DatasetContext`.
Replace the region-box sort with a frozen-set hash of region IDs (vote
regions are small, so this is negligible, but the sort is wasteful).

```python
from vtscore.state.core import get_active_context

epoch = get_active_context()._medias_epoch
region_sig = frozenset(region_boxes_snapshot.keys())

sig = (
    ("epoch", epoch),
    ("votes", tuple(sorted(good_votes)) + tuple(sorted(bad_votes))),
    ("regions", region_sig),
    ("inclusion", inclusion),
    ...
)
```

Vote sets are small (tens to hundreds of items), so sorting them is cheap.

**Risk:** low; the signature is only a cache key; a false miss (different
epoch, same medias) wastes one retrain; a false hit (same epoch, different
medias) cannot happen because the epoch is only bumped on structural change.

---

## Phase 2: Medium effort, high leverage

### 2.1  Cap and defer diversity tree construction (S2, S8)

**Files:** `vtscore/state/diversity_tree.py`,
`vtscore/datasets/load_pipeline.py:968–981`

**Problem:** The diversity tree is always built at load time.  At 100 k
items it already takes several seconds; at 1 M it takes minutes and
allocates GBs.

**Fix (two parts):**

**Part A; smarter defaults.**  In `DiversityTree.__init__`, cap k-means
`_N_INIT` as a function of node size:

```python
def _n_init_for(node_size: int) -> int:
    if node_size > 10_000:
        return 3   # large nodes converge reliably; fewer restarts
    if node_size > 1_000:
        return 5
    return 10  # current default for small nodes
```

Similarly auto-cap `max_depth` so the tree never has more than ~4 000 leaf
nodes regardless of N:

```python
def _auto_max_depth(n: int, k: int, min_node_size: int) -> int:
    import math
    max_leaves = 4_000
    # k^depth ≈ n / min_node_size, capped at max_leaves
    natural = math.ceil(math.log(n / max(min_node_size, 1), max(k, 2)))
    cap = math.ceil(math.log(max_leaves, max(k, 2)))
    return min(DIVERSITY_TREE_MAX_DEPTH, natural, cap)
```

Apply this in `build_diversity_tree_for_context` when the caller doesn't
pass an explicit `max_depth`.

**Part B; skip above a threshold; expose a "Build diversity tree" endpoint.**
In `_build_diversity_tree_stage` (load pipeline):

```python
_DIVERSITY_TREE_AUTO_THRESHOLD = 50_000  # skip auto-build above this

def _build_diversity_tree_stage(ctx, tracker):
    n = len(ctx.medias)
    if n > _DIVERSITY_TREE_AUTO_THRESHOLD:
        # Tree skipped at load time; user can trigger it via POST /api/datasets/.../diversity-tree
        return
    ...build as today...
```

Add a new route `POST /api/datasets/registry/<id>/diversity-tree` that
builds the tree in the background (reusing the existing `JobManager`
pattern) and reports progress via `/api/progress/dataset/<id>`.  The
frontend shows a "Build diversity index" button in the dataset panel when
the tree is absent.

**Risk:** medium; the diversity tree is the autopilot's diversity signal.
Without it, autopilot falls back to score-only sampling.  This is
acceptable and already handled by the `if tree is None` guard in
`diversity_tree_next_sample`.

---

### 2.2  Debounce label sync writes (S12)

**File:** `vtscore/labels/sync.py` (`sync_to_labelset_source`)

**Problem:** `sync_to_labelset_source` fires on every vote and rewrites
the entire JSON labelset file.  At 100 k labels the file is ~50–100 MB;
each vote stalls the vote handler for seconds.

**Fix:** Debounce with a per-detector background flush.

```python
import threading

_flush_timers: dict[str, threading.Timer] = {}  # detector_id → timer
_FLUSH_DELAY = 2.0  # seconds

def sync_to_labelset_source(flush_delay: float = _FLUSH_DELAY) -> None:
    from vtscore.state.core import get_active_detector_context
    det_ctx = get_active_detector_context()
    det_id = det_ctx.detector_id

    # Cancel any pending flush for this detector.
    if det_id in _flush_timers:
        _flush_timers[det_id].cancel()

    def _flush():
        _flush_timers.pop(det_id, None)
        _do_sync_to_labelset_source()  # existing implementation

    timer = threading.Timer(flush_delay, _flush)
    timer.daemon = True
    timer.start()
    _flush_timers[det_id] = timer
```

This means votes accumulate for up to 2 s before a write fires; one write
instead of N.  On process shutdown the timers are daemon threads so they do
not block exit; the settings-source sync (which is already separate) is
responsible for flushing on graceful shutdown.

An immediate flush can be forced by passing `flush_delay=0`; useful in
tests and in the label-export route.

**Risk:** low; labels might be up to 2 s stale in the sync target after a
vote, which is acceptable.

---

### 2.3  Incremental secondary lookups on DatasetContext (S14)

**File:** `vtscore/state/core.py` (`DatasetContext`),
various route files that rebuild `md5_to_media` / `origin_key_to_cid` per
request.

**Problem:** Many routes rebuild `{m["md5"]: m for m in snap.values()}`
(or similar) on every request; O(N) Python iteration with dict construction.

**Fix:** Add maintained secondary indexes to `DatasetContext`:

```python
class DatasetContext:
    __slots__ = (
        ...
        "_md5_index",           # dict[str, int]  md5 → cid
        "_origin_key_index",    # dict[str, int]  stable_origin_key → cid
    )

    def __init__(self, ...):
        ...
        self._md5_index: dict[str, int] = {}
        self._origin_key_index: dict[str, int] = {}
```

Maintained by the same mutation sites that bump `_medias_epoch` (Phase 1.3):

```python
def _add_media(ctx, cid, media):
    ctx.medias[cid] = media
    ctx._medias_epoch += 1
    if md5 := media.get("md5"):
        ctx._md5_index[md5] = cid
    if ok := _stable_origin_key(media):
        ctx._origin_key_index[ok] = cid

def _remove_media(ctx, cid):
    media = ctx.medias.pop(cid, None)
    ctx._medias_epoch += 1
    if media:
        ctx._md5_index.pop(media.get("md5", ""), None)
        ctx._origin_key_index.pop(_stable_origin_key(media), None)
```

Routes and helpers that currently rebuild lookup dicts from the full
snapshot switch to `ctx._md5_index` / `ctx._origin_key_index` directly.

This is a medium refactor because mutation sites are spread across
`load_pipeline.py`, `state/__init__.py`, and `state/media_lookup.py`.

**Risk:** medium; secondary indexes must stay in sync with `medias`.  Any
site that writes to `ctx.medias` directly (bypassing the helper) will
produce stale indexes.  The Phase 1.3 epoch audit surfaces all those sites,
so Phase 1.3 should land first.

---

## Phase 3: Larger architectural work

### 3.1  mmap embedding matrix (S1)

**Files:** `vtscore/datasets/loader_pickle.py`,
`vtscore/embedding/matrix.py`

**Problem:** Embeddings are stored inline in `media["embedding"]` as Python
lists/arrays, then copied into a contiguous `(N, D)` NumPy array on first
access.  At 1 M items × 384-dim the array is 1.5 GB; it must be rebuilt
from the per-item entries on every cold start.

**Fix (two-step):**

**Step A; sidecar `.npy` file.**  During the post-load phase
(after all embeddings are present), write the sorted-by-cid embedding
matrix to `<dataset_path>.emb.npy` if it doesn't already exist.  Format:

```
# Header line (JSON): {"cids": [1, 2, 3, ...], "dim": 384, "dtype": "float32"}
# Body: raw float32 bytes, row-major, shape (N, D)
```

Actually simpler: use `np.save` / `np.load` with a companion `<dataset>.cids.npy`
(int64 array of sorted cids) and `<dataset>.emb.npy` (float32 matrix).

**Step B; mmap load.**  On pickle load, check whether both sidecars exist
and their cid list matches the pickle's media IDs.  If so:

```python
cids = np.load(cids_path)          # int64 array
matrix = np.load(emb_path, mmap_mode='r')  # zero-RAM mmap
ctx._emb_matrix = matrix
ctx._emb_matrix_sorted_ids = list(cids)
ctx._emb_matrix_epoch = ctx._medias_epoch
```

The OS pages in only the rows that scoring actually accesses.  For a
100-item sort query on a 1 M item dataset, only ~40 kB of the 1.5 GB file
is touched.

`get_embedding_matrix` must detect when the matrix is already cached this
way and skip the rebuild.

**Sidecar invalidation:** If the pickle file is newer than the sidecar, or
if the cid list doesn't match, fall back to the in-memory build (and
optionally write fresh sidecars).

**Risk:** high; introduces file-system state alongside pickle files.
Must be careful about:
- Race between sidecar write and concurrent load
- Version drift if the embedder changes (embedding dim changes → sidecar invalid)
- Docker / read-only filesystems (fall back gracefully to in-memory)

A `--no-emb-sidecar` CLI flag (or a settings key) can disable sidecar
writing for environments where the dataset path is read-only.

---

### 3.2  Sparse sort results: top-K API + lazy frontend (S3, S17, S19)

This is the largest single change.  It requires coordinated backend + frontend
work and must be landed as a pair of PRs.

**Backend changes:**

`POST /api/sort/cosine` and `POST /api/sort/learned` currently return:

```json
{
  "results": [{"id": 1, "score": 0.91, "bestRegion": null}, ...],
  "threshold": 0.72
}
```

for all N items.  Change the response to:

```json
{
  "total": 1000000,
  "threshold": 0.72,
  "above_threshold": 312,
  "results": [{"id": 1, "score": 0.91, ...}, ...],
  "has_more_below": true
}
```

where `results` contains only the top `K_ABOVE` items above threshold
plus `K_BELOW` items immediately below (e.g. `K_ABOVE = 500`,
`K_BELOW = 200`).  Clients that need more items fetch:

```
GET /api/sort/page?offset=500&limit=200
```

The full sorted order is computed in the backend as today, but only a
window is transmitted.  The backend holds the full sorted list in the
existing job result (already stored in `AsyncJob.result`).

**Frontend changes:**

`SortStateService` currently holds `SortedItem[] | null`.  Replace with:

```typescript
interface SortWindow {
  total: number;
  threshold: number;
  aboveThreshold: number;
  items: SortedItem[];        // the loaded window
  loadedRange: [number, number]; // [start, end] indices
  hasMore: boolean;
}
```

`media-list.component.ts:rebuildOrderedItems` no longer iterates a
full array; it renders whatever is in `items` and shows a "Load more"
trigger at the end.  `cachedOrderedItems` stays bounded to the loaded
window (≤ 700 items by default).

Virtual scroll on the frontend already handles a fixed window; this just
caps the array size at the API level.

**Complexity:** 2–3 PRs (backend sort API, frontend SortStateService,
frontend media-list).  All three must land atomically or behind a feature
flag.

---

### 3.3  Virtual grid mode (S16)

**File:** `frontend/src/app/components/left-panel/media-list/`

**Problem:** Grid mode renders every item into the DOM.  At > 500 items
Chrome layout stalls; at > 5 000 items the tab may crash.

**Fix:** Use `CdkVirtualScrollViewport` in grid mode with a fixed-height
row that holds `floor(viewport_width / item_width)` grid cells.

The CDK virtual scroller works with 1D lists, not 2D grids.  The standard
approach is to group items into row arrays and give the viewport a list
of rows:

```typescript
// Computed when cachedOrderedItems changes
get virtualRows(): MediaItem[][] {
  const cols = Math.floor(this.viewportWidth / this.gridIconSize);
  const rows: MediaItem[][] = [];
  for (let i = 0; i < this.cachedOrderedItems.length; i += cols) {
    rows.push(this.cachedOrderedItems.slice(i, i + cols));
  }
  return rows;
}
```

The template renders each row as a flex container with `cols` items.
`itemSize` is set to `this.gridIconSize + gap`.

Existing breakpoints (`VIRTUAL_SCROLL_THRESHOLD = 500`) apply to both list
and grid mode.

**Risk:** medium; requires layout changes to the grid template and dynamic
`itemSize` computation when icon-size settings change.

---

## Phase 4: Major infrastructure (1 M+ items)

These are deferred until Phase 1–3 are stable.

### 4.1  Streaming pickle load (S15)

Pickle files > 500 MB should be loaded in chunks.  The embedding matrix
sidecar (Phase 3.1) decouples embeddings from the pickle, enabling the
pickle itself to omit inline embeddings.  Once that is in place, the
pickle file becomes mostly metadata (filenames, origins, md5s) and can
be loaded quickly; embeddings come from the mmap'd sidecar.

### 4.2  Parallel cross-dataset label population (S11)

`populate_label_embeddings` in `vtscore/detectors/labelset_training.py`
resolves and embeds each LabelSet element sequentially.  Replace with a
`ThreadPoolExecutor` (bounded by the same `_embed_gate` that governs
dataset embedding concurrency).  Deduplicate file resolves across elements
with the same origin before dispatching.

### 4.3  Streaming JSON label export (S13)

Replace `LabelSet.to_dict()` → `flask.jsonify()` with a generator that
yields one JSON line per label element, wrapped in `flask.stream_with_context`.
Requires clients to parse a newline-delimited JSON stream (NDJSON) or the
existing `{"labels": [...]}` wrapper using a streaming parser.  The simpler
approach is to add a `?format=ndjson` query param and keep the default
response identical.

---

## Test coverage checklist

Each phase needs targeted tests before merging:

- **1.1**: Unit test: cache key is always the same tuple length regardless
  of N_labels; verify no collision for distinct X/y pairs.
- **1.2**: Unit test: `calculate_gmm_threshold` with N > 50 k produces a
  threshold within 1% of the full-sample result (use a known bimodal
  distribution).
- **1.3**: Integration test: matrix is rebuilt exactly when medias change,
  reused when they don't; epoch counter is O(1) to check.
- **1.4**: Integration test: learned-sort job is not re-fired when called
  twice with the same votes on the same epoch.
- **2.1**: Load test: dataset with 100 k items loads in < 30 s without a
  diversity tree; "Build diversity tree" endpoint completes and the result
  is used by autopilot.
- **2.2**: Integration test: 100 rapid votes produce exactly 1 file write
  within 2 s of the last vote.
- **3.1**: Integration test: load, unload, reload a dataset; matrix is
  re-used from sidecar on second load without rebuilding from per-item entries.
- **3.2**: API test: sort response with 200 k items contains ≤ 700 results;
  `/api/sort/page?offset=700&limit=200` returns the next window.

---

## Open follow-ups

- FAISS / HNSW replacement for diversity tree (long-term S2 fix)
- Columnar `medias` storage (S4): deferred; requires redesign of every
  media-reading call site
- Append-only vote journal for labelset sources (S12 long-term): deferred
  until compaction semantics are defined
- CLI progress bar rate-limiting (S21): trivial, add when touching CLI
