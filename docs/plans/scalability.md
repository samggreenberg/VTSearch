# VTSearch Scalability Brainstorm

**Status:** Brainstorm / open; no fixes shipped yet.

**Scope:** What breaks, slows, or explodes in memory as datasets grow to
100 k / 1 M / 10 M items, and as LabelSets grow to 1 k / 10 k / 100 k labels.
Both GUI and CLI considered. Ordered roughly by severity / likelihood to hurt
first.

Line references are approximate; they will drift as the codebase evolves.

---

## How to read this doc

- Items are grouped by the *resource* they exhaust: **RAM**, **CPU/time**,
  or **network/JSON**.
- Each item has a stable ID (`S#`) for reference from PRs.
- A sub-section at the bottom collects the **recurring root causes**: fixing
  a pattern fixes many items at once.
- "At N" estimates are back-of-envelope with SigLIP/E5 embedding dim ~384
  (float32 = 4 B), sorted-list cost at ~1 µs/item, and JSON encoding at
  ~50 B/element.

---

## RAM explosions

### S1: Embedding matrix: one giant contiguous array per loaded dataset

**File:** `vtscore/embedding/matrix.py`

All embeddings for every media item in the loaded dataset are materialised
into a single `(N, D)` `float32` NumPy array held on `DatasetContext`.

| N | D=384 | D=768 (E5) |
|---|-------|-----------|
| 100 k | 150 MB | 300 MB |
| 1 M | 1.5 GB | 3 GB |
| 10 M | 15 GB | 30 GB |

One loaded dataset already risks OOM.  Two loaded datasets (multi-dataset
mode) doubles it.  If we ever support multiple embedders per item the
matrix fan-out would be worse.

The array is unavoidable for vectorised scoring (matrix-vector dot product),
but it does not need to live in process RAM for the full lifetime of the
dataset.  At 1 M+, it needs to be **memory-mapped from the pickle file**
(or from a companion `.npy` sidecar) so the OS can page it out when unused.

**Fix direction:** Add mmap support to the pickle loader.  For the common
case where all embeddings share the same dtype and dim, write a companion
`dataset.npy` sidecar during the first load; subsequent loads `np.load` it
with `mmap_mode='r'` and hand the matrix directly to `get_embedding_matrix`.
Scoring and sorting never mutate the matrix, so read-only mmap is safe.

---

### S2: Diversity tree lives entirely in RAM

**File:** `vtscore/state/diversity_tree.py`, `vtscore/state/diversity.py`

`DiversityTree` stores all node objects in Python dicts keyed by cid.  At
100 k items the tree has up to ~5 k leaf nodes; the per-node dict overhead
(plus scikit-learn KMeans state) makes the tree substantially larger than
just the embeddings.

More critically, **construction** is O(N × levels × k-means-iterations):
default `k=2, max_depth=10, _N_INIT=10`.  At 100 k items that is ~10 layers
of k-means, each fitting on a subset of vectors.  Empirically this is
already several seconds at 10 k items; at 100 k it will dominate dataset
load time.

At 1 M items the tree construction will likely take minutes and requires
gigabytes of temporary KMeans work arrays.

**Fix direction (short-term):** Cap `max_depth` more aggressively as a
function of N: `max_depth = min(10, max(3, int(log2(N / 20))))` so the tree
stays useful but doesn't blow up.  Also make tree construction **opt-in**:
skip it by default when N > some threshold (e.g. 50 k), and surface a
"Build diversity tree" button in the UI.

**Fix direction (long-term):** Replace with an approximate nearest-neighbour
structure (FAISS IVF index, HNSW) that supports the same "next unseen
cluster" query but can be stored mmap'd.

---

### S3: Sort-result payload: one `{id, score, bestRegion}` per item

**File:** `vtsearch/routes/sorting.py`, `frontend/src/app/services/sort-state.service.ts`

Every sort API call returns `results: [{id, score, bestRegion}, …]` for the
entire dataset in one JSON response.

| N | JSON size (est.) |
|---|-----------------|
| 10 k | 0.5 MB |
| 100 k | 5 MB |
| 1 M | 50 MB |
| 10 M | 500 MB |

The frontend keeps a full `SortedItem[]` array in `SortStateService`.
`media-list.component.ts:rebuildOrderedItems` iterates this array on every
sort result, building `cachedOrderedItems`; an O(N) JS loop.  At 1 M items
this will freeze the browser tab.

**Fix direction:** The sort API should return only the IDs the frontend
actually needs to render: the top-K good items (above threshold) plus a
small window of items below threshold for browsing.  Lazy / paginated sort
results ("give me items 5 000–5 100 in sort order") would let the frontend
request more as the user scrolls.  The frontend would keep a sparse, on-demand
view of the sorted order rather than a full in-memory array.

---

### S4: `medias` dict: one Python dict entry per item

**File:** `vtscore/state/core.py` (`DatasetContext.medias`)

Each media item is a Python `dict` with ~10 keys (id, filename, md5, origin,
origin_name, embedding, media_type, embedder, …).  Python dict overhead is
~250 B per entry, plus the dict itself is ~50 B/slot.  At 1 M items the
`medias` dict alone consumes **several hundred MB** before considering
embedding arrays (which are stored inline as lists or ndarrays).

This is unavoidable with the current design.  The longer-term fix is to
store the dataset as a columnar structure (per-field NumPy arrays or a
Polars/Pandas frame) so metadata is compact and embeddings are in the
already-mmap'd matrix (S1).

**Fix direction (medium-term):** Extract embeddings from the per-item dict
into the embedding matrix on load, replacing `media["embedding"]` with a
row-index pointer.  This alone reduces per-item dict size by ~60–70% at
384-dim.

---

### S5: Threshold cache key contains full training vectors as bytes

**File:** `vtscore/training/thresholds.py:88–89` (`_calibration_cache_key`)

```python
X_bytes = np.stack(X_list).astype(np.float32, copy=False).tobytes()
y_bytes = np.asarray(y_list, dtype=np.float32).tobytes()
return (X_bytes, y_bytes, ...)
```

For N_labels labels at D=384 dim this creates a `(N_labels × D × 4)` byte
string on every threshold computation.  At 10 k labels: ~15 MB blob stored
in the cache key tuple.  At 100 k: 150 MB.  The tuple lives in
`DetectorContext._calibration_threshold_cache` until the next call
invalidates it.

**Fix direction:** Hash the vectors instead of storing them.  Replace
`X_bytes` and `y_bytes` with `hashlib.blake2b(X_bytes).digest()` (fast,
128-bit output) so the cache key is always tiny.  Since the cache already
lives on the detector context and is invalidated by any vote change, the
hash is purely for collision resistance (two different labelsets mapping to
the same key would be a cosmic coincidence).

---

## CPU / time explosions

### S6: Embedding matrix cache invalidation: O(N log N) on every scoring call

**File:** `vtscore/embedding/matrix.py:59–62`

```python
sorted_ids = sorted(ctx.medias.keys())
if cached_matrix is not None and cached_ids == sorted_ids:
    return list(sorted_ids), cached_matrix
```

`sorted(...)` on N integer keys is O(N log N).  `cached_ids == sorted_ids`
is then an O(N) list comparison.  This runs **every time** a scoring or
sorting function calls `get_embedding_matrix`.  After a sort, both panels
call it simultaneously; after a vote, it is called again for retraining.

| N | sorted() cost (est.) |
|---|---------------------|
| 10 k | ~1 ms |
| 100 k | ~20 ms |
| 1 M | ~200 ms |

At 1 M items the cache-hit path alone costs ~200 ms per sort/retrain.

**Fix direction:** Replace sorted-key comparison with an **epoch counter** on
`DatasetContext` that is incremented whenever `medias` is modified (add,
remove).  The matrix caches the epoch at build time; validation is O(1).  The
`_emb_matrix_ids` list can be dropped entirely.

---

### S7: Learned-sort signature: O(N log N) for each sort invocation

**File:** `vtsearch/routes/sorting.py:345,350`

```python
("regions", tuple(sorted(region_boxes_snapshot.items()))),
...
tuple(sorted(snap.keys())),
```

Both `sorted(snap.keys())` and `sorted(region_boxes_snapshot.items())` run
inside `_build_learned_sort_signature`, which is called on every
`/api/sort/learned` request.  The signature is used as a job-cache key; the
sorted-key tuple is also used to check whether training input has changed.

At 1 M items, `sorted(snap.keys())` alone is ~200 ms.  This fires before any
actual work is done.

**Fix direction:** Same epoch counter from S6.  Include the epoch counter (and
vote hash) in the signature instead of sorting all keys.

---

### S8: Diversity tree rebuild: full k-means across all items

**File:** `vtscore/state/diversity.py:82`, `vtscore/state/diversity_tree.py`

See S2 for memory.  The time cost is even more acute: after any dataset load
the pipeline calls `build_diversity_tree_for_context`, which runs hierarchical
k-means on **all N items**.  At k=2, max_depth=10, that is up to 2 048 leaf
nodes, each requiring a k-means fit on its subset.

At 100 k items total, the mid-level nodes each have ~hundreds of items and
k-means converges quickly, but the root-node fit on all 100 k items at
`_N_INIT=10` random starts is already slow.

**Fix direction:** Same as S2.  In the short term: reduce `_N_INIT` to 3 for
nodes above some size; skip the full rebuild when N > 50 k unless explicitly
requested.

---

### S9: GMM threshold: sklearn on N scores

**File:** `vtscore/training/thresholds.py:24–68`

`calculate_gmm_threshold` takes the scored array for **every media item** and
fits a two-component GMM.  At 1 M items, `np.array(scores).reshape(-1, 1)` is
itself 8 MB, and the GMM EM iterations run on the whole array.  This fires on
every retrain.

**Fix direction:** Subsample scores before GMM fitting.  A random 10 k–50 k
sample gives an essentially identical threshold at a fraction of the cost.
Add a `max_gmm_samples: int = 50_000` parameter (default capped).

---

### S10: MLP forward pass: O(N) inference on every retrain

**File:** `vtscore/detectors/training.py` (`_score_all_media`)

Scoring all media requires an MLP forward pass for every item after every
vote.  The batch is large (N × D → N × 1) but even a tiny MLP at N=1 M
will take several hundred milliseconds.

At N=10 M, even with batch GPU inference at 10 M examples/sec, scoring
takes ~1 s per retrain.

**Fix direction:** For very large datasets, **score only the items near the
threshold** (confidence-weighted sampling) and return approximate results.
Alternatively, debounce retraining so it does not fire on every single vote;
accumulate a few votes and retrain once.

---

### S11: Cross-dataset label population: serial I/O, O(N_labels)

**File:** `vtscore/detectors/labelset_training.py` (~line 260)

`populate_label_embeddings` iterates every element in a `LabelSet`,
resolves the file, and embeds it.  For uncached elements this is **serial
I/O + model inference** for each.

| N_labels | at 50 ms/label |
|----------|---------------|
| 1 k | ~50 s |
| 10 k | ~8 min |
| 100 k | ~83 min |

**Fix direction:** Parallelise the resolution/embedding loop using a thread
pool (or an async approach for the I/O phase).  Existing concurrency
infrastructure in `vtscore/concurrency/` (JobManager, ConcurrencyGate) can
be reused.  Also: the double-walk (populate then build_xy) should be merged
into a single pass.

---

### S12: Label sync: full JSON rewrite on every vote

**File:** `vtscore/detectors/label_sync.py`, `vtscore/labels/sync.py`

`sync_to_labelset_source` fires on every vote change.  It serialises the
entire `LabelSet` to JSON and writes to disk.  At 100 k labels, the
serialised labelset JSON is ~50–100 MB.  Writing this on every single vote
makes the vote handler stall for seconds.

**Fix direction:** Debounce the sync write; accumulate votes for e.g. 2 s
then flush once.  For disk-local sources, an **append-only journal** (one
`{id, label}` line per vote) would make the per-vote write O(1) regardless
of labelset size; a compaction step on load reconstructs the full labelset.

---

### S13: Label export: full JSON in memory

**File:** `vtsearch/routes/labels/vote.py:172`, `vtscore/datasets/labelset.py:259`

`LabelSet.to_dict()` builds `[e.to_dict() for e in self.elements]`; a fully
materialised Python list of dicts, then Flask JSON-encodes it into a string
and returns it.  At 100 k labels, the in-memory representation is ~50 MB
before JSON encoding.

**Fix direction:** Stream the JSON response using Flask's
`stream_with_context` + a generator that encodes one element at a time, so
the full payload is never in memory at once.

**Partially shipped:** CLI *autodetect results* now stream via
`--stream-results` (NDJSON / streamed CSV; see
[`cli-stream-massive-images.md`](cli-stream-massive-images.md)).  The
`/api/labels/export` route (this item) still buffers the full JSON; apply the
same streaming pattern there.

---

### S14: `snapshot_medias()` full-dict copies

**File:** `vtscore/state/core.py`

Several routes call `snapshot_medias()` and then build secondary lookup dicts
from the result (e.g. `{m["md5"]: m for m in snap.values()}`).  Each of
these is an O(N) pass.  Most routes do 2–3 such passes per request.

At 1 M items, each pass (even a simple iteration) takes ~100 ms of CPU time
in Python.

**Fix direction:** Add **indexed secondary lookups** on `DatasetContext`:
`md5_to_cid`, `origin_key_to_cid`, etc., maintained incrementally as items
are loaded.  Most routes today rebuild these indexes on every request.

---

### S15: Dataset pickle loading: everything into RAM at once

**File:** `vtscore/datasets/loader_pickle.py`

Pickle files are loaded with `torch.load` / `pickle.load` in one shot.  At
10 M items with 384-dim float32 embeddings, the pickle file is ~15 GB;
loading it in one shot is not feasible on typical hardware.

**Fix direction:** Pickle datasets > some threshold (e.g. 100 k items) should
be loaded in streaming chunks.  Embeddings should be written to a companion
mmap'd `.npy` (see S1) instead of stored inline in the pickle.

**Note:** the *folder* importer's chunked CLI path now enumerates files lazily
(no full file list in RAM); see
[`cli-stream-massive-images.md`](cli-stream-massive-images.md).  This item is
specifically about the *pickle* loader, which still reads the whole file at
once.

---

## Frontend / rendering

### S16: Grid mode: no virtual scrolling, renders all DOM nodes

**File:** `frontend/src/app/components/left-panel/media-list/media-list.component.ts:98–101`

Virtual scrolling is only active in **list mode** when `cachedOrderedItems.length > 500`.
In **grid mode**, every item is rendered into the DOM simultaneously.  At
1 000 items, Chrome typically takes > 1 s to layout the grid; at 10 000 items
the browser may freeze entirely.

**Fix direction:** CDK's `CdkVirtualScrollViewport` can be combined with a
grid layout using `@angular/cdk/scrolling` `FixedSizeVirtualScrollStrategy`
or the `AutoSizeVirtualScrollStrategy`.  Alternatively, the grid can use a
"virtual grid" pattern: render a fixed number of rows (viewport height /
item height) and swap item data as the user scrolls.

---

### S17: `rebuildOrderedItems`: O(N) JS on every sort result

**File:** `frontend/src/app/components/left-panel/media-list/media-list.component.ts:127–164`

Every time sort results arrive or the media list changes, the component
rebuilds `cachedOrderedItems` by iterating the full `sortOrder` array.  For
list mode with virtual scrolling this is fine for rendering (only visible
rows are in the DOM), but the array build itself is O(N) JS.  At 1 M items
the array allocation and the `for … of sortOrder` loop will stall the main
thread.

**Fix direction:** Lazy-evaluate `cachedOrderedItems` as a virtual view over
`sortOrder`; only materialise the window of items the viewport requests.
Combined with S3 (sparse sort results from the API), the array never needs
to be large.

---

### S18: `prefetchVisibleMetadata` fetches all IDs when virtual scroll is off

**File:** `frontend/src/app/components/left-panel/media-list/media-list.component.ts:267–274`

```typescript
if (!this.useVirtualScroll || !this.virtualViewport) {
  const ids = this.cachedOrderedItems.map((item) => item.media.id);
  this.metadataCache.ensureLoaded(ids);
  return;
}
```

When virtual scroll is disabled (list mode < 500 items, or grid mode at any
size), this sends **all IDs** to `ensureLoaded` in one shot.  At 10 k items
that is a single `POST /api/medias/batch` with 10 k IDs; the server must
then pull 10 k rows of metadata, and the response will be large.

`ensureLoaded` does de-duplicate already-cached IDs so this is only expensive
on the first load, but it fires on every `rebuildOrderedItems`.

**Fix direction:** Cap the prefetch to the first N visible items even in
non-virtual mode.  The metadata cache already handles incremental loading;
the component should trust it and not force an eager full-dataset prefetch.

---

### S19: `SortStateService` holds full in-memory `SortedItem[]`

**File:** `frontend/src/app/services/sort-state.service.ts:20,50–51`

`sortOrderSubject` stores the entire sorted array as a `BehaviorSubject`.
Every subscriber (`media-list`, left/right panels) receives the full array on
every sort update.  At 1 M items, this is a multi-MB object held in the
Angular DI graph and cloned/referenced by every subscriber.

**Fix direction:** Companion to S3.  Once the sort API returns only the top
window of results, the service holds a much smaller array.  The service
should expose a paginated query interface: "give me sort positions X–Y".

---

## CLI-specific issues

### S20: CLI autodetect scores every item, serial per detector

**File:** `vtscore/cli.py` (autodetect), `vtsearch/routes/detectors/scoring.py`

`--autodetect` iterates over all Auto-Find detectors and, for each one that
has unresolved labels, resolves+embeds them before scoring all N items.
Label resolution is serial within each detector.  With 10 detectors each
having 1 000 unresolved labels at 50 ms/label: **500 s just for resolution**,
before any inference.

**Partially shipped** (see [`cli-stream-massive-images.md`](cli-stream-massive-images.md)):
`--autodetect --chunk-size N --stream-results` now scores chunk by chunk and
streams hits straight to the exporter, so the *target* side no longer holds all
N items, all hits, or the full export in RAM; folder enumeration is lazy; and
each chunk is embedded one at a time.  **Still open:** the per-detector
*label* resolution below is unchanged.

**Fix direction:** Resolve all detectors' labels in parallel (thread pool).
Bundle duplicate-file resolves across detectors.  Use batch embedding for
labels from the same embedder.

---

### S21: CLI progress bars: O(N) string formatting per update

Not a bottleneck today but worth noting: some progress-reporting paths call
`str.format` with the full medias dict size on every update tick.  At 1 M
items, if updates fire every item, this becomes O(N²) formatting work.
The fix is trivial: only update at a fixed interval (every 1 000 items, or
every 100 ms by wall clock).

---

## Recurring root causes

| Root cause | Items it affects |
|-----------|-----------------|
| **O(N log N) sorted-key comparisons** used as change detection | S6, S7 |
| **Full in-memory arrays / dicts** for every N items | S1, S3, S4, S17, S19 |
| **No streaming** for large JSON payloads | S3, S13 |
| **Serial I/O** where parallelism is easy | S11, S20 |
| **No debouncing** on high-frequency write paths | S12 |
| **No subsampling** for statistical operations (GMM, etc.) | S9 |
| **Diversity tree** unconditional rebuild on every load | S2, S8 |
| **Grid mode** missing virtual scroll | S16 |
| **Oversized cache keys** containing raw vectors | S5 |

---

## Suggested fix order (max-leverage first)

1. **S6 + S7** (epoch counter); trivial code change, eliminates O(N log N)
   hot path from every sort and retrain.  Low risk, high gain.
2. **S1** (mmap embedding matrix); unblocks 1 M+ datasets without OOM.
   Moderate effort.
3. **S5** (hash-based threshold cache key); trivial, eliminates fat cache
   key before it becomes a memory landmine.
4. **S9** (subsample GMM); two-line fix, eliminates a growing cost on every
   retrain.
5. **S2 + S8** (cap/defer diversity tree); makes 100 k+ datasets load
   usably fast; tree can remain available but opt-in.
6. **S3 + S17 + S19** (sparse sort results, lazy ordered items); must be
   done together; unblocks 1 M+ in the frontend.
7. **S12** (debounce label sync); makes voting with large labelsets not
   stall the UI.
8. **S16** (virtual grid); significant frontend work but required for grid
   mode at any large scale.
9. **S11** (parallel label resolution); required for cross-dataset detectors
   with 10 k+ labels.
10. **S15** (streaming pickle load); required for 10 M+ datasets.

---

## Open follow-ups (not yet scoped for implementation)

- FAISS/HNSW replacement for the diversity tree (S2 long-term)
- Columnar storage for `medias` dict (S4)
- Append-only vote journal for label sync sources (S12 long-term)
- Streaming JSON export (S13)
- Paginated sort API (S3 long-term)
- Background MLP inference with result streaming (S10)
