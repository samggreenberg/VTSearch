# Scalability Implementation Plan

**Status:** §1.2 (S9), §2.1 (S2/S8), §2.2 (S12), and §3.3 (S16) shipped; §1.1 (S5)
rejected as a misdiagnosis; §1.3/§1.4 (S6/S7) gated until interactive datasets
approach 1M. Open/next work: §2.3 (S14), §3.1 (S1), §3.2 (S3/S17/S19), §3.4 (S17),
and Phase 4 (§4.1–§4.3) — full design below, followed by the shipped/rejected
one-liners under "What shipped".

**Parent:** [`scalability.md`](scalability.md); the brainstorm that defines all S#
IDs referenced here. **Goal:** Make VTSearch usable at 100k items (near-term) and
survivable at 1M (medium-term) without a full architectural rewrite; 10M requires
Phase 4 work and is deferred. Phases are ordered by leverage-to-effort ratio and are
each shippable independently.

**Target scale (confirmed 2026-06-19, drives what's worth doing):** GUI Train ~50k,
GUI Find ~250k, CLI Find 2M+. A critical re-read of Phase 1 against these numbers
**rejected §1.1** as a misdiagnosis and **gated §1.3/§1.4** as not worth their
correctness risk until interactive datasets actually approach the million-item range
— at 50k–250k the `sorted(medias.keys())` cost they target is 5–80 ms, noise next to
the matmul/argsort. Separately, CLI-specific streaming work (lazy folder enumeration,
per-chunk embed, streaming export — partial fixes for S20/S15/S13 on the
`--autodetect` side) has shipped; see
[`cli-stream-massive-images.md`](cli-stream-massive-images.md).

---

## Open / next work

### 2.3  Incremental secondary lookups on DatasetContext (S14)

**File:** `vtscore/state/core.py` (`DatasetContext`), various route files that rebuild
`md5_to_media` / `origin_key_to_cid` per request.

**Problem:** Many routes rebuild `{m["md5"]: m for m in snap.values()}` (or similar)
on every request; O(N) Python iteration with dict construction.

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

Maintained by the same mutation sites that would bump `_medias_epoch` (§1.3):

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

Routes/helpers that rebuild lookup dicts switch to `ctx._md5_index` /
`ctx._origin_key_index` directly. Medium refactor because mutation sites are spread
across `load_pipeline.py`, `state/__init__.py`, and `state/media_lookup.py`.

**Risk:** medium; secondary indexes must stay in sync with `medias`. Any site that
writes `ctx.medias` directly (bypassing the helper) produces stale indexes. The §1.3
epoch audit surfaces all those sites, so §1.3 should land first.

---

### 3.1  mmap embedding matrix (S1)

**Files:** `vtscore/datasets/loader_pickle.py`, `vtscore/embedding/matrix.py`

**Problem:** Embeddings are stored inline in `media["embedding"]` as Python
lists/arrays, then copied into a contiguous `(N, D)` NumPy array on first access. At
1M items × 384-dim the array is 1.5 GB; it must be rebuilt from per-item entries on
every cold start.

**Fix (two-step):**

**Step A — sidecar `.npy` file.** After all embeddings are present, write the
sorted-by-cid matrix to `<dataset>.emb.npy` (+ companion `<dataset>.cids.npy`, int64
sorted cids) if it doesn't already exist.

**Step B — mmap load.** On pickle load, if both sidecars exist and the cid list
matches the pickle's media IDs:

```python
cids = np.load(cids_path)                  # int64 array
matrix = np.load(emb_path, mmap_mode='r')  # zero-RAM mmap
ctx._emb_matrix = matrix
ctx._emb_matrix_sorted_ids = list(cids)
ctx._emb_matrix_epoch = ctx._medias_epoch
```

The OS pages in only the rows scoring actually accesses (a 100-item sort query on a
1M dataset touches ~40 kB of the 1.5 GB file). `get_embedding_matrix` must detect the
cached-this-way matrix and skip the rebuild.

**Sidecar invalidation:** if the pickle is newer than the sidecar, or the cid list
doesn't match, fall back to the in-memory build (optionally rewrite fresh sidecars).

**Risk:** high; introduces file-system state alongside pickle files. Care needed for:
race between sidecar write and concurrent load; version drift if the embedder changes
(embedding dim changes → sidecar invalid); Docker / read-only filesystems (fall back
to in-memory). A `--no-emb-sidecar` flag (or settings key) disables sidecar writing
where the dataset path is read-only.

---

### 3.2  Sparse sort results: top-K API + lazy frontend (S3, S17, S19)

The largest single change; requires coordinated backend + frontend work landed as a
pair of PRs.

**Backend.** `POST /api/sort/cosine` and `/api/sort/learned` currently return all N
items. Change the response to a windowed shape:

```json
{
  "total": 1000000,
  "threshold": 0.72,
  "above_threshold": 312,
  "results": [{"id": 1, "score": 0.91, ...}, ...],
  "has_more_below": true
}
```

where `results` contains only the top `K_ABOVE` items above threshold plus `K_BELOW`
immediately below (e.g. 500 / 200). Clients fetch more via
`GET /api/sort/page?offset=500&limit=200`. The full sorted order is computed as today
but only a window is transmitted; the backend holds the full list in the existing
`AsyncJob.result`.

**Frontend.** Replace `SortStateService`'s `SortedItem[] | null` with a windowed model:

```typescript
interface SortWindow {
  total: number;
  threshold: number;
  aboveThreshold: number;
  items: SortedItem[];            // the loaded window
  loadedRange: [number, number];  // [start, end] indices
  hasMore: boolean;
}
```

`media-list.component.ts:rebuildOrderedItems` renders whatever is in `items` and shows
a "Load more" trigger at the end; `cachedOrderedItems` stays bounded to the loaded
window (≤700). Virtual scroll already handles a fixed window; this caps the array size
at the API level.

**Complexity:** 2–3 PRs (backend sort API, frontend SortStateService, frontend
media-list). All must land atomically or behind a feature flag.

---

### 3.4  Load Cancel responsiveness during pickle read (S17) — open follow-up

**Files:** `vtscore/datasets/container.py` (`read_container`),
`vtscore/datasets/loader_pickle.py`, `vtscore/datasets/load_pipeline.py`

**Problem (found while investigating the S16 freeze):** the dataset load runs in a
daemon thread and streams SSE progress, so demo-dataset loads stay responsive with a
working Cancel. But on a large `.pkl` two things read as "frozen, Cancel does nothing":

1. **`read_container` is one un-cancellable, no-progress step.** A full ZIP
   `zf.read("medias.pkl")` (DEFLATE of the whole blob) + `safe_pickle_load` of the
   entire dict. While it runs the bar sits at "Reading…" and — because the dev server
   is a single process holding the GIL through that pure-Python work — the Cancel/SSE
   endpoints can't be serviced, so the bar appears stuck and Cancel is inert until it
   finishes.
2. **Cancel is unchecked for the whole import phase.** `_run_importer(...)` runs the
   entire pickle-conversion loop with no `tracker.check_cancelled()` inside; the first
   cancel check is only *after* it returns.

**Fix direction:** add `check_cancelled()` inside the `load_dataset_from_pickle`
conversion loop (it already reports progress every `_progress_interval` items — check
cancel there too), and give the read/decompress phase coarse progress + a cancellation
point (stream the ZIP member, or at minimum surface an honest indeterminate "Reading…"
state). Overlaps with **§4.1 (streaming pickle load)** below.

---

### Phase 4: Major infrastructure (1M+ items)

Deferred until Phase 1–3 are stable.

- **4.1  Streaming pickle load (S15).** Pickle files > 500 MB should load in chunks.
  The embedding-matrix sidecar (§3.1) decouples embeddings from the pickle, so the
  pickle becomes mostly metadata (filenames, origins, md5s) and can load quickly;
  embeddings come from the mmap'd sidecar.
- **4.2  Parallel cross-dataset label population (S11).**
  `populate_label_embeddings` (`vtscore/detectors/labelset_training.py`) resolves and
  embeds each LabelSet element sequentially. Replace with a `ThreadPoolExecutor`
  (bounded by the same `_embed_gate` that governs dataset embedding concurrency);
  dedupe file resolves across elements with the same origin before dispatching.
- **4.3  Streaming JSON label export (S13).** Replace
  `LabelSet.to_dict()` → `flask.jsonify()` with a generator yielding one JSON line per
  element, wrapped in `flask.stream_with_context`. Simplest: add a `?format=ndjson`
  query param and keep the default response identical.

---

## Gated (deferred until 1M-scale interactive use)

### 1.3  Epoch counter for embedding matrix invalidation (S6) — ⏸ GATED

> **Verdict (2026-06-19): defer; not worth the correctness risk at current target
> scale.** The `sorted(ctx.medias.keys())` it removes costs ~5–15 ms at 50k and
> ~30–80 ms at 250k — noise beside the matmul/argsort on the same call. The only
> regime where it matters is CLI Find at 2M+, a *batch* path (latency-insensitive).
> Against that thin payoff, the current check (`cached_ids == sorted_ids`) is
> **self-correcting — it can never serve a stale matrix.** The epoch scheme replaces
> it with a manual invariant every one of ~5 mutation sites must bump; a single miss
> silently serves the **wrong embedding matrix → wrong scores, no error.** Revisit
> only if interactive datasets approach 1M, with the §1.3 test landing first.

**File:** `vtscore/state/core.py` (`DatasetContext`), `vtscore/embedding/matrix.py:58–79`

**Problem:** `get_embedding_matrix` calls `sorted(ctx.medias.keys())` on every
invocation (O(N log N)) to detect a stale cached matrix.

**Fix:** Add an integer `_medias_epoch` to `DatasetContext`, incremented whenever
`medias` is structurally mutated (adds/deletes). The matrix cache stores the epoch at
build time and validates with an O(1) integer compare; `_emb_matrix_ids` is dropped.

```python
__slots__ = (..., "_medias_epoch", "_emb_matrix_epoch", "_emb_matrix")
# __init__: _medias_epoch = 0; _emb_matrix_epoch = -1; _emb_matrix = None
```

All sites that structurally modify `ctx.medias` must increment the epoch:

| Site | File |
|------|------|
| `ctx.medias[cid] = …` (importer write) | `load_pipeline.py` |
| `del ctx.medias[cid]` (None-embedding drop) | `load_pipeline.py:932` |
| `ctx.medias.clear()` | `vtscore/state/__init__.py:134` |
| `collapse_duplicates` (may delete) | `vtscore/state/media_lookup.py` |

Since `medias` is a plain dict with no hooks, a helper `_bump_epoch(ctx)` in `core.py`
keeps this one line per site. On cache hit the return value still needs the id list,
so store `_emb_matrix_sorted_ids` alongside the matrix (the epoch is the validity
signal, not the list comparison) and return the stored list; `sorted(...)` only runs
on cache miss. `get_embedding_matrix_for_snap` reuses the epoch path on the ctx
branch; the temp-dict/cross-dataset branch still sorts once (no epoch for an ad-hoc
dict).

**Risk:** medium; touches `__slots__` and several mutation sites. Must be covered by
tests that verify the matrix rebuilds exactly when medias change.

---

### 1.4  Epoch-based learned-sort signature (S7) — ⏸ GATED (depends on 1.3)

> **Verdict (2026-06-19): defer with 1.3.** Same `sorted(snap.keys())` cost profile
> (negligible ≤250k), and it piggybacks on §1.3's epoch counter, so it inherits the
> same gating and staleness risk. The signature builder now lives in
> `vtscore/detectors/learned_sort.py:build_learned_sort_signature` (the line
> reference below is stale). Note learned-sort is the GUI **Train** path (~50k) — the
> regime where this optimization matters least.

**File:** `vtsearch/routes/sorting.py:340–358` (`_build_learned_sort_signature`)

**Problem:** `tuple(sorted(snap.keys()))` (O(N log N)) and
`tuple(sorted(region_boxes_snapshot.items()))` (O(R log R)) appear inside the
signature checked before every learned-sort job.

**Fix:** Replace `tuple(sorted(snap.keys()))` with `("epoch", ctx._medias_epoch)`;
replace the region-box sort with a frozen-set of region IDs.

```python
epoch = get_active_context()._medias_epoch
region_sig = frozenset(region_boxes_snapshot.keys())
sig = (("epoch", epoch), ("votes", ...), ("regions", region_sig), ("inclusion", ...), ...)
```

Vote sets are small, so sorting them stays cheap.

**Risk:** low; the signature is only a cache key — a false miss wastes one retrain; a
false hit can't happen because the epoch bumps only on structural change.

---

## Open follow-ups

- Frontend "Build diversity index" button (deferred from §2.1 Part B): surface a
  trigger for `POST /api/datasets/registry/<id>/diversity-tree` when a loaded dataset
  has no tree. No existing dataset/tree-status panel hosts it today, so it needs a UI
  home; the endpoint is meanwhile reachable via API/CLI.
- FAISS / HNSW replacement for diversity tree (long-term S2 fix).
- Columnar `medias` storage (S4): deferred; requires redesign of every media-reading
  call site.
- Append-only vote journal for labelset sources (S12 long-term): deferred until
  compaction semantics are defined.
- CLI progress bar rate-limiting (S21): trivial, add when touching CLI.

## Test coverage checklist (open items)

- **1.3**: matrix rebuilt exactly when medias change, reused when they don't; epoch
  check is O(1).
- **1.4**: learned-sort job not re-fired when called twice with the same votes on the
  same epoch.
- **3.1**: load, unload, reload a dataset; matrix re-used from sidecar on second load
  without rebuilding from per-item entries.
- **3.2**: sort response with 200k items contains ≤700 results;
  `/api/sort/page?offset=700&limit=200` returns the next window.

---

## What shipped

- **§1.2  Subsample GMM threshold (S9)** — shipped 2026-06-19. Subsampling lives
  inside `calculate_gmm_threshold` (single point, so every caller benefits), gated by
  `_GMM_MAX_SAMPLES = 50_000` with a seed-42 `default_rng`; the exception-fallback
  median is bounded to the subsample too. The one Phase-1 item with a real,
  scale-justified payoff (the GMM fits the full score distribution, 250k–2M+).
  `tests_lib/sorting/test_gmm_subsample.py`.
- **§2.1  Cap and defer diversity tree construction (S2, S8)** — reload caching
  shipped 2026-06-19 (commit 64cccd5e, PR #1987): the tree is serialized into the
  pickle and restored via `restore_diversity_tree_from_cache` (adopted only when its
  vector set exactly matches the loaded medias, else rebuild). Part A + Part B shipped
  2026-06-25 (backend): `_n_init_for(node_size)` scales k-means restarts down for
  large nodes and `auto_max_depth(...)` caps depth at `_MAX_LEAVES=4000`
  (`vtscore/state/diversity_tree.py`), both structurally no-op on normal/test-sized
  trees; `should_auto_build_diversity_tree(n)` (threshold
  `DIVERSITY_TREE_AUTO_THRESHOLD=50_000`) gates the auto-build, and
  `POST /api/datasets/registry/<id>/diversity-tree` builds on demand in a cancellable
  job. Deferred: the frontend "Build diversity index" button (see Open follow-ups).
- **§2.2  Debounce label sync writes (S12)** — already implemented in
  `vtscore/labels/sync.py`: `sync_to_labelset_source` schedules a per-detector
  `threading.Timer` (`_DEBOUNCE_DELAY = 0.2s`, keyed by `detector_id`), so a rapid
  voting burst collapses into one background push; `flush_pending_label_syncs` drains
  synchronously for tests/shutdown, with an `atexit` flush. 200ms (vs the 2s sketch)
  keeps on-disk labels within a UI tick while coalescing bursts.
- **§3.3  Virtual grid mode (S16)** — shipped. `media-list/` chunks
  `cachedOrderedItems` into fixed-width rows through a `CdkVirtualScrollViewport`
  (one virtual item = one row of `gridColumns` cards; columns from measured viewport
  width, row stride from a real card via `ResizeObserver`; grid-aware
  `scrollToIndex`/selected-scroll/prefetch). List-virtualization threshold dropped
  500 → 150; grid virtualizes above 80. Result: list↔grid toggle ~169 ms (was ~1129),
  grid switch ~18 ms (was ~1814), ~20–34 DOM components instead of 412. (Surfaced
  §3.4/S17 as the deferred backend cancel/progress follow-up, still open above.)
- **§1.1  Hash-based threshold cache key (S5)** — ❌ **REJECTED** (misdiagnosis):
  conflates training-set size with dataset size. `X_list`/`y_list` are labeled
  examples (votes + labelset elements), not the dataset, so the "150 MB at 100k"
  figure never occurs; the cache is a single overwritten slot, not a growing
  structure; and the proposed fix would *add* `np.stack(...).tobytes()` compute on the
  comparison path to save a few MB on a non-bottleneck. Revisit only if labelsets ever
  grow to tens of thousands of elements (a different feature).
