# VTSearch Scalability Plan

**What this is:** The single scalability plan — the former brainstorm catalog and
its separate implementation plan, now consolidated into one file. It catalogs
what breaks, slows, or explodes in memory as datasets grow, defines the
stable `S#` IDs referenced from PRs and from
[`cli-stream-massive-images.md`](cli-stream-massive-images.md), and records the
fix direction + implementation sketch for each item still owed.

**Scope:** What breaks as datasets grow to 100 k / 1 M / 10 M items and as
LabelSets grow to 1 k / 10 k / 100 k labels, GUI and CLI. Items track **future
work only**: shipped items (S2/S8 coverage-atlas auto-defer, S9 GMM subsample,
S16 grid virtual scroll, S18 prefetch cap, S21 CLI progress throttle) have been
pruned per the plan-file policy — git history is their record. `S#` numbering
has gaps where those were removed; that is expected (labels are stable, never
renumbered).

**Target scale (confirmed 2026-06-19 — drives what's worth doing):** GUI Train
~50 k, GUI Find ~250 k, CLI Find 2 M+. This is why **S6/S7 stay gated**: the
`sorted(medias.keys())` cost they target is 5–80 ms at 50 k–250 k — noise beside
the matmul/argsort on the same call — and only matters in CLI Find at 2 M+, a
latency-insensitive batch path.

**Goal:** Usable at 100 k (near-term), survivable at 1 M (medium-term) without a
full architectural rewrite. 10 M requires the Phase-4 infrastructure work and is
deferred. Items are independently shippable.

---

## Suggested work order (open items, max-leverage first)

1. **S14** (incremental secondary indexes on `DatasetContext`); removes per-request
   O(N) dict rebuilds; medium refactor.
3. **S1** (mmap embedding matrix); unblocks 1 M+ datasets without OOM. Also
   decouples embeddings from the pickle, which unblocks S15.
4. **S3 + S17 + S19** (sparse sort results, lazy ordered items); must be done
   together; unblocks 1 M+ in the frontend.
5. **S12** (debounce the detector-JSON label-sync write); makes voting with large
   labelsets not stall the UI.
6. **S11** (parallel label resolution); required for cross-dataset detectors with
   10 k+ labels.
7. **S13** (stream the GUI label-export route); the CLI side already streams.
8. **S15** (streaming pickle load + cancel-check); required for 10 M+ datasets.

**Gated until interactive datasets approach 1 M:** S6, S7 (see below).

---

## How to read the catalog

- Items are grouped by the *resource* they exhaust: **RAM**, **CPU/time**, or
  **frontend/CLI** — ordered roughly by severity.
- Each item has a stable ID (`S#`) for reference from PRs.
- The **recurring root causes** table (bottom) collects patterns: fixing a
  pattern fixes many items at once.
- "At N" estimates are back-of-envelope with SigLIP/E5 embedding dim ~384
  (float32 = 4 B), sorted-list cost at ~1 µs/item, JSON at ~50 B/element. Line
  references are approximate and will drift.

---

## RAM

### S1: mmap embedding matrix — one giant contiguous array per loaded dataset

**Files:** `vtscore/embedding/matrix.py`, `vtscore/datasets/loader_pickle.py`

All embeddings for the loaded dataset are materialised into a single `(N, D)`
`float32` array (`matrix.py:_stack_embeddings`, `np.empty((N, D))`), held on
`DatasetContext`, and rebuilt from per-item entries on every cold start.

| N | D=384 | D=768 (E5) |
|---|-------|-----------|
| 100 k | 150 MB | 300 MB |
| 1 M | 1.5 GB | 3 GB |
| 10 M | 15 GB | 30 GB |

One loaded dataset already risks OOM; multi-dataset mode doubles it. The array is
unavoidable for vectorised scoring, but it does not need to live in RAM for the
dataset's full lifetime — at 1 M+ it should be **memory-mapped**.

**Fix (two-step):**

- **Step A — sidecar `.npy`.** After all embeddings are present, write the
  sorted-by-cid matrix to `<dataset>.emb.npy` (+ companion `<dataset>.cids.npy`,
  int64 sorted cids) if it doesn't already exist.
- **Step B — mmap load.** On pickle load, if both sidecars exist and the cid list
  matches the pickle's media IDs:

  ```python
  cids = np.load(cids_path)                  # int64 array
  matrix = np.load(emb_path, mmap_mode='r')  # zero-RAM mmap
  ctx._emb_matrix = matrix
  ctx._emb_matrix_ids = list(cids)
  ctx._emb_matrix_revision = ctx.media_revision
  ```

  The OS pages in only the rows scoring accesses (a 100-item sort on a 1 M dataset
  touches ~40 kB of a 1.5 GB file). `get_embedding_matrix` must detect the
  cached-this-way matrix and skip the rebuild.

**Sidecar invalidation:** if the pickle is newer than the sidecar, or the cid list
doesn't match, fall back to the in-memory build (optionally rewrite sidecars).

**Risk:** high — introduces filesystem state alongside pickle files. Care for:
race between sidecar write and concurrent load; embedder-dim drift (dim change →
sidecar invalid); Docker / read-only filesystems (fall back to in-memory). A
`--no-emb-sidecar` flag (or settings key) disables writing where the path is
read-only.

**Sidecar cleanup on delete/expiry is already covered.** `unregister_dataset`
(`vtscore/datasets/registry.py`) deletes every file sharing the pkl's stem, not
just the pkl itself — a `<dataset>.emb.npy` / `<dataset>.cids.npy` sidecar named
`ds_<uuid>.emb.npy` next to `ds_<uuid>.pkl` is swept automatically on both the
age-off and manual-delete paths (both route through `unregister_dataset`). No
extra registry field or bookkeeping is needed as long as the sidecar filename
keeps the pkl's stem as a prefix.

---

### S3 / S17 / S19: sparse sort results — top-K API + lazy frontend

**Files:** `vtsearch/routes/sorting.py`, `vtscore/detectors/learned_sort.py`,
`frontend/src/app/services/sort-state.service.ts`,
`frontend/src/app/components/left-panel/media-list/media-list.component.ts`

The largest single change; needs coordinated backend + frontend work. These three
items must land together (behind a flag or atomically).

**Today:** every sort API call returns `results: [{id, score, bestRegion}, …]` for
the entire dataset in one JSON response (`sorting.py` `_cosine_sort`,
`learned_sort` both build the full `results` list). The frontend keeps the full
array: `SortStateService._sortOrder` is a `signal<SortedItem[] | null>`, and
`media-list.component.ts:rebuildOrderedItems` iterates the full `sortOrder` on
every result to build `cachedOrderedItems` — an O(N) JS loop that freezes the tab
at 1 M.

| N | JSON size (est.) |
|---|-----------------|
| 10 k | 0.5 MB |
| 100 k | 5 MB |
| 1 M | 50 MB |
| 10 M | 500 MB |

**Backend fix.** Change the response to a windowed shape:

```json
{
  "total": 1000000,
  "threshold": 0.72,
  "above_threshold": 312,
  "results": [{"id": 1, "score": 0.91}],
  "has_more_below": true
}
```

`results` carries only the top `K_ABOVE` items above threshold plus `K_BELOW`
immediately below (e.g. 500 / 200). Clients fetch more via
`GET /api/sort/page?offset=500&limit=200`. The full sorted order is computed as
today but only a window is transmitted; the backend holds the full list in the
existing `AsyncJob.result`.

**Frontend fix.** Replace `SortStateService`'s array with a windowed model:

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

`rebuildOrderedItems` renders whatever is in `items` and shows a "Load more"
trigger at the end; `cachedOrderedItems` stays bounded to the loaded window
(≤700). Grid/list virtual scroll (shipped) already handles a fixed window; this
caps the array size at the API level.

**Complexity:** 2–3 PRs (backend sort API, frontend SortStateService, frontend
media-list).

---

### S4: `medias` dict — one Python dict entry per item

**File:** `vtscore/state/core.py` (`DatasetContext.medias`, a `MediasDict`)

Each item is a Python `dict` with ~10 keys, embeddings stored inline
(`loader_pickle._build_pickle_full_media` writes `"embeddings"` into each entry).
Dict overhead is ~250 B/entry; at 1 M items the `medias` dict alone consumes
several hundred MB before the embedding arrays.

**Fix (medium-term, long horizon):** Extract embeddings from the per-item dict
into the embedding matrix on load (S1), replacing `media["embedding"]` with a
row-index pointer — ~60–70% smaller per-item dict at 384-dim. The full columnar
rewrite (per-field NumPy arrays or a Polars frame) is deferred: it touches every
media-reading call site.

---

## CPU / time

### S6: epoch counter for embedding-matrix invalidation — ⏸ GATED (partly shipped)

**File:** `vtscore/state/core.py`, `vtscore/embedding/matrix.py`

> **Verdict (re-confirmed): defer.** The invalidation half already shipped safely
> — `media_revision`, bumped automatically by the `MediasDict` subclass on every
> structural change (`core.py:261+`), keys the matrix cache's *validity* via an
> O(1) revision compare (`matrix.py:128`), and the original O(N) id-list
> comparison is gone. Crucially, the auto-bumping `MediasDict` removes the
> staleness risk the earlier gating flagged (a manual invariant every mutation
> site must remember to bump): there is no manual bump to miss.
>
> **What remains** is only the perf tail: `get_embedding_matrix` still calls
> `sorted(ctx.medias.keys())` on *every* call (`matrix.py:116`), before the
> cache-hit return, to produce the returned id list — the O(N log N) the item
> targeted is not eliminated. The fix is small: on a revision cache-hit, return
> the stored `_emb_matrix_ids` instead of re-sorting. But at target scale the
> saving is 5–80 ms on a call dominated by the matmul, so it stays gated until
> interactive datasets approach 1 M. Land the "rebuilds exactly when medias
> change; reused when they don't; O(1) on hit" test with the change.

---

### S7: epoch-based learned-sort signature — ⏸ GATED (with S6)

**File:** `vtscore/detectors/learned_sort.py:143,148` (`build_learned_sort_signature`)

`tuple(sorted(snap.keys()))` (O(N log N)) and
`tuple(sorted(region_boxes_snapshot.items()))` appear inside the signature checked
before every learned-sort job. Same cost profile as S6 (negligible ≤250 k), and
learned-sort is the GUI **Train** path (~50 k) — the regime where this matters
least.

**Fix:** Replace `tuple(sorted(snap.keys()))` with `("epoch", ctx.media_revision)`
and the region-box sort with `frozenset(region_boxes_snapshot.keys())`. **Risk:**
low — the signature is only a cache key; a false miss wastes one retrain, and a
false hit can't happen because `media_revision` bumps on every structural change.

---

### S10: MLP forward pass — O(N) inference on every retrain

**File:** `vtscore/detectors/training.py` (`_score_all_media`, called on every retrain)

Scoring all media requires an MLP forward pass for every item after every vote.
Even a tiny MLP at N=1 M takes several hundred ms; at N=10 M, ~1 s per retrain.

**Fix:** For very large datasets, **score only items near the threshold**
(confidence-weighted sampling) and return approximate results. Alternatively,
**debounce** retraining so it doesn't fire on every single vote — accumulate a few
votes and retrain once.

---

### S11: cross-dataset label population — serial I/O, O(N_labels)

**File:** `vtscore/detectors/labelset_training.py` (`populate_label_embeddings`,
serial `for idx, elem in …` loop)

Resolves the file and embeds each LabelSet element serially — for uncached
elements this is serial I/O + model inference per element.

| N_labels | at 50 ms/label |
|----------|---------------|
| 1 k | ~50 s |
| 10 k | ~8 min |
| 100 k | ~83 min |

**Fix:** Parallelise the resolution/embedding loop with a `ThreadPoolExecutor`,
bounded by the same `_embed_gate` that governs dataset embedding concurrency.
Dedupe file resolves across elements with the same origin before dispatching. Also
merge the double-walk (populate then build_xy) into a single pass.

---

### S12: label sync — full detector-JSON rewrite on every vote

**Files:** `vtscore/detectors/label_sync.py`, `vtscore/labels/sync.py`

**Partly shipped:** the external-source push path *is* debounced and async —
`labels/sync.py` has `_DEBOUNCE_DELAY = 0.2` and a timer-based `_PendingSync`.

**Still open:** `label_sync.sync_labels_to_loaded_detector` still does a full
read-merge-write of the detector JSON per vote (under `_label_sync_write_lock`).
At 100 k labels the serialised labelset is ~50–100 MB; writing it on every vote
stalls the handler for seconds.

**Fix:** Debounce the detector-JSON write the same way the source push is
debounced (accumulate votes ~2 s, flush once). Long-term, an **append-only
journal** (one `{id, label}` line per vote) makes the per-vote write O(1)
regardless of labelset size; a compaction step on load reconstructs the full
labelset. (Journal deferred until compaction semantics are defined.)

---

### S14: incremental secondary lookups on `DatasetContext`

**File:** `vtscore/state/core.py`, plus routes that rebuild lookup dicts
(`vtscore/state/media_lookup.py`, `vtsearch/routes/labels/vote.py:144` does
`{m["md5"]: m for m in all_medias.values()}` per request)

Many routes rebuild `{m["md5"]: m for m in snap.values()}` (or origin-key variants)
on every request — O(N) Python iteration + dict construction, 2–3 passes/request,
~100 ms each at 1 M.

**Fix:** Add maintained secondary indexes to `DatasetContext`:

```python
class DatasetContext:
    __slots__ = (..., "_md5_index", "_origin_key_index")
    # __init__: self._md5_index = {}; self._origin_key_index = {}
```

Maintain them at the same structural-mutation sites that bump `media_revision`
(the `MediasDict` add/remove hooks are the natural home). Routes/helpers that
rebuild lookup dicts switch to `ctx._md5_index` / `ctx._origin_key_index`
directly.

**Risk:** medium — indexes must stay in sync with `medias`. Any site that writes
`ctx.medias` directly must go through the maintained path; the `MediasDict` hook
already centralises structural mutations, so hang the index maintenance there.

---

### S15: dataset pickle loading — everything into RAM at once

**Files:** `vtscore/datasets/loader_pickle.py`, `vtscore/datasets/container.py`
(`read_container`), `vtscore/datasets/load_pipeline.py`

Pickle files load one-shot: `read_container` does a full `zf.read("medias.pkl")` +
`safe_pickle_load` of the whole dict. At 10 M items with 384-dim float32
embeddings the pickle is ~15 GB; loading it in one shot is not feasible on typical
hardware. The chunked variant (`load_dataset_from_pickle_chunked`) still
deserialises the whole pickle once (its own docstring notes this is
"unavoidable").

**Two coupled problems:**

- **Streaming load.** Pickles > some threshold (e.g. 500 MB / 100 k items) should
  load in chunks. The embedding-matrix sidecar (S1) decouples embeddings from the
  pickle, so the pickle becomes mostly metadata (filenames, origins, md5s) and
  loads quickly; embeddings come from the mmap'd sidecar.
- **Cancel responsiveness during read (found while investigating the grid
  freeze).** On a large `.pkl` two things read as "frozen, Cancel does nothing":
  (1) `read_container` is one un-cancellable, no-progress step, and the
  single-process dev server holds the GIL through that pure-Python work so the
  Cancel/SSE endpoints can't be serviced; (2) the pickle-conversion loop
  (`load_dataset_from_pickle`) runs with **no** `tracker.check_cancelled()` inside
  — the first cancel check is only after it returns. Fix: add `check_cancelled()`
  in the conversion loop (it already reports progress every `_progress_interval`
  items — check cancel there too), and give the read/decompress phase coarse
  progress + a cancellation point (stream the ZIP member, or at minimum surface an
  honest indeterminate "Reading…" state).

**Note:** the *folder* importer's chunked CLI path already enumerates files lazily
(no full file list in RAM); see
[`cli-stream-massive-images.md`](cli-stream-massive-images.md). This item is about
the *pickle* loader, which still reads the whole file at once.

---

## CLI

### S20: CLI autodetect scores every item, serial label resolution per detector

**Files:** `vtscore/cli.py` (`_train_detectors_for_first_chunk`, serial
`for det_name in detector_names:` loop), `vtsearch/routes/detectors/scoring.py`

**Partly shipped** (see [`cli-stream-massive-images.md`](cli-stream-massive-images.md)):
`--autodetect --chunk-size N --stream-results` scores chunk by chunk and streams
hits straight to the exporter, so the *target* side no longer holds all N items,
all hits, or the full export in RAM; folder enumeration is lazy; each chunk is
embedded one at a time.

**Still open:** per-detector *label* resolution is serial. With 10 detectors each
having 1 000 unresolved labels at 50 ms/label: **500 s just for resolution**,
before any inference.

**Fix:** Resolve all detectors' labels in parallel (thread pool). Bundle duplicate
file resolves across detectors. Use batch embedding for labels from the same
embedder. Shares the S11 approach.

---

## Open follow-ups (not yet scoped for implementation)

- **FAISS / HNSW replacement for the coverage atlas** (long-term S2 fix): an
  approximate-nearest-neighbour structure supporting the same "next unseen
  cluster" query but storable mmap'd.
- **Frontend "Build coverage atlas" button:** surface a trigger for
  `POST /api/datasets/registry/<id>/coverage-atlas` when a loaded dataset has no
  tree (auto-build is skipped above 50 k). No dataset/tree-status panel hosts it
  today, so it needs a UI home; the endpoint is meanwhile reachable via API/CLI.
- **Columnar `medias` storage** (S4 full rewrite): deferred; requires redesign of
  every media-reading call site.
- **Append-only vote journal** for labelset sources (S12 long-term): deferred
  until compaction semantics are defined.

---

## Recurring root causes

| Root cause | Open items it affects |
|-----------|-----------------------|
| **O(N log N) sorted-key comparisons** used as change detection | S6, S7 |
| **Full in-memory arrays / dicts** for every N items | S1, S3, S4, S17, S19 |
| **No streaming** for large JSON / pickle payloads | S3, S13, S15 |
| **Serial I/O** where parallelism is easy | S11, S20 |
| **No debouncing** on high-frequency write paths | S12 |
| **Per-request rebuild** of secondary lookups | S14 |

---

## Test coverage checklist (open items)

- **S1:** load, unload, reload a dataset; matrix re-used from sidecar on second
  load without rebuilding from per-item entries.
- **S3/S17/S19:** sort response with 200 k items contains ≤700 results;
  `/api/sort/page?offset=700&limit=200` returns the next window.
- **S6:** matrix rebuilt exactly when medias change, reused when they don't; O(1)
  on cache hit (no per-call `sorted()`).
- **S7:** learned-sort job not re-fired when called twice with the same votes on
  the same `media_revision`.
- **S14:** `_md5_index` / `_origin_key_index` stay in sync across add/remove/reload;
  routes read them instead of rebuilding.

---

## Related docs

- [`cli-stream-massive-images.md`](cli-stream-massive-images.md) — implements the
  CLI-specific pieces of **S15** (lazy enumeration), **S20** (chunked scoring),
  and **S13** (streamed export) for `--autodetect`.
