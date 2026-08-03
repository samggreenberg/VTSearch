# VTSearch Scalability Plan

**What this is:** The single scalability plan — the former brainstorm catalog and
its separate implementation plan, now consolidated into one file. It catalogs
what breaks, slows, or explodes in memory as datasets grow, defines the
stable `S#` IDs referenced from PRs and from
[`cli-stream-massive-images.md`](cli-stream-massive-images.md), and records the
fix direction + implementation sketch for each item still owed.

**Scope:** What breaks as datasets grow to 100 k / 1 M / 10 M items and as
LabelSets grow to 1 k / 10 k / 100 k labels, GUI and CLI. Items track **future
work only**: shipped items (S1 mmap embedding-matrix sidecar, S2/S8
coverage-atlas auto-defer, S9 GMM subsample, S14 cached secondary lookups, S16
grid virtual scroll, S18 prefetch cap, S21 CLI progress throttle) have been
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

**This is not a faithful "same UX, just windowed" swap.** A code trace (2026-07)
found many Label/Find surfaces that assume the client holds the *entire* ranked
order. Each needs an explicit decision or a server-assisted redesign — the three
items alone don't cover them:

- **Bulk actions ship the full id list to the backend.** In Find,
  `unverifiedGoodIds()` / `goodIds()` derive the whole above-cutoff id set
  *client-side* and hand it to **Browse**, **To Dataset (promote)**, and
  **Export**. If `above_threshold` exceeds the window's `K_ABOVE`, these
  silently truncate — a correctness bug, not a perf one. Fix: a server-side
  "operate on all-above-threshold" path (the ids come from the backend's full
  list, keyed by the sort-generation token below), not from the client window.
- **Find boundary walk** (`advanceToBoundary`) scans the whole unverified order
  for the nearest unverified item on each side of the cutoff; a window edge
  makes it falsely report "All items reviewed." Needs a server endpoint that
  returns the next unverified item above/below a cutoff, or a guarantee that the
  window always straddles the boundary with slack.
- **Diversity "New" select mode** (`fetchDiversityNext`) POSTs the entire
  `{id: score}` map to `/api/coverage-atlas/next`; a partial window degrades the
  direction signal. Send only the ids the server needs, or move the score lookup
  server-side.
- **`best_region` must stay in the window shape.** Region-aware datasets
  (DINOv2/v3) read it from `sortOrder` for the center-panel overlay
  (`center-panel.ts`); the example JSON above drops it. Keep `best_region` on
  each windowed result. The overlay still can't paint for an item outside the
  loaded window (e.g. after a stripe jump) — acceptable, but note it.
- **Server-side list lifetime.** The sketch says "the backend holds the full
  list in the existing `AsyncJob.result`," but only *learned-sort* is a job.
  **Text-sort and example-sort are synchronous** (`sort_clips`, `example_sort`
  return inline). Paging them means new per-session cached sorted lists with an
  eviction policy, and the ~50 MB @1 M list now stays resident *server-side*
  (× concurrent users) instead of being handed off and GC'd.
- **`/api/sort/page` needs a sort-generation token.** A retrain / re-sort between
  the initial response and a page fetch shifts offsets → duplicate/missing rows.
  Return a generation id and require it on the page URL; a stale token 409s so
  the client refetches from the top.
- **Inclusion slider** currently just moves the line over frozen client-side
  scores (`onInclusionChange`) and recomputes `above_threshold` locally. Under a
  window, moving the cutoff changes which items are "above," so the slider must
  refetch the window (scores stay frozen server-side, so no re-sort — just a new
  slice + count).

**Decisions (2026-07, from the S3/S17/S19 evaluation):**

- **"Bottom" select mode is dropped.** The picker only ever offered Top / Hard /
  New; `'bottom'` was a dead `SelectMode` variant + one branch in
  `autoSelectNext`. Removed, so "walk from the end of the full order" is no
  longer a constraint the window must satisfy. (Landed as the first slice.)
- **The stripe minimap is gated above a large-sort size, not made windowed.**
  The stripe is a full-order minimap by definition; it can't be honestly drawn
  from a window. Above `STRIPE_MAX_ITEMS` it renders a disabled strip with a
  clear tooltip ("Minimap unavailable for large sorts") instead of a wrong
  picture. This also kills its O(N) dot-build loop at scale. (Landed as the
  first slice; the threshold is the natural home for the future window's own
  cutoff.)

**The Train-side windowing has shipped end-to-end.** The sort routes
(`/api/sort`, `/api/example-sort`, `/api/label-file-sort`, `/api/learned-sort`)
window their transmitted `results` above `SORT_WINDOW_THRESHOLD` (aligned with
`STRIPE_MAX_ITEMS`; below it the full ranking is sent unchanged), the frontend
`SortStateService` holds a windowed model, and `media-list` renders the loaded
window + a "Load more" trigger that pages via `GET /api/sort/page`. `best_region`
rides each windowed row. The stripe was already size-gated (slice 1), so the
Train (label-view) flow has no full-order client scan left.

**Server-side Find contract (built, not yet consumed):**

- **Find work-queue ids** — `GET /api/find/queue-ids?filter=unverified_good|good`
  returns the full positive-set ids (rank order) for Browse / To Dataset /
  Export, from frozen `find_scores` + cutoff + verified set.
- **Find boundary walk** — `GET /api/find/boundary-next?side=above|below[&exclude=]`
  returns the next unverified item on the requested face of the cutoff.

**Remaining work — window the Find (find-view) flow:**

`find-label` still returns the full `results` list, because Find's `find-view`
derives its work queue, "just sit and vote" boundary walk, and Browse / To
Dataset / Export id sets from the whole client-side ranking. To window it:

- Window `find-label`'s response (same `windowed_sort_response` helper) and have
  `find-view` install it via `setSortWindow` + wire the media-list "Load more"
  (paging the *unverified* ranking, filtering verified rows out of each page).
- Switch the bulk actions (`unverifiedGoodIds` / `goodIds`) to
  `GET /api/find/queue-ids` and the boundary walk (`advanceToBoundary`) to
  `GET /api/find/boundary-next` — both already built and tested. These are async
  refactors of `find-view`'s currently-synchronous handlers, so land them with
  the windowing atomically (a windowed `find-label` without them would truncate
  Browse/Export and mis-terminate the boundary walk).
- Feed the left-panel `unverifiedGoodCount` from the response's `above_threshold`
  (minus verified) rather than a full-order scan.

---

### S4: `medias` dict — one Python dict entry per item

**File:** `vtscore/state/core.py` (`DatasetContext.medias`, a `MediasDict`)

Each item is a Python `dict` with ~10 keys, embeddings stored inline
(`loader_pickle._build_pickle_full_media` writes `"embeddings"` into each entry).
Dict overhead is ~250 B/entry; at 1 M items the `medias` dict alone consumes
several hundred MB before the embedding arrays.

**Fix (medium-term, long horizon):** Extract embeddings from the per-item dict
into the embedding matrix on load (mirroring the shipped mmap embedding-matrix
sidecar), replacing `media["embedding"]` with a row-index pointer — ~60–70%
smaller per-item dict at 384-dim. The full columnar rewrite (per-field NumPy
arrays or a Polars frame) is deferred: it touches every media-reading call site.

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
  load in chunks. Note the shipped mmap embedding-matrix sidecar does *not* by
  itself shrink this: `medias.pkl` still carries every item's embeddings inline
  (the sidecar is an additional, redundant mmap-able copy used only to skip
  rebuilding the `(N, D)` matrix cache). Making the pickle itself "mostly
  metadata" needs S4's per-item-dict rewrite first (strip `embeddings` from each
  entry in favor of a row-index pointer) — only then does a pickle load stop
  paying to deserialize the embedding bytes at all.
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
- **Share the coverage atlas with the progress cache instead of rebuilding it:**
  `labeling_progress._build_coverage_atlas` clones the dataset context's atlas
  only when the id sets match exactly; otherwise it fits a throwaway private one
  under `_progress_lock`. On a dataset past the 50 k auto-build cutoff (so the
  context has no atlas) that fit dominates a cold progress-cache build —
  measured ~27 s of a 40 s build at 20 k media. It can't write the result back
  to the context from there because `_progress_lock` is held and the lock
  ordering forbids taking `_state_lock` under it. Fix direction: build the atlas
  before acquiring `_progress_lock` and publish it to the context so both paths
  share one structure. Now off the request thread (the plot modal runs it as a
  background job), so this is a cost problem, not a latency-hang.
- **Columnar `medias` storage** (S4 full rewrite): deferred; requires redesign of
  every media-reading call site.
- **Append-only vote journal** for labelset sources (S12 long-term): deferred
  until compaction semantics are defined.

---

## Recurring root causes

| Root cause | Open items it affects |
|-----------|-----------------------|
| **O(N log N) sorted-key comparisons** used as change detection | S6, S7 |
| **Full in-memory arrays / dicts** for every N items | S3, S4, S17, S19 |
| **No streaming** for large JSON / pickle payloads | S3, S13, S15 |
| **Serial I/O** where parallelism is easy | S11, S20 |
| **No debouncing** on high-frequency write paths | S12 |

---

## Test coverage checklist (open items)

- **S3/S17/S19:** sort response with 200 k items contains ≤700 results;
  `/api/sort/page?offset=700&limit=200` returns the next window.
- **S6:** matrix rebuilt exactly when medias change, reused when they don't; O(1)
  on cache hit (no per-call `sorted()`).
- **S7:** learned-sort job not re-fired when called twice with the same votes on
  the same `media_revision`.

---

## Related docs

- [`cli-stream-massive-images.md`](cli-stream-massive-images.md) — implements the
  CLI-specific pieces of **S15** (lazy enumeration), **S20** (chunked scoring),
  and **S13** (streamed export) for `--autodetect`.
