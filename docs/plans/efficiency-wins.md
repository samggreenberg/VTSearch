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

<!-- item-sep -->

- [ ] #2396 — Vectorize region cosine sort (kill per-region Python loop in `cosine_sort_with_boxes`)

<!-- item-sep -->

## Tier 3 — worthwhile, more design judgment needed

<!-- item-sep -->

- [ ] #2397 — Move `/api/labeling-status` cache advancement off the request thread

<!-- item-sep -->

- [ ] #2398 — Efficiency micro-fixes: redundant lookups, repeated snapshots, per-epoch syncs, timers
