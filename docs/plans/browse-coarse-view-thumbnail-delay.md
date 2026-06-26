# Browse coarse-view thumbnail load delay (~3s grayscale bins)

**Status:** Investigation only — root cause hypothesized from code reading, NOT
yet confirmed against a running app. The next contributor should **confirm via
the DevTools Network tab first** (see "How to confirm"), then pick a fix.

## Symptom

When the VTSBrowse canvas first loads (the zoomed-out "top" overview), there's a
solid ~3 seconds where only the grayscale/density-shaded hex bins are visible
before any imagery paints in. The reporter's dataset shows only ~30 bins in that
top view, so "hundreds of requests vs. the 6-connection limit" does **not**
explain it — 30 small thumbnails through 6 connections would be ~250ms.

The bins themselves paint instantly: they're drawn purely from tile cell data
(`count` → colormap), no images involved. The delay is entirely the imagery
filling those bins.

## Primary hypothesis: the coarse view fetches full-res `/image`, not `/thumbnail`

The browse canvas has a resolution-tier switch. When a hex is drawn wider than
the thumbnail's native cap (384 device-px), it stops fetching the capped
`/thumbnail` (which would just upscale a blurry 384px bitmap) and fetches the
**full-resolution `/image`** instead.

- `frontend/src/app/components/browse-canvas/browse-canvas.component.ts:320-322`
  — `useFullResThumbs` getter:
  ```ts
  private get useFullResThumbs(): boolean {
    return 2 * this.targetRadius * this.dpr > BrowseCanvasComponent.THUMB_NATIVE_MAX_DIM; // 384
  }
  ```
- `browse-canvas.component.ts:84` — `THUMB_NATIVE_MAX_DIM = 384`.
- `browse-canvas.component.ts:1428` — the per-cell fetch picks the endpoint:
  ```ts
  const endpoint = this.thumbsAreFullRes ? 'image' : 'thumbnail';
  img.src = this.activeContext.mediaUrl(`/api/medias/${representativeId}/${endpoint}`);
  ```
- `getThumb()` (`:1405-1432`) is called from `drawHex` (`:1233`) on the **first**
  `draw()` — there's no settle/debounce gating it, so the requests fire the
  moment the bins paint.

Why the *top* view specifically is the worst case: it's the coarsest pyramid
level, so the fewest bins cover the whole canvas and each hex is **largest**. The
switch trips when `targetRadius > THUMB_NATIVE_MAX_DIM / (2 * dpr)`. On a Retina
display (`dpr = 2`) that's just `targetRadius > 96` CSS-px — which ~30 bins
spread across a normal-width canvas easily exceeds. So every coarse-view cell
requests `/image`.

What `/image` serves for an image-type item: the **original source bytes with no
downscaling** —
- `vtsearch/routes/media/list.py:501-513` (`media_image`) →
- `vtsearch/routes/media/list.py:457-498` (`_resolve_display_image`) → for image
  types, returns `_resolve_bytes(c)` verbatim (the raw file bytes).

So if the source images are a few MB each, the coarse view is transferring and
full-decoding ~30 multi-MB originals — that's the ~3s. This is the literal answer
to the reporter's "(these are just thumbnails, not full images, right?)": at this
particular zoom, **no — it's fetching full images.**

Corollary prediction: zooming **in** (smaller hexes, below the 384 threshold)
should flip back to the fast `/thumbnail` path and not show the delay. If the
reporter sees the delay only at the most-zoomed-out view, that strongly supports
this hypothesis.

## Secondary suspect: on-the-fly thumbnail generation (only if it's `/thumbnail`)

If the Network tab shows the slow requests are actually `/thumbnail` (not
`/image`), the cause is different: the thumbnails are being **generated per
request** instead of served from precomputed bytes.

- `vtsearch/routes/media/list.py:548-552` — `media_thumbnail` serves precomputed
  `c["thumbnail_bytes"]` directly via `cached_thumbnail_response` (fast, no
  decode) **when present**; otherwise falls back to `_resolve_display_image` +
  `image_thumbnail_response`, which decodes the full source and resizes it
  (`vtsearch/routes/_shared.py:57-103`, `make_image_thumbnail`).
- Datasets missing `thumbnail_bytes`: old pickles, thin loads, undecodable SVGs
  (per the `media_thumbnail` docstring at `:529-533`).
- The server runs **1 gunicorn worker** with `gthread` / 8 threads
  (`gunicorn.conf.py:28-30`). Pillow decode+resize is CPU work largely under the
  GIL, so concurrent thumbnail generation across those threads **serializes** —
  30 on-the-fly generations ≈ 30 sequential decode+resize ops, which can land
  around ~3s.

This suspect is independently worth checking because it would also make *every*
zoom level's first paint slow, not just the coarse top view.

## How to confirm (do this first, before any fix)

Run the app against the reporter's dataset (or any image dataset) and watch the
network while the top view loads:

1. `bash .claude/hooks/ensure-test-deps.sh && python app.py --local` and open the
   browse view; or use the `/run` skill / `/verify` skill to drive it.
2. DevTools → Network, filter to `/api/medias/`, hard-reload the browse view.
3. Inspect the requests that fill the coarse-view bins:
   - **If they're `…/image`** and each is hundreds of KB–MBs and/or slow →
     **primary hypothesis confirmed** (full-res switch). Note typical size and
     time.
   - **If they're `…/thumbnail`** and each is small but slow (high "Waiting/TTFB"
     while the server generates) → **secondary suspect** (on-the-fly generation;
     check whether the dataset's medias carry `thumbnail_bytes`).
   - Note concurrency: are ~6 in flight at once (browser cap) or fewer?
4. Confirm the zoom-dependence: zoom in until hexes are clearly small and reload
   — does the delay vanish and switch to `/thumbnail`? That isolates the tier
   switch as the trigger.
5. (Optional) Add a temporary `console.debug` in `getThumb` logging `endpoint`
   and `representativeId`, or log `this.thumbsAreFullRes` in
   `syncThumbResolutionTier`, to see the tier the coarse view actually selects.

Record the findings in this file (size/time per request, endpoint, tier) so the
fix targets the real cause.

## Fix options (pick after confirming)

If **primary hypothesis** (full-res `/image` at coarse view):

- **A. Add a bounded mid-res thumbnail tier.** Instead of jumping straight from
  384px `/thumbnail` to unbounded full `/image`, introduce an intermediate
  capped size (e.g. 768px or 1024px) so big coarse-view hexes get a sharp-enough
  *but still small* image. Touches: the frontend endpoint/threshold selection in
  `browse-canvas.component.ts` (`useFullResThumbs` / `:1428`) and a backend route
  that serves a larger-capped thumbnail (parameterize `make_image_thumbnail`'s
  `max_dim`, e.g. `/thumbnail?max=768`, and precompute/cache like
  `thumbnail_bytes`). Keeps the overview fast and crisp. **Recommended.**
- **B. Raise the full-res threshold (frontend-only quick win).** Bump
  `THUMB_NATIVE_MAX_DIM` or the `useFullResThumbs` comparison so the overview
  stays on `/thumbnail` longer. Big hexes look slightly soft (upscaled 384px) but
  load fast. Smallest change; degrades sharpness at the coarsest zoom.
- **C. Cap how many full-res cells load at once / prioritize center-out.** Only
  fetch `/image` for the few cells nearest the viewport center first, or keep the
  coarse overview on `/thumbnail` entirely and only use `/image` once the user
  zooms past a tighter threshold. Bounds the burst regardless of source size.

If **secondary suspect** (on-the-fly `/thumbnail` generation):

- **D. Ensure `thumbnail_bytes` is precomputed for the dataset** at ingest (the
  fast path at `media/list.py:548-550`), and/or add a process-scoped cache of
  generated thumbnails on `DatasetContext` so the first generation is paid once.
  Note the **No Persisted Vectors/MLPs** rule does not forbid a thumbnail-bytes
  cache, but keep any in-memory cache process-scoped, not written to disk beyond
  the existing dataset-pickle snapshot.

A combined fix (mid-res tier that's precomputed and cached) addresses both
suspects at once.

## Key file/line references

| What | Location |
|------|----------|
| Tier switch (full-res trigger) | `frontend/.../browse-canvas/browse-canvas.component.ts:320-322` |
| `THUMB_NATIVE_MAX_DIM = 384` | `browse-canvas.component.ts:84` |
| Per-cell endpoint pick + fetch | `browse-canvas.component.ts:1428-1430` |
| `getThumb` (fires on first draw) | `browse-canvas.component.ts:1405-1432` |
| `drawHex` calls `getThumb` | `browse-canvas.component.ts:1233` |
| `/image` route (raw source bytes) | `vtsearch/routes/media/list.py:501-513` |
| `_resolve_display_image` | `vtsearch/routes/media/list.py:457-498` |
| `/thumbnail` route (precomputed vs on-the-fly) | `vtsearch/routes/media/list.py:516-552` |
| `cached_thumbnail_response` (fast path) | `vtsearch/routes/_shared.py:33-54` |
| `image_thumbnail_response` (decode+resize) | `vtsearch/routes/_shared.py:57-103` |
| Single worker + 8 threads (GIL serialization) | `gunicorn.conf.py:28-30` |

## Open follow-ups

- Confirm hypothesis against a running app (see "How to confirm") and record the
  measured endpoint/size/timing here before implementing.
- Decide tier strategy (mid-res cap vs. threshold bump vs. center-out cap).
- If a mid-res tier is added, precompute + cache it so it doesn't reintroduce
  on-the-fly generation cost.
