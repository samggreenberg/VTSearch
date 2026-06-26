# Browse first-paint thumbnail load delay (~3s grayscale bins)

**Status:** Investigation only — cause NOT yet confirmed against a running app.
An earlier draft of this plan blamed a "coarse-view full-resolution `/image`
fetch"; that hypothesis was **wrong and has been retracted** (see "Retracted
hypothesis" below). The next contributor should **confirm via the DevTools
Network tab first** (see "How to confirm"), then pick a fix.

## Symptom

When the VTSBrowse canvas first loads, there's a solid ~3 seconds where only the
grayscale/density-shaded hex bins are visible before any imagery paints in. The
reporter's dataset shows only ~30 bins in that view.

The bins themselves paint instantly: they're drawn purely from tile cell data
(`count` → colormap), no images involved. The delay is entirely the imagery
filling those bins.

## The full-res switch is global, NOT per-pyramid-level (key correction)

The browse canvas decides per-paint whether to fetch the capped `/thumbnail` or
the full-resolution `/image`:

- `browse-canvas.component.ts:320-322` — `useFullResThumbs`:
  ```ts
  private get useFullResThumbs(): boolean {
    return 2 * this.targetRadius * this.dpr > BrowseCanvasComponent.THUMB_NATIVE_MAX_DIM; // 384
  }
  ```

Crucially this reads **only `targetRadius` and `dpr`** — not zoom, not pyramid
level. `targetRadius` is the thumbnail-size knob (default `28` px = the "M" size,
`:79`), held constant across zoom. Level selection (`levelForEffZoom`,
`:757-767`) picks whichever pyramid level keeps each bin near `targetRadius`, so
the on-screen hex is ~28px at **every** zoom. The per-bin *mathematical* area
shrinks going up the pyramid, but it's scaled to a constant *screen* size.

Therefore the full-res switch is the **same decision at every zoom level** — a
global on/off, never a "top of the pyramid" effect. It trips only when
`2 · targetRadius · dpr > 384`, i.e. `targetRadius > 192/dpr`:

| Display | targetRadius needed to trigger full-res | Default "M" = 28 |
|---|---|---|
| Standard (dpr=1) | > 192 px | 56 → **off** |
| Retina (dpr=2)   | > 96 px  | 112 → **off** |

**Conclusion: at the default thumbnail size, the canvas fetches `/thumbnail`
everywhere and never `/image`.** Full-res only engages if the user has enlarged
the thumbnail size to roughly XL (`targetRadius` past ~96 on Retina, ~192
otherwise). So unless the reporter has cranked the size knob up, full-res is not
in play and is not the cause of the delay.

### Retracted hypothesis

The earlier "the coarse/top view has large hexes, so it crosses 384 and fetches
multi-MB `/image` originals" theory was wrong twice over: (1) hex *screen* size
is held ~constant across zoom by level selection, so the top view does not have
larger hexes; (2) `useFullResThumbs` keys off the constant `targetRadius`, not
the actual rendered radius, so it can't be a per-level effect. Do not resurrect
it without first confirming the size knob is actually turned up.

## Leading hypothesis: on-the-fly `/thumbnail` generation (no precomputed bytes)

If the dataset's medias lack precomputed `thumbnail_bytes`, every `/thumbnail`
request decodes the full source image and resizes it at request time:

- `vtsearch/routes/media/list.py:548-552` — `media_thumbnail` serves precomputed
  `c["thumbnail_bytes"]` directly via `cached_thumbnail_response` (fast, no
  decode) **when present**; otherwise falls back to `_resolve_display_image` +
  `image_thumbnail_response`, which decodes the full source and resizes it
  (`vtsearch/routes/_shared.py:57-103`, `make_image_thumbnail`).
- Datasets missing `thumbnail_bytes`: old pickles, thin loads, undecodable SVGs
  (per the `media_thumbnail` docstring at `:529-533`).
- The server runs **1 gunicorn worker**, `gthread`, 8 threads
  (`gunicorn.conf.py:28-30`). Pillow decode+resize is CPU work largely under the
  GIL, so concurrent generation across those threads **serializes**. ~30
  on-the-fly generations of large source photos ≈ ~30 sequential decode+resize
  ops, which can land around ~3s.

This fits the symptom shape: slow on first paint, then fast (ETag 304 +
in-memory `thumbCache`), and recurs when panning into not-yet-cached bins.

**But note:** a normally-ingested dataset *does* get `thumbnail_bytes` at ingest
(docstring `:529`), which would make this path fast. So this hypothesis hinges on
the reporter's dataset actually lacking them — which the Network tab / a quick
`get_media(id).keys()` check will reveal. If `thumbnail_bytes` is present and
`/thumbnail` is still slow, the cause is elsewhere and needs fresh investigation
(candidate avenues: first-thumbnail draw is gated on a late `mediaType()` signal;
tiles contending for connections; projection/meta still settling on first paint).

## How to confirm (do this first, before any fix)

Run the app against the reporter's dataset (or any image dataset) and watch the
network while the view loads:

1. `bash .claude/hooks/ensure-test-deps.sh && python app.py --local` and open the
   browse view; or use the `/run` or `/verify` skill to drive it.
2. DevTools → Network, filter to `/api/medias/`, hard-reload the browse view.
3. Inspect the requests that fill the bins:
   - **Endpoint:** `…/thumbnail` or `…/image`? (Expected: `/thumbnail` at default
     size. If `/image`, the size knob is turned up — confirm and revisit the
     full-res path.)
   - **Timing:** is the time dominated by "Waiting (TTFB)" (server generating the
     thumbnail) vs. "Content Download" (large bytes over the wire)? High TTFB on
     small `/thumbnail` responses ⇒ on-the-fly generation.
   - **Size:** small (KB, precomputed/capped) or large (MB, full source)?
   - **Concurrency:** ~6 in flight (browser cap) or fewer/serialized?
4. Check whether the dataset's medias carry `thumbnail_bytes`: in a Python shell
   against the running context, `get_media(<id>).keys()` (or inspect the loader
   for the source). Absent ⇒ leading hypothesis confirmed.
5. Confirm whether the delay is first-paint-only: pan into fresh bins after the
   first load — does the ~3s recur for newly revealed cells? (Expected yes for
   on-the-fly generation, since only previously-fetched cells are cached.)

Record the measured endpoint / TTFB / size / `thumbnail_bytes` presence in this
file so the fix targets the real cause.

## Fix options (pick after confirming)

If **on-the-fly generation** (no `thumbnail_bytes`):

- **A. Ensure `thumbnail_bytes` is precomputed at ingest** for the dataset's
  media type (the fast path at `media/list.py:548-550`). Best fix if the gap is a
  loader that skips thumbnail generation.
- **B. Process-scoped generated-thumbnail cache** on `DatasetContext` so the
  first on-the-fly generation per item is paid once and reused, even for datasets
  that can't precompute at ingest. Keep it in-memory/process-scoped per the
  **No Persisted Vectors/MLPs** rule (that rule doesn't forbid a thumbnail cache,
  but don't write it to disk beyond the existing dataset-pickle snapshot).
- **C. Generate thumbnails off the request thread** / warm them ahead of the
  first paint so the GIL-serialized decode isn't on the critical path.

If the Network tab shows something else entirely, investigate fresh — do not
assume either of the above.

## Key file/line references

| What | Location |
|------|----------|
| `useFullResThumbs` (global, keys off targetRadius) | `frontend/.../browse-canvas/browse-canvas.component.ts:320-322` |
| `DEFAULT_TARGET_RADIUS = 28` ("M" size) | `browse-canvas.component.ts:79` |
| `THUMB_NATIVE_MAX_DIM = 384` | `browse-canvas.component.ts:84` |
| Level selection holds bin ~targetRadius across zoom | `browse-canvas.component.ts:757-767` |
| Per-cell endpoint pick + fetch | `browse-canvas.component.ts:1428-1430` |
| `getThumb` (fires on first draw) | `browse-canvas.component.ts:1405-1432` |
| `/thumbnail` route (precomputed vs on-the-fly) | `vtsearch/routes/media/list.py:516-552` |
| `cached_thumbnail_response` (fast path) | `vtsearch/routes/_shared.py:33-54` |
| `image_thumbnail_response` (decode+resize) | `vtsearch/routes/_shared.py:57-103` |
| `/image` route (raw source bytes) | `vtsearch/routes/media/list.py:501-513` |
| Single worker + 8 threads (GIL serialization) | `gunicorn.conf.py:28-30` |

## Open follow-ups

- Confirm against a running app (see "How to confirm") and record measured
  endpoint / TTFB / size / `thumbnail_bytes` presence here before implementing.
- If `thumbnail_bytes` is present yet `/thumbnail` is slow, open a fresh
  investigation into the non-generation candidates listed above.
