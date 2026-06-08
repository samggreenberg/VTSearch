# Downscaled thumbnails for grid/list tiles

**Status:** Shipped app-wide, then extended to **precompute at ingest** (see
"Precompute at ingest" below). Image thumbnails are now generated once when
media is loaded and served verbatim, so a fresh browse-canvas zoom no longer
waits on per-tile full-resolution decodes.

## What shipped

Browsing a large image result set (e.g. the "Good" results of a Find with ~209
positives) and zooming the grid up and down repeatedly could exhaust browser
memory and hang the whole machine (Chrome "not responding", OS unresponsive
from swap thrashing). Two compounding causes, both fixed:

1. **Full-resolution tiles.** Grid and list tiles loaded
   `/api/medias/<id>/image`, which streams the *original* source bytes. A
   gallery of many high-res photos forced the browser to download and decode
   every full-size bitmap at once (tens of MB of decoded pixels each).

   Fix: new `GET /api/medias/<id>/thumbnail` route serving a downscaled image
   (longest side ≤ `vtscore.media.image.thumbnail.DEFAULT_MAX_DIM`, currently
   384 px), with an `ETag` + `Cache-Control` so the browser reuses one
   thumbnail per media across scrolls and every zoom level. The full-res
   `/image` route is unchanged and still backs the detail viewer.
   `MediaItemComponent.thumbnailUrl` now points at `/thumbnail`.

2. **Virtualization collapse on zoom.** The grid's CDK virtual scroll uses a
   fixed `itemSize` measured live from the first rendered card. During a zoom
   relayout a card can momentarily report a near-zero height; that value was
   locked in as `itemSize`, making the viewport think each row was a few pixels
   tall and mount nearly every tile at once.

   Fix: `MIN_GRID_ROW_HEIGHT` floor in
   `media-list.component.ts#measureGridLayout` — heights below the floor are
   treated as "not yet laid out" and ignored until a real card measures, so
   `gridHeightMeasured` stays false and the next pass re-measures.

## Whole-app rollout (shipped)

A shared `vtsearch.routes._shared.image_thumbnail_response` helper (ETag +
`Cache-Control` + downscale via `make_image_thumbnail`) now backs every
small-image route, and the previously-full-res "thumbnail" routes finally
produce real image thumbnails:

- **Backend routes that now downscale images:** `/api/medias/<id>/thumbnail`,
  `/api/detectors/<name>/labels/<id>/thumbnail` (in-memory + origin paths,
  backing the right-panel `labelset-list`), and
  `/api/server-media-files/<file>/thumbnail`.
- **Frontend consumers switched from `/image` to `/thumbnail`:** right-panel
  `label-list` (current votes), `browse-selection-panel`, `browse-bin-popup`,
  `browse-canvas`.
- **Intentionally left on full-res `/image`:** the center `image-viewer`
  (detail view) and `label-view`'s crop overlay — the crop maps coordinates
  onto the real pixels, so it needs the original resolution.

## Precompute at ingest (shipped)

The whole-app rollout above downscaled images but still regenerated the
thumbnail from the full-resolution original on every *cold* `/thumbnail`
request (no server-side cache). On the browse canvas this was the dominant
cost behind "zoom into a fresh region and wait for the bin tiles to appear":
each newly visible bin fans out a request that does a full-res PIL decode +
LANCZOS resize, throttled by the browser's ~6-connection-per-origin cap and
the server's thread pool. The delivered 384 px tile is tiny to decode
client-side, so the lag was almost entirely server-side generation + fan-out,
not browser "reformatting".

Audio and video already avoided this: they generate `thumbnail_bytes` at
ingest and serve the stored bytes via `image_response`. Images were the gap.
What shipped brings images to parity:

- **Generate at ingest.** `ImageMediaType.load_media_data` now precomputes
  `thumbnail_bytes` (via `make_image_thumbnail`), and the add-to-pile upload
  route (`routes/media/list.py`) does the same for image uploads. Undecodable
  sources (SVG, corrupt) yield `None` and fall back to request-time generation.
- **Persist it.** `ImageMediaType.pickle_extra_fields` now lists
  `thumbnail_bytes`, and `export_dataset_to_file` was fixed to write *every*
  media type's `pickle_extra_fields` instead of a hardcoded key list. That
  hardcoded list silently dropped `thumbnail_bytes` on export, so even
  audio/video thumbnails were not surviving a pickle round-trip before this
  change — they do now.
- **Serve it directly.** `GET /api/medias/<id>/thumbnail` checks
  `media["thumbnail_bytes"]` first and streams it with no decode/resize
  (`_shared.cached_thumbnail_response`, mimetype sniffed from magic bytes).
  Media without a stored thumbnail (old pickles, thin loads, demo images,
  undecodable sources) fall back to the existing on-demand generation.

**Backwards compatibility:** exported dataset pickles now embed image/audio/
video thumbnails, so they grow by roughly one ~20-40 KB thumbnail per visual
item (marginal next to the full media bytes already stored). Old pickles
without `thumbnail_bytes` still load and simply regenerate thumbnails on
demand. The image `/thumbnail` `ETag` now fingerprints the thumbnail bytes
rather than the source bytes, so caches revalidate once after deploy.

## Open follow-ups

- **Document thumbnails.** The `document` media type has no `image_response`
  and no `thumbnail_bytes`; document tiles get no real thumbnail. Out of scope
  for the image/video responsiveness work; render a first-page image at ingest
  if document tiles ever need a preview.
- **Demo image datasets.** Demo image loaders (`vtscore/media/image/_demo_*`,
  `loader_demo`) build media dicts without routing through
  `load_media_data`, so demo images fall back to request-time thumbnail
  generation. The main folder/server/upload import flows (what populates real
  browse datasets) precompute. Add precompute to the demo path if demo-set
  browse responsiveness matters.
- **Thin loads stay lazy by design.** Thin-mode imports/pickles deliberately
  skip reading bytes, so they carry no `thumbnail_bytes` and regenerate on
  demand — generating thumbnails eagerly there would defeat thin mode.
