# Downscaled thumbnails for grid/list tiles

**Status:** Shipped app-wide. The media-list grid/list (the reported crash
path) plus every other image-thumbnail consumer (Find/Train right panels,
browse views) now serve downscaled thumbnails, and all "thumbnail" routes
finally downscale images.

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

## Open follow-ups

- **Optional server-side thumbnail cache.** Currently each thumbnail is
  generated on demand (PIL decode + resize) and cached only in the browser via
  `ETag`/`Cache-Control`. If first-paint CPU on very large datasets becomes a
  concern, add a process-scoped LRU keyed by media id + content hash. Not
  needed for the reported workload.
