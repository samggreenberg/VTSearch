# Downscaled thumbnails for grid/list tiles

**Status:** Shipped for the media-list grid/list tiles (the reported crash
path). Open follow-up: the other small-thumbnail consumers still request the
full-resolution `/image` route.

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

## Open follow-ups

- **Wire the other thumbnail consumers to `/thumbnail`.** Several views still
  load full-res `/image` for small thumbnails. Lower risk than the grid (they
  render far fewer items), but they'd benefit from the same bounded decode:
  - `right-panel/label-list/label-list.component.ts`
  - `browse-selection-panel/browse-selection-panel.component.ts`
  - `browse-bin-popup/browse-bin-popup.component.ts`
  - `browse-canvas/browse-canvas.component.ts`
  - `label-view/label-view.component.ts`

  Each has spec tests asserting the `/image` URL that would need updating.
- **Optional server-side thumbnail cache.** Currently each thumbnail is
  generated on demand (PIL decode + resize) and cached only in the browser via
  `ETag`/`Cache-Control`. If first-paint CPU on very large datasets becomes a
  concern, add a process-scoped LRU keyed by media id + content hash. Not
  needed for the reported workload.
