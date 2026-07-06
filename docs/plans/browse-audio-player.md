# Browse audio player: squares + anchored hover player

**Status:** design only (not started). Scope confirmed with the user:
*squares + hover player*, player *anchored to the tile*.

## Motivation

On the VTSBrowse map, audio bins render as flat-density **hexes**, and
mousing one plays sound from a `display:none` `<audio>` element — audio
"comes out of nowhere" with no visual player, no play/pause, no volume, no
play-point. We want audio bins to render as **square waveform tiles** (like
image/video), and hovering a bin to raise an **anchored, controllable
player** modelled on the Train center-panel audio player.

## The surprise: audio is *already* a waveform-thumbnail type — except on the map

This proposal is smaller than "add thumbnails to audio," because the
waveform-thumbnail pipeline already exists and already ships:

- `vtscore/media/audio/media_type.py:27` `generate_waveform_thumbnail()`
  renders a min/max-envelope PNG at ingest and stores it as
  `thumbnail_bytes` (round-tripped via `pickle_extra_fields`).
- Served at `/api/medias/<id>/thumbnail` and `/image` through the same
  routes images use (`image_response` hook).
- The **left-panel media grid already displays** audio waveforms:
  `frontend/src/app/components/left-panel/media-item/media-item.component.ts`
  lists `'audio'` among its thumbnail types.
- **Clip-aware already:** `vtscore/datasets/stages/clipper.py:279`
  `_regenerate_clip_thumbnails()` re-renders each clip's waveform from the
  *sliced* bytes, so a clipped track's tile shows only its
  `[clip_start, clip_end)` window. Lazy clips reproduce this on demand.

The **only** place audio is a "no-thumbnail" hex is the VTSBrowse
projection canvas, gated by two lists:

- Backend: `vtscore/projection/pyramid.py:57` `SQUARE_MEDIA_TYPES =
  {"image","video","document"}` + `bin_shape_for_media_type()` (`:60`). The
  comment at `:50-56` deliberately excludes audio ("nobody browses by
  waveform") — so flipping this is a **product reversal**, not a bug fix.
- Frontend: `frontend/src/app/components/browse-canvas/hex-render.util.ts:42`
  `usesThumbnails()` returns true for `'image'|'video'` only.

The rich player we want as the model already exists:
`frontend/src/app/components/center-panel/audio-player/audio-player.component.ts`
— waveform canvas + native `<audio controls>` (play/pause, **volume**,
**scrubber/play-point**) + clip-window looping (`applyClipBounds`,
`startClipEnforcement`), `togglePlayback()`, `adjustVolume()`,
`playingChanged` output.

So the two real builds are (1) the map tile-shape flip and (2) an anchored
hover player — mostly *relocating an existing component*, not new invention.

## Design

### Part A — audio bins as square waveform tiles

1. Add `"audio"` to `SQUARE_MEDIA_TYPES` (`pyramid.py:57`) — or, better,
   replace the three hardcoded type-lists with a single source of truth
   (see "Cleanup" below).
2. Add `'audio'` to `usesThumbnails()` (`hex-render.util.ts:42`).
3. Verify square packing / rep zoom-persistence in `squarebin.py` +
   `pyramid.py` behaves with 128×128 square waveform reps (aspect
   assumptions are image-oriented; waveforms are square so should fit, but
   confirm).
4. Reconcile the color/border handling: `usesThumbnails` types are pinned
   to the grayscale colormap and hide the colormap picker — decide whether
   audio waveform tiles want the same treatment (probably yes).

### Part B — anchored hover player

Upgrade `BrowseHoverPreviewComponent` (the current hidden-`<audio>` host)
into an anchored player, reusing `AudioPlayerComponent`:

- The hover event carries only `cell.rep_id` + `cell.count`; hydrate the
  representative into a `Media` via `MediaMetadataCacheService`
  (`ensureLoaded([id])` / `get(id)`) — the same cache the current hover
  already primes — then feed it to `<vt-audio-player [media]>`.
- Anchor the player overlay to the hovered tile's screen position (the
  hover host already positions at `screenX+16 / screenY-8`).
- **Playhead line over the canvas is net-new:** `AudioPlayerComponent`
  relies on the native scrubber and does not draw a moving playhead on its
  waveform. `audio-crop-overlay.component.ts` shows the canvas-overlay
  pattern to copy if we want the playhead drawn on the waveform itself.

## Problems / risks

1. **Waveform tiles are low-discrimination.** Zoomed out, 128px waveforms
   all read as similar squiggles — far less scannable than image
   thumbnails. This is exactly why `pyramid.py:50-56` excluded them. The
   payoff is the *hover player*, not static-tile browsability; set
   expectations accordingly.
2. **Hover performance.** `AudioPlayerComponent.drawWaveform()` fetches the
   full audio and decodes it client-side (`decode-audio.ts`) to paint its
   interactive waveform. Sweeping the cursor across bins would fire a
   fetch+decode per bin → jank. Mitigation: keep painting the cheap backend
   PNG on the tile; only instantiate the heavy interactive player on a
   **debounced dwell**; reuse cached decodes where possible.
3. **Hover-into-controls.** A player with sliders/buttons means the cursor
   must travel from the tile into the player without hover-out tearing it
   down (classic hover-menu trap). The current code kills audio on
   mouse-leave; the anchored player needs a hover-bridge / dismiss delay so
   the controls are reachable.
4. **Which item plays?** A bin aggregates many items; hover plays only the
   representative today. A visible player invites "page through members,"
   which is the job of the existing click-to-open bin popup
   (`browse-bin-popup.component.ts`, which also hover-plays audio via the
   same hidden-`<audio>` pattern and would want the same upgrade). Decide
   whether member-paging lives in the anchored player or stays in the popup.
5. **Screenshot reshoots.** Changing the Browse map means queueing affected
   shot ids in `docs/user/screenshots-reshoot-queue.md` (no browser in the
   cloud container to reshoot in-session).

## Cleanup opportunity (do this as part of the change)

"Has a browsable thumbnail" is currently smeared across **three
independent hardcoded type-lists that disagree**:

- `media-item.component.ts:46` — image, video, document, **audio**
- `hex-render.util.ts:42` — image, video (missing document!)
- `pyramid.py:57` — image, video, document

There is no `has_thumbnail` capability on `MediaType`
(`vtscore/media/base.py`); `to_dict()` doesn't expose one. The clean move
is to add an explicit `has_thumbnail` / `bin_shape` field to the media-type
API (`to_dict()` → `GET /api/media-types`) and have all three sites consume
it, so audio (and future types) flip in one place instead of a fourth
hardcoded list.

## Suggested phasing

- **Phase 1 — tiles:** add the media-type capability field + reconcile the
  three lists so audio renders as square waveform tiles on the map. Ships
  the visible tile change with the least new UI.
- **Phase 2 — anchored player:** upgrade `BrowseHoverPreviewComponent` to
  the debounced anchored `AudioPlayerComponent` with play/pause + volume +
  scrubber; hover-bridge so controls are reachable.
- **Phase 3 — polish:** canvas playhead line; decide member-paging home;
  apply the same upgrade to the bin-popup hover-play.

## Open follow-ups

- Member-paging: anchored player vs bin popup (Problem 4) — unresolved.
- Canvas playhead over the waveform (Phase 3) — net-new, pattern exists in
  `audio-crop-overlay`.
- Whether `text` (the other hex type) ever gets an analogous treatment — out
  of scope for now.
