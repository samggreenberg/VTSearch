# Browse audio player: squares + top-left now-playing indicator

**Status:** Phase 1, Phase 2, and Phase 4 shipped; Phase 2's anchored floating
player was *replaced* by Phase 4, not layered on top of it — see "What shipped
(Phase 4)" below. Audio bins tile as square waveform thumbnails on the
VTSBrowse map (Phase 1). Hovering an audio bin lifts it (the same hover-enlarge
every thumbnail type gets) and starts the clip playing, with **no on-canvas
player UI**; the only visual feedback is a small waveform + the volume control,
both anchored top-left (Phase 4). Scope confirmed with the user: the earlier
anchored floating player felt redundant with the bin's own hover-enlarge, so it
was deleted rather than polished.

## What shipped (Phase 1)

- **`MediaType.has_thumbnail`** capability (`vtscore/media/base.py`) — the
  single source of truth for the thumbnail vs no-thumbnail distinction. Set
  `True` on image/video/document/audio; `False` on text. Exposed on
  `GET /api/media-types` (`to_dict`) and mirrored on the frontend
  `MediaTypeInfo.has_thumbnail`.
- **`bin_shape_for_media_type`** (`vtscore/projection/pyramid.py`) now derives
  square-vs-hex from `has_thumbnail` via a lazy registry lookup (keeps the
  projection layer Flask/media-free at import). The hardcoded
  `SQUARE_MEDIA_TYPES` frozenset is gone. Audio → square.
- **Frontend `usesThumbnails`** (`hex-render.util.ts`) now includes audio (and
  document, fixing a pre-existing omission that left document square-tiled but
  density-painted). Audio bins on the map paint their waveform PNG; the bin
  popup gains a magnified-waveform preview pane; the colormap picker is hidden
  for audio (grayscale-pinned like other thumbnail types).
- Side effect (desirable): clicking an audio bin now shows a preview pane in
  the bin popup, consistent with image/video.

## What shipped (Phase 2) — superseded by Phase 4

- **Anchored audio hover player** (`BrowseHoverPreviewComponent`). Hovering an
  audio bin opened a panel next to the tile with the bin's waveform PNG plus
  a native `<audio controls>` element — play/pause, volume, and a scrubber (the
  "current play-point"). This replaced the old `display:none` `<audio>` (the
  "sound from nowhere").
- **Dwell + debounce:** the player opened only after the cursor rested ~200ms on
  a bin, and sweeping across bins re-armed the dwell, so a fast pass didn't spawn
  a burst of players/auditions.
- **Hover-bridge:** the panel took pointer events, and a short close-grace after
  the bin-hover cleared let the cursor travel from the tile onto the controls
  without the panel vanishing.
- **Clip-aware:** windowed clips looped within `[clip_start, clip_end]` via the
  shared `applyClipWindow` helper (lazy clip-extent lookup through the metadata
  cache).
- Implementation note: rather than embed the Train `AudioPlayerComponent` (which
  wants a full `Media` object and re-decodes audio client-side per hover), the
  hover player was a lightweight inline panel — the same waveform PNG already on
  the tile + native controls. Visually equivalent, cheaper, no type-plumbing.
- **Why it was removed (Phase 4):** the floating panel duplicated the bin's own
  hover-enlarge (a magnified thumbnail already appears on hover for every
  thumbnail type, audio included) — two "here's what you're hovering" surfaces
  stacked on top of each other. The user asked for the panel to go away
  entirely: hovering should just enlarge the bin and play the clip, with
  feedback confined to a small always-there corner widget instead of new UI
  popping up mid-canvas.

## What shipped (Phase 4) — top-left now-playing indicator

- **Deleted the anchored floating player.** `BrowseHoverPreviewComponent`'s
  audio path no longer renders any DOM; it owns a plain `new Audio()` element
  (never mounted) purely to hear the clip. Hovering an audio bin still lifts it
  via the existing `usesThumbnails()` hover-enlarge — no new visual on the
  canvas itself.
- **`NowPlaying` output**, exported from `browse-hover-preview.component.ts`
  (`{ mediaId, waveUrl }`), emitted when a clip starts and `null` the instant
  hover clears (no hover-bridge grace — there's no panel to bridge onto).
  `browse-bin-popup.component.ts` gained the same output, emitted from its
  existing grid-hover audio path (`onEntryEnter`/`stopAudio`), so the bin-popup
  member grid feeds the same indicator.
- **Top-left now-playing + volume cluster**, in `browse-view.component` inside
  the existing `.browse-tools-left` overlay (alongside "Back to Find" /
  "Rebuild map"): a `.browse-now-playing-wave` `<img>` of whatever clip is
  auditioning (the same waveform PNG painted on its tile) stacked above the
  `.browse-volume` mute+slider control, which **moved from top-right to
  top-left** to sit with it. No per-clip controls (play/pause/scrub) live here
  — deliberately: the cursor can't reach the top-left corner without moving off
  the bin that's making the noise, so any button there would be unreachable
  while relevant.
- **Redundant-by-design overlap with the bin popup:** when the bin-detail popup
  is open, its own member grid still shows the waveform thumbnail for the
  hovered row *and* the top-left indicator shows the same clip. Acknowledged
  and accepted rather than solved — a candidate future merge of "bin popup
  preview" and "now playing" is noted below, not attempted here.

## Motivation (original, Phase 1/2)

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

- **Phase 1 — tiles (SHIPPED):** added the `has_thumbnail` capability +
  reconciled the shape/paint decisions so audio renders as square waveform
  tiles on the map. Visible tile change with the least new UI.
- **Phase 2 — anchored player (SHIPPED, then superseded):** upgraded
  `BrowseHoverPreviewComponent` to a debounced, hover-bridged anchored player
  (waveform + native play/pause + volume + scrubber). Went with a lightweight
  inline panel rather than embedding `AudioPlayerComponent`. Replaced outright
  by Phase 4 rather than polished — see "What shipped (Phase 2) — superseded".
- **Phase 4 — top-left now-playing indicator (SHIPPED):** deleted the anchored
  panel; hovering an audio bin now only enlarges it (shared hover-enlarge) and
  plays sound, with a small waveform + the (relocated) volume control anchored
  top-left as the sole visual feedback. Also upgraded the bin-popup's
  grid-hover audio to feed the same indicator. See "What shipped (Phase 4)".
- ~~Phase 3 — polish~~: superseded by Phase 4; its scope (playhead, panel
  positioning, member-paging) no longer applies to a deleted panel. Any
  surviving pieces are folded into Open follow-ups below.

## Open follow-ups

- **Merge bin-popup preview with now-playing.** Noted as a known, accepted
  redundancy in "What shipped (Phase 4)": with the bin-detail popup open, the
  hovered row's waveform shows both in the popup's own grid and in the
  top-left indicator. A future pass could have the popup point at (or fold
  into) the shared now-playing indicator instead of keeping its own display.
- **Data-drive the frontend from `has_thumbnail`.** Phase 1 added the capability
  to the API + `MediaTypeInfo`, but the frontend still hardcodes the thumbnail
  type set in several spots (`usesThumbnails`, and the per-item `hasThumbnailUrl`
  helpers in bin-popup / selection-panel / label-list / media-item). They now
  all agree on `{image,video,document,audio}`, but a follow-up should collapse
  them onto the served `has_thumbnail` field so a new thumbnail type flips in one
  place.
- **Waveform PNG theming.** `generate_waveform_thumbnail` paints fixed dark
  colors (`_BG_COLOR`/`_WAVE_COLOR`) regardless of light/dark theme, so audio
  square tiles won't match a light-theme map. Consider theme-aware rendering.
  This now also affects the top-left now-playing waveform, which reuses the
  same PNG.
- Whether `text` (the last hex type) ever gets an analogous treatment — out of
  scope for now.
