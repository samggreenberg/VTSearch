# Browse audio player: squares + top-left now-playing indicator

**Status: Phases 1, 2, and 4 shipped; Phase 3 (polish) superseded by Phase 4, not built.** Audio bins now tile as square waveform thumbnails on the VTSBrowse map (Phase 1). Hovering an audio bin lifts it (the shared hover-enlarge every thumbnail type gets) and plays the clip, with **no on-canvas player UI**; the only feedback is a small waveform + volume control anchored top-left (Phase 4). Phase 2's anchored floating player was deleted (felt redundant with the bin's own hover-enlarge), not layered on. Open follow-ups below.

## Open follow-ups

- **Merge bin-popup preview with now-playing.** A known, accepted redundancy:
  with the bin-detail popup open, the hovered row's waveform shows both in the
  popup's own grid and in the top-left indicator. A future pass could have the
  popup point at (or fold into) the shared now-playing indicator instead of
  keeping its own display.
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

## What shipped

Phase 1 — audio bins as square waveform tiles:
- **`MediaType.has_thumbnail`** capability (`vtscore/media/base.py`) — single source of truth for thumbnail vs no-thumbnail; `True` on image/video/document/audio, `False` on text. Exposed on `GET /api/media-types` (`to_dict`) + mirrored on frontend `MediaTypeInfo.has_thumbnail`.
- **`bin_shape_for_media_type`** (`vtscore/projection/pyramid.py`) derives square-vs-hex from `has_thumbnail` via a lazy registry lookup (keeps projection layer Flask/media-free at import); the hardcoded `SQUARE_MEDIA_TYPES` frozenset is gone. Audio → square.
- **Frontend `usesThumbnails`** (`hex-render.util.ts`) now includes audio (and document, fixing a pre-existing omission): audio bins paint their waveform PNG, the bin popup gains a magnified-waveform preview pane, and the colormap picker is hidden for audio (grayscale-pinned).
- Side effect: clicking an audio bin now shows a preview pane in the bin popup, consistent with image/video.

Phase 2 — anchored audio hover player (**superseded, then deleted in Phase 4**):
- `BrowseHoverPreviewComponent` opened a panel next to the hovered tile with the bin's waveform PNG + native `<audio controls>` (play/pause, volume, scrubber), replacing the old `display:none` "sound from nowhere". Dwell (~200ms) + debounce re-armed on sweep; a hover-bridge + close-grace let the cursor reach the controls; windowed clips looped within `[clip_start, clip_end]` via `applyClipWindow`. Built as a lightweight inline panel (waveform PNG + native controls) rather than embedding the Train `AudioPlayerComponent`.
- **Removed in Phase 4** because the floating panel duplicated the bin's own hover-enlarge — two "here's what you're hovering" surfaces stacked. The user asked for it gone: hover should just enlarge the bin and play, with feedback confined to a small always-there corner widget.

Phase 4 — top-left now-playing indicator:
- **Deleted the anchored floating player.** `BrowseHoverPreviewComponent`'s audio path renders no DOM; it owns a never-mounted `new Audio()` purely to hear the clip. Hovering still lifts the bin via `usesThumbnails()` hover-enlarge.
- **`NowPlaying` output** (`{ mediaId, waveUrl }`) from `browse-hover-preview.component.ts`, emitted when a clip starts and `null` the instant hover clears (no bridge grace — no panel to bridge onto). `browse-bin-popup.component.ts` gained the same output from its grid-hover audio path (`onEntryEnter`/`stopAudio`), so the member grid feeds the same indicator.
- **Top-left now-playing + volume cluster** in `browse-view.component`'s `.browse-tools-left` overlay: a `.browse-now-playing-wave` `<img>` of the auditioning clip stacked above `.browse-volume` (mute+slider), which **moved from top-right to top-left**. No per-clip controls (the cursor can't reach the corner without leaving the noisy bin).
- **Redundant-by-design overlap with the bin popup** (hovered row's waveform shows in both the popup grid and the top-left indicator) — accepted, candidate merge noted in Open follow-ups.
