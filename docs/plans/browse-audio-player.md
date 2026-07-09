# Browse audio player: squares + top-left now-playing indicator

**Status:** Core audio-tile + now-playing behavior is in place; remaining work is the open follow-ups below (popup/now-playing merge, `has_thumbnail` data-driving, waveform PNG theming).

## Background

Audio bins tile as square waveform thumbnails on the VTSBrowse map. Hovering an audio bin lifts it (the shared hover-enlarge every thumbnail type gets) and plays the clip, with no on-canvas player UI; the only feedback is a small waveform + volume control anchored top-left. The follow-ups below build on that layout.

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
