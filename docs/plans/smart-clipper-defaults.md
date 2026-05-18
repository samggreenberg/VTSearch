# Smart clipper defaults

*Status: Phase 1, Phase 2, and Phase 3 shipped — picker offers an "Auto
(recommended)" clipper for audio and video that routes each media
through pass-through or tiling based on its own duration, and the
clipper picker itself is hidden behind an "Advanced ▾" toggle in the
importer modal so users who don't need the override don't see it.*

This plan implements [ux-brainstorm.md §1.10](ux-brainstorm.md#110-clipper-default-from-media-type--duration-) ("Clipper default from media type + duration").

## Problem

Most users don't know what a clipper is. The importer modal exposes one
anyway and defaults to `*_default` (pass-through), which is the wrong
choice for a folder of hour-long podcasts: every file becomes a single
embedding and semantic sort is useless. Picking the right clipper
manually means knowing the difference between `sound_default`,
`sound_tiling`, and `sound_clip` — and what good values for `duration`
and `min_overlap` are. The user doesn't have that context.

## Phase 1 — Auto clippers, per-dataset routing (shipped)

Added an **Auto (recommended)** entry to the clipper picker for the two
duration-bearing media types: audio and video. Registered FIRST in each
media type's `CLIPPERS` list so the picker defaults to it. The load
pipeline resolved it to a single concrete clipper for the whole dataset
based on the median media duration.

### Files

- `vtsearch/media/clipper.py` — Added `MediaClipper.resolve_for_durations(durations)` on the base class, default returns `self`.
- `vtsearch/media/audio/clipper.py` — Added `SoundAutoClipper(threshold=30, tile_duration=10)`.
- `vtsearch/media/video/clipper.py` — Added `VideoAutoClipper(threshold=30, tile_duration=10)`.
- `vtsearch/media/audio/__init__.py` — `SoundAutoClipper()` first in `CLIPPERS`.
- `vtsearch/media/video/__init__.py` — `VideoAutoClipper()` first in `CLIPPERS`.
- `vtsearch/datasets/load_pipeline.py` — `_apply_clipper` called `clipper.resolve_for_durations(durations)` once per dataset.

## Phase 2 — Per-media auto routing (shipped)

Moved the auto routing decision from per-dataset (median duration) to
per-media (each item's own duration). A short clip and a long clip in
the same dataset now take different branches.

### Files

- `vtsearch/media/clipper.py` — Added `MediaClipper.resolve_for_media(media)` hook on the base class, default returns `self`. `resolve_for_durations` stays as a no-op base hook reserved for clippers that need a dataset-level decision.
- `vtsearch/media/audio/clipper.py` — `SoundAutoClipper.resolve_for_media(media)` branches on `media["duration"]`, falling back to reading the WAV header if duration is absent. `clip()` is now just `resolve_for_media(media).clip(media)`.
- `vtsearch/media/video/clipper.py` — `VideoAutoClipper.resolve_for_media(media)` mirrors the audio implementation.
- `vtsearch/datasets/load_pipeline.py` — `_apply_clipper` calls `resolve_for_media(media)` per item and tags each clip's origin with the resolved concrete clipper's name and parameters.

### Behavior

- Picker still shows **Auto (recommended)** first.
- For each media in the dataset:
  - `duration <= 30s` → routed through `*_default` (pass-through).
  - `duration > 30s` → routed through `*_tiling` with `duration=10s`.
- Threshold and tile length remain configurable via the chooser's parameter fields (`threshold`, `tile_duration`).
- Each clip records the **resolved** concrete clipper in its origin (`clipper=sound_tiling` or `clipper=sound_default`), not `sound_auto`. Different clips in the same dataset can have different `clipper` values. Cross-dataset replay is deterministic and ignores the original auto policy.
- Direct calls (`SoundAutoClipper().clip(media)` outside the load pipeline) use the same per-media routing.

### Why per-media wins over per-dataset

A dataset with a mix of short voice memos and hour-long podcasts is the
common case (a user dumps a folder of varied recordings). Phase 1 had
to pick one strategy for the whole folder based on the median, which
was wrong for whichever subset fell on the other side of the threshold.
Phase 2 fixes that without changing the user-facing controls or the
origin format.

## Phase 3 — Hide the clipper picker behind Advanced (shipped)

The picker no longer surfaces the clipper button on first sight. Each of
the four importer-modal contexts (generic form, local-folder/files,
server-folder, demo) wraps the clipper button in an **Advanced ▾**
toggle. Clicking the toggle reveals the existing "Use MediaClipper: …"
chooser button; clicking again collapses it. A single
`clipperAdvancedOpen` boolean drives all four contexts (only one is
visible at a time).

If the user has already picked a non-default (non-first-in-list)
clipper, the picker stays visible regardless of the toggle so they can
see and re-edit their selection. The Advanced toggle hides itself in
that case — collapsing wouldn't actually hide the picker, so the toggle
would be a no-op.

### Files

- `frontend/src/app/components/dashboard/dataset-importer-modal/dataset-importer-modal.component.ts` — Added `clipperAdvancedOpen` field plus `isDefaultClipperSelected(context)`, `showClipperPicker(context)`, and `toggleClipperAdvanced()`.
- `frontend/src/app/components/dashboard/dataset-importer-modal/dataset-importer-modal.component.html` — Gated each of the four clipper-button blocks (form, lf, sf, demo) behind the toggle button.
- `frontend/src/app/components/dashboard/dataset-importer-modal/dataset-importer-modal.component.scss` — `.advanced-toggle` link-styled button (subtle muted text, hover underline). The demo context (inline in the embedder row) uses `.advanced-toggle--inline` for a small left margin.

## Open follow-ups

### Per-item routing for image and text (out of scope)

Phase 2's `resolve_for_media` hook only ships on the duration-bearing
media types. Image and text don't have a meaningful "duration", but
analogous signals exist (image aspect ratio for "tile tall images
automatically", paragraph length for "sentence-split long paragraphs").
Out of scope for now; if pursued, follow the same shape as the audio
and video auto clippers — register an `*_auto` clipper first in
`CLIPPERS` and have it override `resolve_for_media`.
