# Smart clipper defaults

*Status: Phase 1 and Phase 2 shipped — picker offers an "Auto
(recommended)" clipper for audio and video that routes each media
through pass-through or tiling based on its own duration. Phase 3
deferred — see Open follow-ups.*

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

## Open follow-ups

### Phase 3 — Hide the clipper picker behind Advanced (deferred)

The ux-brainstorm entry also proposed hiding the clipper button entirely
until the user opens an "Advanced" section, on the theory that the
default is almost always right. Deferred until we have evidence on how
often users tweak the default after Phase 2 lands.

### Per-item routing for image and text (out of scope)

Phase 2's `resolve_for_media` hook only ships on the duration-bearing
media types. Image and text don't have a meaningful "duration", but
analogous signals exist (image aspect ratio for "tile tall images
automatically", paragraph length for "sentence-split long paragraphs").
Out of scope for now; if pursued, follow the same shape as the audio
and video auto clippers — register an `*_auto` clipper first in
`CLIPPERS` and have it override `resolve_for_media`.
