# Smart clipper defaults

*Status: Phase 1 shipped — picker now offers an "Auto (recommended)"
clipper for audio and video that resolves to pass-through or tiling
based on the dataset's median duration. Phase 2 deferred — see Open
follow-ups.*

This plan implements [ux-brainstorm.md §1.10](ux-brainstorm.md#110-clipper-default-from-media-type--duration-) ("Clipper default from media type + duration").

## Problem

Most users don't know what a clipper is. The importer modal exposes one
anyway and defaults to `*_default` (pass-through), which is the wrong
choice for a folder of hour-long podcasts: every file becomes a single
embedding and semantic sort is useless. Picking the right clipper
manually means knowing the difference between `sound_default`,
`sound_tiling`, and `sound_clip` — and what good values for `duration`
and `min_overlap` are. The user doesn't have that context.

## Phase 1 — Auto clippers (shipped)

Add an **Auto (recommended)** entry to the clipper picker for the two
duration-bearing media types: audio and video. Register it FIRST in
each media type's `CLIPPERS` list so the picker defaults to it, and
have the load pipeline resolve it to a concrete clipper for the whole
dataset based on the median media duration.

### Files

- `vtsearch/media/clipper.py` — Added `MediaClipper.resolve_for_durations(durations)` on the base class, default returns `self`.
- `vtsearch/media/audio/clipper.py` — Added `SoundAutoClipper(threshold=30, tile_duration=10)`.
- `vtsearch/media/video/clipper.py` — Added `VideoAutoClipper(threshold=30, tile_duration=10)`.
- `vtsearch/media/audio/__init__.py` — `SoundAutoClipper()` first in `CLIPPERS`.
- `vtsearch/media/video/__init__.py` — `VideoAutoClipper()` first in `CLIPPERS`.
- `vtsearch/datasets/load_pipeline.py` — `_apply_clipper` calls `clipper.resolve_for_durations(durations)` once per dataset and uses the resolved clipper's `name` and `to_dict()` for origin tagging.

### Behavior

- Picker shows **Auto (recommended)** first (the importer modal default-selects the first clipper, and the chooser tab bar prefers the first non-`*_default` clipper).
- When the user accepts Auto, the load pipeline collects `media["duration"]` across the staged medias, takes the median, and resolves:
  - `median <= 30s` → `*_default` (pass-through).
  - `median > 30s` → `*_tiling` with `duration=10s`.
- The threshold and tile length are configurable via the chooser's parameter fields (`threshold`, `tile_duration`).
- Each resulting clip records the **resolved** clipper in its origin (`clipper=sound_tiling` or `clipper=sound_default`), not `sound_auto`. Cross-dataset replay is deterministic and ignores the original auto policy.
- Direct calls (`SoundAutoClipper().clip(media)` outside the load pipeline) fall back to per-media routing using the same threshold, so the clipper still works correctly if used without a dataset context.

### Why per-dataset, not per-media

Phase 1 makes the routing decision once per dataset because that matches the existing clipper UX: the picker takes one choice for the whole import. Per-media routing — "for this video, pass through; for that one, tile" — is a more powerful behavior but it changes the mental model and the origin format. It's deferred to Phase 2.

## What shipped

- `MediaClipper.resolve_for_durations()` hook on the base class.
- `SoundAutoClipper` and `VideoAutoClipper` registered first per media type.
- Load pipeline resolution before clipping begins.
- Origin records the resolved concrete clipper, so replay is stable.

## Open follow-ups

### Phase 2 — Per-media auto routing via clipper options

Today the auto clipper picks one concrete clipper for the whole dataset
based on median duration. A more powerful approach is to expose the
"sometimes clip depending on length" behavior **through the
MediaClipper's options**, so a single clipper instance routes per
media:

- Each media item gets clipped via the strategy that matches its own
  duration: short clips pass through, long clips get tiled.
- The clipper's `parameters` would include the threshold and the tile
  configuration, and `clip(media)` would branch on
  `media["duration"]`.
- This subsumes Phase 1: the picker still shows one "Auto" entry, but
  resolution becomes a no-op (it returns self) and the per-media
  branching lives inside `clip()`.

Open questions for Phase 2:

- **Origin recording**: each clip's origin needs to record which
  branch was taken (`sound_tiling` vs `sound_default`) so cross-dataset
  resolution still works. Either the auto clipper emits the resolved
  branch name in its `clip()` output, or origins carry the auto clipper
  plus a per-clip "branch" marker.
- **UI**: the parameter fields ("threshold", "tile length") stay the
  same — users don't need to know whether the decision is per-dataset
  or per-media. But the description string should change from
  "tile *the dataset* if median > 30s" to "tile *each item* if > 30s".
- **Other media types**: image and text don't have a meaningful
  duration. Should we extend the same pattern (e.g. tile tall images
  automatically, sentence-split long paragraphs)? Out of scope for now
  — but the design should not assume "duration" is the only routing
  signal.

### Phase 3 — Hide the clipper picker behind Advanced (deferred)

The ux-brainstorm entry also proposed hiding the clipper button entirely
until the user opens an "Advanced" section, on the theory that the
default is almost always right. Deferred until we have evidence on how
often users tweak the default after Phase 1 lands.
