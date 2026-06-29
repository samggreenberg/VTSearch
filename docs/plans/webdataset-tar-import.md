# WebDataset-style tar-sharded import (microvent / multivent-raw)

**Status:** Phase A shipped; Phase B shipped (sub-file clip windows, windowed
manifest import, archive-member `MediaSource`, audio-mimetype nuances); the
GUI/docs-polish follow-up (item 5) shipped — user-docs section for the
windowed-manifest schema landed in `docs/user/USER_GUIDE.md`. Only the
cross-browser AAC/`audio/mp4` validation pass (needs a live browser) remains —
see Open follow-ups.

## Problem

Load large WebDataset-style corpora (GRID **microvent**, **multivent-raw**) into
VTSearch for VTSBrowse / similarity search over audio/video events, where the
media lives inside **tar shards** (`microvent/videos/shard_000000.tar`, …), the
embeddings are **precomputed** (512-dim CLAP), and only a **filtered subset** is
wanted. The corpora are far too large to extract a second on-disk copy
(multivent-raw's `videos/` alone is **4.1 TB across 667 shards**), so import and
playback must be **no-extraction**: serve a single tar member on demand.

The pre-existing `local_archive` import path (`vtscore/datasets/archive.py`)
extracts whole archives under `DATA_DIR` — a non-starter at these sizes — and the
byte routes only read a local file path. This plan closes that gap.

## Phase A — shipped (no-extraction byte serving + whole-member import)

What landed:

- **`vtscore/datasets/archive_stream.py`** — streams one tar/zip member with a
  cached `{member: info}` index (metadata only, never bytes), reading just a byte
  *range* by seeking within the member's file object. `read_member`,
  `read_member_range`, `member_size`, `archive_member_ref`,
  `build_archive_member_origin`, and the `local_archive_member` origin name.
- **Unified, archive-aware byte resolution.** `MediaType._resolve_media_bytes`
  (`vtscore/media/base.py`) gained an archive-member branch; the route-level
  `_resolve_bytes` (`vtsearch/routes/media/list.py`) and the crop/seed
  `_resolve_media_bytes` (`vtsearch/routes/media/server.py`) now delegate to it,
  so inline bytes → lazy clip → **archive member** → local path → remote URL is a
  single chain instead of three path-only duplicates.
- **True Range streaming for playback.** `_send_streamed_range` +
  `_archive_member_response` serve `/video` and `/audio` for archive-member media
  by reading only the requested slice out of the shard (a few seconds of playback
  transfers a few seconds of bytes), never extracting or fully buffering.
- **`local_archive_member` importer** — imports a manifest-selected subset of
  members as **whole-member** media with their precomputed vectors, reading **no
  member data** (only each shard's tar headers, to confirm the member exists and
  record its size). md5 is synthesized from `archive::member` (the corpora don't
  use content-based cross-dataset label transfer; see the importer docstring).
- **Manifest reader** —
  `read_npz_archive_member_rows` (`_npz_vectors.py`) reads `{vectors, members,
  archives, filenames?, embedder_name?}`, broadcasting a scalar `archives` for
  one-shard manifests.

Net effect: a filtered subset of either corpus imports with **zero disk growth**
and plays back whole chunks by streaming single members.

## Phase B — shipped (sub-file clip windows + cross-dataset re-derivation)

What landed:

- **Sub-file clip windows (gap 3).** One member can yield multiple media
  carrying `clip_start` / `clip_end`. Both players now seek/loop within the
  window **display-only**: the video player already did
  (`video-player.component.ts`); the audio player gained the same logic
  (`audio-player.component.ts`, `(loadedmetadata)` + a 100 ms boundary poll).
  We never byte-slice these AAC/MP4 members — `clip_recipe`
  (`vtscore/media/lazy_clip.py`) now returns `None` for any archive-member
  media, so the byte routes serve the **whole** member and the player handles
  the window. `batch_medias` already passes `clip_start` / `clip_end` through.

- **Windowed precomputed-embedding import (gap 4).** The manifest schema gained
  optional `clip_start` / `clip_end` / `window_id` columns
  (`_npz_vectors.read_npz_archive_member_rows`, broadcast like `archives`; a
  `NaN`/blank extent means "whole-member row"). The importer fans one member
  into N windowed media, each its own searchable item with its own precomputed
  vector, carrying the clip fields top-level (player) **and** in `origin.params`
  (survives the pickle + feeds the source). The synthesized md5 + `origin_name`
  fold in a per-window suffix (`window_suffix`: `#<window_id>`, else
  `@<clip_start>`) so dedup / voting / display stay unique per window.

- **Archive-member `MediaSource` (gap, cross-dataset re-derivation).**
  `vtscore/datasets/sources/local_archive_member.py` is a manifest-backed
  source that re-supplies a member's (or a specific *window's*) precomputed
  vector by `(archive, member[, window])`, returning a `FetchedItem` with
  `path=None` + `embedding` set. `example_sort_origin`
  (`vtsearch/routes/media/server.py`) now sorts directly on that vector when a
  source returns no path (cropping is rejected — it needs bytes this path never
  materialises). This closes the last unsupported flow (per-media cross-dataset
  "Find" used to 400 with "no media source").

- **Audio mimetype + container nuances (gap 4).** The `/audio` route now picks
  an **audio** `Content-Type` per the member container
  (`_audio_member_mimetype`): `.aac` → `audio/aac`, an MP4-container audio chunk
  (`.mp4` / `.m4a`, which `mimetypes` would call `video/mp4`) → `audio/mp4`,
  `audio/x-wav` normalised to `audio/wav`, unknown → `audio/wav`. multivent-raw
  audio that rides inside the video MP4 is served (as today) through the
  **video** element's audio track.

## Open follow-ups

Tracked here, not in the PR body.

1. **GUI / docs polish (Phase B item 5).** *Shipped.* `docs/user/USER_GUIDE.md`
   now carries an "Archive members, no extraction (WebDataset shards)" section
   under "Loading a dataset" covering the windowed-manifest schema (the `.npz`
   array table, clip windows, display-only playback) and the no-extraction
   import flow. No existing doc screenshot frames this importer's form, so
   nothing was queued in `docs/user/screenshots-reshoot-queue.md`.

2. **Cross-browser AAC / `audio/mp4` validation.** The `/audio` route emits the
   right mimetype, but actual AAC / MP4-audio-track playback across browsers
   still needs a manual pass once the corpora are wired into a live GUI session
   (no browser in the standard cloud container).
