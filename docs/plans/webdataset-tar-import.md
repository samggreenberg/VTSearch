# WebDataset-style tar-sharded import (microvent / multivent-raw)

**Status:** Phase A shipped; Phase B planned (see Open follow-ups).

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

## Open follow-ups (Phase B — sub-file clip windows + full re-derivation)

Tracked here, not in the PR body.

1. **Sub-file clip windows (gap 3).** Let one member yield multiple media items
   carrying `clip_start` / `clip_end`. The serving + frontend side is largely
   already present: the video player seeks/loops on `clip_*`
   (`video-player.component.ts`) and `batch_medias` passes the fields through.
   Audio should use the **same display-only seek** (serve the whole member, seek
   in the player) rather than byte-slicing — the existing `lazy_clip` audio path
   is WAV-only (`_wav_slice`) and these corpora are AAC/MP4, which we do not want
   to decode server-side. Wire `lazy_clip._read_source_bytes` to an archive
   member only if a future windowing recipe needs sliced bytes; for display-only
   windowing nothing more than the clip fields is required.

2. **Windowed precomputed-embedding import (gap 4).** Extend the manifest schema
   with `clip_start` / `clip_end` (and optional `window_id`) so one member fans
   out into N windowed items (≈14 × 10 s CLAP windows per chunk), each its own
   searchable media with its own precomputed vector. The Phase A importer already
   emits one row → one item; windowing is "many rows per member" + carrying the
   clip fields onto each media. Keep md5 unique per *window* (fold the window id
   into the synthesized hash).

3. **Archive-member `MediaSource` for cross-dataset re-derivation.** The
   `MediaSource` / `FetchedItem` contract is path-based, so archive members
   (which have no on-disk path) currently have no source factory. Bytes already
   re-derive via the origin (the byte routes), and a full pickle persists
   embeddings, and the importer's `reload_from_origin` rebuilds the whole dataset
   from the manifest — so the only unsupported flow is **per-media** cross-dataset
   "Find" / example-sort-from-origin (`example_sort_origin` returns 400 "no media
   source"). Closing it needs either a bytes-returning `FetchedItem` extension or
   a manifest-backed source that re-supplies vectors by `(archive, member)`.

4. **Audio mimetype + container nuances.** The `/audio` route now derives the
   mimetype from the member extension (defaulting to `audio/wav`). microvent ships
   demuxed AAC; multivent-raw audio rides in the video MP4 (no separate audio
   dir), so most "audio events" are served through the **video** element's audio
   track. Validate AAC/`audio/mp4` playback across browsers when wiring the GUI.

5. **GUI / docs polish.** The importer registers in the server-tab picker with a
   `.npz` manifest field; add a short user-docs section and (if a screenshot
   frames the importer picker) queue a reshoot in
   `docs/user/screenshots-reshoot-queue.md`.
