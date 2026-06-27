# Browse first-paint thumbnail load delay (~3s grayscale bins)

**Status:** **CONFIRMED + FIXED (2026-06-26)** on branch
`claude/thumbnail-memoise-on-the-fly` (off `dev`, NOT merged/deployed). Cause is
on-the-fly `/thumbnail` generation because the medias carry **no precomputed
`thumbnail_bytes`**, and the per-thumbnail cost scales with **source image
resolution**. Two fixes landed: **B** (runtime memoise — each tile generated at
most once, then streamed) and **A** (persist thumbnails at save — generate any
missing image thumbnail at `export_dataset_to_file` so reloaded datasets stream
bytes with zero regeneration and a fast first paint). See "Confirmed findings"
and "Fix" below. The earlier full-res `/image` hypothesis stays retracted.

## Symptom

When the VTSBrowse canvas first loads, there's a solid ~3 seconds where only the
grayscale/density-shaded hex bins are visible before any imagery paints in. The
reporter's dataset shows only ~30 bins in that view.

The bins themselves paint instantly: they're drawn purely from tile cell data
(`count` → colormap), no images involved. The delay is entirely the imagery
filling those bins.

## Confirmed findings (2026-06-26, rack7n03:11850, Caltech-101 (L) demo)

Measured by driving the live GRID server (tunnel `localhost:11850`) and by
introspecting the dataset pickle with the project loader.

**1. Default browse fetches `/thumbnail`, not `/image` (full-res retraction holds).**
Thumbnail responses are small, 384px-capped JPEGs (6–13 KB; observed dims
384×116, 384×144, …). No multi-MB `/image` originals were requested at the
default thumbnail size. The retracted full-res theory is correctly dead.

**2. The medias have NO `thumbnail_bytes` — even this demo dataset.**
Loading the saved pickle via `load_dataset_from_pickle` and sampling 120 medias:
`thumbnail_bytes` is present as a key but `None` for **0/120**. So every
`/thumbnail` request takes the on-the-fly fallback (`_resolve_display_image` +
`make_image_thumbnail`, decode+resize at request time) — never the fast
`cached_thumbnail_response` byte-stream path.

**3. Root cause — external-dir datasets lose `thumbnail_bytes` on save and never
regenerate them on load.** `_write_demo_cache` strips both `media_bytes` and
`thumbnail_bytes` from the pickle when `store_external` is true
(`vtscore/datasets/loader_demo.py:403`) — true for any image/audio/video
dataset backed by an on-disk media dir (demos, **folder imports**). On load, the
loader resolves `media_bytes` back from the external dir but does **not**
regenerate `thumbnail_bytes`. Confirmed: after load, `media_bytes` is populated
(5–16 KB each) while `thumbnail_bytes` is `None`. Ingest *does* generate
thumbnails (`vtscore/media/image/media_type.py:160` via `make_image_thumbnail`),
but that artifact is discarded for external-dir datasets and not rebuilt. This
is the same gap as the docstring's "old pickles / thin loads", but broader: it
hits **every external-dir-backed image dataset**, including the reporter's
likely folder import.

**4. The delay magnitude is governed by source image resolution.**
Caltech-101 sources are tiny (~0.04–0.11 MP, 402×135 etc.), so even the
on-the-fly path is fast — and that is exactly why a quick test on a demo dataset
*fails to reproduce* the reporter's ~3s. Measured on the live server:

| Test (30 distinct cold thumbs, concurrent) | Wall-clock | per-thumb TTFB |
|---|---|---|
| `/thumbnail` (on-the-fly, Caltech 0.06 MP sources) | **0.24 s** | 40–86 ms |
| `/thumbnail?region=…` (forces regen) | 0.22 s | 34–51 ms |

Generation cost vs. megapixels (`make_image_thumbnail`, single SLURM node,
`VTSEARCH_TORCH_THREADS=1`):

| Source MP | src KB | 1× decode+resize | 30× serial | 30× across 8 threads |
|---|---|---|---|---|
| 0.06 (Caltech) | 37 | 2.4 ms | 72 ms | 53 ms |
| 0.5 | 307 | 21 ms | 637 ms | 153 ms |
| 2.0 | 1214 | 66 ms | 1969 ms | 329 ms |
| 6.0 | 3637 | 107 ms | 3219 ms | 580 ms |
| 12.0 | 7263 | 187 ms | 5604 ms | 1188 ms |
| 24.0 | 14549 | 366 ms | 10970 ms | 2022 ms |

PIL's libjpeg decode + LANCZOS resize release the GIL, so 8 threads give ~3–5×
over serial — but the single-worker server still funnels ~30 multi-MP decodes
into seconds. A dataset of ~6–24 MP photos reaches the reported ~3 s for ~30
bins; real-world it's worse because every request also passes the
`@app.before_request` `_state_lock` state-sync and the browser caps at ~6
concurrent connections. (Caltech can't reproduce ~3 s; a large-photo dataset
will.)

**5. No server-side generated-thumbnail cache.** The on-the-fly branch in
`media_thumbnail` regenerates from scratch every cold request — nothing is
memoised back onto the media dict — so the cost recurs on every fresh
browse-canvas zoom/pan into not-yet-fetched bins, not just first paint. (The
only thing that makes the *second* fetch of the same item fast is the browser's
ETag/304 + its own decoded-image cache, client-side.)

## Retracted hypothesis (kept for the record)

The earlier "coarse/top view has large hexes, crosses 384, fetches multi-MB
`/image` originals" theory was wrong twice over: (1) hex *screen* size is held
~constant across zoom by level selection, so the top view does not have larger
hexes; (2) `useFullResThumbs` keys off the constant `targetRadius`, not the
rendered radius, so it can't be a per-level effect. Full-res only engages if the
user enlarges the thumbnail-size knob past ~XL (`targetRadius > 192/dpr`). Do not
resurrect it.

## Fix (implemented)

The root cause is items 2–3: external-dir datasets serve every tile through
request-time decode+resize. Two complementary fixes landed; **C** was considered
and rejected in favour of **A** (persist beats rederive — see below).

- **A. Persist thumbnails at save (chosen for first paint).**
  `export_dataset_to_file` (`vtscore/datasets/loader.py`, the single choke point
  for every `saved_datasets/*.pkl` write — registry, staging, promote) now
  generates a `thumbnail_bytes` for any **image** media that lacks one, from the
  in-memory `media_bytes`, right before serialising. Image *demos* never produce
  `thumbnail_bytes` at build time (`load_demo_source` doesn't, only the real
  `load_media_data` ingest does) and `_write_demo_cache` strips them, so this is
  where the gap is closed. A re-saved dataset reloads straight onto the fast
  `cached_thumbnail_response` byte-stream path → fast first paint, zero
  load-time regeneration. One-time, offline cost at save. Thumbnails are tiny
  (~10 KB) and immutable per source, so no staleness/invalidation and negligible
  pickle growth. **Requires re-saving** to benefit existing datasets (covered at
  runtime meanwhile by B). Test: `test_image_thumbnail_generated_at_export`.
- **B. Lazy memoise on the request path (runtime, helps existing datasets).**
  On the on-the-fly branch of `media_thumbnail`
  (`vtsearch/routes/media/list.py`), generate once and cache the bytes back onto
  the in-memory media dict's `thumbnail_bytes`, so subsequent cold fetches stream
  instead of re-decoding. Bounds total work to one generation per *viewed* item
  and kills the zoom/pan recurrence (item 5). In-memory only (respects No
  Persisted Vectors/MLPs — the bytes ride only in the live context, never written
  to disk unless the user exports). Does not fix the very first paint by itself;
  A does. Test: `test_on_the_fly_thumbnail_is_memoised`.
- **C. Background warm after load — rejected.** A daemon filling missing
  `thumbnail_bytes` after load (mirroring `_warmup_embedder_async`) was the other
  candidate for first paint, but it pays the decode cost on *every* load/dataset
  switch and, being a background race, doesn't *guarantee* the first paint is
  ready. Persisting at save (A) pays once, offline, and guarantees a fast reload.
  Not implemented.

## How this was confirmed (repro recipe)

1. Tunnel up (`./vtsearch-tunnel.sh`), load the dataset via
   `POST /api/datasets/registry/<id>/load`, then send requests with the
   `X-Dataset-Id: <id>` header (the active context is a per-request header set
   by `active-context.interceptor.ts`, **not** a server-side activate call — a
   curl without it hits the empty default context and looks "not loaded").
2. Fire N concurrent cold `/thumbnail` requests; watch wall-clock + per-request
   `time_starttransfer`. Force the on-the-fly path with `?region=0,0,1,1`.
3. Introspect `thumbnail_bytes` presence + source MP by loading the saved pkl
   with `vtscore.datasets.loader_pickle.load_dataset_from_pickle` and a PIL
   `Image.open` over `media_bytes`.

## Key file/line references

| What | Location |
|------|----------|
| `/thumbnail` route (precomputed vs on-the-fly) | `vtsearch/routes/media/list.py:516-552` |
| `cached_thumbnail_response` (fast byte-stream) | `vtsearch/routes/_shared.py:33-54` |
| `image_thumbnail_response` (decode+resize) | `vtsearch/routes/_shared.py:57-103` |
| `make_image_thumbnail` (None only on decode fail) | `vtscore/media/image/thumbnail.py:63-104` |
| Ingest sets `thumbnail_bytes` | `vtscore/media/image/media_type.py:134-161` |
| External-dir save strips `thumbnail_bytes` | `vtscore/datasets/loader_demo.py:399-405` |
| Embedder warmup pattern (mirror for C) | `vtscore/datasets/load_pipeline.py:_warmup_embedder_async` |
| `useFullResThumbs` (global, keys off targetRadius) | `frontend/.../browse-canvas/browse-canvas.component.ts:320-322` |

## Open follow-ups

- **Deploy:** rebuild not needed (backend-only); restart `app.py` to pick up the
  route change. To make *existing* registry datasets fast on first paint, re-save
  them (re-promote the demo / re-export) so fix A embeds the thumbnails — until
  then they ride fix B (one generation per viewed tile at runtime).
- Audio/video external-dir datasets have the same strip-on-save gap, but their
  thumbnails are waveforms/keyframes (not `make_image_thumbnail`). Fix A only
  covers images; extend the export-time generation per media type if those show
  the same first-paint lag.
