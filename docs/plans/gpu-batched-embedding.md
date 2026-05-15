# GPU batched embedding

Status: **Phase A + B + C landed.** The deferred audio/video bulk
overrides and the DINOv2/v3/EUPE backbone-fusion follow-up are still
open.
Tracks: feature-brainstorm.md §12.2.

The dataset loader already routes embedding through
`MediaEmbedder.embed_media_bulk()` (see `vtsearch/datasets/loader_folder.py`)
and `MediaEmbedder.patch_forward_bulk()` (via `_bulk_patch_forward_files`
at `loader_folder.py:134`). Phase A and B replaced the per-item default
loops with real batched GPU forwards on the image and text embedders
that benefit most. Phase C dropped the tempfile detour from clip
re-embedding inside `_fixup_clip_md5_and_embeddings`, so clipped
sub-items now flow through the same bulk surface as folder loads.

## Phase A — Bulk image + text embedders ✅ Complete

Landed in PR #1341. `_embed_media_bulk_impl` now overrides on:

- `ImageSiglipEmbedder` (`embedder_siglip.py`)
- `ImageSiglip2Embedder` (`embedder_siglip2.py`)
- `ImageClipEmbedder` (`embedder_clip.py`)
- `_Dinov2Base`, `_Dinov3Base` (shared bases — covers single+patch CLS)
- `_EupeBase`
- `TextE5Embedder`, `TextBgeEmbedder` (sentence-transformers natively batches)

Each image embedder grows a `_forward_pil_batch(images)` helper that
runs the processor + model on a list of PIL images and returns an
`(N, D)` numpy array. The shared driver
`vtsearch.media.image._image_bulk.bulk_embed_image_files` decodes
images one-by-one (PIL decode is cheap and inherently per-file), then
calls the batched forward in chunks. Failures are isolated to the
offending image — it's dropped from the batch with `None` in its
position.

**Batch size**: read once at first use from
`VTSEARCH_EMBED_BATCH_SIZE` (default 32) on `MediaEmbedder` as the
`embed_batch_size` property; subclasses may override.

**Progress**: emitted once per *batch*, not per item, with
`(i+1)*batch_size / total` so the bar still ticks. Matches the
existing `"Embedding N/M..."` status string for compatibility.

Audio CLAP + video X-CLIP / LanguageBind: still on the default
per-item loop (see Open follow-ups below).

## Phase B — Bulk patch_forward ✅ Complete

Landed in PR #1341. `_patch_forward_bulk_impl` now overrides on:

- `ImageDinov2PatchEmbedder` → `_Dinov2Base._patch_forward_pil_batch`
- `ImageDinov3PatchEmbedder` → `_Dinov3Base._patch_forward_pil_batch`
- `ImageEupePatchEmbedder` → `_EupeBase._patch_forward_pil_batch`

The driver `bulk_patch_forward_image_files` mirrors the single-vector
path: decode PIL per-file, then call the batched forward in chunks.
`_bulk_patch_forward_files` in `loader_folder.py:134` invokes
`patch_forward_bulk` exactly once per folder load (asserted in
`tests/detectors/test_image_bulk_embedding.py:502`).

Single-vector and patch passes still happen separately — fusing them
into one forward (so DINOv3 doesn't run the backbone 2× per image) is
listed under Open follow-ups.

## Phase C — Clip re-embed ✅ Complete

`_fixup_clip_md5_and_embeddings` in
`vtsearch/datasets/load_pipeline.py` used to write each clip needing
recomputation to `tempfile.mkstemp` and call `embed_file` one clip at
a time through `_reembed_clip`. Two costs piled on top of each other:
serial forward passes (no GPU batching) and tempfile write/unlink
churn dwarfing the decode cost on short clips.

Phase C replaces that with a single `embedder.embed_media_bulk` call
per invocation:

1. The clip loop computes per-clip MD5s from `_clip_content_bytes` (or
   the boundary-hash fallback for metadata-only video clips) and
   collects the clips that need embedding into one batch.
2. `_build_clip_embed_input` builds minimal media dicts that hand the
   embedder the in-memory content directly — `media_bytes` for
   audio/image, `media_string` for text — with **no** `media_path`.
3. `_resolve_clip_embedder` picks the first registered embedder for
   the media type (same fallback chain `embed_file` used).
4. `embed_media_bulk` runs once; results scatter back into the clip
   dicts by index. `None` entries leave the parent embedding intact,
   matching the old `except: pass` contract.
5. Progress events route through the embedder's `_on_progress`,
   scaled back into clip-list coordinates so the existing
   `on_progress(clip_idx, total_clips, "embedding")` API keeps
   reporting against the full clip total.

To make the bulk path work without disk I/O, the relevant single-item
embedders learned to accept in-memory content:

- `bulk_embed_image_files` / `bulk_patch_forward_image_files` in
  `vtsearch/media/image/_image_bulk.py` decode `media_bytes` via
  `io.BytesIO` when present, falling back to `media_path`. Every
  Phase A/B image embedder benefits automatically.
- `AudioClapEmbedder._embed_media_impl` accepts `media_bytes` and
  passes a `BytesIO` to `librosa.load`. Audio still rides the default
  per-item bulk loop (no native CLAP batch override yet) but skips
  the tempfile.
- `TextE5Embedder` and `TextBgeEmbedder` factored their string
  resolution into `_read_text(media)`, which prefers
  `media_string` over `media_path`. Both the single-item and bulk
  hooks share it.

`_reembed_clip` is gone. `_fixup_clip_md5_and_embeddings` no longer
imports `vtsearch.detectors.resolver.embed_file` — the test suite
asserts both via `tests/io/test_clip_reembed_bulk.py`.

### Acceptance

All four acceptance criteria from the original plan are met:

1. `_fixup_clip_md5_and_embeddings` calls `embed_media_bulk` exactly
   once per `(clip-list, media_type)` invocation; `_reembed_clip` is
   deleted and `tempfile.mkstemp` / `embed_file` no longer appear on
   the clip-embed path.
2. The existing clipper tests in `tests/detectors/test_clipper_workflow.py`
   and the converter suites still pass — MD5s, origins, and parent-
   embedding fallback all match.
3. `tests/io/test_clip_reembed_bulk.py` covers the new contract:
   parametrised over audio/image/text, plus failure-fallback and
   no-embedders/no-tempfile cases.
4. Hand-timed GPU win is deferred to follow-up profiling (CLAP bulk
   override pending; for image clips on SigLIP the batched forward
   already shows a measurable improvement out-of-band).

## Open follow-ups (deferred, not in any phase)

- **Audio CLAP + CLAP-Music bulk override**. Decode is the
  bottleneck and adds I/O complexity; smaller GPU win than image
  but still meaningful for big audio imports. `librosa` is happy to
  decode a list serially while the model batches.
- **Video X-CLIP / LanguageBind bulk override**. Tricky because
  X-CLIP at batch 32 with 8 frames each is ~640 MB of activations
  and can OOM on 8 GB cards. Likely wants a smaller default
  `embed_batch_size` (e.g. 8) on the video embedders.
- **Fuse single-vector + patch forward on DINOv2/DINOv3/EUPE**. Today
  the backbone runs twice per image (once for `embed_media_bulk`,
  once for `patch_forward_bulk`). Fusing requires changing the
  loader to call a single combined hook and split the outputs —
  worth it if profiling shows the backbone forward is the dominant
  cost.

## Risks (carried over from Phase A/B)

- **VRAM**: 32 × 224×224 ViT-B is ~80 MB of activations — fine on
  8 GB. Video batches are the OOM risk; see follow-ups.
- **Determinism**: per-batch vs per-item forwards can differ at the
  last-bit level due to kernel choice. Tests assert near-equal
  (`np.allclose(..., atol=1e-5)`), not exact equality.
- **Per-image failures**: one bad PIL decode shouldn't fail the
  whole batch. The decode loop catches per image and replaces the
  spot with `None`; the GPU forward sees a smaller batch. Same
  contract applies to Phase C clip re-embed.
