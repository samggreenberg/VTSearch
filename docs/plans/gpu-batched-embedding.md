# GPU batched embedding

Status: **Phase A + B landed** (PR #1341). Phase C and the deferred
audio/video bulk overrides are still open.
Tracks: feature-brainstorm.md §12.2.

The dataset loader already routes embedding through
`MediaEmbedder.embed_media_bulk()` (see `vtsearch/datasets/loader_folder.py`)
and `MediaEmbedder.patch_forward_bulk()` (via `_bulk_patch_forward_files`
at `loader_folder.py:134`). Phase A and B replaced the per-item default
loops with real batched GPU forwards on the image and text embedders
that benefit most. Phase C addresses the last per-call hotspot —
clip re-embedding inside `_fixup_clip_md5_and_embeddings`.

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

## Phase C — Clip re-embed (next)

`_fixup_clip_md5_and_embeddings` in `vtsearch/datasets/load_pipeline.py:385`
is the worst remaining single-call hotspot. Today, for every clip
flagged `needs_recompute`:

1. `_clip_content_bytes` extracts `media_bytes` (audio/image) or
   `media_string` (text). Video is metadata-only and skips embedding.
2. `_reembed_clip` writes those bytes to a `tempfile.mkstemp` on disk
   with a media-type extension (`.wav` / `.png` / `.txt`).
3. `embed_file(tmp_path, media_type)` resolves the embedder by media
   type and calls `embedder.embed_media(media_from_path(tmp_path))`
   one clip at a time — no bulk hook used, no batching.
4. `os.unlink(tmp_path)` cleans up.

The per-clip tempfile dance defeats GPU batching twice: serial forward
passes, and the tempfile write/read churn dwarfs the actual decode
cost for small clips.

### Refactor

Drop the tempfile entirely and route through the bulk surface that
Phase A already added.

1. **Group clips by `media_type`** before re-embed. Today
   `_fixup_clip_md5_and_embeddings` is already called per `media_type`
   from `_apply_clipper` at line 312, so the grouping is implicit —
   we just need to collect the clips that pass the `needs_embed`
   gate into a single list per call instead of looping.
2. **Build embed-ready media dicts in-memory** from `content_bytes`.
   `embed_media()` expects either `media_path`, `media_bytes`
   (audio/image), or `media_string` (text). The clips already carry
   `media_bytes` / `media_string` in the form the embedders accept,
   so we can construct the dicts directly without going through disk:

   ```python
   # audio / image
   {"media_bytes": content_bytes, "origin_name": clip.get("origin_name", ""), ...}
   # text
   {"media_string": clip["media_string"], "origin_name": ..., ...}
   ```

   Verify each embedder's `_embed_media_impl` actually accepts
   `media_bytes` without a `media_path` — `embedder_clap.py` and the
   image embedders are the consumers; if any of them require a path,
   add a `media_bytes` branch (cheaper than a tempfile).
3. **Call `embedder.embed_media_bulk(media_dicts)`** once per
   `_fixup_clip_md5_and_embeddings` invocation. Resolve the embedder
   via the same `embed_file` fallback chain (`embedders_for_type`
   first hit), but cache the lookup so we don't re-resolve per clip.
4. **Scatter results back into clips** by index: the bulk call
   returns a same-length list of `Optional[np.ndarray]`; assign
   `clip["embedding"] = vec` where `vec is not None`, leaving the
   parent embedding intact otherwise (matches today's
   `except: pass` fallback).
5. **MD5 is still per-clip** and stays where it is — it's a cheap
   `hashlib.md5(content_bytes).hexdigest()` independent of the
   embedder.
6. **Progress** flows through the embedder's `_on_progress` already.
   The existing `on_progress(clip_idx, total_clips, "embedding")`
   callback in the for-loop becomes a single pre-call status update;
   per-batch progress comes from inside `embed_media_bulk`.

### Edge cases

- **Empty `content_bytes`** (video, metadata-only clips): keep the
  existing boundary-tag MD5 path at line 421. Those clips aren't
  re-embedded today and won't be after the refactor — they're just
  skipped in the bulk call.
- **Mixed `recompute` flags**: clips with `recompute=False` but no
  embedding still need re-embedding (`needs_embed` at line 411).
  Build the bulk list from `needs_embed`, not `recompute`.
- **Embedder lookup failure**: if `embedders_for_type(media_type)` is
  empty (`embed_file` returns None today), the bulk path should
  short-circuit and leave embeddings unchanged. Log once, not per
  clip.
- **`_reembed_clip` failure isolation**: today a single bad clip is
  caught by `except Exception: pass`. The bulk path already returns
  `None` per failed item via the `_embed_media_bulk_impl` contract,
  so per-clip isolation is preserved.

### Acceptance

1. `_fixup_clip_md5_and_embeddings` calls `embed_media_bulk` exactly
   once per `(clip-list, media_type)` invocation; no `tempfile.mkstemp`
   or `embed_file` calls remain in `_reembed_clip` (delete the
   function).
2. Existing clipper tests under `tests/io/` and
   `tests/converters/` stay green — MD5s and embeddings are
   identical (`np.allclose`, atol=1e-5) to the pre-refactor path on
   the same clip inputs.
3. Add one parametrised test in `tests/io/test_clip_reembed_bulk.py`
   asserting (a) bulk is called once and (b) failed clips fall back
   to parent embedding.
4. Hand-timed on GPU (out-of-band, not CI): a 200-clip audio
   clipper run completes in < 1/5 of pre-PR wall time.

### Estimated scope

~150 LOC delta in `load_pipeline.py` + one new test file. No ABC
changes — Phase A already established the surface. The only risk is
embedders that currently require `media_path`; a one-line audit of
each `_embed_media_impl` handles it.

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
