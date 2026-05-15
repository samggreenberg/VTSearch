# GPU batched embedding

Status: **In progress** (Phase A landing first).
Tracks: feature-brainstorm.md §12.2.

Today the dataset loader already routes embedding through
`MediaEmbedder.embed_media_bulk()` (see `vtsearch/datasets/loader_folder.py`),
but **no embedder overrides the bulk hook** — every concrete embedder
inherits the default `_embed_media_bulk_impl` that just loops
`embed_media()` per item under the global `_embed_lock`. On GPU that
leaves a 5–10× speedup on the table for any folder/HF import.

This plan adds real batched forward passes to the embedders that benefit
most, without changing any caller. The hook + plumbing already exist;
this is purely a per-embedder override on top.

## Goals

- 5–10× faster folder/HF imports on GPU for image and video datasets.
- Batched patch-forward for the DINOv2 / DINOv3 / EUPE patch variants,
  fusing the two-passes-per-image situation called out in
  `loader_folder.py:_bulk_patch_forward_files`.
- No change to the embedder ABC for callers: `embed_media_bulk()` and
  `patch_forward_bulk()` are the only entry points the loader uses.

## Non-goals

- **Clip re-embed batching** (`_fixup_clip_md5_and_embeddings` in
  `vtsearch/datasets/load_pipeline.py`) — currently writes each clip
  to a tempfile and goes through `embed_file()` one at a time. The win
  is real but the refactor touches audio decode / text / image PIL
  paths and wants its own follow-up PR.
- A unified concurrency story across datasets — `_embed_lock` still
  serialises GPU forwards across embedder instances. Orthogonal.
- New tunables for the user; batch size is an internal knob with one
  env override.

## Phase A — Bulk image + text embedders

Override `_embed_media_bulk_impl` on:

- `ImageSiglipEmbedder` (`embedder_siglip.py`)
- `ImageSiglip2Embedder` (`embedder_siglip2.py`)
- `ImageClipEmbedder` (`embedder_clip.py`)
- `_Dinov2Base`, `_Dinov3Base` (shared bases — covers single+patch CLS)
- `_EupeBase`
- `TextE5Embedder`, `TextBgeEmbedder` (sentence-transformers natively batches)
- Audio CLAP + video X-CLIP / LanguageBind: deferred to a follow-up
  (decode is the bottleneck; smaller win, more I/O complexity).

**Shape**: each image embedder grows a `_forward_pil_batch(images,
batch_size)` helper that runs the processor + model on a list of PIL
images and returns an `(N, D)` numpy array. `_embed_media_bulk_impl`
decodes images one-by-one (PIL decode is cheap and inherently
per-file), then calls the batched forward in chunks. Failures are
isolated to the offending image (we drop it from the batch with `None`
in its position).

**Batch size**: read once at first use from
`VTSEARCH_EMBED_BATCH_SIZE` (default 32). Lives on `MediaEmbedder` as
`embed_batch_size` property; subclasses may override. No per-user
setting yet.

**Progress**: emitted once per *batch*, not per item, with
`(i+1)*batch_size / total` so the bar still ticks. Matches existing
"Embedding N/M..." status string for compatibility.

## Phase B — Bulk patch_forward

`MediaEmbedder` gains:

```python
def patch_forward_bulk(self, medias: list[dict]) -> list[Optional[PatchEmbedOutput]]:
    ...
def _patch_forward_bulk_impl(self, medias) -> list[Optional[PatchEmbedOutput]]:
    # default: loop per-item, emit progress
```

Override on `_Dinov2Base._compute_patch_output_bulk`,
`_Dinov3Base._compute_patch_output_bulk`, `_EupeBase._compute_patch_output_bulk`.
Update `_bulk_patch_forward_files` in `loader_folder.py` to call the
new bulk hook.

Single-vector and patch passes still happen separately for now — fusing
them into a single forward (so DINOv3 doesn't run 2× per image) is a
nice-to-have but requires changing the loader's `embed_media_bulk →
patch_forward_bulk` ordering. Scoped out of Phase B; revisit if
profiling shows it's the dominant cost.

## Phase C (deferred) — Clip re-embed

`_fixup_clip_md5_and_embeddings` in `vtsearch/datasets/load_pipeline.py`
loops per clip, writes to a tempfile, and calls `embed_file()`. For
audio/image clips this is the worst remaining single-call hot spot.

The refactor is:
1. Group clips by media_type.
2. For each group, decode `content_bytes` into in-memory PIL / wav /
   text inputs without touching disk.
3. Call the appropriate bulk surface on the embedder.

Punted because each media type needs its own "bytes → embedder input"
path and audio decoders aren't trivial. Will land as a follow-up plan
doc once Phase A/B are in.

## Risks

- **VRAM**: 32 × 224×224 ViT-B is ~80 MB of activations — fine on 8 GB.
  X-CLIP at batch 32 with 8 frames each is ~640 MB and could OOM on
  small cards; leaving video to a follow-up sidesteps it.
- **Determinism**: per-batch vs per-item forwards can differ at the
  last-bit level due to kernel choice. Tests assert near-equal
  (`np.allclose(..., atol=1e-5)`), not exact equality.
- **Per-image failures**: one bad PIL decode shouldn't fail the whole
  batch. The decode loop catches per image and replaces the spot with
  `None`; the GPU forward sees a smaller batch.

## Acceptance

1. Each new bulk impl produces vectors `np.allclose` to the per-item
   path for the same inputs. Covered by a single shared parametrised
   test in `tests/io/test_bulk_embedding.py`.
2. `_bulk_patch_forward_files` calls `patch_forward_bulk` exactly once
   per folder load (assertion in tests/detectors).
3. `./run-tests.sh` green (existing bulk tests still pass — the
   default-loop contract is preserved for embedders that don't
   override).
4. Hand-timed on GPU (out-of-band, not CI): SigLIP folder of 200
   images < 1/5 of pre-PR wall time.
