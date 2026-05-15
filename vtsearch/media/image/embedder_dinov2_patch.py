"""Image embedder — DINOv2 patch-region mode (``dinov2_patch``).

Produces a CLS vector **plus** a per-patch grid + HAC region tree per
image.  ~30× slower per image and ~100× more storage than the
single-vector variant, but enables region similarity, region-aware MLP
scoring, and (in v2) region voting.

For the single-vector variant of the same backbone see
``embedder_dinov2_single.py``.  Both share weights via
``_dinov2_shared.py``.

DINOv2 weights are **ungated** on Hugging Face.  Image input is the HF
default (224² for the standard processor) with patch size 14 → 16 × 16
= 256 patch tokens.  There is no text encoder, so :attr:`supports_text`
is ``False``.
"""

from __future__ import annotations

from typing import Optional

from vtsearch.media.image._dinov2_shared import _Dinov2Base
from vtsearch.media.image._image_bulk import bulk_patch_forward_image_files
from vtsearch.media.patch_embed import PatchEmbedOutput


class ImageDinov2PatchEmbedder(_Dinov2Base):
    """Embeds images using DINOv2 ViT-B/14 with CLS pooling + patch regions.

    Output dimension: 768.  Per-image side-channel: HAC region tree +
    14 × 14 patch grid (see :func:`vtsearch.media.patch_embed`).
    """

    @property
    def name(self) -> str:
        return "dinov2_patch"

    @property
    def supports_patch_regions(self) -> bool:
        return True

    def _patch_forward_impl(self, media: dict) -> Optional[PatchEmbedOutput]:
        return self._compute_patch_output(media)

    def _patch_forward_bulk_impl(self, medias: list[dict]) -> list[Optional[PatchEmbedOutput]]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return [None] * len(medias)
        with self._embed_lock:
            return bulk_patch_forward_image_files(
                medias,
                forward_pil_batch=self._patch_forward_pil_batch,
                batch_size=self.embed_batch_size,
                on_progress=self._on_progress,
                label="DINOv2 patch",
            )


EMBEDDER = ImageDinov2PatchEmbedder()
