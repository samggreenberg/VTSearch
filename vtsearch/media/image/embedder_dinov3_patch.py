"""Image embedder — DINOv3 patch-region mode (``dinov3_patch``).

Produces a CLS vector **plus** a per-patch grid + HAC region tree per
image.  ~30× slower per image and ~100× more storage than the
single-vector variant, but enables region similarity, region-aware MLP
scoring, and (in v2) region voting.

For the single-vector variant of the same backbone see
``embedder_dinov3_single.py``.  Both share weights via
``_dinov3_shared.py``.

DINOv3 produces high-quality per-patch tokens and exposes a clean
CLS→patch attention map (cleaner than DINOv2's thanks to Gram-anchored
"storage" / register tokens that absorb high-norm artifacts).  We slice
the 4 register tokens out of the token sequence before reshaping to the
14 × 14 patch grid.

DINOv3 weights are gated on Hugging Face — running this embedder requires
the ``HF_TOKEN`` env var to be set to a user token that has accepted the
DINOv3 licence at
``https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m``.
"""

from __future__ import annotations

from typing import Optional

from vtsearch.media.image._dinov3_shared import _Dinov3Base
from vtsearch.media.image._image_bulk import bulk_patch_forward_image_files
from vtsearch.media.patch_embed import PatchEmbedOutput


class ImageDinov3PatchEmbedder(_Dinov3Base):
    """Embeds images using DINOv3 ViT-B/16 with CLS pooling + patch regions.

    Output dimension: 768.  Per-image side-channel: HAC region tree +
    14 × 14 patch grid (see :func:`vtsearch.media.patch_embed`).
    """

    @property
    def name(self) -> str:
        return "dinov3_patch"

    @property
    def supports_patch_regions(self) -> bool:
        return True

    def _patch_forward_impl(self, media: dict) -> Optional[PatchEmbedOutput]:
        return self._compute_patch_output(media)

    def _patch_forward_bulk_impl(
        self, medias: list[dict]
    ) -> list[Optional[PatchEmbedOutput]]:
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
                label="DINOv3 patch",
            )


EMBEDDER = ImageDinov3PatchEmbedder()
