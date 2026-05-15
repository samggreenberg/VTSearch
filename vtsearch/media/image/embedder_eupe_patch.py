"""Image embedder — EUPE patch-region mode (``eupe_patch``).

Produces a CLS vector **plus** a per-patch grid + HAC region tree per
image using the ``facebookresearch/EUPE`` ViT-B/16 backbone.  ~30×
slower per image and ~100× more storage than the single-vector variant,
but enables region similarity, region-aware MLP scoring, and (in v2)
region voting.

For the single-vector variant of the same backbone see
``embedder_eupe_single.py``.  Both share weights via ``_eupe_shared.py``.

EUPE's attention path uses ``torch.nn.functional.scaled_dot_product_attention``
which doesn't return weights, so :attr:`patch_saliency` falls back to a
CLS-cosine-similarity proxy (softmax of each patch's cosine similarity
to the CLS vector).  EUPE outputs are bound by Meta's FAIR Noncommercial
Research Licence — see :attr:`license_notice`.
"""

from __future__ import annotations

from typing import Optional

from vtsearch.media.image._eupe_shared import _EupeBase
from vtsearch.media.image._image_bulk import bulk_patch_forward_image_files
from vtsearch.media.patch_embed import PatchEmbedOutput


class ImageEupePatchEmbedder(_EupeBase):
    """Embeds images using facebookresearch/EUPE ViT-B/16 with CLS + patch regions.

    Output dimension: 768.  Per-image side-channel: HAC region tree +
    14 × 14 patch grid (see :func:`vtsearch.media.patch_embed`).
    """

    @property
    def name(self) -> str:
        return "eupe_patch"

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
        if self._model is None or self._preprocess is None:
            return [None] * len(medias)
        with self._embed_lock:
            return bulk_patch_forward_image_files(
                medias,
                forward_pil_batch=self._patch_forward_pil_batch,
                batch_size=self.embed_batch_size,
                on_progress=self._on_progress,
                label="EUPE patch",
            )


EMBEDDER = ImageEupePatchEmbedder()
