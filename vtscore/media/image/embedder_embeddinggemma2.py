"""Image embedder - EmbeddingGemma 2 (google/embeddinggemma-2), images + text (#4837).

The image half of :mod:`vtscore.media._embeddinggemma2`: images and typed
queries land in the model's one shared 768-d space.  It rides the HF
cross-modal base for the media plumbing (single, bulk and PIL embedding) and
replaces the three forward passes, because this model has no
``get_image_features`` / ``get_text_features`` heads: an image enters the text
backbone as ~280 soft tokens and the vector is the backbone's pooled output.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtscore.config import EMBEDDINGGEMMA2_MODEL_ID, verify_image_processor_backend
from vtscore.media._embeddinggemma2 import load_embeddinggemma2, media_messages, pooled_forward, query_messages
from vtscore.media.embedder import resolve_embed_batch_size
from vtscore.media.image._cross_modal_shared import _CrossModalHFEmbedder

if TYPE_CHECKING:
    from PIL import Image


class ImageEmbeddingGemma2Embedder(_CrossModalHFEmbedder):
    """Embeds images using EmbeddingGemma 2 (google/embeddinggemma-2).

    * Images → 768-dimensional vectors: the vision encoder's soft tokens run
      through the text backbone, mean-pooled.
    * Text queries → 768-dimensional vectors in the same space, with the
      model's retrieval-query prefix.

    Loads the text backbone and the vision encoder (440M parameters), not the
    audio encoder; ``embeddinggemma2_audio`` is the audio half of the same
    checkpoint.
    """

    _label = "EmbeddingGemma 2"

    @property
    def name(self) -> str:
        return "embeddinggemma2"

    @property
    def display_name(self) -> str:
        return "EmbeddingGemma 2 (images)"

    @property
    def model_id(self) -> str:
        return EMBEDDINGGEMMA2_MODEL_ID

    @property
    def embedding_dim(self) -> int:
        return 768

    @property
    def eval_only(self) -> bool:
        """Withheld from the app: too slow to embed with (owner, 2026-10-13, #4837).

        On COCO Better it beat SigLIP by +0.08 to +0.09 session F-beta at every
        balance, but embeds 11x slower on a V100 and about 10 s an image on a
        CPU. It stays registered so the #4837 pile cell and sessions still load
        by name; no picker, default or API listing offers it.
        """
        return True

    @property
    def embed_batch_size(self) -> int:
        """8 by default, not the shared 32: each image is ~2,520 vision patches.

        #4837 measured a batch of 32 peaking at 33 GB on a V100 in fp32, for no
        throughput a batch of 8 does not already reach (6.6 against 6.4 images
        a second).  ``VTSEARCH_EMBED_BATCH_SIZE`` still overrides it.
        """
        return resolve_embed_batch_size(8)

    @property
    def description_wrappers(self) -> list[str]:
        """No wrappers: the query already carries the model's own task prefix.

        The CLIP-style prompt ensemble inherited from the cross-modal base was
        never measured here, and the model was trained on its ``task: search
        result | query:`` prefix rather than on "a photo of ..." templates, so
        Enrich Sort Descriptions is a no-op on this embedder as it is on SigLIP.
        """
        return []

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return
        self._model, self._processor = load_embeddinggemma2(self._on_progress, "image", self._label)
        verify_image_processor_backend(self._processor, embedder=self._label)

    def embed_pil_image(self, image: "Image.Image") -> Optional[np.ndarray]:
        """Embed a PIL Image that is already in memory (e.g. from CIFAR-10)."""
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            return self._forward_pil_batch([image])[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding PIL image (%s)", self._label)
            return None

    def _forward_pil_batch(self, images: list["Image.Image"]) -> np.ndarray:
        rgb = [im.convert("RGB") for im in images]
        return pooled_forward(self._model, self._processor, media_messages("image", rgb))

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            return pooled_forward(self._model, self._processor, query_messages([text]))[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for image (%s)", self._label)
            return None


EMBEDDER = ImageEmbeddingGemma2Embedder()
