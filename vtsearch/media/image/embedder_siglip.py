"""Image embedder — SigLIP (google/siglip-base-patch16-224)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import SIGLIP_MODEL_ID
from vtsearch.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    extract_tensor as _extract_tensor,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
)

if TYPE_CHECKING:
    from PIL import Image
    from transformers import SiglipModel, SiglipProcessor


class ImageSiglipEmbedder(MediaEmbedder):
    """Embeds images using the SigLIP model (google/siglip-base-patch16-224).

    * Images → 768-dimensional vectors via SigLIP's vision encoder.
    * Text queries → 768-dimensional vectors via SigLIP's text encoder.
    * Also exposes :meth:`embed_pil_image` for in-memory PIL Image objects
      (used for PDF rendering and CIFAR-10 demo datasets).
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[SiglipModel] = None
        self._processor: Optional[SiglipProcessor] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "siglip"

    @property
    def media_type_id(self) -> str:
        return "image"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch…", 1, 2):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing transformers…", 2, 2):
            from transformers import SiglipModel, SiglipProcessor  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading SigLIP model weights…")
        with intercept_tqdm_progress(self._on_progress), intercept_weight_loading_progress(
            self._on_progress, "Loading SigLIP model weights…"
        ):
            self._model = load_pretrained_local_first(
                SiglipModel.from_pretrained, SIGLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False
            )
        self._model = self._model.to("cpu")
        self._on_progress("loading", "Loading SigLIP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            from transformers import SiglipImageProcessor, SiglipTokenizer  # noqa: PLC0415

            image_processor = load_pretrained_local_first(
                SiglipImageProcessor.from_pretrained, SIGLIP_MODEL_ID, cache_dir=cache_dir, token=False
            )
            tokenizer = load_pretrained_local_first(
                SiglipTokenizer.from_pretrained, SIGLIP_MODEL_ID, cache_dir=cache_dir, token=False
            )
            self._processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @property
    def description_wrappers(self) -> list[str]:
        return [
            "a photo of {text}",
            "a photograph of {text}",
            "an image of {text}",
            "{text}",
            "a picture of {text}",
        ]

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        file_path = Path(media["media_path"])
        try:
            from PIL import Image  # noqa: PLC0415

            image = Image.open(file_path).convert("RGB")
            return self.embed_pil_image(image)
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", file_path)
            return None

    def embed_pil_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """Embed a PIL Image that is already in memory (e.g. from CIFAR-10)."""
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            image = image.convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                embedding = _extract_tensor(outputs).detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding PIL image")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            inputs = self._processor(text=[text], return_tensors="pt", padding="max_length")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                text_vec = _extract_tensor(self._model.get_text_features(**inputs)).detach().cpu().numpy()[0]
            return text_vec
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for image (SigLIP)")
            return None

    # Internal helper used by loader.py bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor


EMBEDDER = ImageSiglipEmbedder()
