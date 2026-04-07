"""Image embedder — CLIP (openai/clip-vit-base-patch32)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import CLIP_MODEL_ID
from vtsearch.media.base import (
    MediaEmbedder,
    embedder_load_setup,
    extract_tensor as _extract_tensor,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
)

if TYPE_CHECKING:
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor


class ImageClipEmbedder(MediaEmbedder):
    """Embeds images using the CLIP model (openai/clip-vit-base-patch32).

    * Images → 768-dimensional vectors via CLIP's vision encoder.
    * Text queries → 768-dimensional vectors via CLIP's text encoder.
    * Also exposes :meth:`embed_pil_image` for in-memory PIL Image objects
      (used for CIFAR-10 demo datasets and PDF rendering).
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "clip"

    @property
    def media_type_id(self) -> str:
        return "image"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        self._on_progress("loading", "Importing torch…", 1, 2)
        import torch  # noqa: F401, PLC0415

        self._on_progress("loading", "Importing transformers…", 2, 2)
        from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading CLIP model weights…")
        CLIPModel._keys_to_ignore_on_load_unexpected = [r".*position_ids.*"]
        with intercept_tqdm_progress(self._on_progress), intercept_weight_loading_progress(
            self._on_progress, "Loading CLIP model weights…"
        ):
            self._model = load_pretrained_local_first(
                CLIPModel.from_pretrained, CLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False
            )
        self._model = self._model.to("cpu")
        self._on_progress("loading", "Loading CLIP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                CLIPProcessor.from_pretrained, CLIP_MODEL_ID, cache_dir=cache_dir, use_fast=True, token=False
            )

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

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            from PIL import Image  # noqa: PLC0415

            image = Image.open(file_path).convert("RGB")
            return self.embed_pil_image(image)
        except Exception as e:
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
        except Exception as e:
            logging.getLogger(__name__).exception("Error embedding PIL image")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            inputs = self._processor(text=[text], return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                text_vec = _extract_tensor(self._model.get_text_features(**inputs)).detach().cpu().numpy()[0]
            return text_vec
        except Exception as e:
            logging.getLogger(__name__).exception("Error embedding text query for image")
            return None

    # Internal helper used by loader.py bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor
