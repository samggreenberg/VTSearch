"""Image embedder — SigLIP 2 (google/siglip2-base-patch16-224).

Uses :class:`~transformers.AutoModel` / :class:`~transformers.AutoProcessor` so
the embedder loads on any ``transformers`` version that ships SigLIP 2 support
(>= 4.49) without pinning to the concrete ``Siglip2Model`` class name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtsearch.config import SIGLIP2_MODEL_ID
from vtsearch.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    extract_tensor as _extract_tensor,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
)
from vtsearch.media.image._image_bulk import bulk_embed_image_files

if TYPE_CHECKING:
    from PIL import Image


class ImageSiglip2Embedder(MediaEmbedder):
    """Embeds images using SigLIP 2 (google/siglip2-base-patch16-224).

    Successor to SigLIP with stronger transfer performance. Same 768-dim output
    space; uses the model's vision encoder for images and text encoder for
    queries, mirroring :class:`ImageSiglipEmbedder` so callers can swap.
    """

    def __init__(self) -> None:
        super().__init__()
        # Typed ``Any``: transformers stubs miss several processor kwargs we
        # pass at runtime; runtime ``None`` checks guard the calls.
        self._model: Any = None
        self._processor: Any = None

    @property
    def name(self) -> str:
        return "siglip2"

    @property
    def media_type_id(self) -> str:
        return "image"

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch…", 1, 2):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing transformers…", 2, 2):
            from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading SigLIP 2 model weights…")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading SigLIP 2 model weights…"),
        ):
            self._model = load_pretrained_local_first(
                AutoModel.from_pretrained, SIGLIP2_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False
            )
        self._model = self._model.to("cpu")
        self._on_progress("loading", "Loading SigLIP 2 processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                AutoProcessor.from_pretrained, SIGLIP2_MODEL_ID, cache_dir=cache_dir, token=False
            )

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

            with Image.open(file_path) as _img:
                image = _img.convert("RGB")
            return self.embed_pil_image(image)
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", file_path)
            return None

    def embed_pil_image(self, image: Image.Image) -> Optional[np.ndarray]:
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
            logging.getLogger(__name__).exception("Error embedding PIL image (SigLIP 2)")
            return None

    def _forward_pil_batch(self, images: list[Image.Image]) -> np.ndarray:
        import torch  # noqa: PLC0415

        rgb = [im.convert("RGB") for im in images]
        inputs = self._processor(images=rgb, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
            return _extract_tensor(outputs).detach().cpu().numpy()

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return [None] * len(medias)
        with self._embed_lock:
            return bulk_embed_image_files(
                medias,
                forward_pil_batch=self._forward_pil_batch,
                batch_size=self.embed_batch_size,
                on_progress=self._on_progress,
                label="SigLIP 2",
            )

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            inputs = self._processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                text_vec = _extract_tensor(self._model.get_text_features(**inputs)).detach().cpu().numpy()[0]
            return text_vec
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for image (SigLIP 2)")
            return None


EMBEDDER = ImageSiglip2Embedder()
