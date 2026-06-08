"""Image embedder - OpenAI CLIP (openai/clip-vit-base-patch32)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtscore.config import CLIP_MODEL_ID
from vtscore.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    extract_tensor as _extract_tensor,
    hf_token,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
)
from vtscore.media.image._image_bulk import bulk_embed_image_files

if TYPE_CHECKING:
    from PIL import Image


class ImageClipEmbedder(MediaEmbedder):
    """Embeds images using OpenAI CLIP (clip-vit-base-patch32).

    * Images → 512-dimensional vectors via CLIP's vision encoder.
    * Text queries → 512-dimensional vectors via CLIP's text encoder.
    """

    def __init__(self) -> None:
        super().__init__()
        # Typed ``Any``: transformers stubs miss several ``CLIPProcessor.__call__``
        # kwargs we pass at runtime; runtime ``None`` checks guard the calls.
        self._model: Any = None
        self._processor: Any = None

    @property
    def name(self) -> str:
        return "clip"

    @property
    def display_name(self) -> str:
        return "CLIP (general images)"

    @property
    def media_type_id(self) -> str:
        return "image"

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch…", 1, 2):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing transformers…", 2, 2):
            from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading CLIP model weights…")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading CLIP model weights…"),
        ):
            self._model = load_pretrained_local_first(
                CLIPModel.from_pretrained,
                CLIP_MODEL_ID,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=hf_token(),
                on_progress=self._on_progress,
            )
        self._model = self._model.to("cpu")
        self._on_progress("loading", "Loading CLIP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                CLIPProcessor.from_pretrained, CLIP_MODEL_ID, cache_dir=cache_dir, token=hf_token()
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
        from vtscore.media.image._image_bulk import _load_pil, _pil_source_for  # noqa: PLC0415

        source = _pil_source_for(media)
        if source is None:
            return None
        image = _load_pil(source)
        if image is None:
            return None
        return self.embed_pil_image(image)

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
            logging.getLogger(__name__).exception("Error embedding PIL image (CLIP)")
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
                label="CLIP",
            )

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                text_vec = _extract_tensor(self._model.get_text_features(**inputs)).detach().cpu().numpy()[0]
            return text_vec
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for image (CLIP)")
            return None


EMBEDDER = ImageClipEmbedder()
