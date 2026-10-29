"""Shared base for the HF cross-modal image embedders (SigLIP, SigLIP 2, SigLIP2-L, CLIP).

All of them wrap a Hugging Face model/processor pair that exposes
``get_image_features`` / ``get_text_features``, so the full cross-modal
surface (single/bulk image embedding, PIL embedding, text-query embedding)
lives here.  Subclasses provide the identity properties, their own
:meth:`~vtscore.media.embedder.MediaEmbedder._load_models_impl` (the model
and processor classes genuinely differ per backbone), and two class
attributes:

- :attr:`_label` - human-readable backbone name used in progress and log
  messages (e.g. ``"SigLIP 2"``).
- :attr:`_text_processor_kwargs` - tokenization kwargs passed to the
  processor for text queries; the padding/truncation contract differs per
  backbone and changing it silently changes the produced vectors.

``ImageSiglipLEmbedder`` uses the open_clip runtime path instead of an HF
processor and stays outside this base.  The underscore-prefixed filename
keeps this module out of the auto-discovery scan in :mod:`vtscore.media`
(only ``embedder*.py`` files are imported as plugins).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtscore.config import image_processor_call_kwargs
from vtscore.media.embedder import (
    MediaEmbedder,
    embed_autocast,
    extract_tensor as _extract_tensor,
    to_float32,
    to_model_inputs,
)
from vtscore.media.image._image_bulk import bulk_embed_image_files

if TYPE_CHECKING:
    from PIL import Image


class _CrossModalHFEmbedder(MediaEmbedder):
    """Cross-modal (image + text) embedding surface shared by the HF backbones."""

    _label: str
    _text_processor_kwargs: dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        # Typed ``Any``: transformers stubs miss several processor kwargs we
        # pass at runtime; runtime ``None`` checks guard the calls.
        self._model: Any = None
        self._processor: Any = None

    @property
    def media_type_id(self) -> str:
        return "image"

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

    def embed_pil_image(self, image: "Image.Image") -> Optional[np.ndarray]:
        """Embed a PIL Image that is already in memory (e.g. from CIFAR-10)."""
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            image = image.convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt", **image_processor_call_kwargs())
            inputs = to_model_inputs(inputs, self._model)
            with torch.no_grad(), embed_autocast():
                outputs = self._model.get_image_features(**inputs)
                embedding = to_float32(_extract_tensor(outputs).detach()).cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding PIL image (%s)", self._label)
            return None

    def _forward_pil_batch(self, images: list["Image.Image"]) -> np.ndarray:
        """Run the vision encoder on a list of PIL images.

        Returns an ``(N, dim)`` array.  Caller is responsible for batch
        sizing - this runs the whole list in one forward pass.
        """
        import torch  # noqa: PLC0415

        rgb = [im.convert("RGB") for im in images]
        inputs = self._processor(images=rgb, return_tensors="pt", **image_processor_call_kwargs())
        inputs = to_model_inputs(inputs, self._model)
        with torch.no_grad(), embed_autocast():
            outputs = self._model.get_image_features(**inputs)
            return to_float32(_extract_tensor(outputs).detach()).cpu().numpy()

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
                label=self._label,
            )

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            inputs = self._processor(text=[text], return_tensors="pt", **self._text_processor_kwargs)
            inputs = to_model_inputs(inputs, self._model)
            with torch.no_grad(), embed_autocast():
                text_out = _extract_tensor(self._model.get_text_features(**inputs)).detach()
                text_vec = to_float32(text_out).cpu().numpy()[0]
            return text_vec
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for image (%s)", self._label)
            return None
