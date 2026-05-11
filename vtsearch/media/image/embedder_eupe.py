"""Image embedder — EUPE / Meta Perception Encoder (facebook/PE-Core-B16-224).

We expose this model under the slug ``eupe`` (rather than the bare ``pe``)
so the registry key is a distinct, unambiguous proper noun. The underlying
weights are Meta's Perception Encoder Core B/16-224 — a vision encoder
trained on large-scale image data, comparable in size to SigLIP-base.

This embedder uses **only** the vision tower and pools the CLS token from
``last_hidden_state[:, 0]`` to produce a single fixed-size vector per image.
We do not expose a text encoder here — :attr:`supports_text` is ``False`` and
the UI hides text-search affordances for datasets embedded with EUPE.

Meta publishes Perception Encoder with custom modeling code on the Hugging
Face Hub, so we load with ``trust_remote_code=True``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import EUPE_MODEL_ID
from vtsearch.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
)

if TYPE_CHECKING:
    from PIL import Image


class ImageEupeEmbedder(MediaEmbedder):
    """Embeds images using EUPE (Meta Perception Encoder Core B/16-224, CLS-pooled)."""

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "eupe"

    @property
    def media_type_id(self) -> str:
        return "image"

    @property
    def supports_text(self) -> bool:
        return False

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch…", 1, 2):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing transformers…", 2, 2):
            from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading EUPE weights…")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading EUPE weights…"),
        ):
            self._model = load_pretrained_local_first(
                AutoModel.from_pretrained,
                EUPE_MODEL_ID,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
                trust_remote_code=True,
            )
        self._model = self._model.to("cpu")
        self._model.eval()
        self._on_progress("loading", "Loading EUPE image processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                AutoImageProcessor.from_pretrained,
                EUPE_MODEL_ID,
                cache_dir=cache_dir,
                token=False,
                trust_remote_code=True,
            )

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
                outputs = self._model(**inputs)
                # Prefer get_image_features when the loaded class exposes it
                # (Perception Encoder's HF wrapper often does); fall back to
                # CLS-token pooling of last_hidden_state for vision-tower-only
                # variants.
                if hasattr(self._model, "get_image_features"):
                    feats = self._model.get_image_features(**inputs)
                    embedding = feats.detach().cpu().numpy()
                else:
                    cls_token = outputs.last_hidden_state[:, 0]
                    embedding = cls_token.detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding PIL image (EUPE)")
            return None


EMBEDDER = ImageEupeEmbedder()
