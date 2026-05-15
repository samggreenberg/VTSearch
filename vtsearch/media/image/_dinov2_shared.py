"""Shared base class for the two DINOv2 embedder variants.

DINOv2 is exposed as two embedders that share the same backbone but differ
in what they expose:

- ``dinov2_single`` — CLS-pooled single vector per image; fast, small
  storage, no region search.
- ``dinov2_patch``  — same CLS vector plus a per-patch grid + HAC region
  tree; ~30× slower per image and ~100× more storage, but enables region
  similarity and region-aware MLP scoring.

Both share this base.  The underscore-prefixed filename keeps it out of
the auto-discovery scan in :mod:`vtsearch.media` (only ``embedder*.py``
files are imported as plugins) — the variant modules import from here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import DINOV2_MODEL_ID
from vtsearch.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
)
from vtsearch.media.image._image_bulk import bulk_embed_image_files
from vtsearch.media.patch_embed import PatchEmbedOutput, hf_vit_to_patch_output

if TYPE_CHECKING:
    from PIL import Image


class _Dinov2Base(MediaEmbedder):
    """Backbone loader + CLS / patch forward passes for DINOv2.

    Subclasses set :attr:`name` and :attr:`supports_patch_regions`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None

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

        cache_dir = embedder_load_setup(self._on_progress, "Loading DINOv2 model weights…")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading DINOv2 model weights…"),
        ):
            self._model = load_pretrained_local_first(
                AutoModel.from_pretrained, DINOV2_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False
            )
        self._model = self._model.to("cpu")
        self._model.eval()
        self._on_progress("loading", "Loading DINOv2 image processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                AutoImageProcessor.from_pretrained, DINOV2_MODEL_ID, cache_dir=cache_dir, token=False
            )

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

    def embed_pil_image(self, image: "Image.Image") -> Optional[np.ndarray]:
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
                # DINOv2's ``last_hidden_state[:, 0]`` is the CLS token — the
                # global representation used for linear probing.
                cls_token = outputs.last_hidden_state[:, 0]
                embedding = cls_token.detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding PIL image (DINOv2)")
            return None

    def _forward_pil_batch(self, images: list["Image.Image"]) -> np.ndarray:
        import torch  # noqa: PLC0415

        rgb = [im.convert("RGB") for im in images]
        inputs = self._processor(images=rgb, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            cls = outputs.last_hidden_state[:, 0]
            return cls.detach().cpu().numpy()

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
                label="DINOv2",
            )

    def _patch_forward_pil_batch(self, images: list["Image.Image"]) -> list[Optional[PatchEmbedOutput]]:
        import torch  # noqa: PLC0415

        rgb = [im.convert("RGB") for im in images]
        inputs = self._processor(images=rgb, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
        return [hf_vit_to_patch_output(outputs, num_register_tokens=0, batch_index=i) for i in range(len(images))]

    def _compute_patch_output(self, media: dict) -> Optional[PatchEmbedOutput]:
        """Return CLS + per-patch grid + CLS→patch attention saliency.

        Runs one forward pass with ``output_attentions=True`` and reshapes
        ``last_hidden_state[:, 1:]`` into the spatial patch grid.  DINOv2
        has no register tokens, so the token layout is ``[CLS, P1..P_N]``
        and we slice the patch portion at index 1.
        """
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        file_path = Path(media["media_path"])
        try:
            import torch  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415

            with Image.open(file_path) as _img:
                image = _img.convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs, output_attentions=True)
            return hf_vit_to_patch_output(outputs, num_register_tokens=0)
        except Exception:
            logging.getLogger(__name__).exception("Error patch-embedding %s (DINOv2)", file_path)
            return None
