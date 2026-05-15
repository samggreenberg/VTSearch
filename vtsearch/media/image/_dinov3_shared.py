"""Shared base class for the two DINOv3 embedder variants.

DINOv3 is exposed as two embedders that share the same backbone but differ
in what they expose:

- ``dinov3_single`` — CLS-pooled single vector per image; fast, small
  storage, no region search.
- ``dinov3_patch``  — same CLS vector plus a per-patch grid + HAC region
  tree; ~30× slower per image and ~100× more storage, but enables region
  similarity and region-aware MLP scoring.

DINOv3 weights are **gated** on Hugging Face — running either variant
requires the ``HF_TOKEN`` env var to be set to a user token that has
accepted the DINOv3 licence at
``https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import DINOV3_MODEL_ID
from vtsearch.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
)
from vtsearch.media.patch_embed import PatchEmbedOutput, hf_vit_to_patch_output

if TYPE_CHECKING:
    from PIL import Image


# DINOv3 ViT-B uses 4 register tokens ("storage tokens" in the paper)
# inserted between the CLS token and the patch tokens.  The exact count
# lives in the model config (``num_register_tokens``); we hard-code 4
# here because the ViT-B/16 LVD-1689M weight has been published with
# this setting and is what ``DINOV3_MODEL_ID`` points at.  If we ever
# add a second DINOv3 variant we should read it off ``self._model.config``
# at load time instead.
_DINOV3_NUM_REGISTER_TOKENS = 4


class _Dinov3Base(MediaEmbedder):
    """Backbone loader + CLS / patch forward passes for DINOv3.

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

        cache_dir = embedder_load_setup(self._on_progress, "Loading DINOv3 model weights…")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading DINOv3 model weights…"),
        ):
            self._model = load_pretrained_local_first(
                AutoModel.from_pretrained, DINOV3_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False
            )
        self._model = self._model.to("cpu")
        self._model.eval()
        self._on_progress("loading", "Loading DINOv3 image processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                AutoImageProcessor.from_pretrained, DINOV3_MODEL_ID, cache_dir=cache_dir, token=False
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
                # DINOv3 returns hidden states with the CLS token at index 0.
                # Use it as the single-vector image representation (this is the
                # representation linear-probed in the DINOv3 paper).
                cls_token = outputs.last_hidden_state[:, 0]
                embedding = cls_token.detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding PIL image (DINOv3)")
            return None

    def _compute_patch_output(self, media: dict) -> Optional[PatchEmbedOutput]:
        """Return CLS + per-patch grid + CLS→patch attention saliency.

        Runs one forward pass with ``output_attentions=True`` and slices out
        the 4 register tokens between CLS and patches before reshaping to
        the 14 × 14 grid.
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
            return hf_vit_to_patch_output(outputs, num_register_tokens=_DINOV3_NUM_REGISTER_TOKENS)
        except Exception:
            logging.getLogger(__name__).exception("Error patch-embedding %s (DINOv3)", file_path)
            return None
