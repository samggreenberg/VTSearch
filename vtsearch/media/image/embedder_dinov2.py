"""Image embedder — DINOv2 (facebook/dinov2-base).

DINOv2 is the predecessor to DINOv3 and shares its self-supervised
vision-transformer architecture. Unlike DINOv3, the DINOv2 weights are
**ungated** on Hugging Face — anyone can download them without an HF
account, which makes this embedder a friendly out-of-the-box option for
the bundled-image Docker build.

We pool the CLS token from ``last_hidden_state[:, 0]`` for a single
fixed-size vector per image, mirroring how DINOv3 is exposed here.
There is no text encoder, so :attr:`supports_text` is ``False``.

DINOv2 also produces high-quality **per-patch tokens**.  Setting
:attr:`supports_patch_regions` to ``True`` opts the embedder into the
patch-region pipeline — the dataset loader asks for a
:class:`~vtsearch.models.patch_regions.PatchEmbedOutput` (CLS + patch
grid + CLS→patch attention saliency) which is turned into a
hierarchical region set per image.  Image input is the HF default
(224² for the standard processor) with patch size 14 → 16 × 16 = 256
patch tokens.
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
from vtsearch.models.patch_regions import PatchEmbedOutput, hf_vit_to_patch_output

if TYPE_CHECKING:
    from PIL import Image


class ImageDinov2Embedder(MediaEmbedder):
    """Embeds images using DINOv2 ViT-B/14 with CLS-token pooling.

    Output dimension: 768 (ViT-B hidden size).
    """

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "dinov2"

    @property
    def media_type_id(self) -> str:
        return "image"

    @property
    def supports_text(self) -> bool:
        return False

    @property
    def supports_patch_regions(self) -> bool:
        return True

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
                # DINOv2's ``last_hidden_state[:, 0]`` is the CLS token — the
                # global representation used for linear probing.
                cls_token = outputs.last_hidden_state[:, 0]
                embedding = cls_token.detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding PIL image (DINOv2)")
            return None

    def _patch_forward_impl(self, media: dict) -> Optional[PatchEmbedOutput]:
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

            image = Image.open(file_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs, output_attentions=True)
            return hf_vit_to_patch_output(outputs, num_register_tokens=0)
        except Exception:
            logging.getLogger(__name__).exception("Error patch-embedding %s (DINOv2)", file_path)
            return None


EMBEDDER = ImageDinov2Embedder()
