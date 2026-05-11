"""Image embedder — DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m).

DINOv3 is a self-supervised vision transformer that produces dense patch
embeddings. We adapt it for whole-image retrieval by pooling the **CLS token**
from ``last_hidden_state[:, 0]`` — the same global representation used by the
DINOv3 paper for linear-probe classification benchmarks.

No text encoder exists, so :attr:`supports_text` is ``False`` and the UI will
hide text-search affordances for datasets embedded with DINOv3.
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

if TYPE_CHECKING:
    from PIL import Image


class ImageDinov3Embedder(MediaEmbedder):
    """Embeds images using DINOv3 ViT-B/16 with CLS-token pooling.

    Output dimension: 768 (ViT-B hidden size).
    """

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "dinov3"

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
                # DINOv3 returns hidden states with the CLS token at index 0.
                # Use it as the single-vector image representation (this is the
                # representation linear-probed in the DINOv3 paper).
                cls_token = outputs.last_hidden_state[:, 0]
                embedding = cls_token.detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding PIL image (DINOv3)")
            return None


EMBEDDER = ImageDinov3Embedder()
