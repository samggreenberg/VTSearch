"""Image embedder — DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m).

DINOv3 is a self-supervised vision transformer that produces dense patch
embeddings.  We adapt it for whole-image retrieval by pooling the **CLS
token** from ``last_hidden_state[:, 0]`` — the same global representation
used by the DINOv3 paper for linear-probe classification benchmarks.

DINOv3 also produces high-quality **per-patch tokens** and exposes a clean
CLS→patch attention map (cleaner than DINOv2's thanks to Gram-anchored
"storage" / register tokens that absorb high-norm artifacts).  Setting
:attr:`supports_patch_regions` to ``True`` opts the embedder into the
patch-region pipeline — the dataset loader asks for a
:class:`~vtsearch.models.patch_regions.PatchEmbedOutput` (CLS + patch grid
+ CLS→patch attention saliency) which is turned into a hierarchical region
set per image.  DINOv3 has **4 register tokens** between the CLS and
patches; we slice them out before reshaping to the 14 × 14 patch grid.

No text encoder exists, so :attr:`supports_text` is ``False`` and the UI
will hide text-search affordances for datasets embedded with DINOv3.

Weights are gated on Hugging Face — running this embedder requires the
``HF_TOKEN`` env var to be set to a user token that has accepted the
DINOv3 licence at
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
from vtsearch.models.patch_regions import PatchEmbedOutput, hf_vit_to_patch_output

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

    def _patch_forward_impl(self, media: dict) -> Optional[PatchEmbedOutput]:
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

            image = Image.open(file_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs, output_attentions=True)
            return hf_vit_to_patch_output(
                outputs, num_register_tokens=_DINOV3_NUM_REGISTER_TOKENS
            )
        except Exception:
            logging.getLogger(__name__).exception("Error patch-embedding %s (DINOv3)", file_path)
            return None


EMBEDDER = ImageDinov3Embedder()
