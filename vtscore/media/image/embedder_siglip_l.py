"""Image embedder - SigLIP-L (open_clip ``ViT-SO400M-14-SigLIP-384``).

Unlike :mod:`vtscore.media.image.embedder_siglip` (the base SigLIP loaded via
``transformers``), this embedder loads the larger SO400M/384 checkpoint through
``open_clip``.  Loading through open_clip - rather than the transformers port -
means the 1152-dimensional vectors match galleries produced by open_clip's own
model bit-for-bit, so a SigLIP-L detector trained here is directly comparable to
an externally-embedded open_clip gallery.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtscore.config import SIGLIP_L_MODEL_ID, SIGLIP_L_PRETRAINED
from vtscore.media.embedder import (
    IMPORT_MODULE_ESTIMATES,
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    timed_progress,
    to_compute_device,
)
from vtscore.media.image._image_bulk import bulk_embed_image_files

if TYPE_CHECKING:
    from PIL import Image


class ImageSiglipLEmbedder(MediaEmbedder):
    """Embeds images using SigLIP-L (SO400M/384) via ``open_clip``.

    * Images → 1152-dimensional vectors via the vision tower.
    * Text queries → 1152-dimensional vectors via the text tower.
    * Also exposes :meth:`embed_pil_image` for in-memory PIL Image objects
      (used for PDF rendering and CIFAR-10 demo datasets).

    The model / preprocess / tokenizer come straight from open_clip's pretrained
    registry so vectors line up with any gallery embedded by the same open_clip
    checkpoint.  Like every other embedder the loaded model is placed on the
    resolved compute device via :func:`to_compute_device`, so a CUDA host runs
    the forward pass on the GPU while a CPU-only host degrades gracefully.
    """

    def __init__(self) -> None:
        super().__init__()
        # Typed ``Any``: open_clip has no type stubs, so the model / preprocess /
        # tokenizer callables are opaque to pyright; runtime ``None`` checks guard
        # every use.
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "siglip_l"

    @property
    def display_name(self) -> str:
        return "SigLIP-L (SO400M/384)"

    @property
    def model_id(self) -> str:
        return SIGLIP_L_MODEL_ID

    @property
    def media_type_id(self) -> str:
        return "image"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(
            self._on_progress, "loading", "Importing torch…", est_modules=IMPORT_MODULE_ESTIMATES["torch"]
        ):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing open_clip…"):
            import open_clip  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        cache_dir = embedder_load_setup(self._on_progress, "Loading SigLIP-L model weights…")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading SigLIP-L model weights…"),
        ):
            model, _, preprocess = open_clip.create_model_and_transforms(
                SIGLIP_L_MODEL_ID, pretrained=SIGLIP_L_PRETRAINED, cache_dir=cache_dir
            )
            tokenizer = open_clip.get_tokenizer(SIGLIP_L_MODEL_ID)
        model.eval()
        self._model = to_compute_device(model)
        self._preprocess = preprocess
        self._tokenizer = tokenizer

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

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._preprocess is None:
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
        """Embed a PIL Image that is already in memory (e.g. from CIFAR-10)."""
        if self._model is None:
            self.load_models()
        if self._model is None or self._preprocess is None:
            return None
        try:
            return self._forward_pil_batch([image])[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding PIL image (SigLIP-L)")
            return None

    def _forward_pil_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Run SigLIP-L's vision tower on a list of PIL images.

        Returns an ``(N, 1152)`` array.  Caller is responsible for batch
        sizing - this runs the whole list in one forward pass.
        """
        import torch  # noqa: PLC0415

        device = next(self._model.parameters()).device
        batch = torch.stack([self._preprocess(im.convert("RGB")) for im in images]).to(device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
        return features.detach().cpu().numpy()

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[Optional[np.ndarray]]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._preprocess is None:
            return [None] * len(medias)
        with self._embed_lock:
            return bulk_embed_image_files(
                medias,
                forward_pil_batch=self._forward_pil_batch,
                batch_size=self.embed_batch_size,
                on_progress=self._on_progress,
                label="SigLIP-L",
            )

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._tokenizer is None:
            return None
        try:
            import torch  # noqa: PLC0415

            device = next(self._model.parameters()).device
            tokens = self._tokenizer([text]).to(device)
            with torch.no_grad():
                text_vec = self._model.encode_text(tokens).detach().cpu().numpy()[0]
            return text_vec
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for image (SigLIP-L)")
            return None


EMBEDDER = ImageSiglipLEmbedder()
