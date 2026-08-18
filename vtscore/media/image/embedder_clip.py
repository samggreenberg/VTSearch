"""Image embedder - OpenAI CLIP (openai/clip-vit-base-patch32)."""

from __future__ import annotations

from typing import Any

from vtscore.config import CLIP_MODEL_ID
from vtscore.media.embedder import (
    IMPORT_MODULE_ESTIMATES,
    embedder_load_setup,
    hf_token,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
    to_compute_device,
)
from vtscore.media.image._cross_modal_shared import _CrossModalHFEmbedder


class ImageClipEmbedder(_CrossModalHFEmbedder):
    """Embeds images using OpenAI CLIP (clip-vit-base-patch32).

    * Images → 512-dimensional vectors via CLIP's vision encoder.
    * Text queries → 512-dimensional vectors via CLIP's text encoder.
    """

    _label = "CLIP"
    _text_processor_kwargs: dict[str, Any] = {"padding": True, "truncation": True}

    @property
    def name(self) -> str:
        return "clip"

    @property
    def display_name(self) -> str:
        return "CLIP (general images)"

    @property
    def model_id(self) -> str:
        return CLIP_MODEL_ID

    @property
    def embedding_dim(self) -> int:
        return 512

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(
            self._on_progress, "loading", "Importing torch…", est_modules=IMPORT_MODULE_ESTIMATES["torch"]
        ):
            import torch  # noqa: F401, PLC0415

        with timed_progress(
            self._on_progress, "loading", "Importing transformers…", est_modules=IMPORT_MODULE_ESTIMATES["transformers"]
        ):
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
        self._model = to_compute_device(self._model, allow_half=True)
        self._on_progress("loading", "Loading CLIP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                CLIPProcessor.from_pretrained, CLIP_MODEL_ID, cache_dir=cache_dir, token=hf_token()
            )


EMBEDDER = ImageClipEmbedder()
