"""Image embedder - SigLIP 2 (google/siglip2-base-patch16-224).

Uses :class:`~transformers.AutoModel` / :class:`~transformers.AutoProcessor` so
the embedder loads on any ``transformers`` version that ships SigLIP 2 support
(>= 4.49) without pinning to the concrete ``Siglip2Model`` class name.
"""

from __future__ import annotations

from typing import Any

from vtscore.config import SIGLIP2_MODEL_ID
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


class ImageSiglip2Embedder(_CrossModalHFEmbedder):
    """Embeds images using SigLIP 2 (google/siglip2-base-patch16-224).

    Successor to SigLIP with stronger transfer performance. Same 768-dim output
    space; uses the model's vision encoder for images and text encoder for
    queries, mirroring :class:`ImageSiglipEmbedder` so callers can swap.
    """

    _label = "SigLIP 2"
    _text_processor_kwargs: dict[str, Any] = {"padding": "max_length", "truncation": True}

    @property
    def name(self) -> str:
        return "siglip2"

    @property
    def display_name(self) -> str:
        return "SigLIP 2 (general images)"

    @property
    def model_id(self) -> str:
        return SIGLIP2_MODEL_ID

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
            from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading SigLIP 2 model weights…")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading SigLIP 2 model weights…"),
        ):
            self._model = load_pretrained_local_first(
                AutoModel.from_pretrained,
                SIGLIP2_MODEL_ID,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=hf_token(),
                on_progress=self._on_progress,
            )
        self._model = to_compute_device(self._model)
        self._on_progress("loading", "Loading SigLIP 2 processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                AutoProcessor.from_pretrained, SIGLIP2_MODEL_ID, cache_dir=cache_dir, token=hf_token()
            )


EMBEDDER = ImageSiglip2Embedder()
