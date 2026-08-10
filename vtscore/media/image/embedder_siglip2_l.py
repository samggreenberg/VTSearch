"""Image embedder - SigLIP2-L (google/siglip2-so400m-patch14-384).

The SO400M/384 member of the SigLIP 2 family.  Unlike
:mod:`vtscore.media.image.embedder_siglip_l` - which reaches the *v1* SO400M
checkpoint through ``open_clip`` because that is where its canonical weights
live - SigLIP 2 ships a first-party Hugging Face port, so this embedder loads
through :class:`~transformers.AutoModel` / :class:`~transformers.AutoProcessor`
exactly like :mod:`vtscore.media.image.embedder_siglip2` and shares the whole
cross-modal surface with it.
"""

from __future__ import annotations

from typing import Any

from vtscore.config import SIGLIP2_L_MODEL_ID
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


class ImageSiglip2LEmbedder(_CrossModalHFEmbedder):
    """Embeds images using SigLIP2-L (google/siglip2-so400m-patch14-384).

    The large sibling of :class:`ImageSiglip2Embedder`: same vision-encoder /
    text-encoder split and same tokenization contract, but a 400M-parameter
    shape-optimised backbone at 384px producing **1152-dimensional** vectors
    instead of 768.  A detector trained on one is not usable on the other -
    the dimensions differ - so the two are separate registry entries rather
    than a size knob on one.
    """

    _label = "SigLIP2-L"
    _text_processor_kwargs: dict[str, Any] = {"padding": "max_length", "truncation": True}

    @property
    def name(self) -> str:
        return "siglip2_l"

    @property
    def display_name(self) -> str:
        return "SigLIP2-L (SO400M/384)"

    @property
    def model_id(self) -> str:
        return SIGLIP2_L_MODEL_ID

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

        cache_dir = embedder_load_setup(self._on_progress, "Loading SigLIP2-L model weights…")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading SigLIP2-L model weights…"),
        ):
            self._model = load_pretrained_local_first(
                AutoModel.from_pretrained,
                SIGLIP2_L_MODEL_ID,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=hf_token(),
                on_progress=self._on_progress,
            )
        self._model = to_compute_device(self._model)
        self._on_progress("loading", "Loading SigLIP2-L processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                AutoProcessor.from_pretrained, SIGLIP2_L_MODEL_ID, cache_dir=cache_dir, token=hf_token()
            )


EMBEDDER = ImageSiglip2LEmbedder()
