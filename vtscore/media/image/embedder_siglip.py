"""Image embedder - SigLIP (google/siglip-base-patch16-224)."""

from __future__ import annotations

from typing import Any

from vtscore.config import SIGLIP_MODEL_ID, image_processor_load_kwargs, verify_image_processor_backend
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


class ImageSiglipEmbedder(_CrossModalHFEmbedder):
    """Embeds images using the SigLIP model (google/siglip-base-patch16-224).

    * Images → 768-dimensional vectors via SigLIP's vision encoder.
    * Text queries → 768-dimensional vectors via SigLIP's text encoder.
    * Also exposes :meth:`embed_pil_image` for in-memory PIL Image objects
      (used for PDF rendering and CIFAR-10 demo datasets).
    """

    _label = "SigLIP"
    # SigLIP was trained with fixed-length padding and no truncation.
    _text_processor_kwargs: dict[str, Any] = {"padding": "max_length"}

    @property
    def name(self) -> str:
        return "siglip"

    @property
    def display_name(self) -> str:
        return "SigLIP (general images)"

    @property
    def model_id(self) -> str:
        return SIGLIP_MODEL_ID

    @property
    def embedding_dim(self) -> int:
        return 768

    @property
    def is_default(self) -> bool:
        return True

    @property
    def description_wrappers(self) -> list[str]:
        """No wrappers: the CLIP-style prompt ensemble does not help SigLIP.

        #3127 measured enrichment on/off over the image eval corpora and every
        one of the four non-identity templates inherited from
        :class:`~vtscore.media.image._cross_modal_shared._CrossModalHFEmbedder`
        scored *below* the typed query here; the ensemble only netted out to
        -0.001 +/- 0.002 AP because the bare ``"{text}"`` template averages the
        plain query back in.  Dropping them makes the Enrich Sort Descriptions
        setting a no-op on the default image embedder rather than a small,
        silent loss.  See #3341.
        """
        return []

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
            from transformers import SiglipModel, SiglipProcessor  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading SigLIP model weights…")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading SigLIP model weights…"),
        ):
            self._model = load_pretrained_local_first(
                SiglipModel.from_pretrained,
                SIGLIP_MODEL_ID,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=hf_token(),
                on_progress=self._on_progress,
            )
        self._model = to_compute_device(self._model, allow_half=True)
        self._on_progress("loading", "Loading SigLIP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            from transformers import AutoImageProcessor, SiglipTokenizer  # noqa: PLC0415

            # ``AutoImageProcessor`` rather than the concrete ``SiglipImageProcessor``:
            # naming a class pins this to whatever that name means in the installed
            # transformers, and that meaning *changed* — v5 removed the ``Fast``
            # suffix, so ``SiglipImageProcessor`` went from being the PIL
            # implementation to being the torchvision one, silently.  The concrete
            # class also cannot honour a backend request: asked for ``pil`` it warns
            # and hands back torchvision anyway, so the knob above would be inert
            # here alone.  Auto resolves from the model config like every other
            # embedder in this package (#3146).
            image_processor = load_pretrained_local_first(
                AutoImageProcessor.from_pretrained,
                SIGLIP_MODEL_ID,
                cache_dir=cache_dir,
                token=hf_token(),
                **image_processor_load_kwargs(),
            )
            tokenizer = load_pretrained_local_first(
                SiglipTokenizer.from_pretrained, SIGLIP_MODEL_ID, cache_dir=cache_dir, token=hf_token()
            )
            self._processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)
        verify_image_processor_backend(self._processor, embedder="SigLIP")


EMBEDDER = ImageSiglipEmbedder()
