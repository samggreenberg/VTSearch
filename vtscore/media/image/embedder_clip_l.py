"""Image embedder - OpenAI CLIP ViT-L/14 (openai/clip-vit-large-patch14).

**Evaluation only.**  This embedder exists to answer one question, #3292's:
does the ``calibration_fraction`` optimum #3287 measured on SigLIP follow
*single-vector geometry* or merely *the SigLIP family*?  Answering that needs a
single-vector, language-aligned encoder from a different lineage, and CLIP is
the cheapest one -- a softmax/InfoNCE contrastive objective against SigLIP's
pairwise sigmoid loss, different data, different recipe.

It is **not** a production candidate, so :attr:`eval_only` is ``True`` and it is
withheld from every app-facing listing (see
:func:`vtscore.media.embedders_for_type`).  Nothing has evaluated it for the
app, and an eval arm that leaks into the picker is a model users can choose on
the strength of a study that never asked whether it is *good*, only whether it
*agrees*.

Why ``patch14`` and not the already-shipped :mod:`~vtscore.media.image.embedder_clip`
(``clip-vit-base-patch32``): this checkpoint emits **768-d** vectors, exactly
like ``siglip``, which takes output dimensionality off the table as an
explanation for any difference.  The base/32 arm is run *beside* it rather than
instead of it, because the two together separate CLIP's lineage from its
capacity -- see ``docs/experiments/calibration-fraction-3287/REPORT.md``.
"""

from __future__ import annotations

from typing import Any

from vtscore.config import CLIP_L_MODEL_ID, image_processor_load_kwargs, verify_image_processor_backend
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


class ImageClipLargeEmbedder(_CrossModalHFEmbedder):
    """Embeds images using OpenAI CLIP ViT-L/14 (clip-vit-large-patch14).

    * Images -> 768-dimensional vectors via CLIP's vision encoder.
    * Text queries -> 768-dimensional vectors via CLIP's text encoder.

    Same HF ``CLIPModel``/``CLIPProcessor`` pair as :class:`ImageClipEmbedder`;
    only the checkpoint differs.  The tokenization contract is CLIP's, not
    SigLIP's -- ``padding=True`` with truncation, matching the base arm, because
    changing it silently changes the produced text vectors.
    """

    _label = "CLIP ViT-L/14"
    _text_processor_kwargs: dict[str, Any] = {"padding": True, "truncation": True}

    @property
    def name(self) -> str:
        return "clip_l"

    @property
    def display_name(self) -> str:
        return "CLIP ViT-L/14 (evaluation only)"

    @property
    def model_id(self) -> str:
        return CLIP_L_MODEL_ID

    @property
    def embedding_dim(self) -> int:
        return 768

    @property
    def eval_only(self) -> bool:
        return True

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

        cache_dir = embedder_load_setup(self._on_progress, "Loading CLIP ViT-L/14 model weights…")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading CLIP ViT-L/14 model weights…"),
        ):
            self._model = load_pretrained_local_first(
                CLIPModel.from_pretrained,
                CLIP_L_MODEL_ID,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=hf_token(),
                on_progress=self._on_progress,
            )
        self._model = to_compute_device(self._model, allow_half=True)
        self._on_progress("loading", "Loading CLIP ViT-L/14 processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                CLIPProcessor.from_pretrained,
                CLIP_L_MODEL_ID,
                cache_dir=cache_dir,
                token=hf_token(),
                **image_processor_load_kwargs(),
            )
        verify_image_processor_backend(self._processor, embedder="CLIP ViT-L/14")


EMBEDDER = ImageClipLargeEmbedder()
