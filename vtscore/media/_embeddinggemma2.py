"""Shared loader and forward pass for the EmbeddingGemma 2 embedders (#4837).

``google/embeddinggemma-2`` maps text, images and audio into **one** 768-d
space: a 270M text backbone, with a 170M vision and a 300M audio encoder whose
outputs enter that backbone as soft tokens.  So unlike SigLIP or CLAP there is
no separate image or audio *head* to call: every input, a typed query included,
goes through the same forward pass, and the embedding is the masked mean of the
backbone's last hidden state, L2-normalised.  That is the model's own
sentence-transformers recipe (``Transformer -> Pooling(mean, include_prompt) ->
Normalize``), reproduced here on plain ``transformers`` so the app does not
take a sentence-transformers floor of 6.1 for one embedder; #4837 checked the
two agree to cosine 1.0000 on images and on prefixed queries.

Each modality is its own registry entry, because a :class:`MediaEmbedder` has
exactly one media type: ``embeddinggemma2`` (images) and
``embeddinggemma2_audio``.  Each loads only the encoder it needs (the model
card's selective loading, ``vision_config=None`` / ``audio_config=None``), so
the image embedder holds 440M parameters and the audio one 570M.

The underscore-prefixed filename keeps this module out of the embedder
auto-discovery scan; the two embedder modules import from it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from vtscore.config import (
    EMBEDDINGGEMMA2_MODEL_ID,
    EMBEDDINGGEMMA2_QUERY_PREFIX,
    embed_precision,
    image_processor_load_kwargs,
)
from vtscore.media.embedder import (
    IMPORT_MODULE_ESTIMATES,
    embedder_load_setup,
    hf_token,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
    to_compute_device,
    to_model_inputs,
)

if TYPE_CHECKING:
    from vtscore.media.base import ProgressCallback

_log = logging.getLogger(__name__)

#: The ``transformers`` release that added ``EmbeddingGemma2Model``.
MIN_TRANSFORMERS = "5.18.0"

Modality = Literal["image", "audio"]


def _require_transformers() -> None:
    """Fail with the fix in hand when ``transformers`` predates the model.

    Without this, ``AutoModel`` raises a ``ValueError`` about an unrecognised
    ``model_type``, which :func:`load_pretrained_local_first` reads as "not
    cached yet" and answers with a Hub round-trip before failing the same way.
    """
    import transformers  # noqa: PLC0415

    if not hasattr(transformers, "EmbeddingGemma2Model"):
        raise ImportError(
            f"EmbeddingGemma 2 needs transformers>={MIN_TRANSFORMERS}; this environment has "
            f"{transformers.__version__}. Upgrade with: pip install 'transformers>={MIN_TRANSFORMERS}'"
        )


def _weight_dtype() -> Any:
    """``torch.bfloat16`` when bf16 was asked for and runs here, else ``None`` (fp32).

    Never fp16: the model card warns its activations overflow fp16's range and
    come back as NaN or silently degraded vectors, so the fp16 modes of
    ``VTSEARCH_EMBED_PRECISION`` (which #3143 measured on the image encoders
    only) run this model in fp32 instead.  ``embed_precision`` already folds a
    bf16 request on a device without bf16 back to fp32.
    """
    import torch  # noqa: PLC0415

    mode = embed_precision()
    if mode in ("bf16", "autocast_bf16"):
        return torch.bfloat16
    if mode in ("fp16", "autocast_fp16"):
        _log.warning("EmbeddingGemma 2 overflows fp16; using fp32 for VTSEARCH_EMBED_PRECISION=%s", mode)
    return None


def load_embeddinggemma2(on_progress: "ProgressCallback", modality: Modality, label: str) -> tuple[Any, Any]:
    """Load ``(model, processor)`` with the text backbone and *modality*'s encoder only."""
    with timed_progress(on_progress, "loading", "Importing torch…", est_modules=IMPORT_MODULE_ESTIMATES["torch"]):
        import torch  # noqa: F401, PLC0415

    with timed_progress(
        on_progress, "loading", "Importing transformers…", est_modules=IMPORT_MODULE_ESTIMATES["transformers"]
    ):
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

    _require_transformers()
    # The other encoder is never built: its weights are skipped on load
    # (``_keys_to_ignore_on_load_unexpected``), so they cost neither RAM nor time.
    drop = {"audio_config": None} if modality == "image" else {"vision_config": None}
    message = f"Loading {label} model weights…"
    cache_dir = embedder_load_setup(on_progress, message)
    with intercept_tqdm_progress(on_progress), intercept_weight_loading_progress(on_progress, message):
        model = load_pretrained_local_first(
            AutoModel.from_pretrained,
            EMBEDDINGGEMMA2_MODEL_ID,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            token=hf_token(),
            on_progress=on_progress,
            **drop,
        )
    model = to_compute_device(model)
    dtype = _weight_dtype()
    if dtype is not None:
        model = model.to(dtype)
    model.eval()
    on_progress("loading", f"Loading {label} processor…", 0, 0)
    with intercept_tqdm_progress(on_progress):
        processor = load_pretrained_local_first(
            AutoProcessor.from_pretrained,
            EMBEDDINGGEMMA2_MODEL_ID,
            cache_dir=cache_dir,
            token=hf_token(),
            **image_processor_load_kwargs(),
        )
    return model, processor


def query_messages(texts: list[str]) -> list[list[dict]]:
    """One chat per typed query, carrying the model's retrieval-query prefix.

    Queries are the asymmetric side of a search: the model card's
    ``SearchQuery`` prompt, ``task: search result | query: ``.  Images and audio
    take no prefix.
    """
    return [
        [{"role": "user", "content": [{"type": "text", "text": f"{EMBEDDINGGEMMA2_QUERY_PREFIX}{t}"}]}] for t in texts
    ]


def media_messages(kind: Modality, items: list[Any]) -> list[list[dict]]:
    """One chat per image (PIL) or per waveform (16 kHz mono float array)."""
    return [[{"role": "user", "content": [{"type": kind, kind: item}]}] for item in items]


def pooled_forward(model: Any, processor: Any, messages: list[list[dict]], **processor_kwargs: Any) -> np.ndarray:
    """``(N, 768)`` fp32 unit vectors: masked mean of the last hidden state.

    Mean over every unmasked token, the ``<bos>``, the image/audio delimiters
    and the task prefix included, as the checkpoint's own pooling config says
    (``include_prompt: true``).
    """
    import torch  # noqa: PLC0415

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True, **processor_kwargs},
    )
    feed = to_model_inputs(inputs, model)
    with torch.no_grad():
        hidden = model(**feed).last_hidden_state.float()
    mask = feed["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
