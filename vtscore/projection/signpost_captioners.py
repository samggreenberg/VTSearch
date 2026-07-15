"""Generative captioner signpost-text providers (image VLM, audio captioner).

The *generative* tier of the ``object_to_text`` layer (see
``docs/plans/vtsbrowse-toponymy.md`` and :mod:`vtscore.projection.signpost_texts`).
Where :class:`~vtscore.projection.signpost_texts.ZeroShotTagProvider` *matches*
each media against a fixed vocabulary (top-k terms by cosine), these *generate*
a free-text description from the raw media:

* **image** — ``Qwen/Qwen2.5-VL-3B-Instruct``, one instructed catalog line per
  image (the image study's resolved default; the SigLIP tags stay the no-VLM
  fallback and collapse on fine-grained subsets);
* **audio** — ``MU-NLPC/whisper-small-audio-captioning``, one caption per clip
  (the cleanest signs on the audio study's real-world sets).

These are **opt-in per media type**: the ``browse_signpost_captioner`` user
setting flows through :class:`~vtscore.config.CoreConfig` into
:func:`~vtscore.projection.signpost_texts.provider_for`, which selects the
captioner and wraps the tag provider as a fallback (model-load failure or a
per-item decode failure degrades to tags, so a missing model download never
leaves the map blank).

Heavy model deps (torch / transformers / PIL / librosa) are imported **lazily
inside the model seams** (:func:`_load_image_model` etc.), never at module load,
so importing this module is cheap and the library tier stays import-clean.  The
seams are module-level functions precisely so tests can stub the model without a
multi-GB download while the media-decode helpers run for real.  Models are
process-scoped and never persisted (the No-Persisted-Vectors rule); only the
generated *text* is cached on the media dict.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Progress callback shape: ``(current, total, message)``.
ProgressFn = Any

# ---------------------------------------------------------------------------
# Image: Qwen2.5-VL-3B-Instruct
# ---------------------------------------------------------------------------

#: The instructed one-liner prompt the image signpost study locked in: name the
#: type + subject specifically, and quote the key visible text for
#: document/screenshot media (so form/scan regions letter by their heading, not
#: their appearance).  Kept ≤20 words to stay legible as a map sign.
QWEN_PROMPT = (
    "Describe this image in one concise line (at most 20 words) for a browsing catalog. "
    "If it is a screenshot, document, form, or scanned page, say its type and quote the most "
    "important visible text. Otherwise describe the subject specifically (species, breed, "
    "object type). Output only the line."
)

#: Vision-token budget bounds (in 28×28 patches) so a document page doesn't
#: explode the batch — the same envelope the study's framework used.
_QWEN_MIN_PIXELS = 256 * 28 * 28
_QWEN_MAX_PIXELS = 768 * 28 * 28


def _load_image_model(model_id: str, on_progress: ProgressFn | None = None) -> tuple[Any, Any, str]:
    """Load the Qwen2.5-VL model + processor, offline-first.  Model seam (stubbed in tests)."""
    import torch  # noqa: PLC0415
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415

    from vtscore.config import resolve_device  # noqa: PLC0415
    from vtscore.media.embedder import hf_token, load_pretrained_local_first, to_compute_device  # noqa: PLC0415

    device = resolve_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = load_pretrained_local_first(
        Qwen2_5_VLForConditionalGeneration.from_pretrained,
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        token=hf_token(),
        on_progress=on_progress,
    )
    model = to_compute_device(model).eval()
    processor = load_pretrained_local_first(
        AutoProcessor.from_pretrained,
        model_id,
        min_pixels=_QWEN_MIN_PIXELS,
        max_pixels=_QWEN_MAX_PIXELS,
        token=hf_token(),
    )
    # Batched generation on a decoder-only model requires LEFT padding — right
    # padding makes the model continue from pad tokens.
    processor.tokenizer.padding_side = "left"
    return model, processor, device


def _generate_image_captions(
    model: Any, processor: Any, device: str, images: list[Any], prompt: str, max_new_tokens: int
) -> list[str]:
    """Run one greedy generation pass over a batch of PIL images.  Model seam (stubbed in tests)."""
    import torch  # noqa: PLC0415

    messages = [
        [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        for img in images
    ]
    prompts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i_) :] for i_, o in zip(inputs.input_ids, out)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [d.strip() for d in decoded]


# ---------------------------------------------------------------------------
# Audio: MU-NLPC/whisper-small-audio-captioning
# ---------------------------------------------------------------------------

#: The captioner checkpoint is whisper-small based, so it wants 16 kHz mono.
_AUDIO_CAPTION_SR = 16000
#: The checkpoint's training style prefix; forced as the decoder prompt so the
#: vanilla ``WhisperForConditionalGeneration`` emits captions, not transcripts.
_AUDIO_STYLE_PREFIX = "clotho > caption: "


def _load_audio_model(model_id: str, on_progress: ProgressFn | None = None) -> tuple[Any, Any, Any, str, Any]:
    """Load the whisper audio-captioner, offline-first.  Model seam (stubbed in tests)."""
    from transformers import (  # noqa: PLC0415
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperTokenizer,
    )

    from vtscore.config import resolve_device  # noqa: PLC0415
    from vtscore.media.embedder import hf_token, load_pretrained_local_first, to_compute_device  # noqa: PLC0415

    device = resolve_device()
    model = load_pretrained_local_first(
        WhisperForConditionalGeneration.from_pretrained, model_id, token=hf_token(), on_progress=on_progress
    )
    model = to_compute_device(model).eval()
    tokenizer = load_pretrained_local_first(
        WhisperTokenizer.from_pretrained, model_id, language="en", task="transcribe", token=hf_token()
    )
    feature_extractor = load_pretrained_local_first(WhisperFeatureExtractor.from_pretrained, model_id, token=hf_token())
    style_ids = tokenizer("", text_target=_AUDIO_STYLE_PREFIX, return_tensors="pt").labels[:, :-1]
    return model, tokenizer, feature_extractor, device, style_ids


def _generate_audio_captions(
    model: Any, tokenizer: Any, feature_extractor: Any, device: str, style_ids: Any, waveforms: list[np.ndarray]
) -> list[str]:
    """Caption a batch of mono 16 kHz waveforms.  Model seam (stubbed in tests)."""
    import torch  # noqa: PLC0415

    feats = [
        feature_extractor(wav, sampling_rate=_AUDIO_CAPTION_SR, return_tensors="pt").input_features[0]
        for wav in waveforms
    ]
    batch = torch.stack(feats).to(device)
    with torch.no_grad():
        out = model.generate(
            inputs=batch,
            decoder_input_ids=style_ids.to(device).repeat(len(waveforms), 1),
            max_length=80,
        )
    texts = []
    for row in out:
        text = tokenizer.decode(row, skip_special_tokens=True)
        texts.append(text.replace(_AUDIO_STYLE_PREFIX.strip(), "").strip(" :>"))
    return texts


# ---------------------------------------------------------------------------
# Media decode helpers (run for real; tested with tiny fixtures)
# ---------------------------------------------------------------------------


def _load_image(media: dict[str, Any]) -> Any | None:
    """Decode an image media dict to an RGB ``PIL.Image``, or ``None``.

    Reuses the image media type's own source-pick + decode
    (``media_bytes`` | ``media_path``); a media whose bytes/path is missing or
    undecodable simply yields ``None`` and gets no caption.
    """
    from vtscore.media.image._image_bulk import _load_pil, _pil_source_for  # noqa: PLC0415

    source = _pil_source_for(media)
    if source is None:
        return None
    try:
        return _load_pil(source)
    except Exception:
        return None


def _load_audio(media: dict[str, Any]) -> np.ndarray | None:
    """Decode an audio media dict to a mono 16 kHz waveform, or ``None``.

    Mirrors the audio embedders' ``media_bytes`` | ``media_path`` → ``librosa``
    branch; a missing/undecodable clip yields ``None`` and gets no caption.
    """
    blob = media.get("media_bytes")
    if isinstance(blob, (bytes, bytearray)) and blob:
        source: Any = io.BytesIO(bytes(blob))
    else:
        path = media.get("media_path")
        if not path:
            return None
        source = Path(path)
    try:
        import librosa  # noqa: PLC0415

        wav, _sr = librosa.load(source, sr=_AUDIO_CAPTION_SR, mono=True)
        return np.asarray(wav, dtype=np.float32)
    except Exception:
        return None


def _batched(items: list[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageCaptionProvider:
    """Qwen2.5-VL-3B one-line image captioner (a ``SignpostTextProvider``)."""

    name: str = "caption:qwen2.5-vl-3b"
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    prompt: str = QWEN_PROMPT
    max_new_tokens: int = 48
    batch_size: int = 8

    def signature(self, embedder: Any) -> str:
        # Captions are generated from pixels + prompt, independent of the media
        # embedder — so the cache key is the model, not the embedder.
        return self.name

    def build_texts(
        self,
        ids: list[int],
        medias: dict[int, dict[str, Any]],
        matrix: np.ndarray,
        embedder: Any,
        on_progress: ProgressFn | None = None,
    ) -> dict[int, str]:
        resolved = [(mid, img) for mid in ids if (img := _load_image(medias.get(mid, {}))) is not None]
        if not resolved:
            return {}
        model, processor, device = _load_image_model(self.model_id, on_progress)
        out: dict[int, str] = {}
        for chunk in _batched(resolved, self.batch_size):
            caps = _generate_image_captions(
                model, processor, device, [img for _, img in chunk], self.prompt, self.max_new_tokens
            )
            for (mid, _img), cap in zip(chunk, caps):
                if cap and cap.strip():
                    out[mid] = cap.strip()
            if on_progress is not None:
                on_progress(len(out), len(resolved), "Captioning images…")
        return out


@dataclass(frozen=True)
class AudioCaptionProvider:
    """MU-NLPC whisper audio captioner (a ``SignpostTextProvider``)."""

    name: str = "caption:whisper-audio"
    model_id: str = "MU-NLPC/whisper-small-audio-captioning"
    batch_size: int = 16

    def signature(self, embedder: Any) -> str:
        return self.name

    def build_texts(
        self,
        ids: list[int],
        medias: dict[int, dict[str, Any]],
        matrix: np.ndarray,
        embedder: Any,
        on_progress: ProgressFn | None = None,
    ) -> dict[int, str]:
        resolved = [(mid, wav) for mid in ids if (wav := _load_audio(medias.get(mid, {}))) is not None]
        if not resolved:
            return {}
        model, tokenizer, fe, device, style_ids = _load_audio_model(self.model_id, on_progress)
        out: dict[int, str] = {}
        for chunk in _batched(resolved, self.batch_size):
            caps = _generate_audio_captions(model, tokenizer, fe, device, style_ids, [wav for _, wav in chunk])
            for (mid, _wav), cap in zip(chunk, caps):
                if cap and cap.strip():
                    out[mid] = cap.strip()
            if on_progress is not None:
                on_progress(len(out), len(resolved), "Captioning audio…")
        return out


#: The default captioner instances, keyed by media type — imported by
#: :mod:`vtscore.projection.signpost_texts` into its ``_CAPTIONERS`` registry.
IMAGE_CAPTIONER = ImageCaptionProvider()
AUDIO_CAPTIONER = AudioCaptionProvider()


__all__ = [
    "AUDIO_CAPTIONER",
    "IMAGE_CAPTIONER",
    "AudioCaptionProvider",
    "ImageCaptionProvider",
    "QWEN_PROMPT",
]
