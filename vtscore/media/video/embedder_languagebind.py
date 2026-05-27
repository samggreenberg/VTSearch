"""Video embedder - LanguageBind Video (LanguageBind/LanguageBind_Video_V1.5_FT)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from vtscore.config import LANGUAGEBIND_VIDEO_MODEL_ID
from vtscore.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
)
from vtscore.media.video._frame_sampling import sample_video_frames

# CLIP normalization constants (shared with LanguageBind).
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_NUM_FRAMES = 8
_IMAGE_SIZE = 224


def _preprocess_frames(frames: list) -> np.ndarray:
    """Resize, center-crop, and normalise a list of PIL Images.

    Returns an array of shape ``(3, T, H, W)`` ready to be wrapped in a
    batch dimension and converted to a tensor.
    """
    processed = []
    for img in frames:
        w, h = img.size
        # Scale shortest side to _IMAGE_SIZE.
        if w < h:
            new_w = _IMAGE_SIZE
            new_h = int(h * _IMAGE_SIZE / w)
        else:
            new_h = _IMAGE_SIZE
            new_w = int(w * _IMAGE_SIZE / h)
        img = img.resize((new_w, new_h))
        # Center crop to _IMAGE_SIZE x _IMAGE_SIZE.
        left = (new_w - _IMAGE_SIZE) // 2
        top = (new_h - _IMAGE_SIZE) // 2
        img = img.crop((left, top, left + _IMAGE_SIZE, top + _IMAGE_SIZE))
        # Convert to float32 array (H, W, C) in [0, 1].
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # Normalize with CLIP mean/std.
        arr = (arr - _CLIP_MEAN) / _CLIP_STD
        # Transpose to (C, H, W).
        processed.append(arr.transpose(2, 0, 1))

    # Stack into (T, C, H, W) then transpose to (C, T, H, W).
    stacked = np.stack(processed, axis=0)  # (T, C, H, W)
    return stacked.transpose(1, 0, 2, 3)  # (C, T, H, W)


class VideoLanguageBindEmbedder(MediaEmbedder):
    """Embeds videos using the LanguageBind Video model.

    * Videos -> 768-dimensional vectors via LanguageBind's video encoder (8 sampled frames).
    * Text queries -> 768-dimensional vectors via LanguageBind's text encoder.

    Significantly better video-text retrieval quality than X-CLIP on
    MSR-VTT, MSVD, DiDeMo, and ActivityNet benchmarks (ICLR 2024).
    """

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "languagebind"

    @property
    def display_name(self) -> str:
        return "LanguageBind (video)"

    @property
    def media_type_id(self) -> str:
        return "video"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch...", 1, 2):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing transformers...", 2, 2):
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading LanguageBind model weights...")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading LanguageBind model weights..."),
        ):
            self._model = load_pretrained_local_first(
                AutoModel.from_pretrained,
                LANGUAGEBIND_VIDEO_MODEL_ID,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
                trust_remote_code=True,
            )
        self._model = self._model.to("cpu")
        self._on_progress("loading", "Loading LanguageBind tokenizer...", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._tokenizer = load_pretrained_local_first(
                AutoTokenizer.from_pretrained,
                LANGUAGEBIND_VIDEO_MODEL_ID,
                cache_dir=cache_dir,
                token=False,
                trust_remote_code=True,
            )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @property
    def description_wrappers(self) -> list[str]:
        return [
            "a video of {text}",
            "a media showing {text}",
            "{text}",
            "footage of {text}",
            "a video media of {text}",
        ]

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._tokenizer is None:
            return None
        source_repr = media.get("media_path") or media.get("filename") or "<bytes>"
        try:
            import torch  # noqa: PLC0415

            frames = sample_video_frames(media, _NUM_FRAMES)
            if not frames:
                logging.getLogger(__name__).error("Could not extract frames from %s", source_repr)
                return None

            # Preprocess: resize, center-crop, normalize -> (C, T, H, W).
            pixel_values = _preprocess_frames(frames)
            # Add batch dimension -> (1, C, T, H, W).
            pixel_values = torch.tensor(pixel_values, dtype=torch.float32).unsqueeze(0)

            device = next(self._model.parameters()).device
            pixel_values = pixel_values.to(device)
            with torch.no_grad():
                video_features = self._model.get_image_features(pixel_values=pixel_values)
                # L2-normalize (get_image_features returns un-normalized vectors).
                video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
                embedding = video_features.detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", source_repr)
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._tokenizer is None:
            return None
        try:
            import torch  # noqa: PLC0415

            tokens = self._tokenizer(
                [text],
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                text_features = self._model.get_text_features(**tokens)
                # L2-normalize (get_text_features returns un-normalized vectors).
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                text_vec = text_features.detach().cpu().numpy()[0]
            return text_vec
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for video (LanguageBind)")
            return None

    # Internal helper used by loader.py bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._tokenizer


EMBEDDER = VideoLanguageBindEmbedder()
