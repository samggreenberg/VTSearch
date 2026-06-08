"""Video embedder - VideoMAE v2 (OpenGVLab/VideoMAEv2-Base).

VideoMAE v2 is a vision-only video encoder pretrained with a masked
autoencoder objective on a large action-recognition corpus. Compared
to X-CLIP / LanguageBind (which optimise for video-text alignment) it
produces noticeably stronger motion / action features at the cost of a
text tower - there is no paired text encoder, so :meth:`embed_text`
returns ``None`` and :attr:`supports_text` is ``False``. Use it when
the dataset is mostly about *what is happening* rather than *what the
caption says*; pair with image-/text-based detectors for hybrid flows.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from vtscore.config import VIDEOMAE_MODEL_ID
from vtscore.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    hf_token,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
)
from vtscore.media.video._frame_sampling import sample_video_frames

# ImageNet normalisation - VideoMAE's standard preprocessing pipeline.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# VideoMAE v2 Base operates on 16 frames at 224x224.  Tubelet size 2 means
# 8 temporal patches; spatial patch size 16 means 14x14 spatial patches.
_NUM_FRAMES = 16
_IMAGE_SIZE = 224


def _preprocess_frames(frames: list) -> np.ndarray:
    """Resize, center-crop, and normalise PIL Images for VideoMAE v2.

    Returns a ``(T, C, H, W)`` ``float32`` array - the caller wraps it
    in a batch dimension to produce VideoMAE's expected
    ``(B, T, C, H, W)`` input.
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
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        processed.append(arr.transpose(2, 0, 1))  # (C, H, W)
    return np.stack(processed, axis=0)  # (T, C, H, W)


def _pool_features(outputs: Any) -> Any:
    """Reduce a VideoMAE forward-pass output to a single ``(B, D)`` tensor.

    Different VideoMAE checkpoints expose features under different
    attribute names - the original transformers ``VideoMAEModel`` exposes
    ``last_hidden_state``, while OpenGVLab's custom remote-code variant
    sometimes returns a raw tensor or a ``pooler_output``. We probe the
    common shapes and mean-pool patch tokens when needed.
    """
    import torch  # noqa: PLC0415

    if isinstance(outputs, torch.Tensor):
        return outputs if outputs.ndim == 2 else outputs.mean(dim=1)
    pooled = getattr(outputs, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    hidden = getattr(outputs, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        # (B, T*H*W, D) - no CLS token in VideoMAE, so mean-pool every patch.
        return hidden.mean(dim=1)
    # Fallback: assume tuple-like with the feature tensor first.
    first = outputs[0]
    return first if first.ndim == 2 else first.mean(dim=1)


class VideoVideoMAEEmbedder(MediaEmbedder):
    """Embeds videos using VideoMAE v2 (OpenGVLab/VideoMAEv2-Base).

    * Videos -> 768-dimensional vectors via VideoMAE's video encoder
      (16 sampled frames, mean-pooled over patch tokens, L2-normalised).
    * Text queries are **not** supported - VideoMAE has no text tower,
      so :meth:`embed_text` returns ``None`` and the frontend hides
      text-search UI for datasets embedded with this model.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "videomae"

    @property
    def display_name(self) -> str:
        return "VideoMAE v2 (action features)"

    @property
    def media_type_id(self) -> str:
        return "video"

    @property
    def is_default(self) -> bool:
        return False

    @property
    def supports_text(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch...", 1, 2):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing transformers...", 2, 2):
            from transformers import AutoModel  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading VideoMAE v2 model weights...")
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading VideoMAE v2 model weights..."),
        ):
            self._model = load_pretrained_local_first(
                AutoModel.from_pretrained,
                VIDEOMAE_MODEL_ID,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=hf_token(),
                trust_remote_code=True,
                on_progress=self._on_progress,
            )
        self._model = self._model.to("cpu")
        self._model.eval()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None:
            return None
        source_repr = media.get("media_path") or media.get("filename") or "<bytes>"
        try:
            import torch  # noqa: PLC0415

            frames = sample_video_frames(media, _NUM_FRAMES)
            if not frames:
                logging.getLogger(__name__).error("Could not extract frames from %s", source_repr)
                return None

            # (T, C, H, W) -> (1, T, C, H, W)
            pixel_values = _preprocess_frames(frames)
            pixel_values = torch.tensor(pixel_values, dtype=torch.float32).unsqueeze(0)

            device = next(self._model.parameters()).device
            pixel_values = pixel_values.to(device)
            with torch.no_grad():
                outputs = self._model(pixel_values)
                pooled = _pool_features(outputs)
                # L2-normalise so cosine similarity behaves consistently
                # with the other video embedders (xclip, languagebind).
                pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
                embedding = pooled.detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", source_repr)
            return None

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        """VideoMAE is vision-only - text queries are unsupported.

        Returns ``None`` so :func:`vtsearch.routes.media.embed.embed_text`
        and other callers can surface the "this embedder doesn't support
        text" path without a brittle attribute check.
        """
        return None


EMBEDDER = VideoVideoMAEEmbedder()
