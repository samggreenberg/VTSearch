"""Video embedder - X-CLIP (microsoft/xclip-base-patch32)."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from vtscore.config import XCLIP_MODEL_ID
from vtscore.media.embedder import (
    MediaEmbedder,
    embedder_load_setup,
    extract_tensor as _extract_tensor,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
    timed_progress,
)
from vtscore.media.video._frame_sampling import sample_video_frames

_NUM_FRAMES = 8


class VideoXClipEmbedder(MediaEmbedder):
    """Embeds videos using the X-CLIP model (microsoft/xclip-base-patch32).

    * Videos → 768-dimensional vectors via X-CLIP's video encoder (8 sampled frames).
    * Text queries → 768-dimensional vectors via X-CLIP's text encoder.
    """

    def __init__(self) -> None:
        super().__init__()
        # Typed ``Any``: transformers stubs miss several ``XCLIPProcessor.__call__``
        # kwargs we pass at runtime; runtime ``None`` checks guard the calls.
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "xclip"

    @property
    def display_name(self) -> str:
        return "X-CLIP (video)"

    @property
    def media_type_id(self) -> str:
        return "video"

    @property
    def is_default(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        with timed_progress(self._on_progress, "loading", "Importing torch…", 1, 2):
            import torch  # noqa: F401, PLC0415

        with timed_progress(self._on_progress, "loading", "Importing transformers…", 2, 2):
            from transformers import XCLIPModel, XCLIPProcessor  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading X-CLIP model weights…")
        XCLIPModel._keys_to_ignore_on_load_unexpected = [r".*position_ids.*"]
        with (
            intercept_tqdm_progress(self._on_progress),
            intercept_weight_loading_progress(self._on_progress, "Loading X-CLIP model weights…"),
        ):
            self._model = load_pretrained_local_first(
                XCLIPModel.from_pretrained, XCLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False
            )
        self._model = self._model.to("cpu")
        self._on_progress("loading", "Loading X-CLIP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = load_pretrained_local_first(
                XCLIPProcessor.from_pretrained, XCLIP_MODEL_ID, cache_dir=cache_dir, use_fast=False, token=False
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
        if self._model is None or self._processor is None:
            return None
        source_repr = media.get("media_path") or media.get("filename") or "<bytes>"
        try:
            import torch  # noqa: PLC0415

            frames = sample_video_frames(media, _NUM_FRAMES)
            if not frames:
                logging.getLogger(__name__).error("Could not extract frames from %s", source_repr)
                return None

            inputs = self._processor(images=[list(frames)], return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.get_video_features(**inputs)
                embedding = _extract_tensor(outputs).detach().cpu().numpy()
            return embedding[0]
        except Exception:
            logging.getLogger(__name__).exception("Error embedding %s", source_repr)
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            inputs = self._processor(text=[text], return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                text_vec = _extract_tensor(self._model.get_text_features(**inputs)).detach().cpu().numpy()[0]
            return text_vec
        except Exception:
            logging.getLogger(__name__).exception("Error embedding text query for video")
            return None

    # Internal helper used by loader.py bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor


EMBEDDER = VideoXClipEmbedder()
