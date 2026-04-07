"""Video embedder — X-CLIP (microsoft/xclip-base-patch32)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import XCLIP_MODEL_ID
from vtsearch.media.base import (
    MediaEmbedder,
    embedder_load_setup,
    extract_tensor as _extract_tensor,
    intercept_tqdm_progress,
    intercept_weight_loading_progress,
    load_pretrained_local_first,
)

if TYPE_CHECKING:
    from transformers import XCLIPModel, XCLIPProcessor


class VideoXClipEmbedder(MediaEmbedder):
    """Embeds videos using the X-CLIP model (microsoft/xclip-base-patch32).

    * Videos → 768-dimensional vectors via X-CLIP's video encoder (8 sampled frames).
    * Text queries → 768-dimensional vectors via X-CLIP's text encoder.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[XCLIPModel] = None
        self._processor: Optional[XCLIPProcessor] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "xclip"

    @property
    def media_type_id(self) -> str:
        return "video"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_models_impl(self) -> None:
        if self._model is not None:
            return

        self._on_progress("loading", "Importing torch…", 1, 2)
        import torch  # noqa: F401, PLC0415

        self._on_progress("loading", "Importing transformers…", 2, 2)
        from transformers import XCLIPModel, XCLIPProcessor  # noqa: PLC0415

        cache_dir = embedder_load_setup(self._on_progress, "Loading X-CLIP model weights…")
        XCLIPModel._keys_to_ignore_on_load_unexpected = [r".*position_ids.*"]
        with intercept_tqdm_progress(self._on_progress), intercept_weight_loading_progress(
            self._on_progress, "Loading X-CLIP model weights…"
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

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import cv2  # noqa: PLC0415
            import torch  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415

            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                print(f"Error opening video {file_path}")
                return None

            try:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    print(f"Error: could not determine frame count for {file_path}")
                    return None
                num_frames = min(8, max(1, frame_count))
                indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame))
            finally:
                cap.release()

            if not frames:
                print(f"Error: could not extract frames from {file_path}")
                return None

            # X-CLIP expects exactly 8 frames; pad by repeating if we have fewer
            while len(frames) < 8:
                frames.append(frames[-1])

            inputs = self._processor(images=[list(frames)], return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.get_video_features(**inputs)
                embedding = _extract_tensor(outputs).detach().cpu().numpy()
            return embedding[0]
        except Exception as e:
            logging.getLogger(__name__).exception("Error embedding %s", file_path)
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
        except Exception as e:
            logging.getLogger(__name__).exception("Error embedding text query for video")
            return None

    # Internal helper used by loader.py bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor
