"""Video embedder — X-CLIP (microsoft/xclip-base-patch32)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import MODELS_CACHE_DIR, XCLIP_MODEL_ID
from vtsearch.media.base import MediaEmbedder, intercept_tqdm_progress

if TYPE_CHECKING:
    import torch
    from transformers import XCLIPModel, XCLIPProcessor


def _extract_tensor(output: object) -> torch.Tensor:
    """Extract a plain tensor from model output.

    Depending on the transformers version, get_video_features() / get_text_features()
    may return either a raw tensor or a BaseModelOutputWithPooling dataclass.
    This helper handles both cases.
    """
    import torch  # noqa: PLC0415

    if isinstance(output, torch.Tensor):
        return output
    for attr in ("video_embeds", "text_embeds", "pooler_output"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    # Final fallback: treat as tuple-like and return first element
    return output[0]  # type: ignore[index]


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

    def load_models(self) -> None:
        if self._model is not None:
            return
        import gc

        from transformers import XCLIPModel, XCLIPProcessor  # noqa: PLC0415

        from vtsearch.models.loader import ensure_torch_configured

        ensure_torch_configured()
        gc.collect()
        cache_dir = str(MODELS_CACHE_DIR)
        self._on_progress("loading", "Loading X-CLIP model weights…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._model = XCLIPModel.from_pretrained(
                XCLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False
            )
        self._model = self._model.to("cpu")
        self._on_progress("loading", "Loading X-CLIP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = XCLIPProcessor.from_pretrained(
                XCLIP_MODEL_ID, cache_dir=cache_dir, use_fast=False, token=False
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

            inputs = self._processor(videos=list(frames), return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.get_video_features(**inputs)
                embedding = _extract_tensor(outputs).detach().cpu().numpy()
            return embedding[0]
        except Exception as e:
            print(f"Error embedding {file_path}: {e}")
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
            print(f"Error embedding text query for video: {e}")
            return None

    # Internal helper used by loader.py bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor
