"""Video media type — X-CLIP embeddings, MP4/AVI/MOV/WEBM/MKV files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import MODELS_CACHE_DIR, UCF101_SUBSET_DOWNLOAD_SIZE_MB, VIDEO_DIR, XCLIP_MODEL_ID

if TYPE_CHECKING:
    import torch
    from transformers import XCLIPModel, XCLIPProcessor
from vtsearch.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
    intercept_tqdm_progress,
)


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


_VIDEO_MIME_TYPES: dict[str, str] = {
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}


class VideoMediaType(MediaType):
    """Handles video medias using the X-CLIP model (microsoft/xclip-base-patch32).

    * Embeds videos by sampling 8 evenly-spaced frames and running them through
      X-CLIP's video encoder.
    * Embeds text queries via X-CLIP's text encoder (same 768-dim space).
    * Serves medias with MIME types inferred from the file extension.
    """

    def __init__(self) -> None:
        self._model: Optional[XCLIPModel] = None
        self._processor: Optional[XCLIPProcessor] = None
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "video"

    @property
    def name(self) -> str:
        return "Video"

    @property
    def icon(self) -> str:
        return "🎬"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.mp4", "*.avi", "*.mov", "*.webm", "*.mkv"]

    @property
    def folder_import_name(self) -> str:
        return "videos"

    @property
    def tab_title(self) -> str:
        return "Videos"

    @property
    def dir_key(self) -> str:
        return "video_dir"

    @property
    def legacy_bytes_keys(self) -> list[str]:
        return ["video_bytes"]

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    # Shared categories for all S/M/L video demo datasets.
    # All three sizes use the same 10 categories; only the underlying
    # medias differ (disjoint slices within each category's UCF-101 videos).
    _DEMO_CATEGORIES = [
        "ApplyEyeMakeup",
        "ApplyLipstick",
        "Archery",
        "BabyCrawling",
        "BalanceBeam",
        "BandMarching",
        "BaseballPitch",
        "Basketball",
        "BasketballDunk",
        "BenchPress",
    ]

    @property
    def demo_datasets(self) -> list:
        cats = self._DEMO_CATEGORIES
        folder = VIDEO_DIR / "ucf101"
        return [
            DemoDataset(
                id="ucf101_s",
                label="UCF-101 (S)",
                description="Action recognition videos sourced from YouTube, covering sports and everyday activities.",
                categories=cats,
                source="ucf101",
                required_folder=folder,
                slice_start=0,
                slice_end=15,
                download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucf101_m",
                label="UCF-101 (M)",
                description="Action recognition videos sourced from YouTube, covering sports and everyday activities.",
                categories=cats,
                source="ucf101",
                required_folder=folder,
                slice_start=15,
                slice_end=40,
                download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucf101_l",
                label="UCF-101 (L)",
                description="Action recognition videos sourced from YouTube, covering sports and everyday activities.",
                categories=cats,
                source="ucf101",
                required_folder=folder,
                slice_start=40,
                slice_end=150,
                download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
            ),
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def load_demo_source(self, source, categories, slice_start, slice_end, clips, on_progress=None):
        import hashlib  # noqa: PLC0415

        if on_progress is None:
            from vtsearch.utils import update_progress

            on_progress = update_progress

        if source != "ucf101":
            raise ValueError(f"Unsupported video source: {source!r}")

        from vtsearch.datasets.downloader import download_ucf101_subset  # noqa: PLC0415
        from vtsearch.datasets.loader import load_video_metadata_from_folders  # noqa: PLC0415

        video_dir = download_ucf101_subset(on_progress=on_progress)
        metadata = load_video_metadata_from_folders(video_dir, categories)

        by_cat: dict[str, list] = {}
        for fname, meta in sorted(metadata.items()):
            cat = meta["category"]
            by_cat.setdefault(cat, []).append((meta["path"], meta))

        video_files: list[tuple] = []
        for cat in categories:
            video_files.extend(by_cat.get(cat, [])[slice_start:slice_end])

        if getattr(self, "_model", None) is None:
            on_progress("loading", "Loading video embedding model…", 0, 0)
            self.load_models()

        clip_id = 1
        total = len(video_files)
        on_progress("embedding", f"Starting embedding for {total} video files...", 0, total)
        demo_origin: dict = {"importer": "demo", "params": {}}

        for i, (video_path, meta) in enumerate(video_files):
            on_progress(
                "embedding",
                f"Embedding {meta['category']}: {video_path.name} ({i + 1}/{total})",
                i + 1,
                total,
            )
            embedding = self.embed_media(video_path)
            if embedding is None:
                continue
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            media_fields = self.load_media_data(video_path)
            rel_filename = str(video_path.relative_to(video_dir))
            clips[clip_id] = {
                "id": clip_id,
                "type": self.type_id,
                "duration": media_fields["duration"],
                "file_size": len(video_bytes),
                "md5": hashlib.md5(video_bytes).hexdigest(),
                "embedding": embedding,
                "media_bytes": video_bytes,
                "filename": rel_filename,
                "category": meta["category"],
                "origin": demo_origin,
                "origin_name": video_path.name,
            }
            clip_id += 1

        return str(video_dir.absolute())

    # ------------------------------------------------------------------
    # Embeddings
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
        # Materialize any tensors left on the ``meta`` device.
        self._model = self._model.to("cpu")
        self._on_progress("loading", "Loading X-CLIP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = XCLIPProcessor.from_pretrained(
                XCLIP_MODEL_ID, cache_dir=cache_dir, use_fast=False, token=False
            )

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import cv2  # noqa: PLC0415  (lazy import — cv2 is optional)
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

    # internal helper used by loader.py's get_xclip_model() bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    def load_media_data(self, file_path: Path) -> dict:
        with open(file_path, "rb") as f:
            media_bytes = f.read()
        try:
            import cv2  # noqa: PLC0415

            cap = cv2.VideoCapture(str(file_path))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0.0
            finally:
                cap.release()
        except Exception:
            duration = 0.0
        return {"media_bytes": media_bytes, "duration": duration}

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def media_response(self, media: dict) -> MediaResponse:
        filename = media.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".mp4"
        mimetype = _VIDEO_MIME_TYPES.get(ext, "video/mp4")
        data = self._resolve_media_bytes(media)
        if data is None:
            return MediaResponse(data=b"", mimetype=mimetype, download_name=f"media_{media['id']}{ext}")
        return MediaResponse(
            data=data,
            mimetype=mimetype,
            download_name=f"media_{media['id']}{ext}",
        )
