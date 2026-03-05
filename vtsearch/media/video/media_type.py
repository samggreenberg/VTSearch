"""Video media type — MP4/AVI/MOV/WEBM/MKV files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from vtsearch.config import MODELS_CACHE_DIR, UCF101_SUBSET_DOWNLOAD_SIZE_MB, VIDEO_DIR, XCLIP_MODEL_ID
from vtsearch.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
)


_VIDEO_MIME_TYPES: dict[str, str] = {
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


class VideoMediaType(MediaType):
    """Handles video medias — file import, HTTP serving, and demo datasets.

    Embedding is handled by :class:`~vtsearch.media.video.embedder.VideoXClipEmbedder`.
    """

    def __init__(self) -> None:
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
    # Display metadata
    # ------------------------------------------------------------------

    def display_metadata(self, media: dict) -> dict:
        result: dict = {}
        cat = media.get("category")
        if cat and cat not in ("unknown", "custom"):
            result["Category"] = cat
        dur = media.get("duration")
        if dur and dur > 0:
            result["Duration"] = dur
        fs = media.get("file_size")
        if fs:
            result["File Size"] = fs
        return result

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    _DEMO_CATEGORIES = [
        "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
        "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    ]

    @property
    def demo_datasets(self) -> list:
        cats = self._DEMO_CATEGORIES
        folder = VIDEO_DIR / "ucf101"
        return [
            DemoDataset(
                id="ucf101_s", label="UCF-101 (S)",
                description="Action recognition videos sourced from YouTube, covering sports and everyday activities.",
                categories=cats, source="ucf101", required_folder=folder,
                slice_start=0, slice_end=15, download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucf101_m", label="UCF-101 (M)",
                description="Action recognition videos sourced from YouTube, covering sports and everyday activities.",
                categories=cats, source="ucf101", required_folder=folder,
                slice_start=15, slice_end=40, download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucf101_l", label="UCF-101 (L)",
                description="Action recognition videos sourced from YouTube, covering sports and everyday activities.",
                categories=cats, source="ucf101", required_folder=folder,
                slice_start=40, slice_end=150, download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
            ),
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def load_demo_source(self, source, categories, slice_start, slice_end, clips, on_progress=None, embedder=None):
        import hashlib  # noqa: PLC0415

        if on_progress is None:
            from vtsearch.utils import update_progress

            on_progress = update_progress

        if embedder is None:
            from vtsearch.media import embedders_for_type

            avail = embedders_for_type(self.type_id)
            if not avail:
                raise ValueError(f"No embedders registered for media type {self.type_id!r}")
            embedder = avail[0]

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

        if getattr(embedder, "_model", None) is None:
            on_progress("loading", "Loading video embedding model…", 0, 0)
            embedder.load_models()

        clip_id = 1
        total = len(video_files)
        on_progress("embedding", f"Starting embedding for {total} video files...", 0, total)
        demo_origin_template: dict = {"importer": "demo", "params": {}}

        for i, (video_path, meta) in enumerate(video_files):
            rel_name = f"{meta['category']}/{video_path.name}"
            on_progress("embedding", f"Embedding {rel_name} ({i + 1}/{total})", i + 1, total)
            embedding = embedder.embed_media(video_path)
            if embedding is None:
                continue
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            media_fields = self.load_media_data(video_path)
            clips[clip_id] = {
                "id": clip_id,
                "type": self.type_id,
                "embedder": embedder.name,
                "duration": media_fields["duration"],
                "file_size": len(video_bytes),
                "md5": hashlib.md5(video_bytes).hexdigest(),
                "embedding": embedding,
                "media_bytes": video_bytes,
                "filename": rel_name,
                "category": meta["category"],
                "origin": {**demo_origin_template},
                "origin_name": rel_name,
            }
            clip_id += 1

        return str(video_dir.absolute())

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
