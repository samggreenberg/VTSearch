"""Image media type — JPEG/PNG/GIF/BMP/WEBP files."""

from __future__ import annotations

from pathlib import Path


from vtscore.media.base import (
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
)
from vtscore.media.image._demo_sources import build_demo_datasets, load_demo_source


_IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ImageMediaType(MediaType):
    """Handles image medias — file import, HTTP serving, and demo datasets.

    Embedding is handled by :class:`~vtscore.media.image.embedder_siglip.ImageSiglipEmbedder`.
    """

    def __init__(self) -> None:
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "image"

    @property
    def name(self) -> str:
        return "Image"

    @property
    def icon(self) -> str:
        return "image"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]

    @property
    def folder_import_name(self) -> str:
        return "image"

    @property
    def dir_key(self) -> str:
        return "image_dir"

    @property
    def pickle_extra_fields(self) -> list[str]:
        return ["width", "height"]

    # ------------------------------------------------------------------
    # Display metadata
    # ------------------------------------------------------------------

    def display_metadata(self, media: dict) -> dict:
        result: dict = {}
        cat = media.get("category")
        if cat and cat not in ("unknown", "custom"):
            result["Category"] = cat
        w, h = media.get("width"), media.get("height")
        if w and h:
            result["Dimensions"] = f"{w}×{h}"
        fs = media.get("file_size")
        if fs:
            result["File Size"] = fs
        result.update({k: v for k, v in super().display_metadata(media).items() if k not in result})
        return result

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    @property
    def demo_datasets(self) -> list:
        return build_demo_datasets()

    def load_demo_source(
        self,
        source,
        categories,
        slice_start,
        slice_end,
        clips,
        on_progress=None,
        embedder=None,
        slice_frac_start=None,
        slice_frac_end=None,
        **kwargs,
    ):
        return load_demo_source(
            source,
            categories,
            slice_start,
            slice_end,
            clips,
            on_progress=on_progress,
            embedder=embedder,
            slice_frac_start=slice_frac_start,
            slice_frac_end=slice_frac_end,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    def load_media_data(self, file_path: Path, media_bytes: bytes | None = None) -> dict:
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        if media_bytes is None:
            with open(file_path, "rb") as f:
                media_bytes = f.read()
        try:
            with Image.open(io.BytesIO(media_bytes)) as img:
                width, height = img.width, img.height
        except Exception:
            width, height = None, None
        return {
            "media_bytes": media_bytes,
            "duration": 0,
            "width": width,
            "height": height,
        }

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def media_response(self, media: dict) -> MediaResponse:
        filename = media.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".jpg"
        mimetype = _IMAGE_MIME_TYPES.get(ext, "image/jpeg")
        data = self._resolve_media_bytes(media)
        if data is None:
            return MediaResponse(data=b"", mimetype=mimetype, download_name=f"media_{media['id']}{ext}")
        return MediaResponse(
            data=data,
            mimetype=mimetype,
            download_name=f"media_{media['id']}{ext}",
        )
