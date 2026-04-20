"""Document media type — PDF/DOC/PPT files (no embedder)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from vtsearch.media.base import (
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
)

_DOCUMENT_MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".ppt": "application/vnd.ms-powerpoint",
}


class DocumentMediaType(MediaType):
    """Handles document files (PDF, DOC, PPT).

    This media type does **not** have an embedder — :meth:`embed_media` and
    :meth:`embed_text` always return ``None``.  Documents are intended to
    be converted to other media types (images, text, audio) via
    :class:`~vtsearch.converters.base.MediaConverter` subclasses before
    embedding.

    Documents are served as their original binary format with the
    appropriate MIME type.
    """

    def __init__(self) -> None:
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "document"

    @property
    def name(self) -> str:
        return "Document"

    @property
    def icon(self) -> str:
        return "document"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.pdf", "*.doc", "*.ppt"]

    @property
    def folder_import_name(self) -> str:
        return "document"

    @property
    def tab_title(self) -> str:
        return "Documents"

    @property
    def dir_key(self) -> str:
        return "document_dir"

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
        return []

    # ------------------------------------------------------------------
    # Embeddings (not supported)
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        pass

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        return None

    # ------------------------------------------------------------------
    # Media data
    # ------------------------------------------------------------------

    def load_media_data(self, file_path: Path) -> dict:
        with open(file_path, "rb") as f:
            media_bytes = f.read()
        return {"media_bytes": media_bytes, "duration": 0}

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def media_response(self, media: dict) -> MediaResponse:
        filename = media.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".pdf"
        mimetype = _DOCUMENT_MIME_TYPES.get(ext, "application/octet-stream")
        data = self._resolve_media_bytes(media)
        if data is None:
            return MediaResponse(data=b"", mimetype=mimetype, download_name=f"media_{media['id']}{ext}")
        return MediaResponse(
            data=data,
            mimetype=mimetype,
            download_name=f"media_{media['id']}{ext}",
        )
