"""Convert document pages to images."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from vtscore.converters.base import MediaConverter, resolve_media_bytes
from vtscore.utils.optional_deps import agpl_unavailable_message

logger = logging.getLogger(__name__)


class Document2ImageMediaConverter(MediaConverter):
    """Render each page of a PDF/DOC/PPT document as a PNG image.

    .. attribute:: display_name
       :value: "Document \u2192 Images"

    .. attribute:: description
       :value: "Render document pages as images"

    Supported formats:

    * **.pdf** - rendered via PyMuPDF (``fitz``).
    * **.doc** - rendered via PyMuPDF (``fitz``), which can open legacy
      Word files on most platforms.
    * **.ppt** - rendered via PyMuPDF (``fitz``), which can open legacy
      PowerPoint files on most platforms.

    Each page becomes one item in the returned list.
    """

    display_name = "Document \u2192 Images"
    description = "Render document pages as images"
    summary_template = "Render each page of each document as a PNG image."

    @property
    def source_type(self) -> str:
        return "document"

    @property
    def target_type(self) -> str:
        return "image"

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filename = media.get("filename", "document")
        stem = Path(filename).stem

        media_bytes = resolve_media_bytes(media)
        if not media_bytes:
            return []

        try:
            import fitz  # noqa: PLC0415 - PyMuPDF
        except ImportError:
            logger.warning("%s", agpl_unavailable_message("PyMuPDF", "Converting documents to images"))
            return []

        results: list[dict[str, Any]] = []
        try:
            doc = fitz.open(stream=media_bytes, filetype=Path(filename).suffix.lstrip(".") or "pdf")
        except Exception:
            logger.error("Failed to open document %s", filename, exc_info=True)
            return []

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render at 2x zoom for reasonable resolution
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                png_bytes = pix.tobytes("png")

                from PIL import Image  # noqa: PLC0415

                img = Image.open(io.BytesIO(png_bytes))
                width, height = img.width, img.height

                results.append(
                    {
                        "filename": f"{stem}_page_{page_num + 1}.png",
                        "media_bytes": png_bytes,
                        "duration": 0,
                        "width": width,
                        "height": height,
                    }
                )
        finally:
            doc.close()

        return results


CONVERTER = Document2ImageMediaConverter()
