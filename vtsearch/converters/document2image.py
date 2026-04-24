"""Convert document pages to images."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any


from vtsearch.converters.base import MediaConverter


class Document2ImageMediaConverter(MediaConverter):
    """Render each page of a PDF/DOC/PPT document as a PNG image.

    .. attribute:: display_name
       :value: "Document \u2192 Images"

    .. attribute:: converter_description
       :value: "Render document pages as images"

    Supported formats:

    * **.pdf** — rendered via PyMuPDF (``fitz``).
    * **.doc** — rendered via PyMuPDF (``fitz``), which can open legacy
      Word files on most platforms.
    * **.ppt** — rendered via PyMuPDF (``fitz``), which can open legacy
      PowerPoint files on most platforms.

    Each page becomes one item in the returned list.
    """

    display_name = "Document \u2192 Images"
    converter_description = "Render document pages as images"

    @property
    def source_type(self) -> str:
        return "document"

    @property
    def target_type(self) -> str:
        return "image"

    def convert(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        media_bytes = media.get("media_bytes")
        media_path = media.get("media_path")
        filename = media.get("filename", "document")
        stem = Path(filename).stem

        if media_bytes is None and media_path:
            path = Path(media_path)
            if path.exists():
                with open(path, "rb") as f:
                    media_bytes = f.read()

        if not media_bytes:
            return []

        try:
            import fitz  # noqa: PLC0415 — PyMuPDF
        except ImportError:
            print("Document2ImageMediaConverter requires PyMuPDF: pip install PyMuPDF")
            return []

        results: list[dict[str, Any]] = []
        try:
            doc = fitz.open(stream=media_bytes, filetype=Path(filename).suffix.lstrip(".") or "pdf")
        except Exception as e:
            print(f"Error opening document {filename}: {e}")
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
