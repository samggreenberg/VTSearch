"""Extract embedded text from documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtsearch.converters.base import MediaConverter


class Document2TextMediaConverter(MediaConverter):
    """Extract the embedded text content from a PDF/DOC/PPT document.

    No OCR is performed — only text that is already encoded in the
    document structure is extracted.

    Supported formats:

    * **.pdf** — text extracted via PyMuPDF (``fitz``).
    * **.doc** — text extracted via PyMuPDF (``fitz``).
    * **.ppt** — text extracted via PyMuPDF (``fitz``).

    Returns a single-element list containing the concatenated text of
    all pages/slides, or an empty list if no text could be extracted.
    """

    display_name = "Document \u2192 Text"
    converter_description = "Extract embedded text from documents"

    @property
    def source_type(self) -> str:
        return "document"

    @property
    def target_type(self) -> str:
        return "text"

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
            print("Document2TextMediaConverter requires PyMuPDF: pip install PyMuPDF")
            return []

        try:
            doc = fitz.open(stream=media_bytes, filetype=Path(filename).suffix.lstrip(".") or "pdf")
        except Exception as e:
            print(f"Error opening document {filename}: {e}")
            return []

        page_texts: list[str] = []
        try:
            for page in doc:
                text = page.get_text().strip()
                if text:
                    page_texts.append(text)
        finally:
            doc.close()

        if not page_texts:
            return []

        full_text = "\n\n".join(page_texts)
        return [
            {
                "filename": f"{stem}.txt",
                "media_string": full_text,
                "duration": 0,
                "word_count": len(full_text.split()),
                "character_count": len(full_text),
            }
        ]


CONVERTER = Document2TextMediaConverter()
