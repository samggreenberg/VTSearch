"""Extract embedded text from documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from vtscore.converters.base import MediaConverter, resolve_media_bytes
from vtscore.utils.optional_deps import agpl_unavailable_message


class Document2TextMediaConverter(MediaConverter):
    """Extract the embedded text content from a PDF/DOC/PPT document.

    No OCR is performed - only text that is already encoded in the
    document structure is extracted.

    Supported formats:

    * **.pdf** - text extracted via PyMuPDF (``fitz``).
    * **.doc** - text extracted via PyMuPDF (``fitz``).
    * **.ppt** - text extracted via PyMuPDF (``fitz``).

    Returns a single-element list containing the concatenated text of
    all pages/slides, or an empty list if no text could be extracted.
    """

    display_name = "Document \u2192 Text"
    description = "Extract embedded text from documents"
    summary_template = "Extract the embedded text of each document into a single text entry."

    @property
    def source_type(self) -> str:
        return "document"

    @property
    def target_type(self) -> str:
        return "text"

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filename = media.get("filename", "document")
        stem = Path(filename).stem

        media_bytes = resolve_media_bytes(media)
        if not media_bytes:
            return []

        try:
            import fitz  # noqa: PLC0415 - PyMuPDF
        except ImportError:
            print(agpl_unavailable_message("PyMuPDF", "Extracting text from documents"))
            return []

        try:
            doc = fitz.open(stream=media_bytes, filetype=Path(filename).suffix.lstrip(".") or "pdf")
        except Exception as e:
            print(f"Error opening document {filename}: {e}")
            return []

        page_texts: list[str] = []
        try:
            for page in doc:
                # get_text("text") returns str at runtime; the PyMuPDF
                # stub widens to str|list|dict to cover other modes.
                text = cast(str, page.get_text("text")).strip()
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
