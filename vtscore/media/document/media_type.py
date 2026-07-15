"""Document media type - PDF/DOC/PPT files (no embedder)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from vtscore.config import DATA_DIR
from vtscore.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
    demo_slice,
)

#: Industries queried against the UCSF Industry Documents Library for the
#: ``ucsf_documents`` demo.  Owned here (not shared with the old image entry)
#: because the demo now lives on the Document type.
UCSF_DOCUMENTS_CATEGORIES = ["Tobacco", "Food", "Drug", "Chemical", "Fossil Fuel", "Opioids"]

_DOCUMENT_MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".ppt": "application/vnd.ms-powerpoint",
}


class DocumentMediaType(MediaType):
    """Handles document files (PDF, DOC, PPT).

    This media type does **not** have an embedder - :meth:`embed_media` and
    :meth:`embed_text` always return ``None``.  Documents are intended to
    be converted to other media types (images, text, audio) via
    :class:`~vtscore.converters.base.MediaConverter` subclasses before
    embedding.

    Documents are served as their original binary format with the
    appropriate MIME type.
    """

    #: Documents render a first-page thumbnail: a browsable-thumbnail type.
    has_thumbnail = True

    #: A document is a *convert-out* half type: importable but not embeddable.
    #: It must be turned into an image (rasterised pages) or text (extracted
    #: body) before it can be embedded.  Image is the default (first entry).
    converts_to = ["image", "text"]

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
        from vtscore.datasets.downloader.core import UCSF_IDL_DOWNLOAD_SIZE_MB  # noqa: PLC0415

        return [
            DemoDataset(
                id="ucsf_documents_a",
                label="UCSF Documents (A)",
                description="Scanned industry documents (PDF) — convert to page images on load",
                categories=UCSF_DOCUMENTS_CATEGORIES,
                source="ucsf_documents",
                required_folder=DATA_DIR / "ucsf_documents",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=25,
                download_size_mb=UCSF_IDL_DOWNLOAD_SIZE_MB,
            ),
        ]

    def load_demo_source(
        self,
        source,
        categories,
        slice_start,
        slice_end,
        clips,
        on_progress=None,
        embedder=None,  # noqa: ARG002 - documents have no embedder; the converter's target does
        slice_frac_start=None,
        slice_frac_end=None,
        **kwargs,
    ):
        """Populate *clips* with raw-PDF **document** medias for a demo source.

        The only source today is ``ucsf_documents``: short UCSF Industry
        Documents Library PDFs, one document media per PDF.  These are *not*
        embedded here (documents are non-embeddable) — the demo loader applies
        the ``document2image`` converter (see :attr:`converts_to`) on load,
        and the resulting image pages are what get embedded.
        """
        if source != "ucsf_documents":
            raise ValueError(f"Media type 'document' does not support demo source {source!r}")

        from vtscore.datasets.downloader import download_ucsf_documents  # noqa: PLC0415

        if on_progress is None:
            from vtscore.media.base import _noop_progress as _np  # noqa: PLC0415

            on_progress = _np

        docs_dir = download_ucsf_documents(categories, on_progress=on_progress)

        selected: list[tuple[Path, str]] = []
        for cat in categories:
            cat_dir = docs_dir / cat
            if not cat_dir.is_dir():
                continue
            pdfs = sorted(cat_dir.glob("*.pdf"))
            for pdf_path in demo_slice(pdfs, slice_start, slice_end, slice_frac_start, slice_frac_end):
                selected.append((pdf_path, cat))

        clip_id = max(clips.keys(), default=0) + 1
        total = len(selected)
        on_progress("loading", f"Loading {total} documents...", 0, total)
        for i, (pdf_path, cat) in enumerate(selected):
            on_progress("loading", f"Loading {pdf_path.name}", i + 1, total)
            try:
                pdf_bytes = pdf_path.read_bytes()
            except OSError:
                continue
            rel_name = f"{cat}/{pdf_path.name}"
            clips[clip_id] = {
                "id": clip_id,
                "media_type": self.type_id,
                "embedder": "",
                "duration": 0,
                "file_size": len(pdf_bytes),
                "md5": hashlib.md5(pdf_bytes).hexdigest(),
                "embeddings": {},
                "media_bytes": pdf_bytes,
                "media_string": None,
                "filename": rel_name,
                "category": cat,
                "origin_name": rel_name,
            }
            clip_id += 1
        return None

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

    def load_media_data(self, file_path: Path, media_bytes: bytes | None = None) -> dict:
        if media_bytes is None:
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

    def image_response(self, media: dict) -> MediaResponse | None:
        """Return the document's **first page** rasterised to PNG, or *None*.

        A document has no natively-visual bytes the grid / VTSBrowse bin-popup
        can paint (the raw ``application/pdf`` stream is not an image), so the
        media route delegates to this hook the same way it does for audio
        waveforms and video mid-frames.  Only PDFs rasterise (via
        :func:`~vtscore.datasets.pdf.render_pdf_page_png`); ``.doc`` / ``.ppt``
        return ``None`` and fall back to a placeholder.  The rendered page is
        memoised on the media in-memory (never persisted) so repeated fetches
        of the same item skip the re-render.
        """
        thumb = media.get("thumbnail_bytes")
        if not thumb:
            filename = media.get("filename", "")
            ext = Path(filename).suffix.lower() if filename else ".pdf"
            if ext not in ("", ".pdf"):
                return None
            data = self._resolve_media_bytes(media)
            if not data:
                return None
            from vtscore.datasets.pdf import render_pdf_page_png  # noqa: PLC0415

            thumb = render_pdf_page_png(data, page_index=0)
            if thumb:
                media["thumbnail_bytes"] = thumb
        if not thumb:
            return None
        return MediaResponse(
            data=thumb,
            mimetype="image/png",
            download_name=f"media_{media['id']}_page1.png",
        )
