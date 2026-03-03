"""Local-folder importer – scans a directory of media files and embeds them.

When the media type is ``"images"``, PDF files (``*.pdf``) in the folder are
also included: each page is rendered as a separate image and embedded with
CLIP.  The origin for PDF-derived images is ``"pdf"`` (not ``"folder"``) so
that provenance tracks back to the original document.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Iterator

from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.datasets.loader import load_dataset_from_folder, load_dataset_from_folder_chunked


def _load_pdf_images(
    folder: Path,
    medias: dict[int, dict[str, Any]],
    thin: bool = False,
) -> None:
    """Expand all PDFs in *folder* into per-page image medias.

    Each page is rendered at 150 DPI, embedded with CLIP, and appended to
    *medias* with sequential IDs continuing from the current maximum.  The
    ``origin`` is set to ``{"importer": "pdf", "params": {"path": ...}}``
    so the provenance points back to the source document.
    """
    pdf_files = sorted(folder.rglob("*.pdf"))
    if not pdf_files:
        return

    from vtsearch.datasets.pdf import render_pdf_pages  # noqa: PLC0415
    from vtsearch.media import get_by_folder_name  # noqa: PLC0415

    mt = get_by_folder_name("images")
    if getattr(mt, "_model", None) is None:
        mt.load_models()

    media_id = max(medias.keys(), default=0) + 1

    for pdf_path in pdf_files:
        origin = {"importer": "pdf", "params": {"path": str(pdf_path)}}
        pages = render_pdf_pages(pdf_path)
        # Relative path prefix so that PDFs in different subdirectories
        # produce distinct page names.
        pdf_rel = pdf_path.relative_to(folder).as_posix()

        for page_name, pil_image in pages:
            embedding = mt.embed_pil_image(pil_image)
            if embedding is None:
                continue

            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            # page_name is e.g. "doc.pdf-1"; prefix with relative dir
            # so that identically-named PDFs in different folders stay
            # distinct (e.g. "subdir/doc.pdf-1").
            rel_dir = str(Path(pdf_rel).parent)
            if rel_dir and rel_dir != ".":
                full_page_name = f"{rel_dir}/{page_name}"
            else:
                full_page_name = page_name

            media_data: dict[str, Any] = {
                "id": media_id,
                "type": mt.type_id,
                "file_size": len(image_bytes),
                "md5": hashlib.md5(image_bytes).hexdigest(),
                "embedding": embedding,
                "filename": full_page_name,
                "category": "custom",
                "origin": origin,
                "origin_name": full_page_name,
                "media_bytes": None if thin else image_bytes,
                "media_string": None,
                "media_path": str(pdf_path.resolve()),
                "duration": 0,
                "width": pil_image.width,
                "height": pil_image.height,
            }
            medias[media_id] = media_data
            media_id += 1


class FolderDatasetImporter(DatasetImporter):
    """Embed all media files found in a local directory into a dataset.

    The user supplies an absolute filesystem path and selects the media type
    so that the correct file extensions are matched during the scan.

    When the media type is ``"images"``, any ``*.pdf`` files in the folder
    are also processed: each page is rendered as a separate image.
    """

    name = "folder"
    display_name = "Generate from Folder"
    description = "Import media files from a folder."
    icon = "\U0001f4c2"
    fields = [
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            description="Type of media files to scan for in the folder.",
            options=["sounds", "videos", "images", "paragraphs", "documents"],
            default="sounds",
        ),
        ImporterField(
            key="path",
            label="Folder",
            field_type="folder",
            description="Absolute path to the directory containing media files.",
        ),
    ]

    def run(self, field_values: dict, medias: dict, thin: bool = False) -> None:
        folder = Path(field_values["path"])
        media_type = field_values.get("media_type", "sounds")
        has_regular = True
        try:
            load_dataset_from_folder(folder, media_type, medias, thin=thin)
        except ValueError:
            # No regular image files found — PDFs may still be present.
            if media_type != "images":
                raise
            has_regular = False
        if media_type == "images":
            _load_pdf_images(folder, medias, thin=thin)
            if not has_regular and not medias:
                raise ValueError("No image files found in folder")

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        folder = Path(field_values["path"])
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
        self.run(field_values, medias, thin=thin)

    @property
    def supports_chunked(self) -> bool:
        return True

    def run_chunked(
        self, field_values: dict[str, Any], chunk_size: int, thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        folder = Path(field_values["path"])
        media_type = field_values.get("media_type", "sounds")
        try:
            yield from load_dataset_from_folder_chunked(folder, media_type, chunk_size, thin=thin)
        except ValueError:
            if media_type != "images":
                raise
        if media_type == "images":
            chunk: dict[int, dict[str, Any]] = {}
            _load_pdf_images(folder, chunk, thin=thin)
            if chunk:
                yield chunk

    def run_chunked_cli(
        self, field_values: dict[str, Any], chunk_size: int, thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        folder = Path(field_values["path"])
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
        yield from self.run_chunked(field_values, chunk_size, thin=thin)


IMPORTER = FolderDatasetImporter()
