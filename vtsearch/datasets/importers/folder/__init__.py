"""Local-folder importer – scans a directory of media files and embeds them.

When the media type is ``"images"``, PDF files (``*.pdf``) in the folder are
also included: each page is rendered as a separate image and embedded with
CLIP.  The origin for PDF-derived images is ``"pdf"`` (not ``"folder"``) so
that provenance tracks back to the original document.

Converter support
-----------------
When the ``converters`` field value is set (a comma-separated list of
converter names, e.g. ``"video2image,document2image"``), the importer also
scans for source files matching each converter's input type, converts them
to the target media type, embeds the results, and appends them to the
dataset with converter-specific origins.
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
    embedder_name: str = "",
) -> None:
    """Expand all PDFs in *folder* into per-page image medias.

    Each page is rendered at 150 DPI, embedded with the image embedder, and
    appended to *medias* with sequential IDs continuing from the current
    maximum.  The ``origin`` is set to ``{"importer": "pdf", "params":
    {"path": ...}}`` so the provenance points back to the source document.
    """
    pdf_files = sorted(folder.rglob("*.pdf"))
    if not pdf_files:
        return

    from vtsearch.datasets.pdf import render_pdf_pages  # noqa: PLC0415
    from vtsearch.media import embedders_for_type, get_by_folder_name, get_embedder  # noqa: PLC0415

    mt = get_by_folder_name("images")

    # Resolve the embedder
    emb = None
    if embedder_name:
        try:
            emb = get_embedder(embedder_name)
        except KeyError:
            pass
    if emb is None:
        avail = embedders_for_type(mt.type_id)
        if avail:
            emb = avail[0]
    if emb is None:
        return

    if getattr(emb, "_model", None) is None:
        emb.load_models()

    embedder_id = emb.name
    media_id = max(medias.keys(), default=0) + 1

    for pdf_path in pdf_files:
        origin = {"importer": "pdf", "params": {"path": str(pdf_path)}}
        pages = render_pdf_pages(pdf_path)
        # Relative path prefix so that PDFs in different subdirectories
        # produce distinct page names.
        pdf_rel = pdf_path.relative_to(folder).as_posix()

        for page_name, pil_image in pages:
            embedding = emb.embed_pil_image(pil_image)
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
                "embedder": embedder_id,
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


def _run_selected_converters(
    folder: Path,
    media_type: str,
    field_values: dict,
    medias: dict,
    thin: bool = False,
) -> None:
    """Run any user-selected converters from *field_values*."""
    converters_str = field_values.get("converters", "")
    if not converters_str:
        return
    converter_names = [c.strip() for c in converters_str.split(",") if c.strip()]
    if not converter_names:
        return

    from vtsearch.converters.runner import run_converters_on_folder  # noqa: PLC0415

    base_origin = {"importer": "folder", "params": {"path": str(folder), "media_type": media_type}}
    run_converters_on_folder(
        folder_path=folder,
        converter_names=converter_names,
        target_media_type=media_type,
        medias=medias,
        thin=thin,
        base_origin=base_origin,
    )


class FolderDatasetImporter(DatasetImporter):
    """Embed all media files found in a local directory into a dataset.

    The user supplies an absolute filesystem path and selects the media type
    so that the correct file extensions are matched during the scan.

    When the media type is ``"images"``, any ``*.pdf`` files in the folder
    are also processed: each page is rendered as a separate image.

    When converters are selected (via the ``converters`` field value), files
    matching each converter's source type are also scanned, converted, and
    added to the dataset.
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
        emb_name = field_values.get("embedder", "")
        has_regular = True
        try:
            load_dataset_from_folder(folder, media_type, medias, thin=thin, embedder_name=emb_name)
        except ValueError:
            # No regular image files found — PDFs or converters may still produce output.
            if media_type != "images" and not field_values.get("converters"):
                raise
            has_regular = False
        if media_type == "images":
            _load_pdf_images(folder, medias, thin=thin, embedder_name=emb_name)

        # Run any user-selected converters.
        _run_selected_converters(folder, media_type, field_values, medias, thin=thin)

        if not has_regular and not medias:
            raise ValueError(f"No {media_type} files found in folder")

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
        emb_name = field_values.get("embedder", "")
        try:
            yield from load_dataset_from_folder_chunked(
                folder, media_type, chunk_size, thin=thin, embedder_name=emb_name,
            )
        except ValueError:
            if media_type != "images" and not field_values.get("converters"):
                raise
        if media_type == "images":
            chunk: dict[int, dict[str, Any]] = {}
            _load_pdf_images(folder, chunk, thin=thin)
            if chunk:
                yield chunk
        # Run converters and yield as a single chunk.
        converters_str = field_values.get("converters", "")
        if converters_str:
            converter_chunk: dict[int, dict[str, Any]] = {}
            _run_selected_converters(folder, media_type, field_values, converter_chunk, thin=thin)
            if converter_chunk:
                yield converter_chunk

    def run_chunked_cli(
        self, field_values: dict[str, Any], chunk_size: int, thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        folder = Path(field_values["path"])
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
        yield from self.run_chunked(field_values, chunk_size, thin=thin)

    def build_cli_args(self, field_values: dict[str, Any]) -> str:
        base = super().build_cli_args(field_values)
        converters = field_values.get("converters", "")
        if converters:
            base += f" --converters {converters}"
        return base

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        origin = super().build_origin(field_values)
        converters = field_values.get("converters", "")
        if converters:
            origin["params"]["converters"] = converters
        return origin


IMPORTER = FolderDatasetImporter()
