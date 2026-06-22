"""PDF page rendering utility.

Converts each page of a PDF document into a PIL Image at a specified DPI.
Requires the ``pymupdf`` package (``pip install pymupdf``).

Also provides :func:`load_pdf_images_into`, the shared folder-scan helper that
expands every PDF in a directory into per-page image medias (used by the
``server_folder`` importer and the local-archive loader).
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


def render_pdf_pages(pdf_path: Path, dpi: int = 150) -> list[tuple[str, "Image.Image"]]:
    """Render each page of *pdf_path* as a PIL Image.

    Args:
        pdf_path: Path to a ``.pdf`` file.
        dpi: Resolution for rendering.  150 is a good balance of quality and
            performance for CLIP embedding.

    Returns:
        List of ``(page_name, pil_image)`` tuples.  ``page_name`` follows the
        pattern ``"filename.pdf-1"``, ``"filename.pdf-2"``, etc. (1-indexed).
    """
    import fitz  # noqa: PLC0415  (pymupdf)
    from PIL import Image as PILImage  # noqa: PLC0415

    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pages: list[tuple[str, PILImage.Image]] = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix)
            img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
            page_name = f"{pdf_path.name}-{page_num + 1}"
            pages.append((page_name, img))
    finally:
        doc.close()

    return pages


def load_pdf_images_into(
    folder: Path,
    medias: dict[int, dict[str, Any]],
    thin: bool = False,
    recursive: bool = True,
) -> None:
    """Expand all PDFs in *folder* into per-page image medias.

    Each page is rendered at 150 DPI and appended to *medias* with sequential
    IDs continuing from the current maximum.  Pages leave here with
    ``embedding=None``; the framework embed stage fills the vectors in.  The
    ``origin`` is set to ``{"importer": "pdf", "params": {"path": ...}}`` so
    provenance points back to the source document.
    """
    from vtscore.media import get_by_folder_name  # noqa: PLC0415
    from vtscore.security.path_validation import glob_top_level, rglob_follow_symlinks  # noqa: PLC0415

    pdf_files = sorted(rglob_follow_symlinks(folder, "*.pdf") if recursive else glob_top_level(folder, "*.pdf"))
    if not pdf_files:
        return

    mt = get_by_folder_name("image")
    media_id = max(medias.keys(), default=0) + 1

    for pdf_path in pdf_files:
        origin = {"importer": "pdf", "params": {"path": str(pdf_path)}}
        pages = render_pdf_pages(pdf_path)
        pdf_rel = pdf_path.relative_to(folder).as_posix()

        for page_name, pil_image in pages:
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            rel_dir = str(Path(pdf_rel).parent)
            if rel_dir and rel_dir != ".":
                full_page_name = f"{rel_dir}/{page_name}"
            else:
                full_page_name = page_name

            media_data: dict[str, Any] = {
                "id": media_id,
                "media_type": mt.type_id,
                "embedder": "",
                "file_size": len(image_bytes),
                "md5": hashlib.md5(image_bytes).hexdigest(),
                "embeddings": {},
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
