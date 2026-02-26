"""PDF page rendering utility.

Converts each page of a PDF document into a PIL Image at a specified DPI.
Requires the ``pymupdf`` package (``pip install pymupdf``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_name = f"{pdf_path.name}-{page_num + 1}"
            pages.append((page_name, img))
    finally:
        doc.close()

    return pages
