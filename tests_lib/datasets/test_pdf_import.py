"""Tests for PDF-to-image import support.

These tests verify:
- render_pdf_pages converts PDF pages into PIL Images with correct naming
- Folder importer renders PDFs through ``document2image`` when media_type is
  "image", once per page, with the converter's recipe as each page's origin
- PDF-only folders work (no regular image files present)
- PDFs are ignored when media_type is not "image"
- ``load_pdf_images_into`` (the library helper) still follows symlinks
"""

from __future__ import annotations

import unittest.mock as mock

import numpy as np


def _create_test_pdf(path, num_pages=2):
    """Create a minimal PDF at *path* with *num_pages* coloured pages."""
    import fitz

    doc = fitz.open()
    colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    for i in range(num_pages):
        page = doc.new_page(width=100, height=100)
        colour = colours[i % len(colours)]
        page.draw_rect(fitz.Rect(0, 0, 100, 100), color=colour, fill=colour)
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# render_pdf_pages
# ---------------------------------------------------------------------------


class TestRenderPdfPages:
    def test_returns_correct_number_of_pages(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        _create_test_pdf(pdf, num_pages=3)

        from vtscore.datasets.pdf import render_pdf_pages

        pages = render_pdf_pages(pdf)
        assert len(pages) == 3

    def test_page_names_follow_pattern(self, tmp_path):
        pdf = tmp_path / "mydoc.pdf"
        _create_test_pdf(pdf, num_pages=2)

        from vtscore.datasets.pdf import render_pdf_pages

        pages = render_pdf_pages(pdf)
        assert pages[0][0] == "mydoc.pdf-1"
        assert pages[1][0] == "mydoc.pdf-2"

    def test_returns_pil_images(self, tmp_path):
        from PIL import Image

        pdf = tmp_path / "test.pdf"
        _create_test_pdf(pdf, num_pages=1)

        from vtscore.datasets.pdf import render_pdf_pages

        pages = render_pdf_pages(pdf)
        assert isinstance(pages[0][1], Image.Image)

    def test_images_are_rgb(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        _create_test_pdf(pdf, num_pages=1)

        from vtscore.datasets.pdf import render_pdf_pages

        pages = render_pdf_pages(pdf)
        assert pages[0][1].mode == "RGB"

    def test_custom_dpi(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        _create_test_pdf(pdf, num_pages=1)

        from vtscore.datasets.pdf import render_pdf_pages

        pages_low = render_pdf_pages(pdf, dpi=72)
        pages_high = render_pdf_pages(pdf, dpi=300)
        # Higher DPI should produce larger images
        assert pages_high[0][1].width > pages_low[0][1].width
        assert pages_high[0][1].height > pages_low[0][1].height

    def test_single_page_pdf(self, tmp_path):
        pdf = tmp_path / "single.pdf"
        _create_test_pdf(pdf, num_pages=1)

        from vtscore.datasets.pdf import render_pdf_pages

        pages = render_pdf_pages(pdf)
        assert len(pages) == 1
        assert pages[0][0] == "single.pdf-1"


# ---------------------------------------------------------------------------
# Folder importer – PDF integration
# ---------------------------------------------------------------------------


def _write_photo(path) -> None:
    """A real, tiny PNG at *path* (the folder loader decodes what it scans)."""
    from PIL import Image

    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)


def _run_folder(tmp_path, *, thin: bool = False, **field_values) -> dict:
    from vtscore.datasets.importers.server_folder import IMPORTER

    medias: dict = {}
    IMPORTER.run({"path": str(tmp_path), **field_values}, medias, thin=thin)
    return medias


def _pages(medias: dict) -> list[dict]:
    return [m for m in medias.values() if (m.get("origin") or {}).get("importer") == "converter"]


_DOCUMENT_ROW = [{"source_type": "document", "converter": "document2image", "params": {}}]


class TestFolderImporterPdf:
    """A folder's PDFs become page images through ``document2image``, and through nothing else."""

    def test_pdf_pages_added_to_medias(self, tmp_path):
        """An image import that names no source rows renders the folder's PDFs beside its photos."""
        _create_test_pdf(tmp_path / "doc.pdf", num_pages=2)
        _write_photo(tmp_path / "photo.png")

        medias = _run_folder(tmp_path, media_type="image")

        assert len(medias) == 3
        assert len(_pages(medias)) == 2

    def test_pdf_origin_is_the_converter_recipe(self, tmp_path):
        """Each page records which page of which file it is, so a label on it can be resolved."""
        pdf = tmp_path / "report.pdf"
        _create_test_pdf(pdf, num_pages=2)

        pages = _pages(_run_folder(tmp_path, media_type="image"))

        assert [p["origin"]["params"]["converter"] for p in pages] == ["document2image"] * 2
        assert [p["origin"]["params"]["converter_out_index"] for p in pages] == ["0", "1"]
        assert all(p["origin"]["params"]["source_path"] == str(pdf.resolve()) for p in pages)
        assert all(p["origin"]["params"]["parent_importer"] == "server_folder" for p in pages)

    def test_pdf_page_filenames(self, tmp_path):
        _create_test_pdf(tmp_path / "slides.pdf", num_pages=3)

        names = sorted(p["filename"] for p in _pages(_run_folder(tmp_path, media_type="image")))

        assert names == [f"slides.pdf\u2192slides_page_{n}.png" for n in (1, 2, 3)]

    def test_pdf_only_folder(self, tmp_path):
        """A folder with only PDFs (no regular images) should still work."""
        _create_test_pdf(tmp_path / "only.pdf", num_pages=2)

        assert len(_run_folder(tmp_path, media_type="image")) == 2

    def test_named_document_row_gives_each_page_once(self, tmp_path):
        """#5680: the row the Add Dataset dialog sends for a PDF folder used to double every page.

        The importer rendered the PDFs itself *and* ran the row's converter over them.
        """
        _create_test_pdf(tmp_path / "a.pdf", num_pages=2)
        _create_test_pdf(tmp_path / "b.pdf", num_pages=3)

        medias = _run_folder(tmp_path, media_type="image", source_specs=_DOCUMENT_ROW)

        assert len(medias) == 5
        seen = {
            (m["origin"]["params"]["source_file"], m["origin"]["params"]["converter_out_index"])
            for m in medias.values()
        }
        assert len(seen) == 5

    def test_named_rows_are_the_whole_scan(self, tmp_path):
        """A request that names only the image row asked for the photos, not the PDFs."""
        _create_test_pdf(tmp_path / "doc.pdf", num_pages=2)
        _write_photo(tmp_path / "photo.png")

        medias = _run_folder(
            tmp_path, media_type="image", source_specs=[{"source_type": "image", "converter": None, "params": {}}]
        )

        assert [m["filename"] for m in medias.values()] == ["photo.png"]

    def test_chunked_scan_matches_the_whole_one(self, tmp_path):
        from vtscore.datasets.importers.server_folder import IMPORTER

        _create_test_pdf(tmp_path / "doc.pdf", num_pages=2)
        _write_photo(tmp_path / "photo.png")

        chunks = list(IMPORTER.run_chunked({"path": str(tmp_path), "media_type": "image"}, 100))

        assert sum(len(c) for c in chunks) == 3
        assert sum(len(_pages(c)) for c in chunks) == 2

    def test_document_scan_hands_over_the_files(self, tmp_path):
        """Asked for documents, the scan lists the PDFs; the load pipeline converts them before embedding."""
        _create_test_pdf(tmp_path / "doc.pdf", num_pages=2)

        medias = _run_folder(tmp_path, media_type="document")

        assert [m["media_type"] for m in medias.values()] == ["document"]

    def test_pdf_ignored_for_non_image_types(self, tmp_path):
        """An audio import leaves the folder's PDFs alone: documents do not become audio by default."""
        import io
        import struct
        import wave

        _create_test_pdf(tmp_path / "doc.pdf", num_pages=1)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(struct.pack("<" + "h" * 100, *([0] * 100)))
        (tmp_path / "tone.wav").write_bytes(buf.getvalue())

        medias = _run_folder(tmp_path, media_type="audio")

        assert [m["media_type"] for m in medias.values()] == ["audio"]

    def test_pdf_thin_mode_is_a_reference_to_the_source(self, tmp_path):
        """Reference mode tags each page with the file it re-renders from; its bytes go after the embed stage."""
        pdf = tmp_path / "doc.pdf"
        _create_test_pdf(pdf, num_pages=1)

        (page,) = _pages(_run_folder(tmp_path, thin=True, media_type="image"))

        assert page["_lazy_source"] == str(pdf.resolve())
        assert page["media_path"] == str(pdf.resolve())

    def test_pdf_media_fields(self, tmp_path):
        _create_test_pdf(tmp_path / "doc.pdf", num_pages=1)

        (page,) = _pages(_run_folder(tmp_path, media_type="image"))

        assert page["media_type"] == "image"
        assert page["width"] > 0 and page["height"] > 0
        assert len(page["md5"]) == 32
        assert page["origin_name"] == page["filename"]
        assert page["media_bytes"][:8] == b"\x89PNG\r\n\x1a\n"


class TestPdfSymlinkDiscovery:
    """PDF scanning must follow symlinked directories."""

    def _make_fake_image_media_type(self):
        mt = mock.MagicMock()
        mt.type_id = "image"
        mt.folder_import_name = "image"
        mt.file_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]
        mt.load_media_data.return_value = {"media_bytes": b"fake", "duration": 0, "width": 100, "height": 100}
        return mt

    def _make_fake_embedder(self):
        emb = mock.MagicMock()
        emb.name = "siglip"
        emb.media_type_id = "image"
        emb._model = True
        emb.embed_media.return_value = np.zeros(768)
        emb.embed_pil_image.return_value = np.zeros(768)
        emb.embed_media_bulk.side_effect = lambda medias: [emb.embed_media(m) for m in medias]
        return emb

    def test_pdfs_in_symlinked_subdir_are_discovered(self, tmp_path):
        """PDFs inside a symlinked subdirectory should be found and rendered."""
        from contextlib import ExitStack

        from vtscore.datasets.pdf import load_pdf_images_into as _load_pdf_images

        root = tmp_path / "root"
        root.mkdir()

        external = tmp_path / "external"
        external.mkdir()
        _create_test_pdf(external / "linked_doc.pdf", num_pages=2)

        (root / "linked").symlink_to(external)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with ExitStack() as stack:
            stack.enter_context(mock.patch("vtscore.media.get_by_folder_name", return_value=mt))
            stack.enter_context(mock.patch("vtscore.media.get_embedder", return_value=emb))
            stack.enter_context(mock.patch("vtscore.media.embedders_for_type", return_value=[emb]))
            _load_pdf_images(root, medias)

        assert len(medias) == 2
        filenames = {m["filename"] for m in medias.values()}
        assert "linked/linked_doc.pdf-1" in filenames
        assert "linked/linked_doc.pdf-2" in filenames
