"""A label a converter made serves its own bytes, not the file it was converted from (#5680).

The origin of page 2 of a PDF resolves to the PDF.  The label previews and the
positives browse used to read that file and hand it over as the label: a
document where the detector's media are images, so no thumbnail could paint.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vtscore.datasets.importers.server_folder import IMPORTER
from vtscore.datasets.stages.clipper import convert_unembeddable_medias
from vtscore.detectors.positives_browse import _materialize_positive
from vtscore.detectors.resolver import resolve_unit_payload

fitz = pytest.importorskip("fitz")

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "report.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 100), f"page {i + 1}", fontsize=20)
    doc.save(str(path))
    doc.close()
    return path


def _folder_pages(pdf) -> list[dict]:
    """The pages a folder import makes of *pdf*: flat ``converter`` origins."""
    medias: dict = {}
    IMPORTER.run({"path": str(pdf.parent), "media_type": "image"}, medias)
    return list(medias.values())


def _chain_pages(pdf) -> list[dict]:
    """The pages the load pipeline makes of a raw document: ``clipper_chain`` origins."""
    medias: dict = {}
    IMPORTER.run({"path": str(pdf.parent), "media_type": "document"}, medias)
    for media in medias.values():
        media["origin"] = {"importer": "server_folder", "params": {"path": str(pdf.parent), "media_type": "document"}}
    convert_unembeddable_medias(medias)
    return list(medias.values())


@pytest.mark.parametrize("make_pages", [_folder_pages, _chain_pages], ids=["flat-converter", "chain"])
def test_a_converted_page_serves_that_page(pdf, make_pages):
    pages = make_pages(pdf)
    assert len(pages) == 3

    served = [resolve_unit_payload(pdf, page["origin"]) for page in pages]

    assert [payload[1] for payload in served] == [".png"] * 3
    assert [payload[0] for payload in served] == [page["media_bytes"] for page in pages]
    assert len({payload[0] for payload in served}) == 3, "each origin picks its own page back out"


def test_a_plain_origin_serves_the_file(pdf):
    origin = {"importer": "server_folder", "params": {"path": str(pdf.parent)}}

    assert resolve_unit_payload(pdf, origin) == (pdf.read_bytes(), ".pdf")
    assert resolve_unit_payload(pdf, None) == (pdf.read_bytes(), ".pdf")


def test_a_recipe_that_no_longer_matches_serves_nothing(pdf):
    """Better no bytes than the PDF's: nothing can paint those as the label."""
    origin = _folder_pages(pdf)[0]["origin"]
    origin = {**origin, "params": {**origin["params"], "converter_content_hash": "0" * 12, "converter_n_out": "9"}}

    assert resolve_unit_payload(pdf, origin) is None


def test_positives_browse_previews_the_page(pdf):
    page = _folder_pages(pdf)[1]
    elem = SimpleNamespace(origin=page["origin"], origin_name=page["origin_name"], filename=page["filename"], md5="")

    media = _materialize_positive(elem, media_type="image", embedder_name="siglip", cached=np.ones(4, dtype=np.float32))

    assert media is not None
    assert media["media_bytes"] == page["media_bytes"]
    assert media["media_bytes"][:8] == _PNG_MAGIC
