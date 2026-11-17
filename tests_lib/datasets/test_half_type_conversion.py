"""Media of a type nothing embeds (documents) are converted before the embed stage (#5680).

A *convert-out* half type has files to ingest and no embedder of its own.  An
importer that hands such media over as they are - a folder scan asked for
documents, a plugin listing PDFs - used to have every one of them dropped for
want of a vector.  These pin the default conversion that now runs first.
"""

from __future__ import annotations

import json

import pytest

from vtscore.converters import default_converter_for
from vtscore.datasets.importers.base import SourceSpec, with_default_conversions
from vtscore.datasets.stages.clipper import convert_unembeddable_medias, default_conversion_step

fitz = pytest.importorskip("fitz")

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _pdf_bytes(pages: int, label: str = "doc") -> bytes:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 100), f"{label} page {i + 1}", fontsize=20)
    data = doc.tobytes()
    doc.close()
    return data


def _document(mid: int, name: str, pages: int) -> dict:
    data = _pdf_bytes(pages, name)
    return {
        "id": mid,
        "media_type": "document",
        "filename": name,
        "origin_name": name,
        "origin": {"importer": "my_plugin", "params": {"bucket": "b"}},
        "category": "reports",
        "media_bytes": data,
        "media_string": None,
        "embeddings": {},
        "embedder": "",
        "md5": "0" * 32,
        "file_size": len(data),
        "duration": 0,
    }


class TestDefaultConverterFor:
    def test_document_becomes_page_images_by_default(self):
        converter = default_converter_for("document")
        assert converter is not None and converter.name == "document2image"

    def test_a_target_the_type_converts_to_wins(self):
        converter = default_converter_for("document", "text")
        assert converter is not None and converter.name == "document2text"

    def test_a_target_it_cannot_become_falls_back_to_the_default(self):
        converter = default_converter_for("document", "audio")
        assert converter is not None and converter.name == "document2image"

    @pytest.mark.parametrize("media_type", ["image", "audio", "video", "text", "face", "no-such-type"])
    def test_types_that_embed_have_none(self, media_type):
        assert default_converter_for(media_type) is None
        assert default_conversion_step(media_type) is None


class TestWithDefaultConversions:
    _DIRECT = [SourceSpec(source_type="image")]

    def test_an_image_scan_with_no_named_rows_gains_the_document_row(self):
        specs = with_default_conversions({"media_type": "image"}, self._DIRECT, "image")
        assert [(s.source_type, s.converter) for s in specs] == [("image", None), ("document", "document2image")]

    def test_named_rows_are_left_as_they_are(self):
        field_values = {"media_type": "image", "source_specs": [{"source_type": "image", "converter": None}]}
        assert with_default_conversions(field_values, self._DIRECT, "image") == self._DIRECT

    def test_only_the_default_target_gains_a_row(self):
        """Documents become text only when asked: a text scan does not extract every PDF by default."""
        direct = [SourceSpec(source_type="text")]
        assert with_default_conversions({"media_type": "text"}, direct, "text") == direct


class TestConvertUnembeddableMedias:
    def test_documents_become_their_pages(self):
        medias = {1: _document(1, "a.pdf", 2), 2: _document(2, "b.pdf", 1)}

        produced = convert_unembeddable_medias(medias)

        assert produced == {1, 2, 3}
        assert [m["media_type"] for m in medias.values()] == ["image"] * 3
        assert all(m["media_bytes"][:8] == _PNG_MAGIC for m in medias.values())
        assert all(m["id"] == mid for mid, m in medias.items())

    def test_each_page_keeps_its_source_and_records_the_recipe(self):
        medias = {1: _document(1, "a.pdf", 2)}

        convert_unembeddable_medias(medias)

        for index, page in enumerate(medias.values()):
            assert page["origin"]["importer"] == "my_plugin"
            assert page["origin"]["params"]["bucket"] == "b"
            (step,) = json.loads(page["origin"]["params"]["clipper_chain"])
            assert (step["kind"], step["name"], step["out_index"], step["n_out"]) == (
                "converter",
                "document2image",
                index,
                2,
            )
            assert page["category"] == "reports"

    def test_pages_are_stored_like_any_media(self):
        """The bookkeeping a converter does not return is filled in, or the dataset cannot be saved."""
        medias = {1: _document(1, "a.pdf", 1)}

        convert_unembeddable_medias(medias)

        (page,) = medias.values()
        assert page["file_size"] == len(page["media_bytes"])
        assert len(page["md5"]) == 32 and page["md5"] != "0" * 32
        assert page["embeddings"] == {}, "left for the embed stage, with the embedders the load asked for"

    def test_a_text_dataset_gets_the_text(self):
        medias = {1: _document(1, "a.pdf", 2)}

        convert_unembeddable_medias(medias, "text")

        (unit,) = medias.values()
        assert unit["media_type"] == "text"
        assert "a.pdf page 1" in unit["media_string"]

    def test_media_that_embed_are_left_alone(self):
        image = {"id": 1, "media_type": "image", "filename": "p.png", "media_bytes": b"x", "origin": None}
        medias = {1: image, 2: _document(2, "a.pdf", 2)}

        produced = convert_unembeddable_medias(medias)

        assert medias[1] is image
        assert produced == {2, 3}
        assert [m["media_type"] for m in medias.values()] == ["image"] * 3

    def test_nothing_to_convert_is_a_no_op(self):
        medias = {1: {"id": 1, "media_type": "image", "filename": "p.png", "media_bytes": b"x"}}
        before = dict(medias)

        assert convert_unembeddable_medias(medias) == set()
        assert medias == before

    def test_a_chain_with_its_own_converter_is_the_callers_choice(self):
        medias = {1: _document(1, "a.pdf", 2)}
        chain = [{"kind": "converter", "name": "document2text", "params": {}}]

        assert convert_unembeddable_medias(medias, chain_steps=chain) == set()
        assert medias[1]["media_type"] == "document"
