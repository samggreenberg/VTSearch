"""`_stamp_demo_origin` must not clobber a converted media's converter origin.

The demo pickle is written *after* the converter has replaced each source
media with its N converted outputs, so a cached converted demo already carries
the full converter recipe.  Re-stamping the flat demo origin over it discarded
which source file and which sub-output each media came from.
"""

from __future__ import annotations

from typing import Any

from vtscore.datasets.loader_demo import _stamp_demo_origin


def _converter_origin(out_index: int) -> dict[str, Any]:
    return {
        "importer": "converter",
        "params": {
            "converter": "document2image",
            "source_file": "doc.pdf",
            "parent_importer": "demo",
            "parent_name": "ucsf_documents",
            "converter_out_index": str(out_index),
            "converter_n_out": "3",
        },
    }


class TestStampDemoOrigin:
    def test_stamps_plain_medias(self):
        medias: dict[int, dict[str, Any]] = {
            1: {"id": 1},
            2: {"id": 2, "origin": {"importer": "demo", "params": {}}},
        }
        _stamp_demo_origin(medias, "esc50")
        for media in medias.values():
            assert media["origin"] == {"importer": "demo", "params": {"name": "esc50"}}

    def test_records_the_converter_name(self):
        medias: dict[int, dict[str, Any]] = {1: {"id": 1}}
        _stamp_demo_origin(medias, "ucsf_documents", converter_name="document2image")
        assert medias[1]["origin"]["params"] == {
            "name": "ucsf_documents",
            "converter": "document2image",
        }

    def test_each_media_gets_its_own_params_dict(self):
        medias: dict[int, dict[str, Any]] = {1: {"id": 1}, 2: {"id": 2}}
        _stamp_demo_origin(medias, "esc50")
        assert medias[1]["origin"]["params"] is not medias[2]["origin"]["params"]

    def test_preserves_converter_origins(self):
        medias: dict[int, dict[str, Any]] = {i + 1: {"id": i + 1, "origin": _converter_origin(i)} for i in range(3)}
        _stamp_demo_origin(medias, "ucsf_documents", converter_name="document2image")
        for i, mid in enumerate(sorted(medias)):
            params = medias[mid]["origin"]["params"]
            assert medias[mid]["origin"]["importer"] == "converter"
            assert params["source_file"] == "doc.pdf"
            assert params["converter_out_index"] == str(i)

    def test_still_stamps_unconverted_medias_alongside_converted_ones(self):
        medias: dict[int, dict[str, Any]] = {
            1: {"id": 1, "origin": _converter_origin(0)},
            2: {"id": 2},
        }
        _stamp_demo_origin(medias, "ucsf_documents", converter_name="document2image")
        assert medias[1]["origin"]["importer"] == "converter"
        assert medias[2]["origin"]["importer"] == "demo"
