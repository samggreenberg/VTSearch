"""Library-tier tests for ``source_box`` surviving the pickle and reaching the metadata grid.

A converter output may record where it sat in its source as a top-level
``source_box`` (``provenance.SOURCE_BOX_FIELD``): a face crop's normalised box
in its photo (#4748).  ``export_dataset_to_file`` writes an explicit field list,
so the box has to be named there and restored on load, or a saved face dataset
reopens with no way to outline any face on its photo.  Nothing re-derives it:
lazy replay reproduces a crop's bytes, not this metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_pickle import load_dataset_from_pickle
from vtscore.media.face.media_type import FaceMediaType
from vtscore.media.provenance import (
    DERIVED_VIA_LABEL,
    SOURCE_BOX_FIELD,
    SOURCE_BOX_LABEL,
    SOURCE_LABEL,
    describe_source_box,
    provenance_metadata,
)

BOX = [0.125, 0.25, 0.5, 0.75]


def _face(**extra) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    return {
        "id": 1,
        "media_type": "face",
        "duration": 0,
        "file_size": 64,
        "md5": "abc",
        "embedder": "face",
        "embeddings": {"face": rng.standard_normal(16).astype(np.float32)},
        "filename": "group.png→group_face_1.png",
        "category": "test",
        "media_bytes": b"\x89PNG" + b"\x00" * 60,
        "media_string": None,
        "media_path": "/data/photos/group.png",
        "width": 40,
        "height": 40,
        "origin": {
            "importer": "converter",
            "params": {
                "converter": "image2face",
                "source_file": "group.png",
                "source_path": "/data/photos/group.png",
                "parent_importer": "server_folder",
                "parent_path": "/data/photos",
            },
        },
        "origin_name": "group.png→group_face_1.png",
        **extra,
    }


def _round_trip(media: dict[str, Any], tmp_path: Path, thin: bool = False) -> dict[str, Any]:
    container = export_dataset_to_file({1: media}, embedder="face", media_type="face")
    pkl = tmp_path / "faces.pkl"
    pkl.write_bytes(container)
    loaded: dict[int, dict[str, Any]] = {}
    load_dataset_from_pickle(pkl, loaded, thin=thin)
    return loaded[1]


class TestSourceBoxSurvivesPickleRoundTrip:
    def test_box_persists(self, tmp_path: Path):
        out = _round_trip(_face(source_box=BOX), tmp_path)

        assert out[SOURCE_BOX_FIELD] == BOX

    def test_box_persists_through_a_thin_load(self, tmp_path: Path):
        out = _round_trip(_face(source_box=BOX), tmp_path, thin=True)

        assert out[SOURCE_BOX_FIELD] == BOX

    def test_a_tuple_box_comes_back_a_list(self, tmp_path: Path):
        out = _round_trip(_face(source_box=tuple(BOX)), tmp_path)

        assert out[SOURCE_BOX_FIELD] == BOX

    def test_media_without_a_box_gains_none(self, tmp_path: Path):
        out = _round_trip(_face(), tmp_path)

        assert SOURCE_BOX_FIELD not in out

    def test_the_reloaded_face_shows_the_source_box_row(self, tmp_path: Path):
        out = _round_trip(_face(source_box=BOX), tmp_path)

        assert FaceMediaType().display_metadata(out)[SOURCE_BOX_LABEL] == "0.125,0.250,0.500,0.750"


class TestSourceBoxRow:
    def test_the_row_sits_between_source_and_derived_via(self):
        keys = list(provenance_metadata(_face(source_box=BOX)))

        assert keys.index(SOURCE_LABEL) + 1 == keys.index(SOURCE_BOX_LABEL)
        assert keys.index(SOURCE_BOX_LABEL) + 1 == keys.index(DERIVED_VIA_LABEL)

    def test_no_box_no_row(self):
        assert SOURCE_BOX_LABEL not in provenance_metadata(_face())

    def test_a_malformed_box_renders_nothing(self):
        for bad in ([0.1, 0.2, 0.3], "0.1,0.2,0.3,0.4", [0.1, "x", 0.3, 0.4], None):
            assert describe_source_box({SOURCE_BOX_FIELD: bad}) == ""
