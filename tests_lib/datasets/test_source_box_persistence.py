"""Library-tier tests for ``source_box`` and ``face_landmarks`` surviving the pickle.

A converter output may record where it sat in its source as a top-level
``source_box`` (``provenance.SOURCE_BOX_FIELD``): a face crop's normalised box
in its photo (#4748).  A face crop may also keep its five landmarks on the crop
(``provenance.FACE_LANDMARKS_FIELD``, #5644).  ``export_dataset_to_file`` writes
an explicit field list, so each has to be named there and restored on load, or
a saved face dataset reopens with no way to outline any face on its photo, or
to match two faces.  Nothing re-derives either: lazy replay reproduces a crop's
bytes, not this metadata.  The box also renders as a metadata-grid row.
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
    FACE_LANDMARKS_FIELD,
    SOURCE_BOX_FIELD,
    SOURCE_BOX_LABEL,
    SOURCE_LABEL,
    describe_source_box,
    face_landmarks_of,
    provenance_metadata,
)

BOX = [0.125, 0.25, 0.5, 0.75]
POINTS = [[0.3438, 0.4616], [0.6564, 0.4599], [0.5002, 0.6405], [0.3709, 0.8247], [0.6315, 0.8233]]


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


class TestFaceLandmarksSurvivePickleRoundTrip:
    def test_landmarks_persist(self, tmp_path: Path):
        out = _round_trip(_face(source_box=BOX, face_landmarks=POINTS), tmp_path)

        assert out[FACE_LANDMARKS_FIELD] == POINTS
        assert out[SOURCE_BOX_FIELD] == BOX

    def test_landmarks_persist_through_a_thin_load(self, tmp_path: Path):
        out = _round_trip(_face(face_landmarks=POINTS), tmp_path, thin=True)

        assert out[FACE_LANDMARKS_FIELD] == POINTS

    def test_tuple_points_come_back_lists(self, tmp_path: Path):
        out = _round_trip(_face(face_landmarks=tuple(tuple(p) for p in POINTS)), tmp_path)

        assert out[FACE_LANDMARKS_FIELD] == POINTS

    def test_a_face_without_landmarks_reloads_without_them(self, tmp_path: Path):
        """A Face dataset saved before #5644 keeps working, and simply lacks the field."""
        out = _round_trip(_face(source_box=BOX), tmp_path)

        assert FACE_LANDMARKS_FIELD not in out

    def test_malformed_landmarks_are_not_saved(self, tmp_path: Path):
        out = _round_trip(_face(face_landmarks=POINTS[:4]), tmp_path)

        assert FACE_LANDMARKS_FIELD not in out

    def test_landmarks_add_no_metadata_row(self):
        """They are geometry for the compare view, not a number for anyone to read."""
        row = FaceMediaType().display_metadata(_face(face_landmarks=POINTS))

        assert not [k for k in row if "andmark" in k]


class TestFaceLandmarksOf:
    def test_well_formed_points_come_back_as_float_lists(self):
        got = face_landmarks_of({FACE_LANDMARKS_FIELD: [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]})

        assert got == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        assert all(isinstance(v, float) for p in got for v in p)

    def test_points_outside_the_crop_are_kept(self):
        """A face running past the crop's edge has a landmark outside [0, 1]; it is not clamped."""
        points = [[-0.1, 0.4], *POINTS[1:4], [1.2, 0.8]]

        assert face_landmarks_of({FACE_LANDMARKS_FIELD: points}) == points

    def test_the_result_is_a_fresh_list(self):
        media = {FACE_LANDMARKS_FIELD: [list(p) for p in POINTS]}
        face_landmarks_of(media)[0][0] = 99.0

        assert media[FACE_LANDMARKS_FIELD][0][0] == POINTS[0][0]

    def test_malformed_values_read_as_none(self):
        for bad in (
            None,
            "0.1,0.2",
            POINTS[:4],
            [*POINTS[:4], [0.1]],
            [*POINTS[:4], [0.1, "x"]],
            [*POINTS[:4], [0.1, float("nan")]],
            [*POINTS[:4], [float("inf"), 0.2]],
            [*POINTS[:4], 0.5],
        ):
            assert face_landmarks_of({FACE_LANDMARKS_FIELD: bad}) is None, bad
        assert face_landmarks_of({}) is None
