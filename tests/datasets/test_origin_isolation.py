"""Regression tests for H8 — origin dict shared by reference across medias.

The previous implementations of ``_tag_origins`` (in
``vtscore/datasets/load_pipeline.py``), the legacy ingest path in
``vtscore/datasets/ingest.py``, ``ServerFilesDatasetImporter._rewrite_origins``,
and ``_build_folder_media_data`` (in ``vtscore/datasets/loader_folder.py``)
all stamped the same ``origin`` dict reference onto every media they
visited.  A later mutation of ``media["origin"]["params"]`` on any one
of them therefore silently propagated to every sibling — and the
aliasing survived pickle round-trips via backreferences.

These tests pin down the fixed behaviour: each media must own a fresh,
independent ``origin`` (including a fresh ``params``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import app as app_module  # noqa: F401  (sets up Flask test infra via conftest)
from vtscore.datasets.importers.server_files import ServerFilesDatasetImporter
from vtscore.datasets.load_pipeline import _tag_origins
from vtscore.datasets.loader_folder import _build_folder_media_data


def _assert_isolated(*medias: dict[str, Any]) -> None:
    """Every pair must hold distinct ``origin`` and ``params`` dicts."""
    for i, a in enumerate(medias):
        for b in medias[i + 1 :]:
            assert a["origin"] is not b["origin"], "origin dict is shared by reference"
            assert a["origin"]["params"] is not b["origin"]["params"], "origin.params is shared by reference"


class TestTagOriginsIsolation:
    def test_each_media_gets_distinct_origin_and_params(self):
        origin = {"importer": "server_folder", "params": {"path": "/data", "media_type": "audio"}}
        medias = {1: {"filename": "a.wav"}, 2: {"filename": "b.wav"}, 3: {"filename": "c.wav"}}
        _tag_origins(medias, origin)
        _assert_isolated(medias[1], medias[2], medias[3])

    def test_mutating_one_origin_params_does_not_leak_to_siblings(self):
        origin = {"importer": "server_folder", "params": {"path": "/data"}}
        medias = {1: {"filename": "a.wav"}, 2: {"filename": "b.wav"}}
        _tag_origins(medias, origin)

        medias[1]["origin"]["params"]["extra"] = "only-on-1"
        medias[1]["origin"]["importer"] = "mutated"

        assert "extra" not in medias[2]["origin"]["params"]
        assert medias[2]["origin"]["importer"] == "server_folder"

    def test_origin_values_are_preserved(self):
        origin = {"importer": "server_folder", "params": {"path": "/data", "media_type": "audio"}}
        medias = {1: {"filename": "a.wav"}}
        _tag_origins(medias, origin)
        assert medias[1]["origin"] == origin
        # ...but it must not be the same object.
        assert medias[1]["origin"] is not origin
        assert medias[1]["origin"]["params"] is not origin["params"]

    def test_origin_name_falls_back_to_filename(self):
        origin = {"importer": "x", "params": {}}
        medias = {1: {"filename": "a.wav"}, 2: {"filename": "b.wav", "origin_name": "kept"}}
        _tag_origins(medias, origin)
        assert medias[1]["origin_name"] == "a.wav"
        assert medias[2]["origin_name"] == "kept"

    def test_skips_medias_that_already_carry_an_origin(self):
        pre_existing = {"importer": "other", "params": {"k": "v"}}
        origin = {"importer": "new", "params": {"k": "w"}}
        medias = {1: {"filename": "a.wav", "origin": pre_existing}, 2: {"filename": "b.wav"}}
        _tag_origins(medias, origin)
        assert medias[1]["origin"] is pre_existing  # untouched
        assert medias[2]["origin"]["importer"] == "new"
        assert medias[2]["origin"] is not origin

    def test_isolation_survives_pickle_roundtrip(self):
        origin = {"importer": "server_folder", "params": {"path": "/data"}}
        medias = {1: {"filename": "a.wav"}, 2: {"filename": "b.wav"}}
        _tag_origins(medias, origin)

        revived = pickle.loads(pickle.dumps(medias))
        _assert_isolated(revived[1], revived[2])

        revived[1]["origin"]["params"]["extra"] = "only-on-1"
        assert "extra" not in revived[2]["origin"]["params"]


class TestBuildFolderMediaDataIsolation:
    def _build(self, media_id: int, rel_path: str, origin: dict[str, Any] | None) -> dict[str, Any]:
        return _build_folder_media_data(
            media_id=media_id,
            type_id="audio",
            embedder_id="test_embedder",
            embedding=None,
            md5="deadbeef",
            rel_path=rel_path,
            file_path=Path(rel_path),
            file_size=42,
            origin=origin,
        )

    def test_two_calls_share_no_origin_state(self):
        origin = {"importer": "server_folder", "params": {"path": "/data"}}
        a = self._build(1, "a.wav", origin)
        b = self._build(2, "b.wav", origin)
        _assert_isolated(a, b)

    def test_mutating_one_origin_does_not_leak(self):
        origin = {"importer": "server_folder", "params": {"path": "/data"}}
        a = self._build(1, "a.wav", origin)
        b = self._build(2, "b.wav", origin)
        a["origin"]["params"]["clipper"] = "audio_default"
        assert "clipper" not in b["origin"]["params"]
        # Caller-provided origin is also untouched.
        assert "clipper" not in origin["params"]

    def test_none_origin_passes_through(self):
        media = self._build(1, "a.wav", None)
        assert media["origin"] is None


class TestRewriteOriginsIsolation:
    """``ServerFilesDatasetImporter._rewrite_origins`` repoints each media's
    origin at the canonical importer dict — but every media must get its
    own copy, not a shared reference.
    """

    def test_each_media_gets_distinct_origin(self, tmp_path):
        origin = {"importer": "server_files", "params": {"path": "/list.txt"}}
        src_a = tmp_path / "a.wav"
        src_b = tmp_path / "b.wav"
        medias = {
            1: {"origin_name": "a.wav", "filename": "a.wav"},
            2: {"origin_name": "b.wav", "filename": "b.wav"},
        }
        name_to_source = {"a.wav": src_a, "b.wav": src_b}

        importer = ServerFilesDatasetImporter()
        importer._rewrite_origins(medias, name_to_source, origin)

        _assert_isolated(medias[1], medias[2])
        # Mutation isolation.
        medias[1]["origin"]["params"]["mutated"] = "yes"
        assert "mutated" not in medias[2]["origin"]["params"]
        assert "mutated" not in origin["params"]

    def test_skips_medias_with_no_matching_source(self, tmp_path):
        origin = {"importer": "server_files", "params": {"path": "/list.txt"}}
        medias = {
            1: {"origin_name": "a.wav", "filename": "a.wav", "origin": {"existing": True}},
        }
        # name_to_source has no entry for "a.wav" → media must be left alone.
        importer = ServerFilesDatasetImporter()
        importer._rewrite_origins(medias, {}, origin)
        assert medias[1]["origin"] == {"existing": True}
