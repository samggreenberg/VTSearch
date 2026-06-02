"""Tests for the ZIP-based dataset container format."""

from __future__ import annotations

import pickle
import zipfile

import numpy as np

from vtscore.datasets.container import (
    append_projection,
    is_container,
    read_container,
    read_meta,
    read_projection,
    write_container,
)
from vtscore.projection.pyramid import build_pyramid
from vtscore.projection.umap_projection import Projection


def _medias_pkl_bytes(n: int = 5) -> bytes:
    rng = np.random.default_rng(42)
    medias = {
        i: {
            "id": i,
            "media_type": "audio",
            "embedding": rng.standard_normal(8).tolist(),
            "duration": 1.0,
            "file_size": 100,
            "md5": f"md5_{i}",
            "filename": f"clip_{i}.wav",
        }
        for i in range(n)
    }
    return pickle.dumps({"medias": medias})


def _meta() -> dict:
    return {
        "format_version": 1,
        "embedder": "test_embedder",
        "clipper": "test_clipper",
        "media_type": "audio",
        "name": "Test Dataset",
        "created_at": 1700000000.0,
        "expires_at": None,
    }


class TestContainerFormat:
    def test_write_creates_zip(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        assert path.exists()
        assert is_container(path)
        with zipfile.ZipFile(str(path), "r") as zf:
            assert "medias.pkl" in zf.namelist()
            assert "meta.json" in zf.namelist()

    def test_read_container_round_trip(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        data, meta = read_container(path)
        assert "medias" in data
        assert len(data["medias"]) == 5
        assert meta["embedder"] == "test_embedder"
        assert meta["clipper"] == "test_clipper"
        assert meta["name"] == "Test Dataset"

    def test_read_meta_only(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        meta = read_meta(path)
        assert meta["embedder"] == "test_embedder"
        assert meta["format_version"] == 1

    def test_extra_pickle_keys(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(
            path,
            _medias_pkl_bytes(),
            _meta(),
            extra_pickle_keys={"audio_dir": "/data/audio"},
        )
        data, meta = read_container(path)
        assert data["audio_dir"] == "/data/audio"

    def test_is_container_false_for_raw_pickle(self, tmp_path):
        path = tmp_path / "legacy.pkl"
        path.write_bytes(pickle.dumps({"medias": {}}))
        assert not is_container(path)

    def test_read_legacy_pickle(self, tmp_path):
        path = tmp_path / "legacy.pkl"
        medias = {1: {"id": 1, "embedding": [0.1, 0.2]}}
        path.write_bytes(pickle.dumps({"medias": medias}))
        data, meta = read_container(path)
        assert "medias" in data
        assert 1 in data["medias"]
        assert meta.get("embedder") == ""

    def test_expires_at_in_meta(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        meta = _meta()
        meta["expires_at"] = 1800000000.0
        write_container(path, _medias_pkl_bytes(), meta)
        loaded_meta = read_meta(path)
        assert loaded_meta["expires_at"] == 1800000000.0


class TestContainerProjection:
    def _make_projection(self, n: int = 20) -> tuple:
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((n, 2)).astype(np.float32)
        proj = Projection("container-pid", list(range(n)), coords, "test")
        pyr = build_pyramid(proj, n_levels=2)
        return proj, pyr

    def test_append_and_read_projection(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(20), _meta())

        proj, pyr = self._make_projection()
        append_projection(path, proj, pyr)

        loaded = read_projection(path)
        assert loaded is not None
        proj2, pyr2 = loaded
        assert proj2.projection_id == "container-pid"
        np.testing.assert_array_equal(proj2.coords, proj.coords)
        assert pyr2.point_count == pyr.point_count

    def test_read_projection_when_absent(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        assert read_projection(path) is None

    def test_append_overwrites_existing_projection(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(20), _meta())

        proj1, pyr1 = self._make_projection()
        append_projection(path, proj1, pyr1)

        rng = np.random.default_rng(99)
        coords2 = rng.standard_normal((20, 2)).astype(np.float32)
        proj2 = Projection("second-pid", list(range(20)), coords2, "test")
        pyr2 = build_pyramid(proj2, n_levels=2)
        append_projection(path, proj2, pyr2)

        loaded = read_projection(path)
        assert loaded is not None
        assert loaded[0].projection_id == "second-pid"

    def test_legacy_file_uses_sidecar(self, tmp_path):
        path = tmp_path / "legacy.pkl"
        path.write_bytes(pickle.dumps({"medias": {}}))

        proj, pyr = self._make_projection()
        append_projection(path, proj, pyr)

        assert path.with_suffix(".projection").exists()

        loaded = read_projection(path)
        assert loaded is not None
        assert loaded[0].projection_id == "container-pid"


class TestSidecarFallback:
    def test_read_embedder_from_container(self, tmp_path):
        from vtscore.datasets.loader_pickle import read_pkl_embedder

        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        assert read_pkl_embedder(path) == "test_embedder"

    def test_read_clipper_from_container(self, tmp_path):
        from vtscore.datasets.loader_pickle import read_pkl_clipper

        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        assert read_pkl_clipper(path) == "test_clipper"

    def test_read_embedder_from_legacy_sidecar(self, tmp_path):
        from vtscore.datasets.loader_pickle import read_pkl_embedder

        path = tmp_path / "legacy.pkl"
        path.write_bytes(pickle.dumps({"medias": {}}))
        path.with_suffix(".embedder").write_text("legacy_embedder")
        assert read_pkl_embedder(path) == "legacy_embedder"
