"""Tests for the combine-datasets importer.

Covers:
- Importer metadata (name, icon, description, fields)
- Importer is excluded from the generic /api/dataset/importers list
- Combining two pickle datasets with the same media type
- Duplicate detection by MD5 hash
- Media type mismatch rejection
- Fewer than two datasets rejection
- Missing file error handling
- Available-files endpoint
- Combine API endpoint
- Staging endpoints (stage-file, stage-import, staging cleanup)
- CLI support (run_cli)
- build_origin method
"""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pytest


def _unique_bytes(media_id: int) -> bytes:
    """Return unique bytes for a media so MD5 hashes are distinct after reload."""
    return media_id.to_bytes(4, "little") + b"\x00" * 96


def _make_audio_clip(media_id: int, md5: str = "", filename: str = "") -> dict:
    """Return a minimal media dict for testing."""
    if not filename:
        filename = f"clip_{media_id}.wav"
    raw = _unique_bytes(media_id)
    if not md5:
        md5 = hashlib.md5(raw).hexdigest()
    return {
        "id": media_id,
        "media_type": "audio",
        "duration": 1.0,
        "file_size": 1000,
        "md5": md5,
        "embedding": np.array([float(media_id), float(media_id) + 0.5]),
        "media_bytes": raw,
        "media_string": None,
        "media_path": None,
        "filename": filename,
        "category": "test",
        "origin": None,
        "origin_name": filename,
    }


def _make_image_clip(media_id: int, md5: str = "") -> dict:
    """Return a minimal image media dict for testing."""
    raw = b"\x89PNG" + _unique_bytes(media_id)
    if not md5:
        md5 = hashlib.md5(raw).hexdigest()
    return {
        "id": media_id,
        "media_type": "image",
        "duration": 0,
        "file_size": 2000,
        "md5": md5,
        "embedding": np.array([float(media_id)]),
        "media_bytes": raw,
        "media_string": None,
        "media_path": None,
        "filename": f"img_{media_id}.png",
        "category": "test",
        "origin": None,
        "origin_name": f"img_{media_id}.png",
        "width": 32,
        "height": 32,
    }


def _write_pickle_dataset(path, clips_dict):
    """Write a pickle dataset file in the standard format."""
    data = {
        "medias": {
            cid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in media.items()}
            for cid, media in clips_dict.items()
        }
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Importer metadata
# ---------------------------------------------------------------------------


class TestCombineDatasetsMetadata:
    def _get_importer(self):
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        return IMPORTER

    def test_name(self):
        assert self._get_importer().name == "combine_datasets"

    def test_display_name(self):
        assert "Combine" in self._get_importer().display_name

    def test_icon(self):
        assert self._get_importer().icon == "\U0001f500"

    def test_description_mentions_merge_or_combine(self):
        desc = self._get_importer().description.lower()
        assert "merge" in desc or "combine" in desc

    def test_description_mentions_duplicates(self):
        desc = self._get_importer().description.lower()
        assert "duplicate" in desc

    def test_to_dict_includes_icon(self):
        d = self._get_importer().to_dict()
        assert d["icon"] == "\U0001f500"
        assert d["name"] == "combine_datasets"

    def test_fields_include_datasets(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "datasets" in fields


# ---------------------------------------------------------------------------
# Excluded from generic importer list (has dedicated UI)
# ---------------------------------------------------------------------------


class TestCombineDatasetsBuiltinExclusion:
    def test_combine_datasets_has_custom_ui_mode(self):
        from vtscore.datasets.importers import get_importer

        imp = get_importer("combine_datasets")
        assert imp is not None
        assert imp.ui_mode == "custom"

    def test_combine_datasets_not_in_extended_list(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        names = [imp["name"] for imp in data["importers"]]
        assert "combine_datasets" not in names


# ---------------------------------------------------------------------------
# Enabled flag in all-importers endpoint
# ---------------------------------------------------------------------------


class TestCombineDatasetsEnabledFlag:
    def _find_combine(self, client):
        resp = client.get("/api/dataset/all-importers")
        data = resp.get_json()
        for imp in data["importers"]:
            if imp["name"] == "combine_datasets":
                return imp
        return None

    def test_disabled_with_no_saved_datasets(self, client):
        """combine_datasets is disabled when the registry is empty."""
        imp = self._find_combine(client)
        assert imp is not None
        assert imp["enabled"] is False

    def test_disabled_with_one_dataset(self, client):
        """combine_datasets is disabled with only one saved dataset."""
        from vtscore.datasets import registry

        registry.register_dataset(name="solo", media_type="audio", num_items=10, pkl_path="/tmp/solo.pkl")
        try:
            imp = self._find_combine(client)
            assert imp is not None
            assert imp["enabled"] is False
        finally:
            entries = registry.list_datasets()
            for e in entries:
                if e["name"] == "solo":
                    registry.unregister_dataset(e["id"])

    def test_disabled_with_two_different_media_types(self, client):
        """combine_datasets is disabled when two datasets have different media types."""
        from vtscore.datasets import registry

        e1 = registry.register_dataset(name="audio_ds", media_type="audio", num_items=10, pkl_path="/tmp/a.pkl")
        e2 = registry.register_dataset(name="image_ds", media_type="image", num_items=10, pkl_path="/tmp/b.pkl")
        try:
            imp = self._find_combine(client)
            assert imp is not None
            assert imp["enabled"] is False
        finally:
            registry.unregister_dataset(e1["id"])
            registry.unregister_dataset(e2["id"])

    def test_enabled_with_two_same_media_type(self, client):
        """combine_datasets is enabled when two datasets share a media type."""
        from vtscore.datasets import registry

        e1 = registry.register_dataset(name="audio_a", media_type="audio", num_items=10, pkl_path="/tmp/a.pkl")
        e2 = registry.register_dataset(name="audio_b", media_type="audio", num_items=5, pkl_path="/tmp/b.pkl")
        try:
            imp = self._find_combine(client)
            assert imp is not None
            assert imp["enabled"] is True
        finally:
            registry.unregister_dataset(e1["id"])
            registry.unregister_dataset(e2["id"])


# ---------------------------------------------------------------------------
# Core combining logic
# ---------------------------------------------------------------------------


class TestCombineDatasetsRun:
    def test_combine_two_datasets(self, tmp_path):
        """Two datasets with the same media type are merged."""
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1), 2: _make_audio_clip(2)}
        ds2 = {1: _make_audio_clip(3), 2: _make_audio_clip(4)}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        medias: dict = {}
        IMPORTER.run({"datasets": [str(p1), str(p2)]}, medias)

        assert len(medias) == 4
        # IDs should be re-assigned sequentially
        assert set(medias.keys()) == {1, 2, 3, 4}

    def test_deduplication_by_md5(self, tmp_path):
        """Clips with the same MD5 across datasets are included only once."""
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        # Use the same media_id (same bytes) to produce a genuine MD5 duplicate
        dup_clip_a = _make_audio_clip(1, filename="dup_a.wav")
        dup_clip_b = _make_audio_clip(1, filename="dup_b.wav")  # same bytes as dup_a
        ds1 = {1: dup_clip_a, 2: _make_audio_clip(2)}
        ds2 = {1: dup_clip_b, 2: _make_audio_clip(4)}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        medias: dict = {}
        IMPORTER.run({"datasets": [str(p1), str(p2)]}, medias)

        assert len(medias) == 3  # 4 total minus 1 duplicate

    def test_media_type_mismatch_raises(self, tmp_path):
        """Combining audio and image datasets raises ValueError."""
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        ds2 = {1: _make_image_clip(2)}
        p1, p2 = tmp_path / "audio.pkl", tmp_path / "image.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        medias: dict = {}
        with pytest.raises(ValueError, match="Media type mismatch"):
            IMPORTER.run({"datasets": [str(p1), str(p2)]}, medias)

    def test_fewer_than_two_datasets_raises(self, tmp_path):
        """Providing only one dataset raises ValueError."""
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        p1 = tmp_path / "ds1.pkl"
        _write_pickle_dataset(p1, ds1)

        medias: dict = {}
        with pytest.raises(ValueError, match="At least two"):
            IMPORTER.run({"datasets": [str(p1)]}, medias)

    def test_missing_file_raises(self, tmp_path):
        """A non-existent path raises FileNotFoundError."""
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        p1 = tmp_path / "ds1.pkl"
        _write_pickle_dataset(p1, ds1)

        medias: dict = {}
        with pytest.raises(FileNotFoundError):
            IMPORTER.run({"datasets": [str(p1), str(tmp_path / "missing.pkl")]}, medias)

    def test_comma_separated_string_input(self, tmp_path):
        """The datasets field also accepts a comma-separated string."""
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        ds2 = {1: _make_audio_clip(2)}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        medias: dict = {}
        IMPORTER.run({"datasets": f"{p1},{p2}"}, medias)

        assert len(medias) == 2

    def test_empty_dataset_skipped(self, tmp_path):
        """An empty pickle file is skipped without error."""
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        empty = {}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "empty.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, empty)

        # Need a third non-empty dataset to meet minimum of 2 datasets
        ds3 = {1: _make_audio_clip(3)}
        p3 = tmp_path / "ds3.pkl"
        _write_pickle_dataset(p3, ds3)

        medias: dict = {}
        IMPORTER.run({"datasets": [str(p1), str(p2), str(p3)]}, medias)

        assert len(medias) == 2

    def test_preserves_media_data_fields(self, tmp_path):
        """Merged medias retain their original data fields."""
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1, filename="song_a.wav")}
        ds2 = {1: _make_audio_clip(2, filename="song_b.wav")}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        medias: dict = {}
        IMPORTER.run({"datasets": [str(p1), str(p2)]}, medias)

        filenames = {c["filename"] for c in medias.values()}
        assert "song_a.wav" in filenames
        assert "song_b.wav" in filenames

    def test_three_datasets_combined(self, tmp_path):
        """Three datasets combine correctly."""
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        ds2 = {1: _make_audio_clip(2)}
        ds3 = {1: _make_audio_clip(3)}
        p1, p2, p3 = tmp_path / "a.pkl", tmp_path / "b.pkl", tmp_path / "c.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)
        _write_pickle_dataset(p3, ds3)

        medias: dict = {}
        IMPORTER.run({"datasets": [str(p1), str(p2), str(p3)]}, medias)

        assert len(medias) == 3


# ---------------------------------------------------------------------------
# CLI support
# ---------------------------------------------------------------------------


class TestCombineDatasetsCli:
    def test_run_cli_delegates_to_run(self, tmp_path):
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        ds1 = {1: _make_audio_clip(1)}
        ds2 = {1: _make_audio_clip(2)}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        medias: dict = {}
        IMPORTER.run_cli({"datasets": f"{p1},{p2}"}, medias)

        assert len(medias) == 2


# ---------------------------------------------------------------------------
# build_origin
# ---------------------------------------------------------------------------


class TestCombineDatasetsOrigin:
    def test_build_origin_with_list(self):
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        origin = IMPORTER.build_origin({"datasets": ["/a.pkl", "/b.pkl"]})
        assert origin["importer"] == "combine_datasets"
        assert "/a.pkl" in origin["params"]["datasets"]
        assert "/b.pkl" in origin["params"]["datasets"]

    def test_build_origin_with_string(self):
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        origin = IMPORTER.build_origin({"datasets": "/a.pkl,/b.pkl"})
        assert origin["importer"] == "combine_datasets"
        assert origin["params"]["datasets"] == "/a.pkl,/b.pkl"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestAvailableFilesEndpoint:
    def test_returns_list(self, client):
        resp = client.get("/api/dataset/available-files")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "files" in data
        assert isinstance(data["files"], list)

    def test_lists_pkl_files(self, client, tmp_path):
        """When EMBEDDINGS_DIR contains .pkl files, they appear in the list."""
        from vtscore.config import EMBEDDINGS_DIR

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        test_pkl = EMBEDDINGS_DIR / "_test_combine.pkl"
        test_pkl.write_bytes(pickle.dumps({"medias": {}}))
        try:
            resp = client.get("/api/dataset/available-files")
            data = resp.get_json()
            names = [f["name"] for f in data["files"]]
            assert "_test_combine" in names
            # Check fields
            entry = next(f for f in data["files"] if f["name"] == "_test_combine")
            assert "path" in entry
            assert "size_mb" in entry
        finally:
            test_pkl.unlink(missing_ok=True)


class TestCombineEndpoint:
    def test_rejects_fewer_than_two(self, client):
        # Schema-level validation (datasets Length >= 2) → 422.
        resp = client.post(
            "/api/dataset/combine",
            json={"datasets": ["/one.pkl"]},
        )
        assert resp.status_code == 422

    def test_rejects_missing_file(self, client, tmp_path):
        resp = client.post(
            "/api/dataset/combine",
            json={"datasets": ["/nonexistent_a.pkl", "/nonexistent_b.pkl"]},
        )
        assert resp.status_code == 400

    def test_accepts_valid_request(self, client, tmp_path):
        """A valid request returns 200 with ok=True."""
        from unittest.mock import patch

        ds1 = {1: _make_audio_clip(1)}
        ds2 = {1: _make_audio_clip(2)}
        p1, p2 = tmp_path / "ds1.pkl", tmp_path / "ds2.pkl"
        _write_pickle_dataset(p1, ds1)
        _write_pickle_dataset(p2, ds2)

        # Mock the background loader to prevent it from clearing global medias
        with patch("vtsearch.routes.datasets.staging._run_importer_in_background"):
            resp = client.post(
                "/api/dataset/combine",
                json={"datasets": [str(p1), str(p2)]},
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True

    def test_path_traversal_rejected(self, client):
        """Paths outside the allowed directory must be rejected."""
        resp = client.post(
            "/api/dataset/combine",
            json={"datasets": ["/etc/passwd", "/etc/shadow"]},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Staging endpoints
# ---------------------------------------------------------------------------


class TestStageFileEndpoint:
    def test_rejects_missing_file(self, client):
        resp = client.post("/api/dataset/stage-file")
        assert resp.status_code == 400

    def test_stages_valid_pkl(self, client, tmp_path):
        """Uploading a valid .pkl file returns staging metadata."""
        from io import BytesIO

        ds = {1: _make_audio_clip(1), 2: _make_audio_clip(2)}
        buf = BytesIO()
        data = {
            "medias": {
                cid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in m.items()} for cid, m in ds.items()
            }
        }
        pickle.dump(data, buf)
        buf.seek(0)

        resp = client.post(
            "/api/dataset/stage-file",
            data={"file": (buf, "test_ds.pkl")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        result = resp.get_json()
        assert "path" in result
        assert result["name"] == "test_ds.pkl"
        assert result["count"] == 2
        assert result["media_type"] == "audio"

        # Clean up the staging file
        from pathlib import Path

        Path(result["path"]).unlink(missing_ok=True)

    def test_stages_empty_pkl(self, client):
        """An empty pkl returns count=0."""
        from io import BytesIO

        buf = BytesIO()
        pickle.dump({"medias": {}}, buf)
        buf.seek(0)

        resp = client.post(
            "/api/dataset/stage-file",
            data={"file": (buf, "empty.pkl")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["count"] == 0

        from pathlib import Path

        Path(result["path"]).unlink(missing_ok=True)


class TestStageImportEndpoint:
    def test_rejects_unknown_importer(self, client):
        resp = client.post("/api/dataset/stage-import/nonexistent")
        assert resp.status_code == 404

    def test_accepts_valid_importer(self, client, tmp_path):
        """Staging the pickle importer returns 200 with ok=True."""
        from io import BytesIO

        ds = {1: _make_audio_clip(1)}
        buf = BytesIO()
        data = {
            "medias": {
                cid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in m.items()} for cid, m in ds.items()
            }
        }
        pickle.dump(data, buf)
        buf.seek(0)

        resp = client.post(
            "/api/dataset/stage-import/pickle",
            data={"file": (buf, "staged.pkl")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["ok"] is True


class TestStageDemoEndpoint:
    def test_rejects_invalid_name(self, client):
        resp = client.post("/api/dataset/stage-demo/nonexistent_xyz_demo")
        assert resp.status_code == 400

    def test_accepts_valid_demo(self, client):
        """If any demo dataset exists, staging it returns 200."""
        from vtscore.datasets import DEMO_DATASETS

        if not DEMO_DATASETS:
            pytest.skip("No demo datasets configured")
        name = next(iter(DEMO_DATASETS))
        resp = client.post(f"/api/dataset/stage-demo/{name}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True


class TestClearStagingEndpoint:
    def test_clear_staging(self, client):
        """DELETE /api/dataset/staging returns ok."""
        resp = client.delete("/api/dataset/staging")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True

    def test_clear_staging_removes_files(self, client):
        """Staging files are actually removed."""
        from vtscore.datasets.load_pipeline import STAGING_DIR

        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        test_file = STAGING_DIR / "stage_test.pkl"
        test_file.write_bytes(b"dummy")
        assert test_file.exists()

        resp = client.delete("/api/dataset/staging")
        assert resp.status_code == 200
        assert not test_file.exists()


class TestProgressStagingResult:
    def test_staging_result_in_progress(self):
        """update_progress stores staging_result and get_progress returns it."""
        from vtscore.concurrency.progress import get_progress, update_progress

        staging = {"path": "/tmp/test.pkl", "name": "test", "count": 5, "media_type": "audio"}
        update_progress("idle", "done", 100, 100, staging_result=staging)
        progress = get_progress()
        assert progress["staging_result"] == staging

        # Clean up; explicitly clear staging_result (update has merge semantics)
        update_progress("idle", "", 0, 0, staging_result=None)
        assert get_progress()["staging_result"] is None
