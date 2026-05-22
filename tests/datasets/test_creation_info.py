"""Tests for dataset pickle round-trip and status endpoint."""

import pickle

import numpy as np

from vtscore.datasets.importers import get_importer
from vtscore.datasets.importers.base import DatasetImporter, ImporterField
from vtscore.datasets.loader import export_dataset_to_file, load_dataset_from_pickle


# ---------------------------------------------------------------------------
# build_cli_args on the base class
# ---------------------------------------------------------------------------


class _DummyImporter(DatasetImporter):
    name = "test_dummy"
    display_name = "Test Dummy"
    description = "A dummy importer for testing."
    fields = [
        ImporterField("media_type", "Media Type", "select", options=["audio", "image"], default="audio"),
        ImporterField("path", "Folder", "folder"),
    ]

    def run(self, field_values, medias):
        pass


class TestBuildCliArgs:
    def test_basic_cli_args(self):
        imp = _DummyImporter()
        args = imp.build_cli_args({"media_type": "audio", "path": "/data/audio"})
        assert "--importer test_dummy" in args
        assert "--media-type audio" in args
        assert "--path /data/audio" in args

    def test_empty_values_skipped(self):
        imp = _DummyImporter()
        args = imp.build_cli_args({"media_type": "audio", "path": ""})
        assert "--path" not in args

    def test_file_fields_skipped(self):
        """File fields don't translate to CLI flags."""

        class _FileImporter(DatasetImporter):
            name = "file_test"
            display_name = "File Test"
            description = "test"
            fields = [ImporterField("upload", "Upload", "file", accept=".pkl")]

            def run(self, _fv, _c):
                pass

        imp = _FileImporter()
        args = imp.build_cli_args({"upload": "<FileStorage object>"})
        assert "--upload" not in args


# ---------------------------------------------------------------------------
# Real importers produce correct CLI args
# ---------------------------------------------------------------------------


class TestRealImporterCliArgs:
    def test_folder_importer_cli_args(self):
        imp = get_importer("server_folder")
        args = imp.build_cli_args({"media_type": "audio", "path": "/my/folder"})
        assert "--importer server_folder" in args
        assert "--media-type audio" in args
        assert "--path /my/folder" in args
        # `recursive` defaults to true, so the flag is emitted.
        assert "--recursive" in args
        assert "--no-recursive" not in args

    def test_folder_importer_cli_args_no_recursive(self):
        imp = get_importer("server_folder")
        args = imp.build_cli_args({"media_type": "audio", "path": "/my/folder", "recursive": False})
        assert "--no-recursive" in args
        assert " --recursive" not in args

    def test_http_archive_importer_cli_args(self):
        imp = get_importer("http_archive")
        args = imp.build_cli_args({"url": "https://example.com/data.zip", "media_type": "image"})
        assert "--importer http_archive" in args
        assert "--url https://example.com/data.zip" in args
        assert "--media-type image" in args


# ---------------------------------------------------------------------------
# Pickle round-trip: medias survive export -> import
# ---------------------------------------------------------------------------


class TestPickleRoundTrip:
    def _make_clips(self):
        return {
            1: {
                "id": 1,
                "type": "audio",
                "duration": 1.0,
                "file_size": 100,
                "md5": "abc123",
                "embedding": np.zeros(10),
                "filename": "clip_1.wav",
                "category": "test",
                "media_bytes": b"\x00" * 100,
                "media_string": None,
            }
        }

    def test_export_does_not_include_creation_info(self):
        data_bytes = export_dataset_to_file(self._make_clips())
        data = pickle.loads(data_bytes)
        assert "creation_info" not in data

    def test_load_returns_none(self, tmp_path):
        data_bytes = export_dataset_to_file(self._make_clips())
        pkl_path = tmp_path / "test.pkl"
        pkl_path.write_bytes(data_bytes)

        loaded_clips: dict = {}
        result = load_dataset_from_pickle(pkl_path, loaded_clips)
        assert result is None
        assert len(loaded_clips) == 1

    def test_rejects_old_format_pickle(self, tmp_path):
        """Old-style pickles (no wrapping 'medias' key) are rejected."""
        old_data = {
            1: {
                "id": 1,
                "type": "audio",
                "duration": 1.0,
                "file_size": 100,
                "md5": "abc123",
                "embedding": [0.0] * 10,
                "filename": "clip_1.wav",
                "category": "test",
                "wav_bytes": b"\x00" * 100,
            }
        }
        pkl_path = tmp_path / "old.pkl"
        pkl_path.write_bytes(pickle.dumps(old_data))
        loaded_clips: dict = {}
        import pytest

        with pytest.raises(ValueError, match="Invalid pickle format"):
            load_dataset_from_pickle(pkl_path, loaded_clips)


# ---------------------------------------------------------------------------
# Status endpoint no longer includes creation_info
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    def test_status_does_not_include_creation_info(self, client):
        resp = client.get("/api/dataset/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "creation_info" not in data
