"""Tests for the Load Sort window endpoints.

Covers:
- /api/example-sort (generic, media-type-aware)
- /api/server-media-files (listing)
- /api/example-sort-server (server-side example sort)
- /api/detector/server-files/<name> (individual server detector fetch)
"""

import io
import json
import struct
import wave

import numpy as np
import pytest

import app as app_module
from vtsearch.utils import medias


class TestExampleSort:
    """Tests for /api/example-sort (generic upload-based example sort)."""

    def _make_wav_bytes(self, duration=0.1, sample_rate=16000):
        """Create a simple WAV file in memory."""
        num_samples = int(sample_rate * duration)
        samples = [int(32767 * np.sin(2 * np.pi * 440 * i / sample_rate)) for i in range(num_samples)]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{num_samples}h", *samples))
        buf.seek(0)
        return buf

    def test_no_file_returns_400(self, client):
        resp = client.post("/api/example-sort")
        assert resp.status_code == 400
        assert "file" in resp.get_json()["error"].lower()

    def test_example_sort_returns_results_and_threshold(self, client):
        wav_buf = self._make_wav_bytes()
        resp = client.post(
            "/api/example-sort",
            data={"file": (wav_buf, "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert "threshold" in data
        assert len(data["results"]) == app_module.NUM_MEDIAS

    def test_results_sorted_descending(self, client):
        wav_buf = self._make_wav_bytes()
        resp = client.post(
            "/api/example-sort",
            data={"file": (wav_buf, "test.wav")},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        sims = [r["similarity"] for r in data["results"]]
        assert sims == sorted(sims, reverse=True)

    def test_all_media_ids_present(self, client):
        wav_buf = self._make_wav_bytes()
        resp = client.post(
            "/api/example-sort",
            data={"file": (wav_buf, "test.wav")},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        ids = {r["id"] for r in data["results"]}
        assert ids == set(medias.keys())


class TestServerMediaFiles:
    """Tests for /api/server-media-files (listing)."""

    @pytest.fixture(autouse=True)
    def _setup_media_dir(self, tmp_path, monkeypatch):
        from vtsearch.routes import sorting as sort_module

        self._media_dir = tmp_path / "example_media"
        monkeypatch.setattr(sort_module, "SERVER_MEDIA_DIR", self._media_dir)

    def test_empty_dir_returns_empty_list(self, client):
        resp = client.get("/api/server-media-files")
        assert resp.status_code == 200
        assert resp.get_json()["files"] == []

    def test_lists_files_in_dir(self, client):
        self._media_dir.mkdir(parents=True, exist_ok=True)
        (self._media_dir / "test1.wav").write_bytes(b"\x00" * 100)
        (self._media_dir / "test2.mp3").write_bytes(b"\x00" * 200)
        resp = client.get("/api/server-media-files")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["files"]) == 2
        names = {f["filename"] for f in data["files"]}
        assert "test1.wav" in names
        assert "test2.mp3" in names

    def test_files_have_expected_fields(self, client):
        self._media_dir.mkdir(parents=True, exist_ok=True)
        (self._media_dir / "example.wav").write_bytes(b"\x00" * 50)
        resp = client.get("/api/server-media-files")
        f = resp.get_json()["files"][0]
        assert "name" in f
        assert "filename" in f
        assert "size_bytes" in f
        assert f["name"] == "example"
        assert f["filename"] == "example.wav"
        assert f["size_bytes"] == 50

    def test_hidden_files_excluded(self, client):
        self._media_dir.mkdir(parents=True, exist_ok=True)
        (self._media_dir / ".hidden").write_bytes(b"\x00")
        (self._media_dir / "visible.wav").write_bytes(b"\x00")
        resp = client.get("/api/server-media-files")
        files = resp.get_json()["files"]
        assert len(files) == 1
        assert files[0]["filename"] == "visible.wav"


class TestExampleSortServer:
    """Tests for /api/example-sort-server (sort by server-side media file)."""

    @pytest.fixture(autouse=True)
    def _setup_media_dir(self, tmp_path, monkeypatch):
        from vtsearch.routes import sorting as sort_module

        self._media_dir = tmp_path / "example_media"
        self._media_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(sort_module, "SERVER_MEDIA_DIR", self._media_dir)

    def _create_test_wav(self, name="test.wav"):
        """Create a valid WAV file in the server media dir."""
        sample_rate = 16000
        num_samples = int(sample_rate * 0.1)
        samples = [int(32767 * np.sin(2 * np.pi * 440 * i / sample_rate)) for i in range(num_samples)]
        path = self._media_dir / name
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{num_samples}h", *samples))
        return path

    def test_missing_filename_returns_400(self, client):
        resp = client.post("/api/example-sort-server", json={})
        assert resp.status_code == 400
        assert "filename" in resp.get_json()["error"].lower()

    def test_nonexistent_file_returns_404(self, client):
        resp = client.post("/api/example-sort-server", json={"filename": "nope.wav"})
        assert resp.status_code == 404

    def test_sort_returns_results(self, client):
        self._create_test_wav("example.wav")
        resp = client.post("/api/example-sort-server", json={"filename": "example.wav"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert "threshold" in data
        assert len(data["results"]) == app_module.NUM_MEDIAS

    def test_path_traversal_rejected(self, client):
        resp = client.post("/api/example-sort-server", json={"filename": "../../../etc/passwd"})
        assert resp.status_code in (400, 404)


class TestServerDetectorFileGet:
    """Tests for GET /api/detector/server-files/<name> (fetch individual detector)."""

    @pytest.fixture(autouse=True)
    def _setup_det_dir(self, tmp_path):
        from vtsearch import settings

        self._det_dir = tmp_path / "detectors"
        self._det_dir.mkdir(parents=True, exist_ok=True)
        settings.set_detectors_dir(str(self._det_dir))

    def _create_detector(self, name="test_detector"):
        data = {"weights": {"0.weight": [[1.0]], "0.bias": [0.0]}, "threshold": 0.5, "media_type": "audio"}
        (self._det_dir / f"{name}.json").write_text(json.dumps(data))
        return data

    def test_get_existing_detector(self, client):
        expected = self._create_detector("my_det")
        resp = client.get("/api/detector/server-files/my_det")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["weights"] == expected["weights"]
        assert data["threshold"] == expected["threshold"]

    def test_get_nonexistent_returns_404(self, client):
        resp = client.get("/api/detector/server-files/nonexistent")
        assert resp.status_code == 404

    def test_invalid_name_returns_400(self, client):
        resp = client.get("/api/detector/server-files/%%%")
        assert resp.status_code == 400
