"""Tests for the Load Sort window endpoints.

Covers:
- /api/example-sort (generic, media-type-aware)
- /api/server-media-files (listing)
- /api/example-sort-server (server-side example sort)
- /api/detectors/registry (dashboard models for model loading)
- /api/example-sort-origin (sort by origin-based media file)
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

    def test_with_audio_crop_params_sorts_using_clip(self, client):
        # 1s WAV cropped to a 0.05s sub-region — request still succeeds and
        # returns a fully ranked result list.  Server-side cropping is the
        # only path that keeps the sub-region distinguishable from the full
        # clip when both have the same content hash.
        wav_buf = self._make_wav_bytes(duration=1.0)
        resp = client.post(
            "/api/example-sort",
            data={
                "file": (wav_buf, "test.wav"),
                "crop_params": json.dumps({"start": 0.1, "end": 0.15}),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) == app_module.NUM_MEDIAS

    def test_with_invalid_crop_params_falls_back_to_full(self, client):
        wav_buf = self._make_wav_bytes()
        resp = client.post(
            "/api/example-sort",
            data={
                "file": (wav_buf, "test.wav"),
                "crop_params": "not-json",
            },
            content_type="multipart/form-data",
        )
        # Invalid JSON is silently ignored — request still succeeds.
        assert resp.status_code == 200


class TestServerMediaFiles:
    """Tests for /api/server-media-files (listing)."""

    @pytest.fixture(autouse=True)
    def _setup_media_dir(self, tmp_path, monkeypatch):
        from vtsearch.routes import media_server as media_server_module

        self._media_dir = tmp_path / "example_media"
        monkeypatch.setattr(media_server_module, "SERVER_MEDIA_DIR", self._media_dir)

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


class TestServerMediaFileThumbnail:
    """Tests for /api/server-media-files/<filename>/thumbnail."""

    @pytest.fixture(autouse=True)
    def _setup_media_dir(self, tmp_path, monkeypatch):
        from vtsearch.routes import media_server as media_server_module

        self._media_dir = tmp_path / "example_media"
        self._media_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(media_server_module, "SERVER_MEDIA_DIR", self._media_dir)

    def _make_wav(self, name="example.wav", duration=0.1):
        sample_rate = 16000
        num_samples = int(sample_rate * duration)
        samples = [int(32767 * np.sin(2 * np.pi * 440 * i / sample_rate)) for i in range(num_samples)]
        path = self._media_dir / name
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{num_samples}h", *samples))
        return path

    def test_image_returns_file_bytes(self, client):
        from PIL import Image

        path = self._media_dir / "example.png"
        Image.new("RGB", (16, 16), color=(123, 45, 67)).save(path)

        resp = client.get("/api/server-media-files/example.png/thumbnail")
        assert resp.status_code == 200
        assert resp.mimetype == "image/png"
        assert resp.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_audio_returns_waveform_png(self, client):
        self._make_wav("example.wav")
        resp = client.get("/api/server-media-files/example.wav/thumbnail")
        assert resp.status_code == 200
        assert resp.mimetype == "image/png"
        assert resp.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_missing_file_returns_404(self, client):
        resp = client.get("/api/server-media-files/nope.wav/thumbnail")
        assert resp.status_code == 404

    def test_path_traversal_rejected(self, client):
        resp = client.get("/api/server-media-files/..%2F..%2Fetc%2Fpasswd/thumbnail")
        # 400 (rejected by validation) or 404 (path doesn't resolve to a file)
        assert resp.status_code in (400, 404)

    def test_unsupported_extension_returns_404(self, client):
        (self._media_dir / "doc.txt").write_text("hello")
        resp = client.get("/api/server-media-files/doc.txt/thumbnail")
        assert resp.status_code == 404


class TestExampleSortServer:
    """Tests for /api/example-sort-server (sort by server-side media file)."""

    @pytest.fixture(autouse=True)
    def _setup_media_dir(self, tmp_path, monkeypatch):
        from vtsearch.routes import media_server as media_server_module

        self._media_dir = tmp_path / "example_media"
        self._media_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(media_server_module, "SERVER_MEDIA_DIR", self._media_dir)

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

    def test_sort_with_audio_crop_params(self, client):
        # 1s WAV — request a sub-region.  The server crops into a temp
        # file and returns ranked results.
        sample_rate = 16000
        num_samples = sample_rate
        samples = [int(32767 * np.sin(2 * np.pi * 440 * i / sample_rate)) for i in range(num_samples)]
        path = self._media_dir / "longer.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{num_samples}h", *samples))

        resp = client.post(
            "/api/example-sort-server",
            json={"filename": "longer.wav", "crop_params": {"start": 0.1, "end": 0.5}},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) == app_module.NUM_MEDIAS

    def test_upload_with_crop_params_persists_cropped_bytes(self, client):
        # Upload a 1s WAV with crop_params — the saved file should be the
        # cropped sub-region (so seeding/sorting/etc. read the crop directly).
        sample_rate = 16000
        num_samples = sample_rate
        samples = [int(32767 * np.sin(2 * np.pi * 440 * i / sample_rate)) for i in range(num_samples)]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{num_samples}h", *samples))
        buf.seek(0)

        resp = client.post(
            "/api/server-media-files/upload",
            data={
                "file": (buf, "src.wav"),
                "media_type": "audio",
                "crop_params": json.dumps({"start": 0.1, "end": 0.4}),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        saved_name = resp.get_json()["filename"]

        # Verify the saved file is the cropped sub-region (~0.3s).
        with wave.open(str(self._media_dir / saved_name), "rb") as wf:
            saved_duration = wf.getnframes() / wf.getframerate()
        assert saved_duration == pytest.approx(0.3, abs=0.01)


class TestRegistryDetectorsForLoadSort:
    """Tests for the detector registry surface used by the Load Sort window."""

    def test_registry_lists_detectors(self, client):
        """GET /api/detectors/registry returns detectors list."""
        resp = client.get("/api/detectors/registry")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "detectors" in data
        assert isinstance(data["detectors"], list)

    def test_create_and_list_registry_detector(self, client):
        """A registered detector appears in the registry listing."""
        resp = client.post(
            "/api/detectors/registry",
            json={"name": "Test LoadSort Detector", "media_type": "audio"},
        )
        assert resp.status_code == 201

        resp = client.get("/api/detectors/registry")
        detectors = resp.get_json()["detectors"]
        names = [d["name"] for d in detectors]
        assert "Test LoadSort Detector" in names


class TestExampleSortOrigin:
    """Tests for /api/example-sort-origin (sort by origin-resolved media file)."""

    def test_missing_origin_returns_400(self, client):
        resp = client.post("/api/example-sort-origin", json={"key": "test.wav"})
        assert resp.status_code == 400
        assert "origin" in resp.get_json()["error"].lower()

    def test_missing_key_returns_400(self, client):
        resp = client.post(
            "/api/example-sort-origin",
            json={"origin": {"importer": "server_folder", "params": {"path": "/tmp"}}},
        )
        assert resp.status_code == 400
        assert "key" in resp.get_json()["error"].lower()

    def test_invalid_origin_type_returns_400(self, client):
        resp = client.post("/api/example-sort-origin", json={"origin": "not_a_dict", "key": "test.wav"})
        assert resp.status_code == 400

    def test_unknown_origin_importer_returns_400(self, client):
        resp = client.post(
            "/api/example-sort-origin",
            json={"origin": {"importer": "nonexistent_source"}, "key": "test.wav"},
        )
        assert resp.status_code == 400
        assert "source" in resp.get_json()["error"].lower()

    def test_sort_with_folder_origin(self, client, tmp_path):
        """Sort by a media file resolved from a folder origin."""
        # Create a test WAV in a temp folder
        sample_rate = 16000
        num_samples = int(sample_rate * 0.1)
        samples = [int(32767 * np.sin(2 * np.pi * 440 * i / sample_rate)) for i in range(num_samples)]
        wav_path = tmp_path / "test_audio.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{num_samples}h", *samples))

        resp = client.post(
            "/api/example-sort-origin",
            json={
                "origin": {"importer": "server_folder", "params": {"path": str(tmp_path)}},
                "key": "test_audio.wav",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert "threshold" in data
        assert len(data["results"]) == app_module.NUM_MEDIAS
