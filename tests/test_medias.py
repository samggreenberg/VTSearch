import hashlib
import io
import wave

import numpy as np

import app as app_module


class TestInitMedias:
    def test_creates_correct_number_of_medias(self):
        assert len(app_module.medias) == app_module.NUM_MEDIAS

    def test_media_ids_are_1_to_num_medias(self):
        assert set(app_module.medias.keys()) == set(range(1, app_module.NUM_MEDIAS + 1))

    def test_media_frequencies(self):
        for i in range(1, app_module.NUM_MEDIAS + 1):
            expected_freq = 200 + (i - 1) * 50
            assert app_module.medias[i]["frequency"] == expected_freq

    def test_media_durations(self):
        for i in range(1, app_module.NUM_MEDIAS + 1):
            expected_dur = round(1.0 + (i % 5) * 0.5, 1)
            assert app_module.medias[i]["duration"] == expected_dur

    def test_media_has_embedding(self):
        for media in app_module.medias.values():
            emb = media["embedding"]
            assert isinstance(emb, np.ndarray)
            assert len(emb) > 0

    def test_media_has_media_bytes(self):
        for media in app_module.medias.values():
            assert isinstance(media["media_bytes"], bytes)
            assert len(media["media_bytes"]) > 0

    def test_file_size_matches_media_bytes_length(self):
        for media in app_module.medias.values():
            assert media["file_size"] == len(media["media_bytes"])

    def test_deterministic_embeddings(self):
        """CLAP embeddings should be deterministic for the same input audio."""
        from vtsearch.config import DATA_DIR
        from vtsearch.models import embed_audio_file

        media = app_module.medias[1]
        # Re-embed the same WAV and verify the result matches
        temp_path = DATA_DIR / "temp_determ_test.wav"
        temp_path.write_bytes(media["media_bytes"])
        embedding = embed_audio_file(temp_path)
        temp_path.unlink(missing_ok=True)
        np.testing.assert_array_almost_equal(embedding, media["embedding"])


class TestMediaMD5:
    def test_media_has_md5(self):
        for media in app_module.medias.values():
            assert "md5" in media
            assert isinstance(media["md5"], str)
            assert len(media["md5"]) == 32  # MD5 hex string

    def test_md5_matches_media_bytes(self):
        for media in app_module.medias.values():
            expected_md5 = hashlib.md5(media["media_bytes"]).hexdigest()
            assert media["md5"] == expected_md5

    def test_md5_deterministic(self):
        """MD5 should be the same for the same media across re-generation."""
        media = app_module.medias[1]
        # Regenerate the WAV with the same parameters and verify MD5
        wav_bytes = app_module.generate_wav(media["frequency"], media["duration"])
        assert hashlib.md5(wav_bytes).hexdigest() == media["md5"]


class TestListMedias:
    def test_returns_all_medias(self, client):
        resp = client.get("/api/medias")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == app_module.NUM_MEDIAS

    def test_media_fields(self, client):
        resp = client.get("/api/medias")
        data = resp.get_json()
        for media in data:
            assert "id" in media
            assert "md5" in media
            assert "filename" in media
            assert "custom_metadata" in media
            # Type-specific fields appear in custom_metadata
            cm = media["custom_metadata"]
            assert "Frequency" in cm
            assert "Duration" in cm
            assert "File Size" in cm

    def test_does_not_expose_media_bytes(self, client):
        resp = client.get("/api/medias")
        data = resp.get_json()
        for media in data:
            assert "media_bytes" not in media

    def test_does_not_expose_embedding(self, client):
        resp = client.get("/api/medias")
        data = resp.get_json()
        for media in data:
            assert "embedding" not in media


class TestMediaAudio:
    def test_returns_wav_for_valid_id(self, client):
        resp = client.get("/api/medias/1/audio")
        assert resp.status_code == 200
        assert resp.content_type == "audio/wav"

    def test_wav_is_valid(self, client):
        resp = client.get("/api/medias/1/audio")
        buf = io.BytesIO(resp.data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

    def test_returns_404_for_invalid_id(self, client):
        resp = client.get("/api/medias/9999/audio")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["error"] == "not found"

    def test_returns_404_for_zero_id(self, client):
        resp = client.get("/api/medias/0/audio")
        assert resp.status_code == 404
