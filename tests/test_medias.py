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


class TestApplyCustomMetadataMD5:
    """Tests for apply_custom_metadata_md5: use MD5 from custom_metadata when provided."""

    def test_replaces_md5_from_custom_metadata(self):
        from vtsearch.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calculated_hash", "custom_metadata": {"md5": "provided_hash", "title": "foo"}},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 1
        assert media_dict[1]["md5"] == "provided_hash"
        # md5 should be removed from custom_metadata to avoid duplicate display
        assert "md5" not in media_dict[1]["custom_metadata"]
        assert media_dict[1]["custom_metadata"]["title"] == "foo"

    def test_skips_when_no_custom_metadata(self):
        from vtsearch.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calculated_hash"},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 0
        assert media_dict[1]["md5"] == "calculated_hash"

    def test_skips_when_custom_metadata_has_no_md5(self):
        from vtsearch.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calculated_hash", "custom_metadata": {"title": "foo"}},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 0
        assert media_dict[1]["md5"] == "calculated_hash"

    def test_skips_when_custom_metadata_md5_is_empty(self):
        from vtsearch.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calculated_hash", "custom_metadata": {"md5": ""}},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 0
        assert media_dict[1]["md5"] == "calculated_hash"

    def test_handles_mixed_medias(self):
        from vtsearch.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calc1", "custom_metadata": {"md5": "provided1"}},
            2: {"md5": "calc2", "custom_metadata": {"title": "no md5 here"}},
            3: {"md5": "calc3"},
            4: {"md5": "calc4", "custom_metadata": {"md5": "provided4"}},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 2
        assert media_dict[1]["md5"] == "provided1"
        assert "md5" not in media_dict[1]["custom_metadata"]
        assert media_dict[2]["md5"] == "calc2"
        assert media_dict[3]["md5"] == "calc3"
        assert media_dict[4]["md5"] == "provided4"
        assert "md5" not in media_dict[4]["custom_metadata"]

    def test_skips_when_custom_metadata_md5_is_none(self):
        from vtsearch.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calculated_hash", "custom_metadata": {"md5": None}},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 0
        assert media_dict[1]["md5"] == "calculated_hash"


class TestCustomMetadataMapInLoader:
    """Tests that custom_metadata_map in load_dataset_from_folder skips MD5 calculation."""

    def test_md5_from_custom_metadata_map_used(self, tmp_path):
        """When custom_metadata_map provides md5, the loader uses it."""
        from vtsearch.datasets.loader import load_dataset_from_folder
        from vtsearch.audio.generator import generate_wav

        # Create a WAV file in tmp_path
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(generate_wav(440, 0.5))

        medias_dict: dict = {}
        custom_md5 = "custom_provided_md5_value_here"
        cm_map = {"test.wav": {"md5": custom_md5, "source": "external_db"}}

        load_dataset_from_folder(
            tmp_path, "sounds", medias_dict, custom_metadata_map=cm_map,
        )

        assert len(medias_dict) == 1
        media = next(iter(medias_dict.values()))
        # MD5 should be from custom_metadata_map, not computed from file bytes
        assert media["md5"] == custom_md5
        # custom_metadata should be attached to the media
        assert media["custom_metadata"]["md5"] == custom_md5
        assert media["custom_metadata"]["source"] == "external_db"

    def test_empty_md5_in_custom_metadata_map_falls_through(self, tmp_path):
        """When custom_metadata_map has empty md5, the loader computes it."""
        from vtsearch.datasets.loader import load_dataset_from_folder
        from vtsearch.audio.generator import generate_wav

        wav_path = tmp_path / "test.wav"
        wav_bytes = generate_wav(440, 0.5)
        wav_path.write_bytes(wav_bytes)

        medias_dict: dict = {}
        cm_map = {"test.wav": {"md5": "", "source": "external_db"}}

        load_dataset_from_folder(
            tmp_path, "sounds", medias_dict, custom_metadata_map=cm_map,
        )

        media = next(iter(medias_dict.values()))
        expected_md5 = hashlib.md5(wav_bytes).hexdigest()
        assert media["md5"] == expected_md5
        # custom_metadata should still be attached
        assert media["custom_metadata"]["source"] == "external_db"

    def test_no_custom_metadata_map_computes_md5(self, tmp_path):
        """Without custom_metadata_map, md5 is computed normally."""
        from vtsearch.datasets.loader import load_dataset_from_folder
        from vtsearch.audio.generator import generate_wav

        wav_path = tmp_path / "test.wav"
        wav_bytes = generate_wav(440, 0.5)
        wav_path.write_bytes(wav_bytes)

        medias_dict: dict = {}
        load_dataset_from_folder(tmp_path, "sounds", medias_dict)

        media = next(iter(medias_dict.values()))
        expected_md5 = hashlib.md5(wav_bytes).hexdigest()
        assert media["md5"] == expected_md5
        assert "custom_metadata" not in media

    def test_custom_metadata_map_with_relative_path_key(self, tmp_path):
        """custom_metadata_map keys can be relative paths (subdir/file.wav)."""
        from vtsearch.datasets.loader import load_dataset_from_folder
        from vtsearch.audio.generator import generate_wav

        subdir = tmp_path / "sub"
        subdir.mkdir()
        wav_path = subdir / "test.wav"
        wav_path.write_bytes(generate_wav(440, 0.5))

        medias_dict: dict = {}
        custom_md5 = "relpath_provided_md5"
        cm_map = {"sub/test.wav": {"md5": custom_md5}}

        load_dataset_from_folder(
            tmp_path, "sounds", medias_dict, custom_metadata_map=cm_map,
        )

        media = next(iter(medias_dict.values()))
        assert media["md5"] == custom_md5


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


class TestAddToPile:
    """Tests for POST /api/medias/add-to-pile."""

    def test_add_existing_media_to_good(self, client):
        """Uploading a file whose MD5 matches an existing media votes it good."""
        media = app_module.medias[1]
        data = {"label": "good"}
        resp = client.post(
            "/api/medias/add-to-pile",
            data={**data, "file": (io.BytesIO(media["media_bytes"]), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["ok"] is True
        assert result["is_new"] is False
        assert result["media_id"] == 1
        assert 1 in app_module.good_votes

    def test_add_existing_media_to_bad(self, client):
        """Uploading a file whose MD5 matches an existing media votes it bad."""
        media = app_module.medias[2]
        data = {"label": "bad"}
        resp = client.post(
            "/api/medias/add-to-pile",
            data={**data, "file": (io.BytesIO(media["media_bytes"]), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["ok"] is True
        assert result["is_new"] is False
        assert result["media_id"] == 2
        assert 2 in app_module.bad_votes

    def test_add_new_media_to_good(self, client):
        """Uploading a file with new MD5 embeds it, inserts it, and votes good."""
        # Generate a unique WAV that doesn't match any existing media
        wav_bytes = app_module.generate_wav(12345, 0.3)
        initial_count = len(app_module.medias)
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "good", "file": (io.BytesIO(wav_bytes), "new_sound.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        result = resp.get_json()
        assert result["ok"] is True
        assert result["is_new"] is True
        new_id = result["media_id"]
        assert new_id in app_module.medias
        assert len(app_module.medias) == initial_count + 1
        assert new_id in app_module.good_votes
        # Verify the new media has proper fields
        new_media = app_module.medias[new_id]
        assert new_media["md5"] == hashlib.md5(wav_bytes).hexdigest()
        assert new_media["filename"] == "new_sound.wav"
        assert new_media["origin"]["importer"] == "add_to_pile"
        assert isinstance(new_media["embedding"], np.ndarray)

    def test_add_new_media_to_bad(self, client):
        """Uploading a new file and voting it bad."""
        wav_bytes = app_module.generate_wav(54321, 0.2)
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "bad", "file": (io.BytesIO(wav_bytes), "bad_sound.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        result = resp.get_json()
        assert result["ok"] is True
        assert result["is_new"] is True
        assert result["media_id"] in app_module.bad_votes

    def test_missing_file(self, client):
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "good"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "No file" in resp.get_json()["error"]

    def test_missing_label(self, client):
        wav_bytes = app_module.generate_wav(999, 0.1)
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"file": (io.BytesIO(wav_bytes), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "label" in resp.get_json()["error"]

    def test_invalid_label(self, client):
        wav_bytes = app_module.generate_wav(999, 0.1)
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "maybe", "file": (io.BytesIO(wav_bytes), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "label" in resp.get_json()["error"]

    def test_empty_file(self, client):
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "good", "file": (io.BytesIO(b""), "empty.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "Empty" in resp.get_json()["error"]

    def test_no_dataset_loaded(self, client):
        """When no dataset is loaded, adding new media fails gracefully."""
        saved = dict(app_module.medias)
        app_module.medias.clear()
        try:
            wav_bytes = app_module.generate_wav(999, 0.1)
            resp = client.post(
                "/api/medias/add-to-pile",
                data={"label": "good", "file": (io.BytesIO(wav_bytes), "test.wav")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 400
            assert "No dataset" in resp.get_json()["error"]
        finally:
            app_module.medias.update(saved)
