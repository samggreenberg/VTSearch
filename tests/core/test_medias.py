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
        from vtscore.config import DATA_DIR
        from vtscore.embedding import embed_audio_file

        media = app_module.medias[1]
        # Re-embed the same WAV and verify the result matches
        temp_path = DATA_DIR / "temp_determ_test.wav"
        temp_path.write_bytes(media["media_bytes"])
        embedding = embed_audio_file(temp_path)
        temp_path.unlink(missing_ok=True)
        assert embedding is not None
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
        from vtscore.datasets.loader import apply_custom_metadata_md5

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
        from vtscore.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calculated_hash"},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 0
        assert media_dict[1]["md5"] == "calculated_hash"

    def test_skips_when_custom_metadata_has_no_md5(self):
        from vtscore.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calculated_hash", "custom_metadata": {"title": "foo"}},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 0
        assert media_dict[1]["md5"] == "calculated_hash"

    def test_skips_when_custom_metadata_md5_is_empty(self):
        from vtscore.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calculated_hash", "custom_metadata": {"md5": ""}},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 0
        assert media_dict[1]["md5"] == "calculated_hash"

    def test_handles_mixed_medias(self):
        from vtscore.datasets.loader import apply_custom_metadata_md5

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
        from vtscore.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calculated_hash", "custom_metadata": {"md5": None}},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 0
        assert media_dict[1]["md5"] == "calculated_hash"

    def test_replaces_md5_from_uppercase_key(self):
        from vtscore.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calculated_hash", "custom_metadata": {"MD5": "upper_hash", "title": "foo"}},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 1
        assert media_dict[1]["md5"] == "upper_hash"
        assert "MD5" not in media_dict[1]["custom_metadata"]
        assert media_dict[1]["custom_metadata"]["title"] == "foo"

    def test_lowercase_md5_takes_priority_over_uppercase(self):
        from vtscore.datasets.loader import apply_custom_metadata_md5

        media_dict = {
            1: {"md5": "calculated_hash", "custom_metadata": {"md5": "lower_hash", "MD5": "upper_hash"}},
        }
        count = apply_custom_metadata_md5(media_dict)
        assert count == 1
        # lowercase "md5" should win
        assert media_dict[1]["md5"] == "lower_hash"
        assert "md5" not in media_dict[1]["custom_metadata"]
        # uppercase key remains since lowercase was used
        assert media_dict[1]["custom_metadata"]["MD5"] == "upper_hash"


class TestCustomMetadataMapInLoader:
    """Tests that custom_metadata_map in load_dataset_from_folder skips MD5 calculation."""

    def test_md5_from_custom_metadata_map_used(self, tmp_path):
        """When custom_metadata_map provides md5, the loader uses it."""
        from vtscore.datasets.loader import load_dataset_from_folder
        from vtscore.media.audio.audio_generator import generate_wav

        # Create a WAV file in tmp_path
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(generate_wav(440, 0.5))

        medias_dict: dict = {}
        custom_md5 = "custom_provided_md5_value_here"
        cm_map = {"test.wav": {"md5": custom_md5, "source": "external_db"}}

        load_dataset_from_folder(
            tmp_path,
            "audio",
            medias_dict,
            custom_metadata_map=cm_map,
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
        from vtscore.datasets.loader import load_dataset_from_folder
        from vtscore.media.audio.audio_generator import generate_wav

        wav_path = tmp_path / "test.wav"
        wav_bytes = generate_wav(440, 0.5)
        wav_path.write_bytes(wav_bytes)

        medias_dict: dict = {}
        cm_map = {"test.wav": {"md5": "", "source": "external_db"}}

        load_dataset_from_folder(
            tmp_path,
            "audio",
            medias_dict,
            custom_metadata_map=cm_map,
        )

        media = next(iter(medias_dict.values()))
        expected_md5 = hashlib.md5(wav_bytes).hexdigest()
        assert media["md5"] == expected_md5
        # custom_metadata should still be attached
        assert media["custom_metadata"]["source"] == "external_db"

    def test_no_custom_metadata_map_computes_md5(self, tmp_path):
        """Without custom_metadata_map, md5 is computed normally."""
        from vtscore.datasets.loader import load_dataset_from_folder
        from vtscore.media.audio.audio_generator import generate_wav

        wav_path = tmp_path / "test.wav"
        wav_bytes = generate_wav(440, 0.5)
        wav_path.write_bytes(wav_bytes)

        medias_dict: dict = {}
        load_dataset_from_folder(tmp_path, "audio", medias_dict)

        media = next(iter(medias_dict.values()))
        expected_md5 = hashlib.md5(wav_bytes).hexdigest()
        assert media["md5"] == expected_md5
        assert "custom_metadata" not in media

    def test_custom_metadata_map_with_relative_path_key(self, tmp_path):
        """custom_metadata_map keys can be relative paths (subdir/file.wav)."""
        from vtscore.datasets.loader import load_dataset_from_folder
        from vtscore.media.audio.audio_generator import generate_wav

        subdir = tmp_path / "sub"
        subdir.mkdir()
        wav_path = subdir / "test.wav"
        wav_path.write_bytes(generate_wav(440, 0.5))

        medias_dict: dict = {}
        custom_md5 = "relpath_provided_md5"
        cm_map = {"sub/test.wav": {"md5": custom_md5}}

        load_dataset_from_folder(
            tmp_path,
            "audio",
            medias_dict,
            custom_metadata_map=cm_map,
        )

        media = next(iter(medias_dict.values()))
        assert media["md5"] == custom_md5

    def test_uppercase_md5_key_in_custom_metadata_map(self, tmp_path):
        """When custom_metadata_map provides MD5 (uppercase), the loader uses it."""
        from vtscore.datasets.loader import load_dataset_from_folder
        from vtscore.media.audio.audio_generator import generate_wav

        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(generate_wav(440, 0.5))

        medias_dict: dict = {}
        custom_md5 = "uppercase_provided_md5"
        cm_map = {"test.wav": {"MD5": custom_md5, "source": "external_db"}}

        load_dataset_from_folder(
            tmp_path,
            "audio",
            medias_dict,
            custom_metadata_map=cm_map,
        )

        assert len(medias_dict) == 1
        media = next(iter(medias_dict.values()))
        assert media["md5"] == custom_md5
        assert media["custom_metadata"]["source"] == "external_db"


class TestListMediaIds:
    """Tests for the lightweight ``GET /api/medias/ids`` listing.

    This endpoint replaced the previous unpaginated ``GET /api/medias`` -
    full per-item metadata now comes from ``POST /api/medias/batch`` for
    just the IDs the viewport needs.
    """

    def test_returns_all_medias(self, client):
        resp = client.get("/api/medias/ids")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == app_module.NUM_MEDIAS

    def test_stub_fields_only(self, client):
        """Each stub carries id + media_type and (optionally) embedder - nothing else."""
        resp = client.get("/api/medias/ids")
        data = resp.get_json()
        allowed = {"id", "media_type", "embedder"}
        for media in data:
            assert "id" in media
            assert "media_type" in media
            extra = set(media.keys()) - allowed
            assert not extra, f"unexpected heavy fields in /ids response: {extra}"

    def test_does_not_expose_heavy_fields(self, client):
        resp = client.get("/api/medias/ids")
        data = resp.get_json()
        for media in data:
            for heavy in (
                "media_bytes",
                "embedding",
                "filename",
                "md5",
                "custom_metadata",
                "thumbnail_bytes",
                "description",
                "origin_name",
            ):
                assert heavy not in media

    def test_unpaginated_endpoint_is_gone(self, client):
        """The old ``GET /api/medias`` route no longer exists."""
        resp = client.get("/api/medias")
        assert resp.status_code == 404


class TestBatchMedias:
    def test_returns_requested_ids(self, client):
        resp = client.post("/api/medias/batch", json={"ids": [1, 2]})
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 2
        returned_ids = {m["id"] for m in data}
        assert returned_ids == {1, 2}

    def test_returns_full_metadata(self, client):
        resp = client.post("/api/medias/batch", json={"ids": [1]})
        data = resp.get_json()
        assert len(data) == 1
        media = data[0]
        assert "id" in media
        assert "md5" in media
        assert "filename" in media
        assert "custom_metadata" in media
        assert "media_bytes" not in media
        assert "embedding" not in media

    def test_unknown_ids_omitted(self, client):
        resp = client.post("/api/medias/batch", json={"ids": [1, 99999]})
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["id"] == 1

    def test_empty_ids(self, client):
        resp = client.post("/api/medias/batch", json={"ids": []})
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_missing_ids_field(self, client):
        # Marshmallow-validated route: missing required ``ids`` field
        # surfaces as 422 with the flask-smorest error envelope.
        resp = client.post("/api/medias/batch", json={"foo": "bar"})
        assert resp.status_code == 422

    def test_non_list_ids(self, client):
        # Type-coercion failure on ``ids`` → 422 with the ``errors`` envelope.
        resp = client.post("/api/medias/batch", json={"ids": "not a list"})
        assert resp.status_code == 422


class TestMediaThumbnailBytes:
    """Test medias have waveform thumbnail_bytes generated at init."""

    def test_all_medias_have_thumbnail_bytes(self):
        for media in app_module.medias.values():
            assert "thumbnail_bytes" in media
            assert isinstance(media["thumbnail_bytes"], bytes)
            assert len(media["thumbnail_bytes"]) > 0

    def test_thumbnail_bytes_is_valid_png(self):
        media = app_module.medias[1]
        thumb = media["thumbnail_bytes"]
        # PNG magic number
        assert thumb[:8] == b"\x89PNG\r\n\x1a\n"

    def test_thumbnail_bytes_not_exposed_in_api(self, client):
        resp = client.post("/api/medias/batch", json={"ids": list(range(1, app_module.NUM_MEDIAS + 1))})
        data = resp.get_json()
        for media in data:
            assert "thumbnail_bytes" not in media


class TestMediaImage:
    """Tests for GET /api/medias/<id>/image (audio waveform thumbnails)."""

    def test_returns_png_for_audio_media(self, client):
        resp = client.get("/api/medias/1/image")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_png_is_valid(self, client):
        resp = client.get("/api/medias/1/image")
        assert resp.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_returns_404_for_invalid_id(self, client):
        resp = client.get("/api/medias/9999/image")
        assert resp.status_code == 404

    def test_consistent_responses(self, client):
        """Same media should return the same thumbnail."""
        resp1 = client.get("/api/medias/1/image")
        resp2 = client.get("/api/medias/1/image")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.data == resp2.data

    def test_fallback_generates_thumbnail_on_the_fly(self, client):
        """When thumbnail_bytes is missing, the endpoint generates it on the fly."""
        media = app_module.medias[1]
        saved_thumb = media.pop("thumbnail_bytes", None)
        try:
            resp = client.get("/api/medias/1/image")
            assert resp.status_code == 200
            assert resp.content_type == "image/png"
        finally:
            if saved_thumb is not None:
                media["thumbnail_bytes"] = saved_thumb


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
        # The global 404 handler in ``app.py`` normalises NotFound
        # exceptions on ``/api/`` paths to ``error_response(exc.name, 404)``
        # - that's the JSON-for-SPA hook that overrides werkzeug's HTML
        # 404. The flask-smorest message passed to ``abort()`` is dropped
        # in favour of the canonical reason phrase.
        resp = client.get("/api/medias/9999/audio")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["error"] == "Not Found"

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
        # Multipart body parsed by the handler (not a marshmallow schema),
        # so the failure surfaces as 400 with the standard
        # flask-smorest ``message`` envelope.
        assert "No file" in resp.get_json()["message"]

    def test_missing_label(self, client):
        wav_bytes = app_module.generate_wav(999, 0.1)
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"file": (io.BytesIO(wav_bytes), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "label" in resp.get_json()["message"]

    def test_invalid_label(self, client):
        wav_bytes = app_module.generate_wav(999, 0.1)
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "maybe", "file": (io.BytesIO(wav_bytes), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "label" in resp.get_json()["message"]

    def test_empty_file(self, client):
        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "good", "file": (io.BytesIO(b""), "empty.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "Empty" in resp.get_json()["message"]

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
            assert "No dataset" in resp.get_json()["message"]
        finally:
            app_module.medias.update(saved)

    def test_concurrent_uploads_same_md5_no_duplicate(self):
        """H32 regression: two concurrent uploads of identical bytes must
        not produce duplicate medias.

        The first MD5 lookup happens outside ``_state_lock`` and embedding
        takes the full unlocked window with it. Without a re-check inside
        the lock, two parallel requests both miss the existing-cid hit and
        both insert - yielding two medias with identical md5/embedding.
        The fix re-checks ``md5_lookup`` under ``_state_lock`` immediately
        before the insert and routes the loser into the existing-cid
        branch.
        """
        import threading
        from unittest.mock import patch

        from vtscore.media import embedders_for_type
        from vtscore.state.core import (
            get_active_context,
            get_active_detector_context,
            set_thread_dataset_context,
            set_thread_detector_context,
        )

        audio_embedder = embedders_for_type("audio")[0]

        wav_bytes = app_module.generate_wav(77777, 0.25)
        initial_count = len(app_module.medias)

        # Capture the test thread's contexts so the worker threads resolve
        # the same DatasetContext/DetectorContext when ``before_request``
        # falls through to the thread-local (no X-Dataset-Id header in
        # these requests).
        ds_ctx = get_active_context()
        det_ctx = get_active_detector_context()

        # Embedding must block both threads simultaneously inside the
        # unlocked window so the race is reproduced deterministically.
        barrier = threading.Barrier(2)
        original_embed = audio_embedder.embed_media

        def _blocking_embed(media_dict):
            barrier.wait(timeout=5)
            return original_embed(media_dict)

        results: list[tuple[int, dict]] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def _do_post():
            try:
                set_thread_dataset_context(ds_ctx)
                set_thread_detector_context(det_ctx)
                # Each worker thread uses its own test client. The fixture
                # ``client`` is opened in the main thread, and Flask's test
                # client preserves request contexts on its own ExitStack -
                # invoking it from worker threads pushes contexts onto the
                # workers' ContextVars and breaks the main-thread teardown.
                # The conftest ``client`` fixture auto-injects context
                # headers; freshly-built test clients don't, so we attach
                # them explicitly here (H34 - vote/pile-add now requires
                # ``X-Dataset-Id`` / ``X-Detector-Id``).
                with app_module.app.test_client() as c:
                    r = c.post(
                        "/api/medias/add-to-pile",
                        data={"label": "good", "file": (io.BytesIO(wav_bytes), "race.wav")},
                        content_type="multipart/form-data",
                        headers={
                            "X-Dataset-Id": ds_ctx.dataset_id,
                            "X-Detector-Id": det_ctx.detector_id,
                        },
                    )
                with lock:
                    results.append((r.status_code, r.get_json()))
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        with patch.object(audio_embedder, "embed_media", side_effect=_blocking_embed):
            t1 = threading.Thread(target=_do_post)
            t2 = threading.Thread(target=_do_post)
            t1.start()
            t2.start()
            t1.join(timeout=15)
            t2.join(timeout=15)

        assert not errors, f"worker threads raised: {errors}"
        assert len(results) == 2

        # Exactly one new media inserted (the regression: was 2 before fix).
        assert len(app_module.medias) == initial_count + 1, (
            f"H32: duplicate insert - medias grew by {len(app_module.medias) - initial_count}, expected 1"
        )

        # Both requests succeed and report the same media id.
        media_ids = {r[1]["media_id"] for r in results}
        assert len(media_ids) == 1, f"requests reported different ids: {media_ids}"

        # One winner (201, is_new=True), one loser (200, is_new=False).
        statuses = sorted(r[0] for r in results)
        assert statuses == [200, 201]
        is_new_flags = sorted(r[1]["is_new"] for r in results)
        assert is_new_flags == [False, True]

        # The single inserted media carries the uploaded bytes' md5.
        inserted_id = next(iter(media_ids))
        assert app_module.medias[inserted_id]["md5"] == hashlib.md5(wav_bytes).hexdigest()
        assert inserted_id in app_module.good_votes

    def _setup_loaded_detector(self, client, name="H33Detector"):
        """Create a detector, register it, load it, and return its registry id."""
        from tests import load_detector_and_wait

        client.post(
            "/api/detectors",
            json={"name": name, "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={"name": name, "media_type": "audio", "text_query": "test"},
        )
        mid = res.get_json()["detector"]["id"]
        load_detector_and_wait(client, mid)
        return mid, name

    def test_existing_media_label_synced_to_disk(self, client):
        """H33 regression - MD5-match branch.

        Without :func:`_sync_pile_label_to_storage` the vote applied to the
        existing media never reaches the detector's JSON file, so the next
        rehydration (dataset/detector switch, mtime advance) silently drops
        it.  Assert the labelset on disk contains an entry keyed by the
        matched media's origin with the right label.
        """
        from vtscore.datasets.labelset import LabelSet, element_key, media_element_key
        from vtscore.detectors.store import _detector_path, _read_detector

        media = app_module.medias[1]
        _, name = self._setup_loaded_detector(client)

        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "good", "file": (io.BytesIO(media["media_bytes"]), "match.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200

        data = _read_detector(_detector_path(name))
        assert data is not None, "detector JSON not written"
        ls = LabelSet.from_dict(data.get("labelset") or {})
        target_key = media_element_key(media)
        match = next((el for el in ls.elements if element_key(el) == target_key), None)
        assert match is not None, "matched-media label was not persisted to disk"
        assert match.label == "good"

    def test_new_media_label_synced_to_disk(self, client):
        """H33 regression - new-media branch.

        Newly inserted media items are session-only (not in the dataset
        pickle), so the only durable record of an add-to-pile vote on them
        is the detector's on-disk labelset.  Assert the labelset contains
        an ``add_to_pile`` origin entry with the right label.
        """
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.store import _detector_path, _read_detector

        wav_bytes = app_module.generate_wav(33333, 0.2)
        _, name = self._setup_loaded_detector(client)

        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "bad", "file": (io.BytesIO(wav_bytes), "fresh.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201

        data = _read_detector(_detector_path(name))
        assert data is not None, "detector JSON not written"
        ls = LabelSet.from_dict(data.get("labelset") or {})
        match = next(
            (
                el
                for el in ls.elements
                if (el.origin or {}).get("importer") == "add_to_pile" and el.origin_name == "fresh.wav"
            ),
            None,
        )
        assert match is not None, "add-to-pile origin entry missing from on-disk labelset"
        assert match.label == "bad"

    def test_label_survives_rehydration(self, client):
        """H33 regression - end-to-end symptom.

        ``ensure_votes_match_active_dataset`` rehydrates from the detector
        file when the (dataset, detector) pair changes or the file's mtime
        advances.  Before the fix, ``add_media_to_pile`` only mutated
        in-memory dicts, so a forced rehydrate dropped the just-applied
        vote.  Force a rehydration by clearing the cached labelset and
        re-running the hook; the vote must come back from disk.
        """
        from vtscore.detectors.dataset_sync import ensure_votes_match_active_dataset
        from vtscore.state.core import get_active_detector_context

        media = app_module.medias[2]
        self._setup_loaded_detector(client)

        resp = client.post(
            "/api/medias/add-to-pile",
            data={"label": "good", "file": (io.BytesIO(media["media_bytes"]), "rehydrate.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert 2 in app_module.good_votes

        # Invalidate the detector context's caches and drop the in-memory
        # votes so the rehydrate hook must round-trip through the file.
        det_ctx = get_active_detector_context()
        det_ctx.good_votes.clear()
        det_ctx.bad_votes.clear()
        det_ctx.cached_labelset = None
        det_ctx.cached_labelset_mtime = 0.0
        det_ctx.votes_dataset_id = ""

        ensure_votes_match_active_dataset()
        assert 2 in app_module.good_votes, "label vanished on rehydration - H33 regressed"
