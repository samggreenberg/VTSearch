"""Multi-media coverage tests.

Exercises key workflows (text sort, learned sort, vote, label export/import,
example sort, extract, localize) against image, text, video, and document
media types — not just audio.  Also covers previously-untested endpoints:

  - POST /api/example-sort-server
  - POST /api/example-sort-origin
  - POST /api/extract
  - POST /api/auto-extract
  - POST /api/auto-localize
"""

from __future__ import annotations

import io

import numpy as np

from helpers import (
    make_image_media,
    make_text_media,
    make_video_media,
    make_png_bytes,
    make_wav_bytes,
)
from vtsearch.state import (
    good_votes,
    bad_votes,
    medias,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate(media_factory, count=10, start_id=1):
    """Clear medias and populate with the given factory."""
    saved = dict(medias)
    medias.clear()
    try:
        for i in range(start_id, start_id + count):
            medias[i] = media_factory(i)
        yield
    finally:
        medias.clear()
        medias.update(saved)


# ===================================================================
# TEXT SORT across media types
# ===================================================================


class TestTextSortMultiMedia:
    """POST /api/sort should work for any media type with a text embedder."""

    def test_text_sort_image_medias(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            for i in range(1, 11):
                medias[i] = make_image_media(i)
            resp = client.post("/api/sort", json={"text": "a photo of a cat"})
            assert resp.status_code == 200
            data = resp.get_json()
            assert "results" in data
            assert len(data["results"]) == 10
        finally:
            medias.clear()
            medias.update(saved)

    def test_text_sort_text_medias(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            for i in range(1, 11):
                medias[i] = make_text_media(i)
            resp = client.post("/api/sort", json={"text": "breaking news"})
            assert resp.status_code == 200
            data = resp.get_json()
            assert "results" in data
            assert len(data["results"]) == 10
        finally:
            medias.clear()
            medias.update(saved)

    def test_text_sort_video_medias(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            for i in range(1, 11):
                medias[i] = make_video_media(i)
            resp = client.post("/api/sort", json={"text": "a dog running"})
            assert resp.status_code == 200
            data = resp.get_json()
            assert "results" in data
            assert len(data["results"]) == 10
        finally:
            medias.clear()
            medias.update(saved)


# ===================================================================
# VOTE + LEARNED SORT across media types
# ===================================================================


class TestLearnedSortMultiMedia:
    """POST /api/learned-sort should train on image/text/video embeddings."""

    def test_learned_sort_image_medias(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            rng = np.random.default_rng(42)
            for i in range(1, 21):
                m = make_image_media(i)
                m["embedding"] = rng.standard_normal(512).astype("float32")
                medias[i] = m
            # Vote on some
            for i in [1, 2, 3]:
                good_votes[i] = None
            for i in [18, 19, 20]:
                bad_votes[i] = None
            resp = client.post("/api/learned-sort", json={"wait": True})
            assert resp.status_code == 200
            data = resp.get_json()
            assert "results" in data
            assert len(data["results"]) == 20
        finally:
            medias.clear()
            medias.update(saved)

    def test_learned_sort_text_medias(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            rng = np.random.default_rng(99)
            for i in range(1, 21):
                m = make_text_media(i)
                m["embedding"] = rng.standard_normal(512).astype("float32")
                medias[i] = m
            for i in [1, 2, 3]:
                good_votes[i] = None
            for i in [18, 19, 20]:
                bad_votes[i] = None
            resp = client.post("/api/learned-sort", json={"wait": True})
            assert resp.status_code == 200
            data = resp.get_json()
            assert "results" in data
        finally:
            medias.clear()
            medias.update(saved)


# ===================================================================
# LABEL EXPORT with non-audio medias
# ===================================================================


class TestLabelExportMultiMedia:
    """GET /api/labels/export should include origin info for all media types."""

    def test_export_image_labels(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            for i in range(1, 6):
                medias[i] = make_image_media(i)
            good_votes[1] = None
            good_votes[2] = None
            bad_votes[5] = None
            resp = client.get("/api/labels/export")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "labels" in data
            labels = data["labels"]
            assert len(labels) == 3
            for lbl in labels:
                assert "origin" in lbl
                assert "origin_name" in lbl
        finally:
            medias.clear()
            medias.update(saved)


# ===================================================================
# /api/medias listing for non-audio types
# ===================================================================


class TestMediasListingMultiMedia:
    """GET /api/medias should correctly list non-audio media."""

    def test_image_medias_listed(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            for i in range(1, 4):
                medias[i] = make_image_media(i)
            resp = client.get("/api/medias/ids")
            assert resp.status_code == 200
            data = resp.get_json()
            assert len(data) == 3
            for item in data:
                assert item["type"] == "image"
                assert "embedding" not in item
                assert "media_bytes" not in item
        finally:
            medias.clear()
            medias.update(saved)

    def test_text_medias_listed(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            for i in range(1, 4):
                medias[i] = make_text_media(i)
            resp = client.get("/api/medias/ids")
            assert resp.status_code == 200
            data = resp.get_json()
            assert len(data) == 3
            for item in data:
                assert item["type"] == "text"
                assert "media_string" not in item
        finally:
            medias.clear()
            medias.update(saved)

    def test_video_medias_listed(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            for i in range(1, 4):
                medias[i] = make_video_media(i)
            resp = client.get("/api/medias/ids")
            assert resp.status_code == 200
            data = resp.get_json()
            assert len(data) == 3
            for item in data:
                assert item["type"] == "video"
        finally:
            medias.clear()
            medias.update(saved)


# ===================================================================
# EXAMPLE SORT endpoints (file upload + server-file + origin)
# ===================================================================


class TestExampleSortUpload:
    """POST /api/example-sort — upload a file for similarity sorting."""

    def test_upload_wav_example_sort(self, client):
        wav = make_wav_bytes()
        data = {"file": (io.BytesIO(wav), "example.wav")}
        resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "results" in body

    def test_upload_png_example_sort(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            for i in range(1, 11):
                medias[i] = make_image_media(i)
            png = make_png_bytes()
            data = {"file": (io.BytesIO(png), "example.png")}
            resp = client.post("/api/example-sort", data=data, content_type="multipart/form-data")
            assert resp.status_code == 200
            body = resp.get_json()
            assert "results" in body
        finally:
            medias.clear()
            medias.update(saved)

    def test_no_file_returns_400(self, client):
        resp = client.post("/api/example-sort", content_type="multipart/form-data")
        assert resp.status_code == 400


class TestExampleSortServer:
    """POST /api/example-sort-server — sort by a server-side file."""

    def test_missing_filename_returns_422(self, client):
        # The example-sort-server schema declares ``filename`` as required,
        # so an empty body fails schema-level validation with the standard
        # flask-smorest 422 envelope.
        resp = client.post("/api/example-sort-server", json={})
        assert resp.status_code == 422
        assert "filename" in resp.get_json()["errors"]["json"]

    def test_nonexistent_file_returns_404(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "vtsearch.routes.media.server._get_server_media_dir",
            lambda: tmp_path,
        )
        resp = client.post("/api/example-sort-server", json={"filename": "nope.wav"})
        assert resp.status_code == 404

    def test_path_traversal_rejected(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "vtsearch.routes.media.server._get_server_media_dir",
            lambda: tmp_path,
        )
        resp = client.post("/api/example-sort-server", json={"filename": "../../etc/passwd"})
        assert resp.status_code == 400

    def test_valid_file_returns_results(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "vtsearch.routes.media.server._get_server_media_dir",
            lambda: tmp_path,
        )
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(make_wav_bytes())
        resp = client.post("/api/example-sort-server", json={"filename": "test.wav"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data


class TestExampleSortOrigin:
    """POST /api/example-sort-origin — sort by media resolved from an origin."""

    def test_missing_origin_returns_422(self, client):
        # The example-sort-origin schema declares ``origin`` as required.
        resp = client.post("/api/example-sort-origin", json={"key": "foo.wav"})
        assert resp.status_code == 422
        assert "origin" in resp.get_json()["errors"]["json"]

    def test_missing_key_returns_422(self, client):
        # The example-sort-origin schema declares ``key`` as required.
        resp = client.post(
            "/api/example-sort-origin",
            json={"origin": {"importer": "server_folder", "params": {"path": "/tmp"}}},
        )
        assert resp.status_code == 422
        assert "key" in resp.get_json()["errors"]["json"]

    def test_unknown_origin_type_returns_400(self, client):
        # Origin shape itself is permissive; the handler rejects unknown
        # importer types with a 400 + standard ``message`` envelope.
        resp = client.post(
            "/api/example-sort-origin",
            json={
                "origin": {"importer": "nonexistent_source_type", "params": {}},
                "key": "test.wav",
            },
        )
        assert resp.status_code == 400

    def test_valid_folder_origin_returns_results(self, client, tmp_path):
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(make_wav_bytes())
        resp = client.post(
            "/api/example-sort-origin",
            json={
                "origin": {"importer": "server_folder", "params": {"path": str(tmp_path)}},
                "key": "test.wav",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data


# ===================================================================
# EXTRACT / AUTO-EXTRACT / AUTO-LOCALIZE endpoints
# ===================================================================


class TestExtractEndpoint:
    """POST /api/extract — run a single extractor."""

    def test_missing_extractor_type_returns_422(self, client):
        resp = client.post("/api/extract", json={"name": "test", "config": {"foo": 1}})
        assert resp.status_code == 422
        assert "extractor_type" in str(resp.get_json()).lower()

    def test_missing_config_returns_422(self, client):
        resp = client.post("/api/extract", json={"name": "test", "extractor_type": "image_class"})
        assert resp.status_code == 422
        assert "config" in str(resp.get_json()).lower()

    def test_unknown_type_returns_400(self, client):
        resp = client.post(
            "/api/extract",
            json={"name": "test", "extractor_type": "not_real", "config": {"x": 1}},
        )
        assert resp.status_code == 400

    def test_media_type_mismatch_returns_400(self, client):
        resp = client.post(
            "/api/extract",
            json={
                "name": "test",
                "extractor_type": "image_class",
                "config": {"target_class": "person"},
            },
        )
        # Default medias are audio; image_class extractor expects image
        assert resp.status_code == 400
        assert "media type" in resp.get_json()["message"].lower()

    def test_no_medias_returns_400(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            resp = client.post(
                "/api/extract",
                json={
                    "name": "test",
                    "extractor_type": "image_class",
                    "config": {"target_class": "person"},
                },
            )
            assert resp.status_code == 400
            assert "No medias" in resp.get_json()["message"]
        finally:
            medias.clear()
            medias.update(saved)


class TestAutoExtractEndpoint:
    """POST /api/auto-extract — run all autorun extractors."""

    def test_no_medias_returns_400(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            resp = client.post("/api/auto-extract")
            assert resp.status_code == 400
            assert "No medias" in resp.get_json()["message"]
        finally:
            medias.clear()
            medias.update(saved)

    def test_no_autorun_extractors_returns_400(self, client):
        resp = client.post("/api/auto-extract")
        assert resp.status_code == 400
        assert "No autorun extractors" in resp.get_json()["message"]


class TestAutoLocalizeEndpoint:
    """POST /api/auto-localize — run all autorun localizers."""

    def test_no_medias_returns_400(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            resp = client.post("/api/auto-localize")
            assert resp.status_code == 400
            assert "No medias" in resp.get_json()["message"]
        finally:
            medias.clear()
            medias.update(saved)

    def test_no_autorun_localizers_returns_400(self, client):
        resp = client.post("/api/auto-localize")
        assert resp.status_code == 400
        assert "No autorun localizers" in resp.get_json()["message"]


# ===================================================================
# MIXED-MEDIA DATASET operations
# ===================================================================


class TestVoteOnNonAudioMedia:
    """Voting on image/text/video medias should work identically to audio."""

    def test_vote_good_on_image(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            for i in range(1, 6):
                medias[i] = make_image_media(i)
            resp = client.post("/api/medias/1/vote", json={"vote": "good"})
            assert resp.status_code == 200
            resp = client.get("/api/votes")
            data = resp.get_json()
            assert 1 in data["good"]
        finally:
            medias.clear()
            medias.update(saved)

    def test_vote_bad_on_text(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            for i in range(1, 6):
                medias[i] = make_text_media(i)
            resp = client.post("/api/medias/3/vote", json={"vote": "bad"})
            assert resp.status_code == 200
            resp = client.get("/api/votes")
            data = resp.get_json()
            assert 3 in data["bad"]
        finally:
            medias.clear()
            medias.update(saved)
