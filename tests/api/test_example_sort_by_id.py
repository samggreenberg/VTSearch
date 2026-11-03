"""Tests for the right-click ``Use this as detector seed`` endpoints.

Covers:

* ``POST /api/example-sort-by-id``; sort by similarity to an
  already-loaded media (in-memory embedding path, no re-embed).
* ``POST /api/server-media-files/from-media-id``; materialise a loaded
  media's bytes to ``example_media/`` for the new-detector seed flow.
"""

from __future__ import annotations

import pytest

from vtsearch.state import medias


@pytest.fixture
def loaded_media_id(reset_state):
    """Return the id of a loaded media. The autouse ``reset_state`` fixture
    seeds the snapshot from ``_test_medias_snapshot``; pick the first id."""
    if not medias:
        pytest.skip("No medias loaded in test environment")
    return next(iter(medias.keys()))


class TestExampleSortByIdEndpoint:
    def test_missing_media_id(self, client):
        resp = client.post("/api/example-sort-by-id", json={})
        assert resp.status_code == 422
        assert "media_id" in resp.get_json()["errors"]["json"]

    def test_unknown_media_id(self, client, loaded_media_id):
        resp = client.post("/api/example-sort-by-id", json={"media_id": 999_999})
        assert resp.status_code == 400
        assert "not loaded" in resp.get_json()["message"].lower()

    def test_no_medias_loaded(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            resp = client.post("/api/example-sort-by-id", json={"media_id": 1})
            assert resp.status_code == 400
            assert "no medias loaded" in resp.get_json()["message"].lower()
        finally:
            medias.update(saved)

    def test_success_uses_in_memory_embedding(self, client, loaded_media_id):
        """Happy path: the existing embedding is reused, no fetch / re-embed."""
        resp = client.post(
            "/api/example-sort-by-id",
            json={"media_id": loaded_media_id},
        )
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert "results" in body
        assert "threshold" in body
        # Cosine similarity to self is 1.0; the query item should rank #1.
        assert body["results"][0]["id"] == loaded_media_id

    def test_media_without_embedding_returns_400(self, client, loaded_media_id):
        """Cropping is the only path that can re-derive; without crop, an
        embedding-less media is unsortable."""
        saved = medias[loaded_media_id].copy()
        medias[loaded_media_id]["embeddings"] = {}
        medias[loaded_media_id].pop("embedding", None)
        try:
            resp = client.post(
                "/api/example-sort-by-id",
                json={"media_id": loaded_media_id},
            )
            assert resp.status_code == 400
            assert "no embedding" in resp.get_json()["message"].lower()
        finally:
            medias[loaded_media_id] = saved


class TestServerMediaFileFromMediaIdEndpoint:
    def test_missing_media_id(self, client):
        resp = client.post("/api/server-media-files/from-media-id", json={})
        assert resp.status_code == 422
        assert "media_id" in resp.get_json()["errors"]["json"]

    def test_unknown_media_id(self, client):
        resp = client.post("/api/server-media-files/from-media-id", json={"media_id": 999_999})
        assert resp.status_code == 400
        assert "not loaded" in resp.get_json()["message"].lower()

    def test_media_without_bytes_returns_404(self, client, loaded_media_id):
        """A loaded media that has no resolvable bytes (no media_bytes,
        no media_path, no media_string) is a 404; we can't materialise it."""
        saved = medias[loaded_media_id].copy()
        for key in ("media_bytes", "media_path", "media_string"):
            medias[loaded_media_id].pop(key, None)
        try:
            resp = client.post(
                "/api/server-media-files/from-media-id",
                json={"media_id": loaded_media_id},
            )
            assert resp.status_code == 404
        finally:
            medias[loaded_media_id] = saved

    def test_success_materialises_bytes(self, client, loaded_media_id, tmp_path, monkeypatch):
        """Happy path: media bytes are written under example_media/ and the
        saved filename + original name come back in the response."""
        # Redirect the per-user example_media/ dir so we don't pollute disk.
        import vtscore.security.path_validation as paths_mod

        monkeypatch.setattr(paths_mod, "example_media_dir", lambda: tmp_path / "example_media")

        # Make sure the media has resolvable bytes.
        media = medias[loaded_media_id]
        if not media.get("media_bytes") and not media.get("media_path"):
            media["media_bytes"] = b"fake content for materialisation test"

        resp = client.post(
            "/api/server-media-files/from-media-id",
            json={"media_id": loaded_media_id},
        )
        assert resp.status_code == 201, resp.get_json()
        body = resp.get_json()
        assert body["filename"]
        assert body["original_name"]

        saved_path = tmp_path / "example_media" / body["filename"]
        assert saved_path.exists()
        assert saved_path.stat().st_size > 0

    def test_a_face_says_whether_it_is_a_hard_view(self, client, loaded_media_id, tmp_path, monkeypatch):
        """A face comes back with its Face Quality verdict (#5372); any other media has none."""
        import vtscore.security.path_validation as paths_mod
        from vtscore.media.face import imagekit_bridge

        monkeypatch.setattr(paths_mod, "example_media_dir", lambda: tmp_path / "example_media")
        monkeypatch.setattr(imagekit_bridge, "hard_view_of_file", lambda path: None)
        media = medias[loaded_media_id]
        if not media.get("media_bytes") and not media.get("media_path"):
            media["media_bytes"] = b"fake content for materialisation test"

        def post():
            resp = client.post("/api/server-media-files/from-media-id", json={"media_id": loaded_media_id})
            assert resp.status_code == 201, resp.get_json()
            return resp.get_json()

        assert "face_hard_view" not in post()
        monkeypatch.setitem(media, "media_type", "face")
        compat = imagekit_bridge.COMPAT_METADATA_KEY
        for quality, verdict in ((11.0, True), (23.0, False)):
            meta = {imagekit_bridge.QUALITY_METADATA_KEY: quality, compat: "oss-arcface-mbf:1"}
            monkeypatch.setitem(media, "custom_metadata", meta)
            assert post()["face_hard_view"] is verdict
        monkeypatch.setitem(media, "custom_metadata", {})
        assert post()["face_hard_view"] is None


class TestUploadedFaceHardView:
    def test_an_uploaded_face_says_whether_it_is_a_hard_view(self, client, monkeypatch):
        """An upload sent as a face is read by the provider for its verdict (#5372); any other upload is not."""
        import io

        from vtscore.media.face import imagekit_bridge

        seen = []
        monkeypatch.setattr(imagekit_bridge, "hard_view_of_file", lambda path: seen.append(path) or True)

        def upload(**form):
            resp = client.post(
                "/api/server-media-files/upload",
                data={"file": (io.BytesIO(b"\x89PNG fake"), "face.png"), **form},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 201, resp.get_json()
            return resp.get_json()

        assert "face_hard_view" not in upload() and "face_hard_view" not in upload(media_type="image")
        assert upload(media_type="face")["face_hard_view"] is True
        assert len(seen) == 1 and seen[0].name.endswith(".png")
