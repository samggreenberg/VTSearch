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
        import vtsearch.routes.media.server as server_module

        monkeypatch.setattr(server_module, "SERVER_MEDIA_DIR", tmp_path / "example_media")

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
