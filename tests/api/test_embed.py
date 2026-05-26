"""Tests for the on-demand embedding API (``POST /api/embed``)."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from vtscore.media import _embedder_registry
from vtscore.media.embedder import MediaEmbedder


class _FakeImageEmbedder(MediaEmbedder):
    """Image-modality fake that records the path it was asked to embed.

    Returns a deterministic 4-d vector for both media and text inputs so
    tests can assert exact response shapes.  Setting ``return_none_media``
    or ``return_none_text`` lets individual tests exercise the failure
    branches.
    """

    name = "fake_image"  # type: ignore[assignment]
    media_type_id = "image"  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self._model = "loaded"
        self.last_media_path: str | None = None
        self.last_text: str | None = None
        self.return_none_media = False
        self.return_none_text = False
        self._supports_text = True

    @property
    def supports_text(self) -> bool:
        return self._supports_text

    def _load_models_impl(self) -> None:  # pragma: no cover; pre-loaded
        return

    def _embed_media_impl(self, media):
        self.last_media_path = media.get("media_path")
        if self.return_none_media:
            return None
        return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    def embed_text(self, text):
        self.last_text = text
        if self.return_none_text:
            return None
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)


@pytest.fixture
def fake_image_embedder():
    """Register a fake image embedder for one test, then unregister it."""
    emb = _FakeImageEmbedder()
    saved_name = _embedder_registry.pop("fake_image", None)
    _embedder_registry["fake_image"] = emb
    try:
        yield emb
    finally:
        _embedder_registry.pop("fake_image", None)
        if saved_name is not None:
            _embedder_registry["fake_image"] = saved_name


class TestEmbedMediaUpload:
    """Multipart upload mode."""

    def test_embeds_uploaded_image(self, client, fake_image_embedder):
        resp = client.post(
            "/api/embed",
            data={"embedder": "fake_image", "file": (BytesIO(b"\x89PNGfake"), "pic.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["embedder"] == "fake_image"
        assert body["media_type"] == "image"
        assert body["dim"] == 4
        assert body["embedding"] == [1.0, 2.0, 3.0, 4.0]
        # norm = sqrt(1+4+9+16) = sqrt(30)
        assert body["norm"] == pytest.approx(np.sqrt(30.0), rel=1e-5)
        # Embedder saw a real file on disk that was cleaned up afterwards.
        assert fake_image_embedder.last_media_path is not None

    def test_unknown_embedder_returns_404(self, client):
        resp = client.post(
            "/api/embed",
            data={"embedder": "nope", "file": (BytesIO(b"x"), "pic.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 404
        body = resp.get_json()
        assert "Unknown embedder 'nope'" in body["error"]
        assert "Available:" in body["error"]

    def test_missing_embedder_field_returns_400(self, client):
        resp = client.post(
            "/api/embed",
            data={"file": (BytesIO(b"x"), "pic.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "embedder is required" in resp.get_json()["error"]

    def test_missing_file_returns_400(self, client, fake_image_embedder):
        resp = client.post(
            "/api/embed",
            data={"embedder": "fake_image"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "file is required" in resp.get_json()["error"]

    def test_wrong_extension_returns_400(self, client, fake_image_embedder):
        """An audio file uploaded to an image embedder is rejected fast."""
        resp = client.post(
            "/api/embed",
            data={"embedder": "fake_image", "file": (BytesIO(b"RIFF"), "sound.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["expected_media_type"] == "image"
        assert body["detected_media_type"] == "audio"

    def test_unknown_extension_passes_through_to_embedder(self, client, fake_image_embedder):
        """A file with no recognised extension is still attempted."""
        resp = client.post(
            "/api/embed",
            data={"embedder": "fake_image", "file": (BytesIO(b"x"), "blob.xyz")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["dim"] == 4

    def test_embedder_returns_none_yields_400(self, client, fake_image_embedder):
        fake_image_embedder.return_none_media = True
        resp = client.post(
            "/api/embed",
            data={"embedder": "fake_image", "file": (BytesIO(b"\x89PNGfake"), "pic.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "could not embed" in body["error"]
        assert body["media_type"] == "image"


class TestEmbedTextJson:
    """JSON body mode."""

    def test_embeds_text(self, client, fake_image_embedder):
        resp = client.post(
            "/api/embed",
            json={"embedder": "fake_image", "text": "a cat"},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["embedder"] == "fake_image"
        assert body["media_type"] == "image"
        assert body["dim"] == 4
        assert body["embedding"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert fake_image_embedder.last_text == "a cat"

    def test_unknown_embedder_returns_404(self, client):
        resp = client.post("/api/embed", json={"embedder": "nope", "text": "hi"})
        assert resp.status_code == 404
        assert "Unknown embedder 'nope'" in resp.get_json()["error"]

    def test_missing_embedder_returns_400(self, client):
        resp = client.post("/api/embed", json={"text": "hi"})
        assert resp.status_code == 400
        assert "embedder is required" in resp.get_json()["error"]

    def test_missing_text_returns_400(self, client, fake_image_embedder):
        resp = client.post("/api/embed", json={"embedder": "fake_image"})
        assert resp.status_code == 400
        assert "text is required" in resp.get_json()["error"]

    def test_non_text_embedder_returns_400(self, client, fake_image_embedder):
        fake_image_embedder._supports_text = False
        resp = client.post(
            "/api/embed",
            json={"embedder": "fake_image", "text": "a cat"},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["supports_text"] is False
        assert "does not support text" in body["error"]

    def test_embedder_returns_none_yields_500(self, client, fake_image_embedder):
        fake_image_embedder.return_none_text = True
        resp = client.post(
            "/api/embed",
            json={"embedder": "fake_image", "text": "a cat"},
        )
        assert resp.status_code == 500
        assert "returned no vector" in resp.get_json()["error"]

    def test_invalid_json_returns_400(self, client):
        resp = client.post(
            "/api/embed",
            data="not json",
            content_type="application/json",
        )
        assert resp.status_code == 400
