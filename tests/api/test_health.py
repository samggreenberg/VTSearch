"""Tests for the ``/healthz`` (liveness) and ``/readyz`` (readiness) probes."""

from unittest.mock import patch

import app as app_module  # noqa: F401 - triggers conftest side effects


class TestLiveness:
    """``GET /healthz`` is always 200 while the process is up."""

    def test_returns_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "ok"}

    def test_does_not_touch_models(self, client):
        """Liveness must not trigger embedder loading."""
        with patch("vtsearch.routes.health.predict_embedders_to_preload") as m:
            resp = client.get("/healthz")
        assert resp.status_code == 200
        m.assert_not_called()


class TestReadiness:
    """``GET /readyz`` returns 200 only when every sub-check passes."""

    def test_ready_with_empty_registry(self, client):
        """An empty dataset/detector registry implies no embedders are required.

        ``conftest.reset_state`` clears both registries before each test, so
        ``predict_embedders_to_preload()`` returns ``[]`` and the models
        check passes trivially.
        """
        resp = client.get("/readyz")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ready"
        assert body["checks"]["data_dir"]["ok"] is True
        assert body["checks"]["models"]["ok"] is True

    def test_not_ready_when_embedder_pending(self, client):
        """If an embedder is expected but not loaded, return 503."""
        with patch(
            "vtsearch.routes.health.predict_embedders_to_preload",
            return_value=["clap"],
        ):
            # In the test session embedders' load_models is stubbed, so _model
            # remains None - exactly the "still loading" condition we want.
            resp = client.get("/readyz")
        assert resp.status_code == 503
        body = resp.get_json()
        assert body["status"] == "not_ready"
        assert body["checks"]["models"]["ok"] is False
        assert "clap" in body["checks"]["models"]["detail"]

    def test_not_ready_when_data_dir_unwritable(self, client, tmp_path):
        """If the data dir is not writable, return 503.

        ``os.access`` is patched directly because the test container often
        runs as root, where chmod bits are bypassed and the natural
        approach of ``chmod 0o500`` doesn't actually make a dir unwritable.
        """
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        with (
            patch("vtsearch.routes.health.DATA_DIR", ro_dir),
            patch("vtsearch.routes.health.os.access", return_value=False),
        ):
            resp = client.get("/readyz")
        assert resp.status_code == 503
        body = resp.get_json()
        assert body["status"] == "not_ready"
        assert body["checks"]["data_dir"]["ok"] is False
        assert "not writable" in body["checks"]["data_dir"]["detail"]

    def test_reports_loaded_embedder_as_ok(self, client):
        """When the expected embedder has a non-None ``_model``, models check passes."""
        from vtscore.media import get_embedder

        emb = get_embedder("clap")
        sentinel = object()
        with (
            patch(
                "vtsearch.routes.health.predict_embedders_to_preload",
                return_value=["clap"],
            ),
            patch.object(emb, "_model", sentinel, create=True),
        ):
            resp = client.get("/readyz")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["checks"]["models"]["ok"] is True
        assert "clap" in body["checks"]["models"]["detail"]
