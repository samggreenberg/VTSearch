"""Tests for the solo-mediaType streamlining setting and CLI fallback.

Covers the per-user ``solo_media_type`` + ``solo_media_type_explicit`` pair,
the process-level CLI fallback (``set_cli_solo_media_type``), the resolver
(``get_effective_solo_media_type``), the route handler
(``PUT /api/settings``), and the preload extension
(``preload_predicted_embedders(extra_media_types=...)``).
"""

from __future__ import annotations

import json

import pytest

import app as app_module  # noqa: F401 - triggers conftest media init
from vtsearch import settings as settings_mod


@pytest.fixture(autouse=True)
def _reset_cli_solo():
    """The CLI fallback is process-global; clear it around each test."""
    settings_mod.set_cli_solo_media_type(None)
    yield
    settings_mod.set_cli_solo_media_type(None)


class TestSoloMediaTypeSettings:
    def test_default_is_none_and_not_explicit(self):
        assert settings_mod.get_solo_media_type() is None
        assert settings_mod.get_solo_media_type_explicit() is False
        assert settings_mod.get_effective_solo_media_type() is None

    def test_apply_user_solo_media_type_sets_both_fields(self, isolated_settings):
        settings_mod.apply_user_solo_media_type("image")
        assert settings_mod.get_solo_media_type() == "image"
        assert settings_mod.get_solo_media_type_explicit() is True
        assert settings_mod.get_effective_solo_media_type() == "image"

        # Persisted to the per-user file (not the server file).
        raw = json.loads(isolated_settings.read_text())
        assert raw["solo_media_type"] == "image"
        assert raw["solo_media_type_explicit"] is True

    def test_apply_user_solo_media_type_none_means_show_everything(self):
        settings_mod.apply_user_solo_media_type(None)
        assert settings_mod.get_solo_media_type() is None
        assert settings_mod.get_solo_media_type_explicit() is True
        assert settings_mod.get_effective_solo_media_type() is None

    def test_apply_user_solo_media_type_empty_string_normalises_to_none(self):
        settings_mod.apply_user_solo_media_type("")
        assert settings_mod.get_solo_media_type() is None
        assert settings_mod.get_solo_media_type_explicit() is True


class TestCliFallback:
    def test_cli_fallback_applies_when_not_explicit(self):
        settings_mod.set_cli_solo_media_type("audio")
        assert settings_mod.get_cli_solo_media_type() == "audio"
        # User hasn't touched their setting -> CLI value wins.
        assert settings_mod.get_solo_media_type_explicit() is False
        assert settings_mod.get_effective_solo_media_type() == "audio"

    def test_explicit_user_choice_overrides_cli_fallback(self):
        settings_mod.set_cli_solo_media_type("audio")
        settings_mod.apply_user_solo_media_type("image")
        assert settings_mod.get_effective_solo_media_type() == "image"

    def test_user_can_opt_out_against_cli_fallback(self):
        # CLI says solo=image, but user explicitly picks "show everything".
        settings_mod.set_cli_solo_media_type("image")
        settings_mod.apply_user_solo_media_type(None)
        assert settings_mod.get_effective_solo_media_type() is None
        # The CLI value is still recorded; only the resolver hides it.
        assert settings_mod.get_cli_solo_media_type() == "image"

    def test_cli_fallback_clears_to_none(self):
        settings_mod.set_cli_solo_media_type("video")
        settings_mod.set_cli_solo_media_type(None)
        assert settings_mod.get_cli_solo_media_type() is None
        assert settings_mod.get_effective_solo_media_type() is None

    def test_cli_fallback_strips_whitespace(self):
        settings_mod.set_cli_solo_media_type("  ")
        assert settings_mod.get_cli_solo_media_type() is None


class TestSettingsApiRoute:
    def test_get_settings_includes_effective_solo_media_type(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        body = res.get_json()
        assert "effective_solo_media_type" in body
        assert body["effective_solo_media_type"] is None

    def test_put_solo_media_type_via_api(self, client):
        res = client.put("/api/settings", json={"solo_media_type": "image"})
        assert res.status_code == 200
        body = res.get_json()
        assert body["solo_media_type"] == "image"
        assert body["solo_media_type_explicit"] is True
        assert body["effective_solo_media_type"] == "image"

    def test_put_solo_media_type_null_clears(self, client):
        # First set it, then clear it explicitly.
        client.put("/api/settings", json={"solo_media_type": "image"})
        res = client.put("/api/settings", json={"solo_media_type": None})
        assert res.status_code == 200
        body = res.get_json()
        assert body["solo_media_type"] is None
        # Still marked explicit so a future CLI fallback wouldn't reapply.
        assert body["solo_media_type_explicit"] is True
        assert body["effective_solo_media_type"] is None

    def test_put_invalid_solo_media_type_returns_400(self, client):
        res = client.put("/api/settings", json={"solo_media_type": "garbage"})
        assert res.status_code == 400
        body = res.get_json()
        assert "garbage" in str(body)

    def test_get_returns_cli_fallback_when_user_not_explicit(self, client):
        settings_mod.set_cli_solo_media_type("audio")
        res = client.get("/api/settings")
        assert res.status_code == 200
        body = res.get_json()
        assert body["effective_solo_media_type"] == "audio"
        # The raw per-user value still shows None / not-explicit.
        assert body["solo_media_type"] is None
        assert body["solo_media_type_explicit"] is False

    def test_effective_solo_media_type_is_read_only(self, client):
        # Sending effective_solo_media_type in PUT body is silently ignored.
        res = client.put(
            "/api/settings",
            json={"effective_solo_media_type": "image"},
        )
        assert res.status_code == 200
        body = res.get_json()
        # Did not mutate the raw value.
        assert body["solo_media_type"] is None
        assert body["solo_media_type_explicit"] is False


class TestPreloadIntegration:
    def test_predict_includes_extra_media_type_default_embedder(self):
        from vtscore.embedding.loader import predict_embedders_to_preload
        from vtscore.media import embedders_for_type

        # With empty registries, predict yields nothing.
        assert predict_embedders_to_preload() == []

        # Asking for an image extra should yield the image default embedder.
        opts = embedders_for_type("image")
        if not opts:
            pytest.skip("No image embedder registered")
        default_image_embedder = opts[0].name

        result = predict_embedders_to_preload(extra_media_types=["image"])
        assert default_image_embedder in result
        # Extras come first in the predicted order.
        assert result[0] == default_image_embedder
