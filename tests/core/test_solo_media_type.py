"""Tests for the solo-mediaType server restriction and its CLI override.

Covers the server-tier ``solo_media_type`` setting, the process-level CLI
override (``set_cli_solo_media_type``), the resolver
(``get_effective_solo_media_type``), the settings route (which surfaces the
effective value read-only and refuses to change it), and the preload
extension (``preload_predicted_embedders(extra_media_types=...)``).
"""

from __future__ import annotations

import json

import pytest

import app as app_module  # noqa: F401  (triggers conftest media init)
from vtsearch import settings as settings_mod


@pytest.fixture(autouse=True)
def _reset_cli_solo():
    """The CLI override is process-global; clear it around each test."""
    settings_mod.set_cli_solo_media_type(None)
    yield
    settings_mod.set_cli_solo_media_type(None)


class TestSoloMediaTypeSettings:
    def test_default_is_none(self):
        assert settings_mod.get_solo_media_type() is None
        assert settings_mod.get_effective_solo_media_type() is None

    def test_is_a_server_tier_setting(self):
        assert "solo_media_type" in settings_mod._SERVER_KEYS

    def test_persisted_value_is_effective(self, isolated_settings):
        settings_mod.set_solo_media_type("image")
        assert settings_mod.get_effective_solo_media_type() == "image"

        # Persisted to the server settings file, not a per-user file.
        raw = json.loads(settings_mod.SETTINGS_PATH.read_text())
        assert raw["solo_media_type"] == "image"

    def test_persisted_empty_string_normalises_to_none(self):
        settings_mod.set_solo_media_type("   ")
        assert settings_mod.get_effective_solo_media_type() is None


class TestCliOverride:
    def test_cli_override_applies_with_no_persisted_value(self):
        settings_mod.set_cli_solo_media_type("audio")
        assert settings_mod.get_cli_solo_media_type() == "audio"
        assert settings_mod.get_effective_solo_media_type() == "audio"

    def test_cli_override_wins_over_persisted_value(self):
        settings_mod.set_solo_media_type("image")
        settings_mod.set_cli_solo_media_type("audio")
        assert settings_mod.get_effective_solo_media_type() == "audio"
        # The persisted value is untouched; only the resolver prefers the CLI.
        assert settings_mod.get_solo_media_type() == "image"

    def test_clearing_cli_override_falls_back_to_persisted(self):
        settings_mod.set_solo_media_type("image")
        settings_mod.set_cli_solo_media_type("video")
        settings_mod.set_cli_solo_media_type(None)
        assert settings_mod.get_cli_solo_media_type() is None
        assert settings_mod.get_effective_solo_media_type() == "image"

    def test_cli_override_strips_whitespace(self):
        settings_mod.set_cli_solo_media_type("  ")
        assert settings_mod.get_cli_solo_media_type() is None
        assert settings_mod.get_effective_solo_media_type() is None


class TestSettingsApiRoute:
    def test_get_settings_includes_solo_media_type(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        body = res.get_json()
        assert "solo_media_type" in body
        assert body["solo_media_type"] is None

    def test_get_returns_cli_override(self, client):
        settings_mod.set_cli_solo_media_type("audio")
        res = client.get("/api/settings")
        assert res.status_code == 200
        assert res.get_json()["solo_media_type"] == "audio"

    def test_get_returns_persisted_server_value(self, client):
        settings_mod.set_solo_media_type("image")
        res = client.get("/api/settings")
        assert res.status_code == 200
        assert res.get_json()["solo_media_type"] == "image"

    def test_put_solo_media_type_is_ignored(self, client):
        """It is admin-set: PUT can neither set nor clear it."""
        res = client.put("/api/settings", json={"solo_media_type": "image"})
        assert res.status_code == 200
        assert res.get_json()["solo_media_type"] is None
        assert settings_mod.get_solo_media_type() is None

    def test_put_cannot_clear_the_server_restriction(self, client):
        settings_mod.set_solo_media_type("image")
        res = client.put("/api/settings", json={"solo_media_type": None})
        assert res.status_code == 200
        assert res.get_json()["solo_media_type"] == "image"
        assert settings_mod.get_effective_solo_media_type() == "image"

    def test_solo_media_type_is_not_in_the_update_schema(self):
        from vtsearch.schemas.settings import SettingsUpdateSchema

        assert "solo_media_type" not in SettingsUpdateSchema().fields


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
