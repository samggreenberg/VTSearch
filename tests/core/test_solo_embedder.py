"""Tests for the per-mediaType solo-embedder streamlining setting.

Mirrors the structure of :mod:`tests.core.test_solo_media_type`: covers the
per-user ``solo_embedder_per_media_type`` dict, the process-level CLI
fallback (:func:`set_cli_solo_embedder`), the resolver
(:func:`get_effective_solo_embedders`), the route handler
(``PUT /api/settings``), and the preload extension
(``extra_embedders`` argument to :func:`predict_embedders_to_preload`).
"""

from __future__ import annotations

import json

import pytest

import app as app_module  # noqa: F401  (triggers conftest media init)
from vtsearch import settings as settings_mod


def _first_image_embedder() -> str:
    from vtscore.media import embedders_for_type

    opts = embedders_for_type("image")
    if not opts:
        pytest.skip("No image embedder registered")
    return opts[0].name


def _first_audio_embedder() -> str:
    from vtscore.media import embedders_for_type

    opts = embedders_for_type("audio")
    if not opts:
        pytest.skip("No audio embedder registered")
    return opts[0].name


@pytest.fixture(autouse=True)
def _reset_cli_solo_embedders():
    """The CLI fallback is process-global; clear it around each test."""
    settings_mod._cli_solo_embedders.clear()
    yield
    settings_mod._cli_solo_embedders.clear()


class TestSoloEmbedderSettings:
    def test_default_is_empty(self):
        assert settings_mod.get_solo_embedder_per_media_type() == {}
        assert settings_mod.get_effective_solo_embedders() == {}
        assert settings_mod.get_effective_solo_embedder("image") is None

    def test_apply_user_value_persists(self, isolated_settings):
        emb = _first_image_embedder()
        settings_mod.apply_user_solo_embedder_per_media_type({"image": emb})
        assert settings_mod.get_solo_embedder_per_media_type() == {"image": emb}
        assert settings_mod.get_effective_solo_embedder("image") == emb

        raw = json.loads(isolated_settings.read_text())
        assert raw["solo_embedder_per_media_type"] == {"image": emb}

    def test_apply_none_clears_every_entry(self):
        emb = _first_image_embedder()
        settings_mod.apply_user_solo_embedder_per_media_type({"image": emb})
        settings_mod.apply_user_solo_embedder_per_media_type(None)
        assert settings_mod.get_solo_embedder_per_media_type() == {}
        assert settings_mod.get_effective_solo_embedders() == {}

    def test_empty_value_preserved_as_opt_out_sentinel(self):
        emb = _first_image_embedder()
        settings_mod.apply_user_solo_embedder_per_media_type({"image": emb, "audio": ""})
        # The empty entry is kept so it can override a CLI fallback for
        # ``audio``; see ``test_user_can_opt_out_of_cli_fallback_for_one_type``.
        assert settings_mod.get_solo_embedder_per_media_type() == {"image": emb, "audio": ""}
        # The effective resolver strips the empty entry; only non-empty
        # values appear in the merged map.
        assert settings_mod.get_effective_solo_embedders() == {"image": emb}


class TestCliSoloEmbedderFallback:
    def test_cli_fallback_applies_when_user_has_no_entry(self):
        emb = _first_image_embedder()
        settings_mod.set_cli_solo_embedder("image", emb)
        assert settings_mod.get_cli_solo_embedders() == {"image": emb}
        # User hasn't touched their setting -> CLI value wins.
        assert settings_mod.get_effective_solo_embedder("image") == emb

    def test_user_entry_overrides_cli_for_same_type(self):
        first = _first_image_embedder()
        from vtscore.media import embedders_for_type

        opts = embedders_for_type("image")
        if len(opts) < 2:
            pytest.skip("Need two image embedders to test override")
        second = opts[1].name
        settings_mod.set_cli_solo_embedder("image", first)
        settings_mod.apply_user_solo_embedder_per_media_type({"image": second})
        assert settings_mod.get_effective_solo_embedder("image") == second

    def test_cli_and_user_entries_for_different_types_both_apply(self):
        img = _first_image_embedder()
        aud = _first_audio_embedder()
        settings_mod.set_cli_solo_embedder("audio", aud)
        settings_mod.apply_user_solo_embedder_per_media_type({"image": img})
        eff = settings_mod.get_effective_solo_embedders()
        assert eff == {"image": img, "audio": aud}

    def test_user_can_opt_out_of_cli_fallback_for_one_type(self):
        # CLI pins image=X but user explicitly clears it; audio CLI value
        # still applies because the user said nothing about audio.
        img = _first_image_embedder()
        aud = _first_audio_embedder()
        settings_mod.set_cli_solo_embedder("image", img)
        settings_mod.set_cli_solo_embedder("audio", aud)
        settings_mod.apply_user_solo_embedder_per_media_type({"image": ""})
        eff = settings_mod.get_effective_solo_embedders()
        assert "image" not in eff
        assert eff.get("audio") == aud

    def test_set_cli_solo_embedder_empty_clears(self):
        emb = _first_image_embedder()
        settings_mod.set_cli_solo_embedder("image", emb)
        settings_mod.set_cli_solo_embedder("image", None)
        assert settings_mod.get_cli_solo_embedders() == {}


class TestSettingsApiRoute:
    def test_get_includes_effective_map(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        body = res.get_json()
        assert body["effective_solo_embedder_per_media_type"] == {}

    def test_put_persists_and_returns_effective_map(self, client):
        emb = _first_image_embedder()
        res = client.put("/api/settings", json={"solo_embedder_per_media_type": {"image": emb}})
        assert res.status_code == 200
        body = res.get_json()
        assert body["solo_embedder_per_media_type"] == {"image": emb}
        assert body["effective_solo_embedder_per_media_type"] == {"image": emb}

    def test_put_clears_with_empty_dict(self, client):
        emb = _first_image_embedder()
        client.put("/api/settings", json={"solo_embedder_per_media_type": {"image": emb}})
        res = client.put("/api/settings", json={"solo_embedder_per_media_type": {}})
        assert res.status_code == 200
        body = res.get_json()
        assert body["solo_embedder_per_media_type"] == {}
        assert body["effective_solo_embedder_per_media_type"] == {}

    def test_put_clears_with_null(self, client):
        emb = _first_image_embedder()
        client.put("/api/settings", json={"solo_embedder_per_media_type": {"image": emb}})
        res = client.put("/api/settings", json={"solo_embedder_per_media_type": None})
        assert res.status_code == 200
        body = res.get_json()
        assert body["solo_embedder_per_media_type"] == {}

    def test_put_rejects_unknown_media_type(self, client):
        emb = _first_image_embedder()
        res = client.put(
            "/api/settings",
            json={"solo_embedder_per_media_type": {"garbage": emb}},
        )
        assert res.status_code == 400
        assert "garbage" in str(res.get_json())

    def test_put_rejects_unknown_embedder(self, client):
        res = client.put(
            "/api/settings",
            json={"solo_embedder_per_media_type": {"image": "not-a-real-embedder"}},
        )
        assert res.status_code == 400
        assert "not-a-real-embedder" in str(res.get_json())

    def test_put_rejects_embedder_for_wrong_type(self, client):
        aud = _first_audio_embedder()
        res = client.put(
            "/api/settings",
            json={"solo_embedder_per_media_type": {"image": aud}},
        )
        assert res.status_code == 400

    def test_effective_map_is_read_only(self, client):
        emb = _first_image_embedder()
        # Sending effective_solo_embedder_per_media_type is silently ignored.
        res = client.put(
            "/api/settings",
            json={"effective_solo_embedder_per_media_type": {"image": emb}},
        )
        assert res.status_code == 200
        body = res.get_json()
        assert body["solo_embedder_per_media_type"] == {}
        assert body["effective_solo_embedder_per_media_type"] == {}

    def test_get_returns_cli_fallback_when_user_has_no_entry(self, client):
        emb = _first_image_embedder()
        settings_mod.set_cli_solo_embedder("image", emb)
        res = client.get("/api/settings")
        body = res.get_json()
        assert body["effective_solo_embedder_per_media_type"] == {"image": emb}
        # The raw per-user value remains untouched.
        assert body["solo_embedder_per_media_type"] == {}


class TestPreloadIntegration:
    def test_predict_includes_extra_embedders(self):
        from vtscore.embedding.loader import predict_embedders_to_preload

        emb = _first_image_embedder()
        # With empty registries and no extras, nothing is predicted.
        assert predict_embedders_to_preload() == []

        result = predict_embedders_to_preload(extra_embedders=[emb])
        assert emb in result
        # Extras come first (extras list precedes media-type fallbacks).
        assert result[0] == emb

    def test_predict_skips_unknown_extra_embedders(self):
        from vtscore.embedding.loader import predict_embedders_to_preload

        result = predict_embedders_to_preload(extra_embedders=["does-not-exist"])
        assert result == []
