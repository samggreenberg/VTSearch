"""Tests for the server-tier ``browse_signpost_vocab`` setting.

The custom signpost tag vocabulary is a deployment-level choice an operator
makes for a whole instance (a domain taxonomy every user of the server should
see on the map), not a per-user preference. It therefore lives on
:class:`~vtsearch.settings_models.ServerSettings`, is read-only over the API,
and is set by editing the server settings file.
"""

from __future__ import annotations

import vtsearch.settings as settings_mod
from vtsearch.schemas.settings import AppSettingsSchema, SettingsUpdateSchema
from vtsearch.settings_models import SIGNPOST_VOCAB_MAX_TERMS, ServerSettings, UserSettings


class TestTier:
    def test_lives_on_the_server_model(self):
        assert "browse_signpost_vocab" in ServerSettings.model_fields
        assert "browse_signpost_vocab" not in UserSettings.model_fields

    def test_routes_to_the_server_tier(self):
        assert "browse_signpost_vocab" in settings_mod._SERVER_KEYS

    def test_writes_land_in_the_server_settings_file(self, isolated_settings):
        settings_mod.set_browse_signpost_vocab({"audio": ["dog barking", "rain"]})
        assert settings_mod.get_browse_signpost_vocab() == {"audio": ["dog barking", "rain"]}
        # Same value for every user, since it is not per-user state.
        assert settings_mod.get_all()["browse_signpost_vocab"] == {"audio": ["dog barking", "rain"]}


class TestNormalization:
    def test_trims_dedupes_and_drops_empty_types(self, isolated_settings):
        # Whitespace trimmed, blanks and duplicates dropped (order preserved),
        # and a media type left with no usable terms is omitted entirely.
        settings_mod.set_browse_signpost_vocab({"audio": ["  dog  ", "dog", "", "  ", "rain"], "image": ["   "]})
        stored = settings_mod.get_browse_signpost_vocab()
        assert stored["audio"] == ["dog", "rain"]
        assert "image" not in stored

    def test_caps_term_count(self, isolated_settings):
        # Each term costs one text embed at build time, so the list is capped.
        settings_mod.set_browse_signpost_vocab({"image": [f"term{i}" for i in range(SIGNPOST_VOCAB_MAX_TERMS + 50)]})
        assert len(settings_mod.get_browse_signpost_vocab()["image"]) == SIGNPOST_VOCAB_MAX_TERMS


class TestApiContract:
    def test_not_settable_via_put(self, client, isolated_settings):
        """A PUT body carrying browse_signpost_vocab is silently ignored:
        the update schema excludes it, so it never reaches a setter."""
        settings_mod.set_browse_signpost_vocab({"audio": ["rain"]})
        resp = client.put("/api/settings", json={"browse_signpost_vocab": {"audio": ["hijacked"]}})
        assert resp.status_code == 200
        assert settings_mod.get_browse_signpost_vocab() == {"audio": ["rain"]}
        assert resp.get_json()["browse_signpost_vocab"] == {"audio": ["rain"]}

    def test_get_reports_the_configured_vocabulary(self, client, isolated_settings):
        settings_mod.set_browse_signpost_vocab({"image": ["cat", "car"]})
        body = client.get("/api/settings").get_json()
        assert body["browse_signpost_vocab"] == {"image": ["cat", "car"]}

    def test_schema_marks_it_read_only(self):
        # Dumpable (so the Server settings tab can report it) but not loadable
        # (set in the server settings file, never PUT).
        app_field = AppSettingsSchema().fields["browse_signpost_vocab"]
        assert app_field.dump_only is True
        assert "browse_signpost_vocab" not in SettingsUpdateSchema().fields

    def test_stale_per_user_copy_does_not_shadow_the_server_value(self, client, isolated_settings):
        """A user file written before the tier move still carries the key.

        ``get_all`` layers the per-user cache over the server one, so such a
        leftover would be reported by GET while projection builds (which read
        the accessor) used the operator's value. The load-time sanitizer drops
        server-tier keys out of a per-user file, so the stale copy never
        reaches the cache in the first place (issue #3413).
        """
        import json as _json

        settings_mod.set_browse_signpost_vocab({"audio": ["rain"]})
        isolated_settings._user.parent.mkdir(parents=True, exist_ok=True)
        isolated_settings._user.write_text(_json.dumps({"browse_signpost_vocab": {"audio": ["stale"]}}) + "\n")
        settings_mod.reset()  # re-read both tiers from disk

        assert client.get("/api/settings").get_json()["browse_signpost_vocab"] == {"audio": ["rain"]}
        assert settings_mod.get_browse_signpost_vocab() == {"audio": ["rain"]}
