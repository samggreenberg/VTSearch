"""Tests for the server-tier ``hide_ingest_eta`` switch (issue #4233).

Covers the persisted setting, its path into the library tier
(``CoreConfig.hide_ingest_eta`` → ``ingest_eta_hidden()``, which the ingest
paths turn into ``publish_eta=False``), and the API contract: read-only,
reported by ``GET /api/settings`` and ignored by ``PUT``. The flag / env forms
and their precedence are covered with the other overrides in
``test_admin_overrides.py``.
"""

from __future__ import annotations

from vtscore.concurrency.progress import ingest_eta_hidden
from vtscore.config import CoreConfig
from vtsearch import settings as settings_mod


class TestEffectiveResolution:
    def test_default_is_off(self, isolated_settings):
        assert settings_mod.get_effective_hide_ingest_eta() is False

    def test_persisted_setting_turns_it_on(self, isolated_settings):
        settings_mod.set_hide_ingest_eta(True)
        assert settings_mod.get_cli_hide_ingest_eta() is None
        assert settings_mod.get_effective_hide_ingest_eta() is True


class TestLibraryTier:
    def test_reaches_core_config(self, isolated_settings):
        assert CoreConfig.from_settings().hide_ingest_eta is False
        settings_mod.set_hide_ingest_eta(True)
        assert CoreConfig.from_settings().hide_ingest_eta is True

    def test_drives_the_ingest_policy(self, isolated_settings):
        assert ingest_eta_hidden() is False
        settings_mod.set_hide_ingest_eta(True)
        assert ingest_eta_hidden() is True


class TestApiContract:
    def test_not_settable_via_put(self, client, isolated_settings):
        resp = client.put("/api/settings", json={"hide_ingest_eta": True})
        assert resp.status_code == 200
        assert settings_mod.get_hide_ingest_eta() is False

    def test_get_reports_the_effective_value(self, client, isolated_settings):
        assert client.get("/api/settings").get_json()["hide_ingest_eta"] is False
        settings_mod.set_hide_ingest_eta(True)
        assert client.get("/api/settings").get_json()["hide_ingest_eta"] is True
