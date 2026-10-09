"""Tests for the server-tier ``autopilot_only`` lock (issue #4666).

Covers the persisted setting and the API contract: read-only, reported by
``GET /api/settings`` (which the SPA reads to drop Train's Manual tab and
Test's Review tab) and ignored by ``PUT``. The flag / env forms and their
precedence are covered with the other overrides in ``test_admin_overrides.py``.
"""

from __future__ import annotations

from vtsearch import settings as settings_mod


class TestEffectiveResolution:
    def test_default_is_off(self, isolated_settings):
        assert settings_mod.get_effective_autopilot_only() is False

    def test_persisted_setting_turns_it_on(self, isolated_settings):
        settings_mod.set_autopilot_only(True)
        assert settings_mod.get_cli_autopilot_only() is None
        assert settings_mod.get_effective_autopilot_only() is True


class TestApiContract:
    def test_not_settable_via_put(self, client, isolated_settings):
        resp = client.put("/api/settings", json={"autopilot_only": True})
        assert resp.status_code == 200
        assert settings_mod.get_autopilot_only() is False

    def test_get_reports_the_effective_value(self, client, isolated_settings):
        assert client.get("/api/settings").get_json()["autopilot_only"] is False
        settings_mod.set_autopilot_only(True)
        assert client.get("/api/settings").get_json()["autopilot_only"] is True
