"""Tests for the server-tier ``support_email`` contact setting.

Covers the persisted server-tier value, the process-level override set by
``--support-email`` / ``VTSEARCH_SUPPORT_EMAIL``
(``set_cli_support_email``), the resolver (``get_effective_support_email``),
and the API contract (read-only: exposed in ``GET /api/settings`` but not
settable via ``PUT``). This is the address the Help modal's "Email us" link
is pre-addressed to (issue #2357).
"""

from __future__ import annotations

import pytest

import app as app_module  # noqa: F401  (triggers conftest media init)
from vtsearch import settings as settings_mod
from vtsearch.schemas.settings import AppSettingsSchema, SettingsUpdateSchema
from vtsearch.settings_models import DEFAULT_SUPPORT_EMAIL


@pytest.fixture(autouse=True)
def _reset_cli_support_email():
    """The CLI/env override is process-global; clear it around each test."""
    settings_mod.set_cli_support_email(None)
    yield
    settings_mod.set_cli_support_email(None)


class TestEffectiveResolution:
    def test_default_is_project_address(self):
        assert settings_mod.get_effective_support_email() == DEFAULT_SUPPORT_EMAIL
        assert DEFAULT_SUPPORT_EMAIL == "sam.greenberg@gmail.com"

    def test_cli_override_wins_over_persisted(self, isolated_settings):
        settings_mod.set_support_email("persisted@example.com")
        assert settings_mod.get_support_email() == "persisted@example.com"
        settings_mod.set_cli_support_email("cli@example.com")
        assert settings_mod.get_effective_support_email() == "cli@example.com"

    def test_falls_back_to_persisted_when_no_override(self, isolated_settings):
        settings_mod.set_support_email("ops@example.com")
        assert settings_mod.get_effective_support_email() == "ops@example.com"

    def test_override_clears_back_to_persisted(self, isolated_settings):
        settings_mod.set_support_email("ops@example.com")
        settings_mod.set_cli_support_email("cli@example.com")
        settings_mod.set_cli_support_email(None)
        assert settings_mod.get_effective_support_email() == "ops@example.com"

    def test_blank_override_is_treated_as_unset(self, isolated_settings):
        settings_mod.set_support_email("ops@example.com")
        settings_mod.set_cli_support_email("   ")
        assert settings_mod.get_cli_support_email() is None
        assert settings_mod.get_effective_support_email() == "ops@example.com"


class TestApiContract:
    def test_not_settable_via_put(self, client, isolated_settings):
        """A PUT body carrying support_email is silently ignored:
        the schema excludes it, so it never reaches a setter."""
        settings_mod.set_support_email("ops@example.com")
        resp = client.put("/api/settings", json={"support_email": "attacker@example.com"})
        assert resp.status_code == 200
        assert settings_mod.get_support_email() == "ops@example.com"

    def test_get_reflects_effective_value(self, client):
        settings_mod.set_cli_support_email("cli@example.com")
        body = client.get("/api/settings").get_json()
        assert body["support_email"] == "cli@example.com"

    def test_get_reflects_default(self, client):
        body = client.get("/api/settings").get_json()
        assert body["support_email"] == DEFAULT_SUPPORT_EMAIL

    def test_schema_marks_it_read_only(self):
        # Dumpable (so the Help modal can address its mailto link) but not
        # loadable (set via CLI/env/file, never PUT).
        app_field = AppSettingsSchema().fields["support_email"]
        assert app_field.dump_only is True
        assert "support_email" not in SettingsUpdateSchema().fields
