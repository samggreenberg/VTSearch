"""Tests for the server-tier ``dataset_max_age_days`` retention setting.

Covers the persisted server-tier value, the process-level override set by
``--dataset-max-age-days`` / ``VTSEARCH_DATASET_MAX_AGE_DAYS``
(``set_cli_dataset_max_age_days``), the resolver
(``get_effective_dataset_max_age_days``), and the API contract (read-only:
exposed in ``GET /api/settings`` but not settable via ``PUT``).
"""

from __future__ import annotations

import pytest

from vtsearch import settings as settings_mod
from vtsearch.schemas.settings import AppSettingsSchema, SettingsUpdateSchema


@pytest.fixture(autouse=True)
def _reset_cli_max_age():
    """The CLI/env override is process-global; clear it around each test."""
    settings_mod.set_cli_dataset_max_age_days(None)
    yield
    settings_mod.set_cli_dataset_max_age_days(None)


class TestEffectiveResolution:
    def test_default_is_none_never_expires(self):
        assert settings_mod.get_effective_dataset_max_age_days() is None

    def test_cli_override_wins_over_persisted(self, isolated_settings):
        settings_mod.set_dataset_max_age_days(30)
        assert settings_mod.get_dataset_max_age_days() == 30
        settings_mod.set_cli_dataset_max_age_days(14)
        assert settings_mod.get_effective_dataset_max_age_days() == 14

    def test_falls_back_to_persisted_when_no_override(self, isolated_settings):
        settings_mod.set_dataset_max_age_days(7)
        assert settings_mod.get_effective_dataset_max_age_days() == 7

    def test_override_clears_to_none(self, isolated_settings):
        settings_mod.set_dataset_max_age_days(7)
        settings_mod.set_cli_dataset_max_age_days(14)
        settings_mod.set_cli_dataset_max_age_days(None)
        assert settings_mod.get_effective_dataset_max_age_days() == 7


class TestApiContract:
    def test_not_settable_via_put(self, client, isolated_settings):
        """A PUT body carrying dataset_max_age_days is silently ignored:
        the schema excludes it, so it never reaches a setter."""
        settings_mod.set_dataset_max_age_days(None)
        resp = client.put("/api/settings", json={"dataset_max_age_days": 99})
        assert resp.status_code == 200
        assert settings_mod.get_dataset_max_age_days() is None

    def test_get_reflects_effective_value(self, client):
        settings_mod.set_cli_dataset_max_age_days(14)
        body = client.get("/api/settings").get_json()
        assert body["dataset_max_age_days"] == 14

    def test_schema_marks_it_read_only(self):
        # Dumpable (so the dashboard can gate its Age-Off column) but not
        # loadable (set via CLI/env/file, never PUT).
        app_field = AppSettingsSchema().fields["dataset_max_age_days"]
        assert app_field.dump_only is True
        assert "dataset_max_age_days" not in SettingsUpdateSchema().fields
