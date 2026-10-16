"""Directory path settings tests.

Covers saved_datasets_dir and detectors_dir settings: their accessors, and
that the API reports them but does not write them.
"""

from __future__ import annotations

import json

import pytest

from vtsearch import settings as settings_mod

# ---------------------------------------------------------------------------
# Directory path settings
# ---------------------------------------------------------------------------


class TestDirectorySettings:
    def test_saved_datasets_dir_default(self):
        path = settings_mod.get_saved_datasets_dir()
        assert str(path).endswith("saved_datasets")

    def test_detectors_dir_default(self):
        path = settings_mod.get_detectors_dir()
        assert str(path).endswith("detectors")

    def test_set_saved_datasets_dir(self, isolated_settings):
        settings_mod.set_saved_datasets_dir("/tmp/my_datasets")
        assert str(settings_mod.get_saved_datasets_dir()) == "/tmp/my_datasets"

        raw = json.loads(isolated_settings.read_text())
        assert raw["saved_datasets_dir"] == "/tmp/my_datasets"

    def test_set_detectors_dir(self, isolated_settings):
        settings_mod.set_detectors_dir("/tmp/my_models")
        assert str(settings_mod.get_detectors_dir()) == "/tmp/my_models"

        raw = json.loads(isolated_settings.read_text())
        assert raw["detectors_dir"] == "/tmp/my_models"

    def test_directory_settings_persist_across_reset(self, isolated_settings):
        settings_mod.set_saved_datasets_dir("/tmp/persist_test")
        settings_mod.reset()
        assert str(settings_mod.get_saved_datasets_dir()) == "/tmp/persist_test"

    def test_get_all_includes_directory_settings(self):
        data = settings_mod.get_all()
        assert "saved_datasets_dir" in data
        assert "detectors_dir" in data


class TestDirectorySettingsAPI:
    def test_get_settings_includes_dirs(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "saved_datasets_dir" in data
        assert "detectors_dir" in data

    @pytest.mark.parametrize("key", ["saved_datasets_dir", "detectors_dir"])
    def test_put_leaves_dir_unchanged(self, client, key):
        """Server-tier storage paths are the operator's, set in the settings
        file; PUT drops them like any key it does not accept (#4965)."""
        before = client.get("/api/settings").get_json()[key]
        res = client.put("/api/settings", json={key: "/tmp/elsewhere", "volume": 0.4})
        assert res.status_code == 200
        assert res.get_json()[key] == before
        assert res.get_json()["volume"] == 0.4
