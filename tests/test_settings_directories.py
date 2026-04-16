"""Directory path settings tests.

Covers saved_datasets_dir, detectors_dir, trainable_models_dir settings
and their corresponding API endpoints.
"""

from __future__ import annotations

import json

import app as app_module  # noqa: F401 — triggers conftest media init
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

    def test_trainable_models_dir_default(self):
        path = settings_mod.get_trainable_models_dir()
        assert str(path).endswith("trainable_models")

    def test_set_saved_datasets_dir(self, isolated_settings):
        settings_mod.set_saved_datasets_dir("/tmp/my_datasets")
        assert str(settings_mod.get_saved_datasets_dir()) == "/tmp/my_datasets"

        raw = json.loads(isolated_settings.read_text())
        assert raw["saved_datasets_dir"] == "/tmp/my_datasets"

    def test_set_detectors_dir(self, isolated_settings):
        settings_mod.set_detectors_dir("/tmp/my_detectors")
        assert str(settings_mod.get_detectors_dir()) == "/tmp/my_detectors"

        raw = json.loads(isolated_settings.read_text())
        assert raw["detectors_dir"] == "/tmp/my_detectors"

    def test_set_trainable_models_dir(self, isolated_settings):
        settings_mod.set_trainable_models_dir("/tmp/my_models")
        assert str(settings_mod.get_trainable_models_dir()) == "/tmp/my_models"

        raw = json.loads(isolated_settings.read_text())
        assert raw["trainable_models_dir"] == "/tmp/my_models"

    def test_directory_settings_persist_across_reset(self, isolated_settings):
        settings_mod.set_saved_datasets_dir("/tmp/persist_test")
        settings_mod.reset()
        assert str(settings_mod.get_saved_datasets_dir()) == "/tmp/persist_test"

    def test_get_all_includes_directory_settings(self):
        data = settings_mod.get_all()
        assert "saved_datasets_dir" in data
        assert "detectors_dir" in data
        assert "trainable_models_dir" in data


class TestDirectorySettingsAPI:
    def test_get_settings_includes_dirs(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "saved_datasets_dir" in data
        assert "detectors_dir" in data
        assert "trainable_models_dir" in data

    def test_update_saved_datasets_dir(self, client):
        res = client.put("/api/settings", json={"saved_datasets_dir": "/tmp/new_ds"})
        assert res.status_code == 200
        assert res.get_json()["saved_datasets_dir"] == "/tmp/new_ds"

        res2 = client.get("/api/settings")
        assert res2.get_json()["saved_datasets_dir"] == "/tmp/new_ds"

    def test_update_detectors_dir(self, client):
        res = client.put("/api/settings", json={"detectors_dir": "/tmp/new_det"})
        assert res.status_code == 200
        assert res.get_json()["detectors_dir"] == "/tmp/new_det"

    def test_update_trainable_models_dir(self, client):
        res = client.put("/api/settings", json={"trainable_models_dir": "/tmp/new_tm"})
        assert res.status_code == 200
        assert res.get_json()["trainable_models_dir"] == "/tmp/new_tm"

    def test_update_dir_empty_string_rejected(self, client):
        res = client.put("/api/settings", json={"saved_datasets_dir": ""})
        assert res.status_code == 400

    def test_update_dir_whitespace_only_rejected(self, client):
        res = client.put("/api/settings", json={"detectors_dir": "   "})
        assert res.status_code == 400

    def test_update_dir_non_string_rejected(self, client):
        res = client.put("/api/settings", json={"trainable_models_dir": 123})
        assert res.status_code == 400

