"""Tests for the persistent settings module.

Covers:
- Settings file read/write (vtsearch.settings)
- Volume persistence
- Autorun processor recipes: add, remove, list, CLI command generation
- ensure_autorun_processors_imported (lazy import on autodetect)
- Flask API routes: GET/PUT /api/settings,
  GET/POST/DELETE /api/settings/autorun-processors
"""

from __future__ import annotations

import json

import pytest

import app as app_module  # noqa: F401 — triggers conftest media init
from vtsearch import settings as settings_mod


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Settings module unit tests
# ---------------------------------------------------------------------------


class TestSettingsModule:
    def test_defaults_when_no_file(self):
        data = settings_mod.get_all()
        assert data["volume"] == 1.0
        assert data["autorun_processors"] == []

    def test_get_set_volume(self, isolated_settings):
        settings_mod.set_volume(0.42)
        assert settings_mod.get_volume() == pytest.approx(0.42)

        # Persisted to disk
        raw = json.loads(isolated_settings.read_text())
        assert raw["volume"] == pytest.approx(0.42)

    def test_volume_clamped(self):
        settings_mod.set_volume(5.0)
        assert settings_mod.get_volume() == 1.0

        settings_mod.set_volume(-3.0)
        assert settings_mod.get_volume() == 0.0

    def test_get_set_inclusion(self, isolated_settings):
        settings_mod.set_inclusion(5)
        assert settings_mod.get_inclusion() == 5

        # Persisted to disk
        raw = json.loads(isolated_settings.read_text())
        assert raw["inclusion"] == 5

    def test_inclusion_clamped(self):
        settings_mod.set_inclusion(100)
        assert settings_mod.get_inclusion() == 10

        settings_mod.set_inclusion(-100)
        assert settings_mod.get_inclusion() == -10

    def test_inclusion_default(self):
        assert settings_mod.get_inclusion() == 0

    def test_inclusion_persists_across_reset(self, isolated_settings):
        settings_mod.set_inclusion(7)
        settings_mod.reset()
        assert settings_mod.get_inclusion() == 7

    def test_add_autorun_processor(self, isolated_settings):
        settings_mod.add_autorun_processor(
            "my det", "detector_file", {"file": "/tmp/det.json"}
        )
        procs = settings_mod.get_autorun_processors()
        assert len(procs) == 1
        assert procs[0]["processor_name"] == "my det"
        assert procs[0]["processor_importer"] == "detector_file"
        assert procs[0]["field_values"]["file"] == "/tmp/det.json"

    def test_add_overwrites_same_name(self):
        settings_mod.add_autorun_processor("a", "detector_file", {"file": "1.json"})
        settings_mod.add_autorun_processor("a", "detector_file", {"file": "2.json"})
        procs = settings_mod.get_autorun_processors()
        assert len(procs) == 1
        assert procs[0]["field_values"]["file"] == "2.json"

    def test_remove_autorun_processor(self):
        settings_mod.add_autorun_processor("x", "detector_file", {"file": "x.json"})
        assert settings_mod.remove_autorun_processor("x") is True
        assert settings_mod.get_autorun_processors() == []

    def test_remove_nonexistent(self):
        assert settings_mod.remove_autorun_processor("nope") is False

    def test_to_settings_json(self):
        entry = {
            "processor_name": "my detector",
            "processor_importer": "detector_file",
            "field_values": {"file": "/path/to/det.json"},
        }
        snippet = settings_mod.to_settings_json(entry)
        import json

        parsed = json.loads(snippet)
        assert parsed["processor_name"] == "my detector"
        assert parsed["processor_importer"] == "detector_file"
        assert parsed["field_values"]["file"] == "/path/to/det.json"

    def test_to_settings_json_with_spaces(self):
        entry = {
            "processor_name": "det",
            "processor_importer": "detector_file",
            "field_values": {"file": "/my path/det.json"},
        }
        snippet = settings_mod.to_settings_json(entry)
        import json

        parsed = json.loads(snippet)
        assert parsed["field_values"]["file"] == "/my path/det.json"

    def test_persistence_survives_reset(self, isolated_settings):
        settings_mod.set_volume(0.7)
        settings_mod.add_autorun_processor("p", "detector_file", {"file": "p.json"})

        # Simulate restart
        settings_mod.reset()

        assert settings_mod.get_volume() == pytest.approx(0.7)
        procs = settings_mod.get_autorun_processors()
        assert len(procs) == 1
        assert procs[0]["processor_name"] == "p"

    def test_get_set_calibrate_count(self, isolated_settings):
        settings_mod.set_calibrate_count(5)
        assert settings_mod.get_calibrate_count() == 5

        # Persisted to disk
        raw = json.loads(isolated_settings.read_text())
        assert raw["calibrate_count"] == 5

    def test_calibrate_count_clamped(self):
        settings_mod.set_calibrate_count(200)
        assert settings_mod.get_calibrate_count() == 100

        settings_mod.set_calibrate_count(0)
        assert settings_mod.get_calibrate_count() == 1

    def test_calibrate_count_default(self):
        assert settings_mod.get_calibrate_count() == 2

    def test_calibrate_count_persists_across_reset(self, isolated_settings):
        settings_mod.set_calibrate_count(10)
        settings_mod.reset()
        assert settings_mod.get_calibrate_count() == 10

    def test_get_set_safe_thresholds(self, isolated_settings):
        settings_mod.set_safe_thresholds(True)
        assert settings_mod.get_safe_thresholds() is True

        raw = json.loads(isolated_settings.read_text())
        assert raw["safe_thresholds"] is True

    def test_safe_thresholds_default(self):
        assert settings_mod.get_safe_thresholds() is False

    def test_get_set_show_thumbnails_left(self, isolated_settings):
        settings_mod.set_show_thumbnails_left(True)
        assert settings_mod.get_show_thumbnails_left() is True

        raw = json.loads(isolated_settings.read_text())
        assert raw["show_thumbnails_left"] is True

    def test_show_thumbnails_left_default(self):
        assert settings_mod.get_show_thumbnails_left() is False

    def test_show_thumbnails_left_persists_across_reset(self, isolated_settings):
        settings_mod.set_show_thumbnails_left(True)
        settings_mod.reset()
        assert settings_mod.get_show_thumbnails_left() is True

    def test_get_set_show_thumbnails_right(self, isolated_settings):
        settings_mod.set_show_thumbnails_right(False)
        assert settings_mod.get_show_thumbnails_right() is False

        raw = json.loads(isolated_settings.read_text())
        assert raw["show_thumbnails_right"] is False

    def test_show_thumbnails_right_default(self):
        assert settings_mod.get_show_thumbnails_right() is True

    def test_show_thumbnails_right_persists_across_reset(self, isolated_settings):
        settings_mod.set_show_thumbnails_right(False)
        settings_mod.reset()
        assert settings_mod.get_show_thumbnails_right() is False

    def test_get_defaults(self):
        defaults = settings_mod.get_defaults()
        assert defaults["volume"] == 1.0
        assert defaults["theme"] == "dark"
        assert defaults["calibrate_count"] == 2
        assert defaults["calibration_fraction"] == 0.5
        assert defaults["safe_thresholds"] is False
        assert defaults["show_thumbnails_left"] is False
        assert defaults["show_thumbnails_right"] is True
        assert defaults["autoload_media_types"] == []
        assert defaults["autopilot_top_greens"] == 3
        assert defaults["autopilot_hard_reds"] == 4
        assert "autorun_processors" not in defaults

    def test_get_set_autoload_media_types(self, isolated_settings):
        settings_mod.set_autoload_media_types(["audio", "video"])
        assert settings_mod.get_autoload_media_types() == ["audio", "video"]

        raw = json.loads(isolated_settings.read_text())
        assert raw["autoload_media_types"] == ["audio", "video"]

    def test_autoload_media_types_default(self):
        assert settings_mod.get_autoload_media_types() == []

    def test_autoload_media_types_invalid(self):
        with pytest.raises(ValueError):
            settings_mod.set_autoload_media_types(["invalid_type"])

    def test_autoload_media_types_clear(self):
        settings_mod.set_autoload_media_types(["video"])
        settings_mod.set_autoload_media_types([])
        assert settings_mod.get_autoload_media_types() == []

    def test_autoload_media_types_persists_across_reset(self, isolated_settings):
        settings_mod.set_autoload_media_types(["image", "audio"])
        settings_mod.reset()
        assert settings_mod.get_autoload_media_types() == ["image", "audio"]

    def test_autoload_media_types_all_valid_types(self):
        settings_mod.set_autoload_media_types(["audio", "image", "paragraph", "video"])
        assert settings_mod.get_autoload_media_types() == ["audio", "image", "paragraph", "video"]

    def test_toggle_autoload_media_type(self):
        result = settings_mod.toggle_autoload_media_type("audio")
        assert result == ["audio"]
        result = settings_mod.toggle_autoload_media_type("video")
        assert result == ["audio", "video"]
        result = settings_mod.toggle_autoload_media_type("audio")
        assert result == ["video"]

    def test_toggle_autoload_media_type_invalid(self):
        with pytest.raises(ValueError):
            settings_mod.toggle_autoload_media_type("invalid")

    def test_autoload_media_types_deduplicates(self):
        settings_mod.set_autoload_media_types(["audio", "audio", "video"])
        assert settings_mod.get_autoload_media_types() == ["audio", "video"]

    def test_get_set_autopilot_top_greens(self, isolated_settings):
        settings_mod.set_autopilot_top_greens(20)
        assert settings_mod.get_autopilot_top_greens() == 20

        raw = json.loads(isolated_settings.read_text())
        assert raw["autopilot_top_greens"] == 20

    def test_autopilot_top_greens_clamped(self):
        settings_mod.set_autopilot_top_greens(0)
        assert settings_mod.get_autopilot_top_greens() == 1

        settings_mod.set_autopilot_top_greens(-5)
        assert settings_mod.get_autopilot_top_greens() == 1

    def test_autopilot_top_greens_default(self):
        assert settings_mod.get_autopilot_top_greens() == 3

    def test_autopilot_top_greens_persists_across_reset(self, isolated_settings):
        settings_mod.set_autopilot_top_greens(25)
        settings_mod.reset()
        assert settings_mod.get_autopilot_top_greens() == 25

    def test_get_set_autopilot_hard_reds(self, isolated_settings):
        settings_mod.set_autopilot_hard_reds(15)
        assert settings_mod.get_autopilot_hard_reds() == 15

        raw = json.loads(isolated_settings.read_text())
        assert raw["autopilot_hard_reds"] == 15

    def test_autopilot_hard_reds_clamped(self):
        settings_mod.set_autopilot_hard_reds(0)
        assert settings_mod.get_autopilot_hard_reds() == 1

        settings_mod.set_autopilot_hard_reds(-5)
        assert settings_mod.get_autopilot_hard_reds() == 1

    def test_autopilot_hard_reds_default(self):
        assert settings_mod.get_autopilot_hard_reds() == 4

    def test_autopilot_hard_reds_persists_across_reset(self, isolated_settings):
        settings_mod.set_autopilot_hard_reds(30)
        settings_mod.reset()
        assert settings_mod.get_autopilot_hard_reds() == 30

    def test_corrupt_settings_file(self, isolated_settings):
        isolated_settings.write_text("not json!!!")
        settings_mod.reset()
        # Should fall back to defaults
        assert settings_mod.get_volume() == 1.0
        assert settings_mod.get_autorun_processors() == []


# ---------------------------------------------------------------------------
# ensure_autorun_processors_imported
# ---------------------------------------------------------------------------


class TestEnsureAutorunProcessorsImported:
    def test_imports_detector_file_processor(self, tmp_path):
        """An autorun processor recipe with detector_file should be imported."""
        from vtsearch.utils import autorun_detectors

        # Create a fake detector JSON
        det_weights = {
            "0.weight": [[0.1] * 512],
            "0.bias": [0.0],
            "2.weight": [[0.2]],
            "2.bias": [0.1],
        }
        det_path = tmp_path / "test_det.json"
        det_path.write_text(json.dumps({
            "media_type": "audio",
            "weights": det_weights,
            "threshold": 0.5,
        }))

        settings_mod.add_autorun_processor(
            "settings_test_det", "detector_file", {"file": str(det_path)}
        )

        imported = settings_mod.ensure_autorun_processors_imported()
        assert "settings_test_det" in imported
        assert "settings_test_det" in autorun_detectors

    def test_skips_already_imported(self, tmp_path):
        """If a detector with the same name already exists, skip it."""
        from vtsearch.utils import add_autorun_detector

        add_autorun_detector("existing", "audio", {"0.weight": [[0.1]], "0.bias": [0.0]}, 0.5)

        settings_mod.add_autorun_processor(
            "existing", "detector_file", {"file": "/nonexistent.json"}
        )

        imported = settings_mod.ensure_autorun_processors_imported()
        assert imported == []

    def test_handles_bad_importer_gracefully(self):
        """Unknown importer name should not crash."""
        settings_mod.add_autorun_processor(
            "bad_proc", "totally_fake_importer", {"file": "x.json"}
        )

        imported = settings_mod.ensure_autorun_processors_imported()
        assert imported == []


# ---------------------------------------------------------------------------
# Flask API routes
# ---------------------------------------------------------------------------


class TestSettingsAPI:
    def test_get_settings(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "volume" in data
        assert "autorun_processors" in data

    def test_update_volume(self, client):
        res = client.put(
            "/api/settings",
            json={"volume": 0.65},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["volume"] == pytest.approx(0.65)

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["volume"] == pytest.approx(0.65)

    def test_update_inclusion(self, client):
        res = client.put(
            "/api/settings",
            json={"inclusion": 5},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["inclusion"] == 5

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["inclusion"] == 5

    def test_update_inclusion_clamped(self, client):
        res = client.put("/api/settings", json={"inclusion": 99})
        assert res.status_code == 200
        assert res.get_json()["inclusion"] == 10

    def test_update_inclusion_invalid(self, client):
        res = client.put(
            "/api/settings",
            json={"inclusion": "not a number"},
        )
        assert res.status_code == 400

    def test_update_volume_invalid(self, client):
        res = client.put(
            "/api/settings",
            json={"volume": "not a number"},
        )
        assert res.status_code == 400

    def test_update_calibrate_count(self, client):
        res = client.put("/api/settings", json={"calibrate_count": 5})
        assert res.status_code == 200
        data = res.get_json()
        assert data["calibrate_count"] == 5

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["calibrate_count"] == 5

    def test_update_calibrate_count_clamped(self, client):
        res = client.put("/api/settings", json={"calibrate_count": 999})
        assert res.status_code == 200
        assert res.get_json()["calibrate_count"] == 100

    def test_update_calibrate_count_invalid(self, client):
        res = client.put("/api/settings", json={"calibrate_count": "not a number"})
        assert res.status_code == 400

    def test_update_empty_body(self, client):
        res = client.put(
            "/api/settings",
            data="",
            content_type="application/json",
        )
        assert res.status_code == 400

    def test_add_autorun_processor(self, client):
        res = client.post(
            "/api/settings/autorun-processors",
            json={
                "processor_name": "api_test",
                "processor_importer": "detector_file",
                "field_values": {"file": "/tmp/det.json"},
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["processor_name"] == "api_test"
        assert "settings_json" in data

    def test_add_autorun_processor_missing_name(self, client):
        res = client.post(
            "/api/settings/autorun-processors",
            json={"processor_importer": "detector_file", "field_values": {}},
        )
        assert res.status_code == 400

    def test_add_autorun_processor_missing_importer(self, client):
        res = client.post(
            "/api/settings/autorun-processors",
            json={"processor_name": "x", "field_values": {}},
        )
        assert res.status_code == 400

    def test_list_autorun_processors(self, client):
        # Add one first
        client.post(
            "/api/settings/autorun-processors",
            json={
                "processor_name": "list_test",
                "processor_importer": "detector_file",
                "field_values": {"file": "x.json"},
            },
        )

        res = client.get("/api/settings/autorun-processors")
        assert res.status_code == 200
        data = res.get_json()
        assert any(p["processor_name"] == "list_test" for p in data["autorun_processors"])

    def test_delete_autorun_processor(self, client):
        client.post(
            "/api/settings/autorun-processors",
            json={
                "processor_name": "del_test",
                "processor_importer": "detector_file",
                "field_values": {"file": "x.json"},
            },
        )

        res = client.delete("/api/settings/autorun-processors/del_test")
        assert res.status_code == 200

        # Verify it's gone
        res2 = client.get("/api/settings/autorun-processors")
        data = res2.get_json()
        assert not any(p["processor_name"] == "del_test" for p in data["autorun_processors"])

    def test_delete_nonexistent(self, client):
        res = client.delete("/api/settings/autorun-processors/nope")
        assert res.status_code == 404

    def test_get_settings_includes_settings_json(self, client):
        client.post(
            "/api/settings/autorun-processors",
            json={
                "processor_name": "cmd_test",
                "processor_importer": "detector_file",
                "field_values": {"file": "det.json"},
            },
        )

        res = client.get("/api/settings")
        data = res.get_json()
        proc = next(p for p in data["autorun_processors"] if p["processor_name"] == "cmd_test")
        import json

        parsed = json.loads(proc["settings_json"])
        assert parsed["processor_name"] == "cmd_test"
        assert parsed["processor_importer"] == "detector_file"

    def test_get_defaults(self, client):
        res = client.get("/api/settings/defaults")
        assert res.status_code == 200
        data = res.get_json()
        assert data["volume"] == 1.0
        assert data["theme"] == "dark"
        assert data["calibrate_count"] == 2
        assert data["calibration_fraction"] == 0.5
        assert data["safe_thresholds"] is False
        assert data["autoload_media_types"] == []
        assert "autorun_processors" not in data

    def test_update_safe_thresholds(self, client):
        res = client.put("/api/settings", json={"safe_thresholds": True})
        assert res.status_code == 200
        assert res.get_json()["safe_thresholds"] is True

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["safe_thresholds"] is True

    def test_update_safe_thresholds_false(self, client):
        client.put("/api/settings", json={"safe_thresholds": True})
        res = client.put("/api/settings", json={"safe_thresholds": False})
        assert res.status_code == 200
        assert res.get_json()["safe_thresholds"] is False

    def test_update_show_thumbnails_left(self, client):
        res = client.put("/api/settings", json={"show_thumbnails_left": True})
        assert res.status_code == 200
        assert res.get_json()["show_thumbnails_left"] is True

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["show_thumbnails_left"] is True

    def test_update_show_thumbnails_left_false(self, client):
        client.put("/api/settings", json={"show_thumbnails_left": True})
        res = client.put("/api/settings", json={"show_thumbnails_left": False})
        assert res.status_code == 200
        assert res.get_json()["show_thumbnails_left"] is False

    def test_update_show_thumbnails_right(self, client):
        res = client.put("/api/settings", json={"show_thumbnails_right": True})
        assert res.status_code == 200
        assert res.get_json()["show_thumbnails_right"] is True

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["show_thumbnails_right"] is True

    def test_update_show_thumbnails_right_false(self, client):
        client.put("/api/settings", json={"show_thumbnails_right": True})
        res = client.put("/api/settings", json={"show_thumbnails_right": False})
        assert res.status_code == 200
        assert res.get_json()["show_thumbnails_right"] is False

    def test_get_settings_includes_show_thumbnails(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "show_thumbnails_left" in data
        assert "show_thumbnails_right" in data

    def test_update_autoload_media_types(self, client):
        res = client.put("/api/settings", json={"autoload_media_types": ["audio", "video"]})
        assert res.status_code == 200
        assert res.get_json()["autoload_media_types"] == ["audio", "video"]

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["autoload_media_types"] == ["audio", "video"]

    def test_update_autoload_media_types_all_types(self, client):
        res = client.put("/api/settings", json={"autoload_media_types": ["audio", "image", "paragraph", "video"]})
        assert res.status_code == 200
        assert res.get_json()["autoload_media_types"] == ["audio", "image", "paragraph", "video"]

    def test_update_autoload_media_types_clear(self, client):
        client.put("/api/settings", json={"autoload_media_types": ["video"]})
        res = client.put("/api/settings", json={"autoload_media_types": []})
        assert res.status_code == 200
        assert res.get_json()["autoload_media_types"] == []

    def test_update_autoload_media_types_invalid(self, client):
        res = client.put("/api/settings", json={"autoload_media_types": ["invalid"]})
        assert res.status_code == 400

    def test_update_autoload_media_types_not_list(self, client):
        res = client.put("/api/settings", json={"autoload_media_types": "audio"})
        assert res.status_code == 400

    def test_get_settings_includes_autoload_media_types(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        assert "autoload_media_types" in res.get_json()

    def test_update_autopilot_top_greens(self, client):
        res = client.put("/api/settings", json={"autopilot_top_greens": 20})
        assert res.status_code == 200
        assert res.get_json()["autopilot_top_greens"] == 20

        res2 = client.get("/api/settings")
        assert res2.get_json()["autopilot_top_greens"] == 20

    def test_update_autopilot_top_greens_clamped(self, client):
        res = client.put("/api/settings", json={"autopilot_top_greens": 0})
        assert res.status_code == 200
        assert res.get_json()["autopilot_top_greens"] == 1

    def test_update_autopilot_top_greens_invalid(self, client):
        res = client.put("/api/settings", json={"autopilot_top_greens": "not a number"})
        assert res.status_code == 400

    def test_update_autopilot_hard_reds(self, client):
        res = client.put("/api/settings", json={"autopilot_hard_reds": 15})
        assert res.status_code == 200
        assert res.get_json()["autopilot_hard_reds"] == 15

        res2 = client.get("/api/settings")
        assert res2.get_json()["autopilot_hard_reds"] == 15

    def test_update_autopilot_hard_reds_clamped(self, client):
        res = client.put("/api/settings", json={"autopilot_hard_reds": 0})
        assert res.status_code == 200
        assert res.get_json()["autopilot_hard_reds"] == 1

    def test_update_autopilot_hard_reds_invalid(self, client):
        res = client.put("/api/settings", json={"autopilot_hard_reds": "not a number"})
        assert res.status_code == 400

    def test_get_settings_includes_autopilot(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "autopilot_top_greens" in data
        assert "autopilot_hard_reds" in data

    def test_get_defaults_includes_autopilot(self, client):
        res = client.get("/api/settings/defaults")
        assert res.status_code == 200
        data = res.get_json()
        assert data["autopilot_top_greens"] == 3
        assert data["autopilot_hard_reds"] == 4
