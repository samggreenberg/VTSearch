"""Settings API route tests.

Covers Flask API routes: GET/PUT /api/settings,
GET/POST/DELETE /api/settings/autorun-processors.
"""

from __future__ import annotations

import json

import pytest

import app as app_module  # noqa: F401 — triggers conftest media init

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
                "processor_importer": "server_detector_file",
                "field_values": {"filepath": "/tmp/det.json"},
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
            json={"processor_importer": "server_detector_file", "field_values": {}},
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
                "processor_importer": "server_detector_file",
                "field_values": {"filepath": "x.json"},
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
                "processor_importer": "server_detector_file",
                "field_values": {"filepath": "x.json"},
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
                "processor_importer": "server_detector_file",
                "field_values": {"filepath": "det.json"},
            },
        )

        res = client.get("/api/settings")
        data = res.get_json()
        proc = next(p for p in data["autorun_processors"] if p["processor_name"] == "cmd_test")

        parsed = json.loads(proc["settings_json"])
        assert parsed["processor_name"] == "cmd_test"
        assert parsed["processor_importer"] == "server_detector_file"

    def test_get_defaults(self, client):
        res = client.get("/api/settings/defaults")
        assert res.status_code == 200
        data = res.get_json()
        assert data["volume"] == 1.0
        assert data["theme"] == "dark"
        assert data["calibrate_count"] == 2
        assert data["calibration_fraction"] == 0.5
        assert data["safe_thresholds"] is False
        assert isinstance(data["focus_mode_left"], dict)
        for v in data["focus_mode_left"].values():
            assert v == "click"
        assert isinstance(data["focus_mode_right"], dict)
        for v in data["focus_mode_right"].values():
            assert v == "click"
        assert "autorun_processors" not in data
        assert "saved_datasets_dir" not in data
        assert "detectors_dir" not in data
        assert "trainable_models_dir" not in data

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

    def test_update_show_metadata(self, client):
        res = client.put("/api/settings", json={"show_metadata": False})
        assert res.status_code == 200
        assert res.get_json()["show_metadata"] is False

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["show_metadata"] is False

    def test_update_show_metadata_true(self, client):
        client.put("/api/settings", json={"show_metadata": False})
        res = client.put("/api/settings", json={"show_metadata": True})
        assert res.status_code == 200
        assert res.get_json()["show_metadata"] is True

    def test_update_view_mode_left_per_type(self, client):
        res = client.put("/api/settings", json={"view_mode_left": {"audio": "grid", "image": "list"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["view_mode_left"]["audio"] == "grid"
        assert data["view_mode_left"]["image"] == "list"

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["view_mode_left"]["audio"] == "grid"

    def test_update_view_mode_left_legacy_scalar(self, client):
        res = client.put("/api/settings", json={"view_mode_left": "grid"})
        assert res.status_code == 200
        data = res.get_json()
        # All types should now be "grid"
        for v in data["view_mode_left"].values():
            assert v == "grid"

    def test_update_view_mode_left_invalid(self, client):
        res = client.put("/api/settings", json={"view_mode_left": {"audio": "invalid"}})
        assert res.status_code == 400

    def test_update_view_mode_left_invalid_scalar(self, client):
        res = client.put("/api/settings", json={"view_mode_left": "invalid"})
        assert res.status_code == 400

    def test_update_view_mode_right_per_type(self, client):
        res = client.put("/api/settings", json={"view_mode_right": {"audio": "list", "video": "grid"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["view_mode_right"]["audio"] == "list"
        assert data["view_mode_right"]["video"] == "grid"

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["view_mode_right"]["audio"] == "list"

    def test_update_view_mode_right_legacy_scalar(self, client):
        res = client.put("/api/settings", json={"view_mode_right": "list"})
        assert res.status_code == 200
        data = res.get_json()
        for v in data["view_mode_right"].values():
            assert v == "list"

    def test_update_view_mode_right_invalid(self, client):
        res = client.put("/api/settings", json={"view_mode_right": {"audio": "invalid"}})
        assert res.status_code == 400

    def test_get_settings_includes_view_modes(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "show_metadata" in data
        assert "view_mode_left" in data
        assert "view_mode_right" in data

    def test_update_grid_icon_size_left_per_type(self, client):
        res = client.put("/api/settings", json={"grid_icon_size_left": {"audio": "XS", "image": "XL"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["grid_icon_size_left"]["audio"] == "XS"
        assert data["grid_icon_size_left"]["image"] == "XL"

        res2 = client.get("/api/settings")
        assert res2.get_json()["grid_icon_size_left"]["audio"] == "XS"

    def test_update_grid_icon_size_left_scalar(self, client):
        res = client.put("/api/settings", json={"grid_icon_size_left": "L"})
        assert res.status_code == 200
        data = res.get_json()
        for v in data["grid_icon_size_left"].values():
            assert v == "L"

    def test_update_grid_icon_size_left_invalid(self, client):
        res = client.put("/api/settings", json={"grid_icon_size_left": {"audio": "HUGE"}})
        assert res.status_code == 400

    def test_update_grid_icon_size_left_invalid_scalar(self, client):
        res = client.put("/api/settings", json={"grid_icon_size_left": "invalid"})
        assert res.status_code == 400

    def test_update_grid_icon_size_right_per_type(self, client):
        res = client.put("/api/settings", json={"grid_icon_size_right": {"audio": "S", "video": "L"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["grid_icon_size_right"]["audio"] == "S"
        assert data["grid_icon_size_right"]["video"] == "L"

        res2 = client.get("/api/settings")
        assert res2.get_json()["grid_icon_size_right"]["audio"] == "S"

    def test_update_grid_icon_size_right_invalid(self, client):
        res = client.put("/api/settings", json={"grid_icon_size_right": {"audio": "TINY"}})
        assert res.status_code == 400

    def test_get_settings_includes_grid_icon_size(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "grid_icon_size_left" in data
        assert "grid_icon_size_right" in data

    def test_update_focus_mode_left_per_type(self, client):
        res = client.put("/api/settings", json={"focus_mode_left": {"audio": "hover", "image": "click"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["focus_mode_left"]["audio"] == "hover"
        assert data["focus_mode_left"]["image"] == "click"

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["focus_mode_left"]["audio"] == "hover"

    def test_update_focus_mode_left_legacy_scalar(self, client):
        res = client.put("/api/settings", json={"focus_mode_left": "hover"})
        assert res.status_code == 200
        data = res.get_json()
        # All types should now be "hover"
        for v in data["focus_mode_left"].values():
            assert v == "hover"

    def test_update_focus_mode_left_invalid(self, client):
        res = client.put("/api/settings", json={"focus_mode_left": {"audio": "invalid"}})
        assert res.status_code == 400

    def test_update_focus_mode_left_invalid_scalar(self, client):
        res = client.put("/api/settings", json={"focus_mode_left": "invalid"})
        assert res.status_code == 400

    def test_update_focus_mode_right_per_type(self, client):
        res = client.put("/api/settings", json={"focus_mode_right": {"audio": "hover", "video": "click"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["focus_mode_right"]["audio"] == "hover"
        assert data["focus_mode_right"]["video"] == "click"

        # Verify it persisted
        res2 = client.get("/api/settings")
        assert res2.get_json()["focus_mode_right"]["audio"] == "hover"

    def test_update_focus_mode_right_legacy_scalar(self, client):
        res = client.put("/api/settings", json={"focus_mode_right": "hover"})
        assert res.status_code == 200
        data = res.get_json()
        for v in data["focus_mode_right"].values():
            assert v == "hover"

    def test_update_focus_mode_right_invalid(self, client):
        res = client.put("/api/settings", json={"focus_mode_right": {"audio": "invalid"}})
        assert res.status_code == 400

    def test_get_settings_includes_focus_modes(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "focus_mode_left" in data
        assert isinstance(data["focus_mode_left"], dict)
        assert "focus_mode_right" in data
        assert isinstance(data["focus_mode_right"], dict)

    # --- Panel percentage API tests ---

    def test_update_panel_pct_left_per_type(self, client):
        res = client.put("/api/settings", json={"panel_pct_left": {"audio": 200, "image": 350}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["panel_pct_left"]["audio"] == 200
        assert data["panel_pct_left"]["image"] == 350

        res2 = client.get("/api/settings")
        assert res2.get_json()["panel_pct_left"]["audio"] == 200

    def test_update_panel_pct_right_per_type(self, client):
        res = client.put("/api/settings", json={"panel_pct_right": {"audio": 250, "video": 400}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["panel_pct_right"]["audio"] == 250
        assert data["panel_pct_right"]["video"] == 400

    def test_update_panel_pct_left_scalar(self, client):
        res = client.put("/api/settings", json={"panel_pct_left": 300})
        assert res.status_code == 200
        for v in res.get_json()["panel_pct_left"].values():
            assert v == 300

    def test_update_panel_pct_left_invalid(self, client):
        res = client.put("/api/settings", json={"panel_pct_left": {"audio": "invalid"}})
        assert res.status_code == 400

    def test_update_panel_pct_left_out_of_range(self, client):
        res = client.put("/api/settings", json={"panel_pct_left": 100})
        assert res.status_code == 400
        res = client.put("/api/settings", json={"panel_pct_left": 600})
        assert res.status_code == 400

    def test_get_settings_includes_panel_pct(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "panel_pct_left" in data
        assert isinstance(data["panel_pct_left"], dict)
        assert "panel_pct_right" in data
        assert isinstance(data["panel_pct_right"], dict)

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

    def test_update_autopilot_enabled(self, client):
        res = client.put("/api/settings", json={"autopilot_enabled": False})
        assert res.status_code == 200
        assert res.get_json()["autopilot_enabled"] is False

        res2 = client.get("/api/settings")
        assert res2.get_json()["autopilot_enabled"] is False

    def test_update_autopilot_enabled_true(self, client):
        client.put("/api/settings", json={"autopilot_enabled": False})
        res = client.put("/api/settings", json={"autopilot_enabled": True})
        assert res.status_code == 200
        assert res.get_json()["autopilot_enabled"] is True

    def test_update_autopilot_goal_diversity(self, client):
        res = client.put("/api/settings", json={"autopilot_goal_diversity": 60})
        assert res.status_code == 200
        assert res.get_json()["autopilot_goal_diversity"] == 60

        res2 = client.get("/api/settings")
        assert res2.get_json()["autopilot_goal_diversity"] == 60

    def test_update_autopilot_goal_diversity_clamped(self, client):
        res = client.put("/api/settings", json={"autopilot_goal_diversity": 0})
        assert res.status_code == 200
        assert res.get_json()["autopilot_goal_diversity"] == 1

    def test_update_autopilot_goal_diversity_invalid(self, client):
        res = client.put("/api/settings", json={"autopilot_goal_diversity": "not a number"})
        assert res.status_code == 400

    def test_get_settings_includes_autopilot(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "autopilot_enabled" in data
        assert "autopilot_top_greens" in data
        assert "autopilot_hard_reds" in data
        assert "autopilot_goal_diversity" in data

    def test_get_defaults_includes_autopilot(self, client):
        res = client.get("/api/settings/defaults")
        assert res.status_code == 200
        data = res.get_json()
        assert data["autopilot_enabled"] is True
        assert data["autopilot_top_greens"] == 3
        assert data["autopilot_hard_reds"] == 4
        assert data["autopilot_goal_diversity"] == 40

    def test_get_defaults_excludes_directory_settings(self, client):
        res = client.get("/api/settings/defaults")
        assert res.status_code == 200
        data = res.get_json()
        assert "saved_datasets_dir" not in data
        assert "detectors_dir" not in data
        assert "trainable_models_dir" not in data


