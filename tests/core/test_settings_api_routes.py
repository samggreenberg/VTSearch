"""Settings API route tests.

Covers Flask API routes: GET/PUT /api/settings,
including autofind_detectors persistence.
"""

from __future__ import annotations


import pytest

import app as app_module  # noqa: F401  (triggers conftest media init)

# Flask API routes
# ---------------------------------------------------------------------------


class TestSettingsAPI:
    def test_get_settings(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "volume" in data
        assert "autofind_detectors" in data

    def test_get_settings_tolerates_stale_scalar_dict_field(self, client, isolated_settings):
        """A stale scalar where a per-media-type dict is expected must not 500.

        Older settings files (or hand-edits) can carry e.g.
        ``browse_bin_shape: "hex"`` from before the field became a
        ``{media_type: shape}`` dict. Marshmallow's ``Dict`` field calls
        ``.items()`` while dumping, so the bare string used to crash the
        whole endpoint with ``AttributeError: 'str' object has no attribute
        'items'`` → 500. The read path now coerces it to ``{}`` (equivalent
        to "never set") instead.
        """
        import json as _json

        from vtsearch import settings as settings_mod

        isolated_settings._user.parent.mkdir(parents=True, exist_ok=True)
        isolated_settings._user.write_text(_json.dumps({"browse_bin_shape": "hex"}) + "\n")
        settings_mod.reset()  # force the in-memory cache to re-read the corrupt file

        res = client.get("/api/settings")
        assert res.status_code == 200
        assert res.get_json()["browse_bin_shape"] == {}

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
        # SettingsUpdate schema catches the type mismatch → 422.
        assert res.status_code == 422

    def test_update_inclusion_no_detector_browse_mode(self, client):
        """A bulk settings save without an identified detector must not 400.

        Reproduces the VTSBrowser theme-switch bug: the browser has a dataset
        loaded but no detector, so Angular's ``activeContextInterceptor``
        sends ``X-Dataset-Id`` but omits ``X-Detector-Id``. The settings-modal
        ``save()`` echoes the whole settings blob back on every change
        (including ``inclusion``, which routes to the active detector
        context). With no detector identified, the route resolves the frozen
        request-missing detector sentinel; applying ``inclusion`` used to
        raise ``RequestMissingContextError`` → 400. The inclusion cache write
        is now skipped when no detector is present (the value still persists
        to the per-user settings store).
        """
        from vtscore.state.core import set_thread_detector_context

        # Drop the thread-local detector context the conftest installs so the
        # resolver falls through to the request-missing sentinel, matching a
        # production Flask request thread with no detector.
        set_thread_detector_context(None)
        res = client.put(
            "/api/settings",
            json={"theme": "dark", "inclusion": 3},
            headers={"X-Detector-Id": ""},
        )
        assert res.status_code == 200
        assert res.get_json()["inclusion"] == 3

        # The value still persisted to the per-user settings store.
        res2 = client.get("/api/settings")
        assert res2.get_json()["inclusion"] == 3

    def test_update_volume_invalid(self, client):
        res = client.put(
            "/api/settings",
            json={"volume": "not a number"},
        )
        assert res.status_code == 422

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
        assert res.status_code == 422

    def test_update_empty_body(self, client):
        res = client.put(
            "/api/settings",
            data="",
            content_type="application/json",
        )
        # Empty body is a legitimate no-op PUT under the new schema;
        # every key is optional, so nothing to apply. Returns 200 with
        # the current settings dict.
        assert res.status_code == 200

    def test_update_autofind_detectors(self, client):
        res = client.put(
            "/api/settings",
            json={"autofind_detectors": ["model-a", "model-b"]},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["autofind_detectors"] == ["model-a", "model-b"]

    def test_update_autofind_detectors_invalid(self, client):
        res = client.put(
            "/api/settings",
            json={"autofind_detectors": "not a list"},
        )
        # List-of-string validation runs in the schema → 422.
        assert res.status_code == 422

    def test_update_autofind_exporter_valid(self, client):
        res = client.put("/api/settings", json={"autofind_exporter": "server_json_file"})
        assert res.status_code == 200
        assert res.get_json()["autofind_exporter"] == "server_json_file"

    def test_update_autofind_exporter_empty_clears(self, client):
        client.put("/api/settings", json={"autofind_exporter": "server_json_file"})
        res = client.put("/api/settings", json={"autofind_exporter": ""})
        assert res.status_code == 200
        assert res.get_json()["autofind_exporter"] == ""

    def test_update_autofind_exporter_unknown_rejected(self, client):
        res = client.put("/api/settings", json={"autofind_exporter": "no_such_exporter"})
        assert res.status_code == 400

    def test_update_autofind_exporter_field_values(self, client):
        res = client.put(
            "/api/settings",
            json={
                "autofind_exporter_field_values": {
                    "server_json_file": {"filepath": "/tmp/out.json"},
                }
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["autofind_exporter_field_values"]["server_json_file"]["filepath"] == "/tmp/out.json"

    def test_autofind_exporter_excluded_from_defaults(self, client):
        res = client.get("/api/settings/defaults")
        assert res.status_code == 200
        data = res.get_json()
        assert "autofind_exporter" not in data
        assert "autofind_exporter_field_values" not in data

    def test_get_defaults(self, client):
        res = client.get("/api/settings/defaults")
        assert res.status_code == 200
        data = res.get_json()
        assert data["volume"] == 1.0
        # Theme defaults to ``"system"``; the frontend resolves this to
        # the OS ``prefers-color-scheme`` at render time.
        assert data["theme"] == "system"
        assert data["calibrate_count"] == 1
        assert data["calibration_fraction"] == 0.5
        assert data["safe_thresholds"] is False
        assert isinstance(data["focus_mode_left"], dict)
        for v in data["focus_mode_left"].values():
            assert v == "click"
        assert isinstance(data["focus_mode_right"], dict)
        for v in data["focus_mode_right"].values():
            assert v == "click"
        assert "autofind_detectors" not in data
        assert "saved_datasets_dir" not in data
        assert "detectors_dir" not in data

    def test_update_import_defaults_by_media_type(self, client):
        # PUT round-trips the nested defaults dict through the JSON
        # schema, then GET sees the same value (proving the field
        # reaches the per-user JSON store).
        payload = {
            "import_defaults_by_media_type": {
                "image": {
                    "embedder": "siglip",
                    "clipper": "image_grid_clipper",
                    "clipper_params": {"rows": 2, "cols": 2},
                    "source_specs": [
                        {
                            "source_type": "video",
                            "converter": "video_to_image",
                            "params": {"n_clips": "3"},
                        },
                    ],
                },
            },
        }
        res = client.put("/api/settings", json=payload)
        assert res.status_code == 200
        assert res.get_json()["import_defaults_by_media_type"] == payload["import_defaults_by_media_type"]

        res2 = client.get("/api/settings")
        assert res2.get_json()["import_defaults_by_media_type"] == payload["import_defaults_by_media_type"]

    def test_get_defaults_import_defaults_empty(self, client):
        # Factory defaults: nothing saved means an empty dict, not a
        # missing key; the frontend reads ``settings.import_defaults...``
        # directly and shouldn't have to defend against ``undefined``.
        res = client.get("/api/settings/defaults")
        assert res.status_code == 200
        assert res.get_json().get("import_defaults_by_media_type") == {}

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

    def test_update_label_hint_dismissed(self, client):
        # Default is False; the hint shows on first session.
        initial = client.get("/api/settings").get_json()
        assert initial["label_hint_dismissed"] is False

        res = client.put("/api/settings", json={"label_hint_dismissed": True})
        assert res.status_code == 200
        assert res.get_json()["label_hint_dismissed"] is True

        # Persists across reads.
        res2 = client.get("/api/settings")
        assert res2.get_json()["label_hint_dismissed"] is True

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

    def test_update_view_mode_popup_per_type(self, client):
        res = client.put("/api/settings", json={"view_mode_popup": {"audio": "list", "image": "grid"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["view_mode_popup"]["audio"] == "list"
        assert data["view_mode_popup"]["image"] == "grid"

        res2 = client.get("/api/settings")
        assert res2.get_json()["view_mode_popup"]["audio"] == "list"

    def test_update_view_mode_popup_invalid(self, client):
        res = client.put("/api/settings", json={"view_mode_popup": {"audio": "invalid"}})
        assert res.status_code == 400

    def test_update_grid_icon_size_popup_per_type(self, client):
        res = client.put("/api/settings", json={"grid_icon_size_popup": {"audio": "S", "video": "L"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["grid_icon_size_popup"]["audio"] == "S"
        assert data["grid_icon_size_popup"]["video"] == "L"

        res2 = client.get("/api/settings")
        assert res2.get_json()["grid_icon_size_popup"]["audio"] == "S"

    def test_update_grid_icon_size_popup_invalid(self, client):
        res = client.put("/api/settings", json={"grid_icon_size_popup": {"audio": "TINY"}})
        assert res.status_code == 400

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
        res = client.put("/api/settings", json={"panel_pct_left": 20000})
        assert res.status_code == 400

    def test_update_panel_pct_left_wide_panel(self, client):
        """Widths above the old 500px cap persist (frontend resize allows them)."""
        res = client.put("/api/settings", json={"panel_pct_left": {"image": 520}})
        assert res.status_code == 200
        assert res.get_json()["panel_pct_left"]["image"] == 520

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
        assert res.status_code == 422

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
        assert res.status_code == 422

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
        assert res.status_code == 422

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
        assert "detectors_dir" not in data


class TestServerSettingsReadOnly:
    """The read-only "Server" settings tab is fed by GET /api/settings.

    These assert the server-tier keys are exposed on read and ignored on
    write so the frontend can render them as immutable reference values.
    """

    def test_get_settings_includes_server_tier(self, client):
        data = client.get("/api/settings").get_json()
        for key in (
            "saved_datasets_dir",
            "detectors_dir",
            "max_concurrent_dataset_downloads",
            "max_concurrent_dataset_embeddings",
            "autofind_detectors",
            "hidden_plugins",
        ):
            assert key in data, key

    def test_hidden_plugins_reflects_persisted_setting(self, client):
        from vtsearch import settings as settings_mod

        settings_mod.set_cli_hidden_plugins(None)
        try:
            settings_mod.set_hidden_plugins({"converters": ["audio2image"]})
            data = client.get("/api/settings").get_json()
            assert data["hidden_plugins"] == {"converters": ["audio2image"]}
        finally:
            settings_mod.set_cli_hidden_plugins(None)

    def test_hidden_plugins_unions_cli_flags(self, client):
        from vtsearch import settings as settings_mod

        settings_mod.set_cli_hidden_plugins(None)
        try:
            settings_mod.set_hidden_plugins({"converters": ["audio2image"]})
            settings_mod.add_cli_hidden_plugin("converters", "video2image")
            settings_mod.add_cli_hidden_plugin("embedders", "clap")
            data = client.get("/api/settings").get_json()
            # Effective view: persisted ∪ CLI, names sorted within each family.
            assert data["hidden_plugins"]["converters"] == ["audio2image", "video2image"]
            assert data["hidden_plugins"]["embedders"] == ["clap"]
        finally:
            settings_mod.set_cli_hidden_plugins(None)

    def test_hidden_plugins_is_not_writable_via_api(self, client):
        from vtsearch import settings as settings_mod

        settings_mod.set_cli_hidden_plugins(None)
        try:
            res = client.put("/api/settings", json={"hidden_plugins": {"embedders": ["clap"]}})
            # Unknown-on-write key is silently dropped (schema excludes it).
            assert res.status_code == 200
            assert settings_mod.get_hidden_plugins() == {}
            assert res.get_json()["hidden_plugins"] == {}
        finally:
            settings_mod.set_cli_hidden_plugins(None)


class TestBrowserSettings:
    """Per-media-type VTSBrowse prefs: bin shape, colormap, and cell size."""

    def test_update_browse_bin_shape_per_type(self, client):
        res = client.put("/api/settings", json={"browse_bin_shape": {"audio": "square", "image": "hex"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["browse_bin_shape"]["audio"] == "square"
        assert data["browse_bin_shape"]["image"] == "hex"

        # Persisted, and a second key write merges rather than clobbers.
        res2 = client.put("/api/settings", json={"browse_bin_shape": {"audio": "square", "video": "square"}})
        assert res2.get_json()["browse_bin_shape"]["video"] == "square"
        assert client.get("/api/settings").get_json()["browse_bin_shape"]["audio"] == "square"

    def test_update_browse_bin_shape_invalid(self, client):
        res = client.put("/api/settings", json={"browse_bin_shape": {"audio": "triangle"}})
        assert res.status_code == 400

    def test_update_browse_colormap_per_type(self, client):
        res = client.put("/api/settings", json={"browse_colormap": {"audio": "ocean", "image": "heat"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["browse_colormap"]["audio"] == "ocean"
        assert data["browse_colormap"]["image"] == "heat"
        assert client.get("/api/settings").get_json()["browse_colormap"]["audio"] == "ocean"

    def test_update_browse_colormap_invalid(self, client):
        res = client.put("/api/settings", json={"browse_colormap": {"audio": "rainbow"}})
        assert res.status_code == 400

    def test_update_browse_icon_size_per_type(self, client):
        res = client.put("/api/settings", json={"browse_icon_size": {"audio": "XL", "image": "XS"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["browse_icon_size"]["audio"] == "XL"
        assert data["browse_icon_size"]["image"] == "XS"

    def test_update_browse_icon_size_is_uppercased(self, client):
        # Mirrors grid_icon_size: lowercase input is normalised to the enum.
        res = client.put("/api/settings", json={"browse_icon_size": {"audio": "l"}})
        assert res.status_code == 200
        assert res.get_json()["browse_icon_size"]["audio"] == "L"

    def test_update_browse_icon_size_invalid(self, client):
        res = client.put("/api/settings", json={"browse_icon_size": {"audio": "huge"}})
        assert res.status_code == 400

    def test_update_browse_icon_size_accepts_larger_levels(self, client):
        # The browse canvas walks nine zoom levels (XS..5XL), four beyond the
        # grid icon size's XS..XL; the backend must accept the larger labels the
        # bigger/smaller buttons persist.
        res = client.put("/api/settings", json={"browse_icon_size": {"audio": "2XL", "image": "5xl"}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["browse_icon_size"]["audio"] == "2XL"
        assert data["browse_icon_size"]["image"] == "5XL"

    def test_update_browse_thumbnail_border_per_type(self, client):
        res = client.put("/api/settings", json={"browse_thumbnail_border": {"image": 3, "video": 0}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["browse_thumbnail_border"]["image"] == 3
        assert data["browse_thumbnail_border"]["video"] == 0
        # The written value persists across a fresh read (the frontend sends the
        # full merged map; the backend stores it verbatim).
        assert client.get("/api/settings").get_json()["browse_thumbnail_border"]["image"] == 3

    def test_update_browse_thumbnail_border_clamped(self, client):
        # Out-of-range values are clamped into 0..8 rather than rejected.
        res = client.put("/api/settings", json={"browse_thumbnail_border": {"image": 100, "video": -5}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["browse_thumbnail_border"]["image"] == 8
        assert data["browse_thumbnail_border"]["video"] == 0

    def test_update_browse_compact_per_type(self, client):
        res = client.put("/api/settings", json={"browse_compact": {"audio": False, "image": True}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["browse_compact"]["audio"] is False
        assert data["browse_compact"]["image"] is True
        # Persists across a fresh read.
        assert client.get("/api/settings").get_json()["browse_compact"]["audio"] is False

    def test_get_settings_includes_browser_prefs(self, client):
        data = client.get("/api/settings").get_json()
        # Present as (possibly empty) dicts so the frontend can index by type.
        for key in (
            "browse_bin_shape",
            "browse_colormap",
            "browse_icon_size",
            "browse_thumbnail_border",
            "browse_compact",
        ):
            assert key in data
            assert isinstance(data[key], dict)

    def test_defaults_have_empty_browser_prefs(self, client):
        data = client.get("/api/settings/defaults").get_json()
        for key in (
            "browse_bin_shape",
            "browse_colormap",
            "browse_icon_size",
            "browse_thumbnail_border",
            "browse_compact",
        ):
            assert data.get(key, {}) == {}
