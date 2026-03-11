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
        settings_mod.add_autorun_processor("my det", "server_detector_file", {"filepath": "/tmp/det.json"})
        procs = settings_mod.get_autorun_processors()
        assert len(procs) == 1
        assert procs[0]["processor_name"] == "my det"
        assert procs[0]["processor_importer"] == "server_detector_file"
        assert procs[0]["field_values"]["filepath"] == "/tmp/det.json"

    def test_add_overwrites_same_name(self):
        settings_mod.add_autorun_processor("a", "server_detector_file", {"filepath": "1.json"})
        settings_mod.add_autorun_processor("a", "server_detector_file", {"filepath": "2.json"})
        procs = settings_mod.get_autorun_processors()
        assert len(procs) == 1
        assert procs[0]["field_values"]["filepath"] == "2.json"

    def test_remove_autorun_processor(self):
        settings_mod.add_autorun_processor("x", "server_detector_file", {"filepath": "x.json"})
        assert settings_mod.remove_autorun_processor("x") is True
        assert settings_mod.get_autorun_processors() == []

    def test_remove_nonexistent(self):
        assert settings_mod.remove_autorun_processor("nope") is False

    def test_to_settings_json(self):
        entry = {
            "processor_name": "my detector",
            "processor_importer": "server_detector_file",
            "field_values": {"filepath": "/path/to/det.json"},
        }
        snippet = settings_mod.to_settings_json(entry)
        import json

        parsed = json.loads(snippet)
        assert parsed["processor_name"] == "my detector"
        assert parsed["processor_importer"] == "server_detector_file"
        assert parsed["field_values"]["filepath"] == "/path/to/det.json"

    def test_to_settings_json_with_spaces(self):
        entry = {
            "processor_name": "det",
            "processor_importer": "server_detector_file",
            "field_values": {"filepath": "/my path/det.json"},
        }
        snippet = settings_mod.to_settings_json(entry)
        import json

        parsed = json.loads(snippet)
        assert parsed["field_values"]["filepath"] == "/my path/det.json"

    def test_persistence_survives_reset(self, isolated_settings):
        settings_mod.set_volume(0.7)
        settings_mod.add_autorun_processor("p", "server_detector_file", {"filepath": "p.json"})

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

    def test_get_set_show_metadata(self, isolated_settings):
        settings_mod.set_show_metadata(False)
        assert settings_mod.get_show_metadata() is False

        raw = json.loads(isolated_settings.read_text())
        assert raw["show_metadata"] is False

    def test_show_metadata_default(self):
        assert settings_mod.get_show_metadata() is True

    def test_show_metadata_persists_across_reset(self, isolated_settings):
        settings_mod.set_show_metadata(False)
        settings_mod.reset()
        assert settings_mod.get_show_metadata() is False

    def test_get_view_mode_left_default(self):
        result = settings_mod.get_view_mode_left()
        assert isinstance(result, dict)
        # All types default to "list"
        for v in result.values():
            assert v == "list"

    def test_get_view_mode_right_default(self):
        result = settings_mod.get_view_mode_right()
        assert isinstance(result, dict)
        # All types default to "grid"
        for v in result.values():
            assert v == "grid"

    def test_set_view_mode_left_per_type(self, isolated_settings):
        settings_mod.set_view_mode_left({"audio": "grid", "image": "list"})
        result = settings_mod.get_view_mode_left()
        assert result["audio"] == "grid"
        assert result["image"] == "list"

        raw = json.loads(isolated_settings.read_text())
        assert raw["view_mode_left"]["audio"] == "grid"
        assert raw["view_mode_left"]["image"] == "list"

    def test_set_view_mode_right_per_type(self, isolated_settings):
        settings_mod.set_view_mode_right({"audio": "list", "video": "grid"})
        result = settings_mod.get_view_mode_right()
        assert result["audio"] == "list"
        assert result["video"] == "grid"

        raw = json.loads(isolated_settings.read_text())
        assert raw["view_mode_right"]["audio"] == "list"

    def test_view_mode_left_legacy_scalar(self, isolated_settings):
        """Legacy string value is accepted and expanded to all types."""
        settings_mod.set_view_mode_left("grid")
        result = settings_mod.get_view_mode_left()
        for v in result.values():
            assert v == "grid"

    def test_view_mode_right_legacy_scalar(self, isolated_settings):
        """Legacy string value is accepted and expanded to all types."""
        settings_mod.set_view_mode_right("list")
        result = settings_mod.get_view_mode_right()
        for v in result.values():
            assert v == "list"

    def test_view_mode_left_persists_across_reset(self, isolated_settings):
        settings_mod.set_view_mode_left({"audio": "grid"})
        settings_mod.reset()
        assert settings_mod.get_view_mode_left()["audio"] == "grid"

    def test_view_mode_right_persists_across_reset(self, isolated_settings):
        settings_mod.set_view_mode_right({"audio": "list"})
        settings_mod.reset()
        assert settings_mod.get_view_mode_right()["audio"] == "list"

    def test_view_mode_left_invalid_mode(self):
        with pytest.raises(ValueError):
            settings_mod.set_view_mode_left({"audio": "invalid"})

    def test_view_mode_right_invalid_mode(self):
        with pytest.raises(ValueError):
            settings_mod.set_view_mode_right({"audio": "invalid"})

    def test_view_mode_invalid_scalar(self):
        with pytest.raises(ValueError):
            settings_mod.set_view_mode_left("invalid")

    def test_view_mode_invalid_media_type(self):
        with pytest.raises(ValueError):
            settings_mod.set_view_mode_left({"nonexistent_type": "grid"})

    def test_get_grid_columns_left_default(self):
        result = settings_mod.get_grid_columns_left()
        assert isinstance(result, dict)
        for v in result.values():
            assert v == 2

    def test_get_grid_columns_right_default(self):
        result = settings_mod.get_grid_columns_right()
        assert isinstance(result, dict)
        for v in result.values():
            assert v == 2

    def test_set_grid_columns_left_per_type(self, isolated_settings):
        settings_mod.set_grid_columns_left({"audio": 3, "image": 1})
        result = settings_mod.get_grid_columns_left()
        assert result["audio"] == 3
        assert result["image"] == 1

        raw = json.loads(isolated_settings.read_text())
        assert raw["grid_columns_left"]["audio"] == 3
        assert raw["grid_columns_left"]["image"] == 1

    def test_set_grid_columns_right_per_type(self, isolated_settings):
        settings_mod.set_grid_columns_right({"audio": 1, "video": 3})
        result = settings_mod.get_grid_columns_right()
        assert result["audio"] == 1
        assert result["video"] == 3

        raw = json.loads(isolated_settings.read_text())
        assert raw["grid_columns_right"]["audio"] == 1

    def test_grid_columns_left_legacy_scalar(self, isolated_settings):
        settings_mod.set_grid_columns_left(3)
        result = settings_mod.get_grid_columns_left()
        for v in result.values():
            assert v == 3

    def test_grid_columns_right_legacy_scalar(self, isolated_settings):
        settings_mod.set_grid_columns_right(1)
        result = settings_mod.get_grid_columns_right()
        for v in result.values():
            assert v == 1

    def test_grid_columns_left_persists_across_reset(self, isolated_settings):
        settings_mod.set_grid_columns_left({"audio": 3})
        settings_mod.reset()
        assert settings_mod.get_grid_columns_left()["audio"] == 3

    def test_grid_columns_right_persists_across_reset(self, isolated_settings):
        settings_mod.set_grid_columns_right({"audio": 1})
        settings_mod.reset()
        assert settings_mod.get_grid_columns_right()["audio"] == 1

    def test_grid_columns_left_invalid_value(self):
        with pytest.raises(ValueError):
            settings_mod.set_grid_columns_left({"audio": "invalid"})

    def test_grid_columns_right_invalid_value(self):
        with pytest.raises(ValueError):
            settings_mod.set_grid_columns_right({"audio": "invalid"})

    def test_grid_columns_invalid_scalar(self):
        with pytest.raises(ValueError):
            settings_mod.set_grid_columns_left("invalid")

    def test_grid_columns_invalid_media_type(self):
        with pytest.raises(ValueError):
            settings_mod.set_grid_columns_left({"nonexistent_type": 3})

    def test_grid_columns_out_of_range(self):
        with pytest.raises(ValueError):
            settings_mod.set_grid_columns_left({"audio": 0})
        with pytest.raises(ValueError):
            settings_mod.set_grid_columns_left({"audio": 7})
        with pytest.raises(ValueError):
            settings_mod.set_grid_columns_left(0)
        with pytest.raises(ValueError):
            settings_mod.set_grid_columns_left(7)

    def test_grid_columns_allows_up_to_six(self, isolated_settings):
        settings_mod.set_grid_columns_left({"audio": 6})
        assert settings_mod.get_grid_columns_left()["audio"] == 6
        settings_mod.set_grid_columns_right(5)
        for v in settings_mod.get_grid_columns_right().values():
            assert v == 5

    def test_get_focus_mode_left_default(self):
        result = settings_mod.get_focus_mode_left()
        assert isinstance(result, dict)
        # All types default to "click"
        for v in result.values():
            assert v == "click"

    def test_get_focus_mode_right_default(self):
        result = settings_mod.get_focus_mode_right()
        assert isinstance(result, dict)
        # All types default to "click"
        for v in result.values():
            assert v == "click"

    def test_set_focus_mode_left_per_type(self, isolated_settings):
        settings_mod.set_focus_mode_left({"audio": "hover", "image": "click"})
        result = settings_mod.get_focus_mode_left()
        assert result["audio"] == "hover"
        assert result["image"] == "click"

        raw = json.loads(isolated_settings.read_text())
        assert raw["focus_mode_left"]["audio"] == "hover"
        assert raw["focus_mode_left"]["image"] == "click"

    def test_set_focus_mode_right_per_type(self, isolated_settings):
        settings_mod.set_focus_mode_right({"audio": "hover", "video": "click"})
        result = settings_mod.get_focus_mode_right()
        assert result["audio"] == "hover"
        assert result["video"] == "click"

        raw = json.loads(isolated_settings.read_text())
        assert raw["focus_mode_right"]["audio"] == "hover"

    def test_focus_mode_left_legacy_scalar(self, isolated_settings):
        """Legacy string value is accepted and expanded to all types."""
        settings_mod.set_focus_mode_left("hover")
        result = settings_mod.get_focus_mode_left()
        for v in result.values():
            assert v == "hover"

    def test_focus_mode_right_legacy_scalar(self, isolated_settings):
        """Legacy string value is accepted and expanded to all types."""
        settings_mod.set_focus_mode_right("hover")
        result = settings_mod.get_focus_mode_right()
        for v in result.values():
            assert v == "hover"

    def test_focus_mode_left_persists_across_reset(self, isolated_settings):
        settings_mod.set_focus_mode_left({"audio": "hover"})
        settings_mod.reset()
        assert settings_mod.get_focus_mode_left()["audio"] == "hover"

    def test_focus_mode_right_persists_across_reset(self, isolated_settings):
        settings_mod.set_focus_mode_right({"audio": "hover"})
        settings_mod.reset()
        assert settings_mod.get_focus_mode_right()["audio"] == "hover"

    def test_focus_mode_left_invalid_mode(self):
        with pytest.raises(ValueError):
            settings_mod.set_focus_mode_left({"audio": "invalid"})

    def test_focus_mode_right_invalid_mode(self):
        with pytest.raises(ValueError):
            settings_mod.set_focus_mode_right({"audio": "invalid"})

    def test_focus_mode_invalid_scalar(self):
        with pytest.raises(ValueError):
            settings_mod.set_focus_mode_left("invalid")

    def test_focus_mode_invalid_media_type(self):
        with pytest.raises(ValueError):
            settings_mod.set_focus_mode_left({"nonexistent_type": "hover"})

    # --- Panel percentage settings ---

    def test_get_panel_pct_left_default(self):
        result = settings_mod.get_panel_pct_left()
        assert isinstance(result, dict)
        for v in result.values():
            assert v is None

    def test_get_panel_pct_right_default(self):
        result = settings_mod.get_panel_pct_right()
        assert isinstance(result, dict)
        for v in result.values():
            assert v is None

    def test_set_panel_pct_left_per_type(self, isolated_settings):
        settings_mod.set_panel_pct_left({"audio": 0.25, "image": 0.3})
        result = settings_mod.get_panel_pct_left()
        assert result["audio"] == pytest.approx(0.25)
        assert result["image"] == pytest.approx(0.3)

        raw = json.loads(isolated_settings.read_text())
        assert raw["panel_pct_left"]["audio"] == pytest.approx(0.25)
        assert raw["panel_pct_left"]["image"] == pytest.approx(0.3)

    def test_set_panel_pct_right_per_type(self, isolated_settings):
        settings_mod.set_panel_pct_right({"audio": 0.2, "video": 0.35})
        result = settings_mod.get_panel_pct_right()
        assert result["audio"] == pytest.approx(0.2)
        assert result["video"] == pytest.approx(0.35)

        raw = json.loads(isolated_settings.read_text())
        assert raw["panel_pct_right"]["audio"] == pytest.approx(0.2)

    def test_set_panel_pct_left_scalar(self, isolated_settings):
        settings_mod.set_panel_pct_left(0.3)
        result = settings_mod.get_panel_pct_left()
        for v in result.values():
            assert v == pytest.approx(0.3)

    def test_set_panel_pct_null_clears(self, isolated_settings):
        settings_mod.set_panel_pct_left({"audio": 0.25})
        settings_mod.set_panel_pct_left({"audio": None})
        assert settings_mod.get_panel_pct_left()["audio"] is None

    def test_panel_pct_persists_across_reset(self, isolated_settings):
        settings_mod.set_panel_pct_left({"audio": 0.25})
        settings_mod.reset()
        assert settings_mod.get_panel_pct_left()["audio"] == pytest.approx(0.25)

    def test_panel_pct_out_of_range(self):
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left({"audio": 0.01})
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left({"audio": 0.95})
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left(0.01)
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left(0.95)

    def test_panel_pct_invalid_media_type(self):
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left({"nonexistent_type": 0.25})

    def test_panel_pct_invalid_value(self):
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left({"audio": "invalid"})

    def test_panel_pct_clamped_on_read(self, isolated_settings):
        """Out-of-range values stored on disk are clamped when read."""
        settings_mod.set_panel_pct_left({"audio": 0.5})
        # Manually write an out-of-range value to disk
        raw = json.loads(isolated_settings.read_text())
        raw["panel_pct_left"]["audio"] = 0.99
        isolated_settings.write_text(json.dumps(raw))
        settings_mod.reset()
        result = settings_mod.get_panel_pct_left()
        assert result["audio"] == pytest.approx(0.80)

    def test_get_defaults(self):
        defaults = settings_mod.get_defaults()
        assert defaults["volume"] == 1.0
        assert defaults["theme"] == "dark"
        assert defaults["calibrate_count"] == 2
        assert defaults["calibration_fraction"] == 0.5
        assert defaults["safe_thresholds"] is False
        assert defaults["show_metadata"] is True
        assert isinstance(defaults["view_mode_left"], dict)
        for v in defaults["view_mode_left"].values():
            assert v == "list"
        assert isinstance(defaults["view_mode_right"], dict)
        for v in defaults["view_mode_right"].values():
            assert v == "grid"
        assert isinstance(defaults["grid_columns_left"], dict)
        for v in defaults["grid_columns_left"].values():
            assert v == 2
        assert isinstance(defaults["grid_columns_right"], dict)
        for v in defaults["grid_columns_right"].values():
            assert v == 2
        assert isinstance(defaults["focus_mode_left"], dict)
        for v in defaults["focus_mode_left"].values():
            assert v == "click"
        assert isinstance(defaults["focus_mode_right"], dict)
        for v in defaults["focus_mode_right"].values():
            assert v == "click"
        assert isinstance(defaults["panel_pct_left"], dict)
        for v in defaults["panel_pct_left"].values():
            assert v is None
        assert isinstance(defaults["panel_pct_right"], dict)
        for v in defaults["panel_pct_right"].values():
            assert v is None
        assert defaults["autoload_media_types"] == []
        assert defaults["autopilot_enabled"] is True
        assert defaults["autopilot_top_greens"] == 3
        assert defaults["autopilot_hard_reds"] == 4
        assert "autorun_processors" not in defaults
        # Directory settings excluded from defaults (not reset by Default button)
        assert "saved_datasets_dir" not in defaults
        assert "detectors_dir" not in defaults
        assert "trainable_models_dir" not in defaults

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

    def test_get_set_autopilot_enabled(self, isolated_settings):
        # Default is True
        assert settings_mod.get_autopilot_enabled() is True

        settings_mod.set_autopilot_enabled(False)
        assert settings_mod.get_autopilot_enabled() is False

        raw = json.loads(isolated_settings.read_text())
        assert raw["autopilot_enabled"] is False

    def test_autopilot_enabled_persists_across_reset(self, isolated_settings):
        settings_mod.set_autopilot_enabled(False)
        settings_mod.reset()
        assert settings_mod.get_autopilot_enabled() is False

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
        det_path.write_text(
            json.dumps(
                {
                    "media_type": "audio",
                    "weights": det_weights,
                    "threshold": 0.5,
                }
            )
        )

        settings_mod.add_autorun_processor("settings_test_det", "server_detector_file", {"filepath": str(det_path)})

        imported = settings_mod.ensure_autorun_processors_imported()
        assert "settings_test_det" in imported
        assert "settings_test_det" in autorun_detectors

    def test_skips_already_imported(self, tmp_path):
        """If a detector with the same name already exists, skip it."""
        from vtsearch.utils import add_autorun_detector

        add_autorun_detector("existing", "audio", {"0.weight": [[0.1]], "0.bias": [0.0]}, 0.5)

        settings_mod.add_autorun_processor("existing", "server_detector_file", {"filepath": "/nonexistent.json"})

        imported = settings_mod.ensure_autorun_processors_imported()
        assert imported == []

    def test_handles_bad_importer_gracefully(self):
        """Unknown importer name should not crash."""
        settings_mod.add_autorun_processor("bad_proc", "totally_fake_importer", {"filepath": "x.json"})

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
        import json

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
        assert data["autoload_media_types"] == []
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

    def test_update_grid_columns_left_per_type(self, client):
        res = client.put("/api/settings", json={"grid_columns_left": {"audio": 3, "image": 1}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["grid_columns_left"]["audio"] == 3
        assert data["grid_columns_left"]["image"] == 1

        res2 = client.get("/api/settings")
        assert res2.get_json()["grid_columns_left"]["audio"] == 3

    def test_update_grid_columns_left_scalar(self, client):
        res = client.put("/api/settings", json={"grid_columns_left": 3})
        assert res.status_code == 200
        data = res.get_json()
        for v in data["grid_columns_left"].values():
            assert v == 3

    def test_update_grid_columns_left_invalid(self, client):
        res = client.put("/api/settings", json={"grid_columns_left": {"audio": "invalid"}})
        assert res.status_code == 400

    def test_update_grid_columns_left_invalid_scalar(self, client):
        res = client.put("/api/settings", json={"grid_columns_left": "invalid"})
        assert res.status_code == 400

    def test_update_grid_columns_left_out_of_range(self, client):
        res = client.put("/api/settings", json={"grid_columns_left": 0})
        assert res.status_code == 400
        res = client.put("/api/settings", json={"grid_columns_left": 7})
        assert res.status_code == 400

    def test_update_grid_columns_right_per_type(self, client):
        res = client.put("/api/settings", json={"grid_columns_right": {"audio": 1, "video": 3}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["grid_columns_right"]["audio"] == 1
        assert data["grid_columns_right"]["video"] == 3

        res2 = client.get("/api/settings")
        assert res2.get_json()["grid_columns_right"]["audio"] == 1

    def test_update_grid_columns_right_invalid(self, client):
        res = client.put("/api/settings", json={"grid_columns_right": {"audio": "invalid"}})
        assert res.status_code == 400

    def test_get_settings_includes_grid_columns(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "grid_columns_left" in data
        assert "grid_columns_right" in data

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
        res = client.put("/api/settings", json={"panel_pct_left": {"audio": 0.25, "image": 0.3}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["panel_pct_left"]["audio"] == pytest.approx(0.25)
        assert data["panel_pct_left"]["image"] == pytest.approx(0.3)

        res2 = client.get("/api/settings")
        assert res2.get_json()["panel_pct_left"]["audio"] == pytest.approx(0.25)

    def test_update_panel_pct_right_per_type(self, client):
        res = client.put("/api/settings", json={"panel_pct_right": {"audio": 0.2, "video": 0.35}})
        assert res.status_code == 200
        data = res.get_json()
        assert data["panel_pct_right"]["audio"] == pytest.approx(0.2)
        assert data["panel_pct_right"]["video"] == pytest.approx(0.35)

    def test_update_panel_pct_left_scalar(self, client):
        res = client.put("/api/settings", json={"panel_pct_left": 0.3})
        assert res.status_code == 200
        for v in res.get_json()["panel_pct_left"].values():
            assert v == pytest.approx(0.3)

    def test_update_panel_pct_left_null(self, client):
        res = client.put("/api/settings", json={"panel_pct_left": {"audio": None}})
        assert res.status_code == 200
        assert res.get_json()["panel_pct_left"]["audio"] is None

    def test_update_panel_pct_left_invalid(self, client):
        res = client.put("/api/settings", json={"panel_pct_left": {"audio": "invalid"}})
        assert res.status_code == 400

    def test_update_panel_pct_left_out_of_range(self, client):
        res = client.put("/api/settings", json={"panel_pct_left": 0.01})
        assert res.status_code == 400
        res = client.put("/api/settings", json={"panel_pct_left": 0.95})
        assert res.status_code == 400

    def test_get_settings_includes_panel_pct(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "panel_pct_left" in data
        assert isinstance(data["panel_pct_left"], dict)
        assert "panel_pct_right" in data
        assert isinstance(data["panel_pct_right"], dict)

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

    def test_get_settings_includes_autopilot(self, client):
        res = client.get("/api/settings")
        assert res.status_code == 200
        data = res.get_json()
        assert "autopilot_enabled" in data
        assert "autopilot_top_greens" in data
        assert "autopilot_hard_reds" in data

    def test_get_defaults_includes_autopilot(self, client):
        res = client.get("/api/settings/defaults")
        assert res.status_code == 200
        data = res.get_json()
        assert data["autopilot_enabled"] is True
        assert data["autopilot_top_greens"] == 3
        assert data["autopilot_hard_reds"] == 4

    def test_get_defaults_excludes_directory_settings(self, client):
        res = client.get("/api/settings/defaults")
        assert res.status_code == 200
        data = res.get_json()
        assert "saved_datasets_dir" not in data
        assert "detectors_dir" not in data
        assert "trainable_models_dir" not in data


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
