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

    def test_autorun_detector_names_default_empty(self):
        assert settings_mod.get_autorun_detector_names() == []

    def test_add_autorun_detector_name(self, isolated_settings):
        settings_mod.add_autorun_detector_name("det-a")
        assert settings_mod.get_autorun_detector_names() == ["det-a"]

    def test_add_autorun_detector_name_idempotent(self, isolated_settings):
        settings_mod.add_autorun_detector_name("det-a")
        settings_mod.add_autorun_detector_name("det-a")
        assert settings_mod.get_autorun_detector_names() == ["det-a"]

    def test_remove_autorun_detector_name(self, isolated_settings):
        settings_mod.add_autorun_detector_name("det-a")
        assert settings_mod.remove_autorun_detector_name("det-a") is True
        assert settings_mod.get_autorun_detector_names() == []

    def test_remove_autorun_detector_name_nonexistent(self):
        assert settings_mod.remove_autorun_detector_name("nope") is False

    def test_is_autorun_detector(self, isolated_settings):
        settings_mod.add_autorun_detector_name("det-a")
        assert settings_mod.is_autorun_detector("det-a") is True
        assert settings_mod.is_autorun_detector("det-b") is False

    def test_autorun_detector_names_persists_across_reset(self, isolated_settings):
        settings_mod.add_autorun_detector_name("det-a")
        settings_mod.add_autorun_detector_name("det-b")
        settings_mod.reset()
        assert settings_mod.get_autorun_detector_names() == ["det-a", "det-b"]

    def test_set_autorun_detector_names(self, isolated_settings):
        settings_mod.set_autorun_detector_names(["x", "y", "z"])
        assert settings_mod.get_autorun_detector_names() == ["x", "y", "z"]

    def test_set_autorun_detector_names_deduplicates(self, isolated_settings):
        settings_mod.set_autorun_detector_names(["x", "y", "x"])
        assert settings_mod.get_autorun_detector_names() == ["x", "y"]

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

    def test_get_grid_icon_size_left_default(self):
        result = settings_mod.get_grid_icon_size_left()
        assert isinstance(result, dict)
        for v in result.values():
            assert v == "M"

    def test_get_grid_icon_size_right_default(self):
        result = settings_mod.get_grid_icon_size_right()
        assert isinstance(result, dict)
        for v in result.values():
            assert v == "M"

    def test_set_grid_icon_size_left_per_type(self, isolated_settings):
        settings_mod.set_grid_icon_size_left({"audio": "XS", "image": "XL"})
        result = settings_mod.get_grid_icon_size_left()
        assert result["audio"] == "XS"
        assert result["image"] == "XL"

        raw = json.loads(isolated_settings.read_text())
        assert raw["grid_icon_size_left"]["audio"] == "XS"
        assert raw["grid_icon_size_left"]["image"] == "XL"

    def test_set_grid_icon_size_right_per_type(self, isolated_settings):
        settings_mod.set_grid_icon_size_right({"audio": "S", "video": "L"})
        result = settings_mod.get_grid_icon_size_right()
        assert result["audio"] == "S"
        assert result["video"] == "L"

        raw = json.loads(isolated_settings.read_text())
        assert raw["grid_icon_size_right"]["audio"] == "S"

    def test_grid_icon_size_left_scalar(self, isolated_settings):
        settings_mod.set_grid_icon_size_left("XL")
        result = settings_mod.get_grid_icon_size_left()
        for v in result.values():
            assert v == "XL"

    def test_grid_icon_size_right_scalar(self, isolated_settings):
        settings_mod.set_grid_icon_size_right("XS")
        result = settings_mod.get_grid_icon_size_right()
        for v in result.values():
            assert v == "XS"

    def test_grid_icon_size_left_persists_across_reset(self, isolated_settings):
        settings_mod.set_grid_icon_size_left({"audio": "L"})
        settings_mod.reset()
        assert settings_mod.get_grid_icon_size_left()["audio"] == "L"

    def test_grid_icon_size_right_persists_across_reset(self, isolated_settings):
        settings_mod.set_grid_icon_size_right({"audio": "S"})
        settings_mod.reset()
        assert settings_mod.get_grid_icon_size_right()["audio"] == "S"

    def test_grid_icon_size_left_invalid_value(self):
        with pytest.raises(ValueError):
            settings_mod.set_grid_icon_size_left({"audio": "HUGE"})

    def test_grid_icon_size_right_invalid_value(self):
        with pytest.raises(ValueError):
            settings_mod.set_grid_icon_size_right({"audio": "TINY"})

    def test_grid_icon_size_invalid_scalar(self):
        with pytest.raises(ValueError):
            settings_mod.set_grid_icon_size_left("invalid")

    def test_grid_icon_size_invalid_media_type(self):
        with pytest.raises(ValueError):
            settings_mod.set_grid_icon_size_left({"nonexistent_type": "M"})

    def test_grid_icon_size_case_insensitive(self, isolated_settings):
        settings_mod.set_grid_icon_size_left({"audio": "xs"})
        assert settings_mod.get_grid_icon_size_left()["audio"] == "XS"
        settings_mod.set_grid_icon_size_right("xl")
        for v in settings_mod.get_grid_icon_size_right().values():
            assert v == "XL"

    def test_grid_icon_size_all_valid_sizes(self, isolated_settings):
        for size in ("XS", "S", "M", "L", "XL"):
            settings_mod.set_grid_icon_size_left({"audio": size})
            assert settings_mod.get_grid_icon_size_left()["audio"] == size

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
            assert v == 260

    def test_get_panel_pct_right_default(self):
        result = settings_mod.get_panel_pct_right()
        assert isinstance(result, dict)
        for v in result.values():
            assert v == 300

    def test_set_panel_pct_left_per_type(self, isolated_settings):
        settings_mod.set_panel_pct_left({"audio": 200, "image": 350})
        result = settings_mod.get_panel_pct_left()
        assert result["audio"] == 200
        assert result["image"] == 350

        raw = json.loads(isolated_settings.read_text())
        assert raw["panel_pct_left"]["audio"] == 200
        assert raw["panel_pct_left"]["image"] == 350

    def test_set_panel_pct_right_per_type(self, isolated_settings):
        settings_mod.set_panel_pct_right({"audio": 250, "video": 400})
        result = settings_mod.get_panel_pct_right()
        assert result["audio"] == 250
        assert result["video"] == 400

        raw = json.loads(isolated_settings.read_text())
        assert raw["panel_pct_right"]["audio"] == 250

    def test_set_panel_pct_left_scalar(self, isolated_settings):
        settings_mod.set_panel_pct_left(300)
        result = settings_mod.get_panel_pct_left()
        for v in result.values():
            assert v == 300

    def test_panel_pct_persists_across_reset(self, isolated_settings):
        settings_mod.set_panel_pct_left({"audio": 200})
        settings_mod.reset()
        assert settings_mod.get_panel_pct_left()["audio"] == 200

    def test_panel_pct_out_of_range(self):
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left({"audio": 100})
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left({"audio": 600})
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left(100)
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left(600)

    def test_panel_pct_invalid_media_type(self):
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left({"nonexistent_type": 250})

    def test_panel_pct_invalid_value(self):
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left({"audio": "invalid"})

    def test_panel_pct_clamped_on_read(self, isolated_settings):
        """Out-of-range values stored on disk are clamped when read."""
        settings_mod.set_panel_pct_left({"audio": 300})
        # Manually write an out-of-range value to disk
        raw = json.loads(isolated_settings.read_text())
        raw["panel_pct_left"]["audio"] = 700
        isolated_settings.write_text(json.dumps(raw))
        settings_mod.reset()
        result = settings_mod.get_panel_pct_left()
        assert result["audio"] == 500

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
        assert isinstance(defaults["grid_icon_size_left"], dict)
        for v in defaults["grid_icon_size_left"].values():
            assert v == "M"
        assert isinstance(defaults["grid_icon_size_right"], dict)
        for v in defaults["grid_icon_size_right"].values():
            assert v == "M"
        assert isinstance(defaults["focus_mode_left"], dict)
        for v in defaults["focus_mode_left"].values():
            assert v == "click"
        assert isinstance(defaults["focus_mode_right"], dict)
        for v in defaults["focus_mode_right"].values():
            assert v == "click"
        assert isinstance(defaults["panel_pct_left"], dict)
        for v in defaults["panel_pct_left"].values():
            assert v == 260
        assert isinstance(defaults["panel_pct_right"], dict)
        for v in defaults["panel_pct_right"].values():
            assert v == 300
        assert defaults["autopilot_enabled"] is True
        assert defaults["autopilot_top_greens"] == 3
        assert defaults["autopilot_hard_reds"] == 4
        assert defaults["autopilot_goal_diversity"] == 40
        assert "autorun_processors" not in defaults
        # Directory settings excluded from defaults (not reset by Default button)
        assert "saved_datasets_dir" not in defaults
        assert "detectors_dir" not in defaults
        assert "trainable_models_dir" not in defaults

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

    def test_get_set_autopilot_goal_diversity(self, isolated_settings):
        settings_mod.set_autopilot_goal_diversity(60)
        assert settings_mod.get_autopilot_goal_diversity() == 60

        raw = json.loads(isolated_settings.read_text())
        assert raw["autopilot_goal_diversity"] == 60

    def test_autopilot_goal_diversity_clamped(self):
        settings_mod.set_autopilot_goal_diversity(0)
        assert settings_mod.get_autopilot_goal_diversity() == 1

        settings_mod.set_autopilot_goal_diversity(-5)
        assert settings_mod.get_autopilot_goal_diversity() == 1

    def test_autopilot_goal_diversity_default(self):
        assert settings_mod.get_autopilot_goal_diversity() == 40

    def test_autopilot_goal_diversity_persists_across_reset(self, isolated_settings):
        settings_mod.set_autopilot_goal_diversity(80)
        settings_mod.reset()
        assert settings_mod.get_autopilot_goal_diversity() == 80

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

        # Create a fake detector JSON (origin-based with weight fallback)
        fake_o = [{"origin": {"importer": "test"}, "origin_name": "t.wav", "filename": "t.wav", "md5": "abc"}]
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
                    "good_origins": fake_o,
                    "bad_origins": fake_o,
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


