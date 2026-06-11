"""Tests for the persistent settings module.

Covers:
- Settings file read/write (vtsearch.settings)
- Volume persistence
- autorun_detectors list management
- Flask API routes: GET/PUT /api/settings
"""

from __future__ import annotations

import json

import pytest

import app as app_module  # noqa: F401  (triggers conftest media init)
from vtsearch import settings as settings_mod


# ---------------------------------------------------------------------------
# Settings module unit tests
# ---------------------------------------------------------------------------


class TestSettingsModule:
    def test_defaults_when_no_file(self):
        data = settings_mod.get_all()
        assert data["volume"] == 1.0
        assert data["autorun_detectors"] == []

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

    def test_autorun_detectors_default_empty(self):
        assert settings_mod.get_autorun_detectors() == []

    def test_add_autorun_trainable_model(self, isolated_settings):
        settings_mod.add_autorun_detector("model-a")
        assert settings_mod.get_autorun_detectors() == ["model-a"]

    def test_add_autorun_trainable_model_idempotent(self, isolated_settings):
        settings_mod.add_autorun_detector("model-a")
        settings_mod.add_autorun_detector("model-a")
        assert settings_mod.get_autorun_detectors() == ["model-a"]

    def test_remove_autorun_trainable_model(self, isolated_settings):
        settings_mod.add_autorun_detector("model-a")
        assert settings_mod.remove_autorun_detector("model-a") is True
        assert settings_mod.get_autorun_detectors() == []

    def test_remove_autorun_trainable_model_nonexistent(self):
        assert settings_mod.remove_autorun_detector("nope") is False

    def test_is_autorun_trainable_model(self, isolated_settings):
        settings_mod.add_autorun_detector("model-a")
        assert settings_mod.is_autorun_detector("model-a") is True
        assert settings_mod.is_autorun_detector("model-b") is False

    def test_autorun_detectors_persists_across_reset(self, isolated_settings):
        settings_mod.add_autorun_detector("model-a")
        settings_mod.add_autorun_detector("model-b")
        settings_mod.reset()
        assert settings_mod.get_autorun_detectors() == ["model-a", "model-b"]

    def test_set_autorun_detectors(self, isolated_settings):
        settings_mod.set_autorun_detectors(["x", "y", "z"])
        assert settings_mod.get_autorun_detectors() == ["x", "y", "z"]

    def test_set_autorun_detectors_deduplicates(self, isolated_settings):
        settings_mod.set_autorun_detectors(["x", "y", "x"])
        assert settings_mod.get_autorun_detectors() == ["x", "y"]

    def test_max_concurrent_downloads_default_from_hardware(self, isolated_settings):
        """With no override on disk, the default scales with cpu_count (cap 4)."""
        import os

        expected = max(1, min(4, os.cpu_count() or 1))
        assert settings_mod.get_max_concurrent_dataset_downloads() == expected
        # The hardware-derived default must NOT be persisted to disk; only
        # explicit ``set_*`` calls write keys. (The fixture pre-writes a few
        # path keys; we just check our key is absent.)
        raw = isolated_settings.read_text() if isolated_settings.exists() else ""
        assert "max_concurrent_dataset_downloads" not in raw

    def test_max_concurrent_embeddings_default_from_hardware(self, isolated_settings):
        """The default mirrors the hardware-derived value and is not persisted.

        The concrete scaling (CPU cores, total RAM) is unit-tested in
        ``tests_lib/datasets/test_concurrency_defaults.py``; here we only assert
        the settings layer surfaces that value and doesn't write it to disk.
        """
        from vtscore.embedding.loader import default_concurrent_embeddings

        assert settings_mod.get_max_concurrent_dataset_embeddings() == default_concurrent_embeddings()
        raw = isolated_settings.read_text() if isolated_settings.exists() else ""
        assert "max_concurrent_dataset_embeddings" not in raw

    def test_max_concurrent_explicit_value_wins_over_hardware_default(self, isolated_settings):
        """An explicit ``set_*`` call overrides the hardware-derived default."""
        settings_mod.set_max_concurrent_dataset_downloads(7)
        settings_mod.set_max_concurrent_dataset_embeddings(3)
        assert settings_mod.get_max_concurrent_dataset_downloads() == 7
        assert settings_mod.get_max_concurrent_dataset_embeddings() == 3
        # And it survives a reset (read back from disk).
        settings_mod.reset()
        assert settings_mod.get_max_concurrent_dataset_downloads() == 7
        assert settings_mod.get_max_concurrent_dataset_embeddings() == 3

    def test_persistence_survives_reset(self, isolated_settings):
        settings_mod.set_volume(0.7)
        settings_mod.add_autorun_detector("p")

        # Simulate restart
        settings_mod.reset()

        assert settings_mod.get_volume() == pytest.approx(0.7)
        assert settings_mod.get_autorun_detectors() == ["p"]

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
        assert settings_mod.get_calibrate_count() == 1

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

    def test_label_hint_dismissed_default(self):
        assert settings_mod.get_label_hint_dismissed() is False

    def test_get_set_label_hint_dismissed(self, isolated_settings):
        settings_mod.set_label_hint_dismissed(True)
        assert settings_mod.get_label_hint_dismissed() is True

        raw = json.loads(isolated_settings.read_text())
        assert raw["label_hint_dismissed"] is True

    def test_label_hint_dismissed_persists_across_reset(self, isolated_settings):
        settings_mod.set_label_hint_dismissed(True)
        settings_mod.reset()
        assert settings_mod.get_label_hint_dismissed() is True

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
            settings_mod.set_panel_pct_left({"audio": 20000})
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left(100)
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left(20000)

    def test_panel_pct_wide_panel_allowed(self, isolated_settings):
        """Widths above the old 500px cap are accepted (no upper-bound clip)."""
        settings_mod.set_panel_pct_left({"audio": 520})
        assert settings_mod.get_panel_pct_left()["audio"] == 520

    def test_panel_pct_invalid_media_type(self):
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left({"nonexistent_type": 250})

    def test_panel_pct_invalid_value(self):
        with pytest.raises(ValueError):
            settings_mod.set_panel_pct_left({"audio": "invalid"})  # pyright: ignore[reportArgumentType]

    def test_panel_pct_clamped_on_read(self, isolated_settings):
        """Out-of-range values stored on disk are clamped when read."""
        settings_mod.set_panel_pct_left({"audio": 300})
        # Manually write an out-of-range value to disk
        raw = json.loads(isolated_settings.read_text())
        raw["panel_pct_left"]["audio"] = 20000
        isolated_settings.write_text(json.dumps(raw))
        settings_mod.reset()
        result = settings_mod.get_panel_pct_left()
        assert result["audio"] == 10000

    def test_get_defaults(self):
        defaults = settings_mod.get_defaults()
        assert defaults["volume"] == 1.0
        # Theme defaults to ``"system"``; the frontend resolves this to
        # the OS ``prefers-color-scheme`` at render time.
        assert defaults["theme"] == "system"
        assert defaults["calibrate_count"] == 1
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
        assert "autorun_detectors" not in defaults
        # Directory settings excluded from defaults (not reset by Default button)
        assert "saved_datasets_dir" not in defaults
        assert "detectors_dir" not in defaults

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
        assert settings_mod.get_autorun_detectors() == []

    def test_last_embedder_per_media_type_default(self):
        assert settings_mod.get_last_embedder_per_media_type() == {}
        assert settings_mod.get_last_embedder_for_media_type("image") == ""

    def test_set_last_embedder_for_media_type(self, isolated_settings):
        settings_mod.set_last_embedder_for_media_type("image", "siglip")
        assert settings_mod.get_last_embedder_for_media_type("image") == "siglip"
        assert settings_mod.get_last_embedder_per_media_type() == {"image": "siglip"}

        raw = json.loads(isolated_settings.read_text())
        assert raw["last_embedder_per_media_type"] == {"image": "siglip"}

    def test_last_embedder_per_media_type_independent_keys(self, isolated_settings):
        settings_mod.set_last_embedder_for_media_type("image", "siglip")
        settings_mod.set_last_embedder_for_media_type("audio", "clap")
        assert settings_mod.get_last_embedder_for_media_type("image") == "siglip"
        assert settings_mod.get_last_embedder_for_media_type("audio") == "clap"

    def test_last_embedder_for_media_type_ignores_empty_args(self, isolated_settings):
        settings_mod.set_last_embedder_for_media_type("image", "siglip")
        # Empty media_type or embedder is a no-op.
        settings_mod.set_last_embedder_for_media_type("", "anything")
        settings_mod.set_last_embedder_for_media_type("image", "")
        assert settings_mod.get_last_embedder_for_media_type("image") == "siglip"

    def test_last_embedder_per_media_type_persists_across_reset(self, isolated_settings):
        settings_mod.set_last_embedder_for_media_type("image", "siglip")
        settings_mod.reset()
        assert settings_mod.get_last_embedder_for_media_type("image") == "siglip"

    def test_import_defaults_by_media_type_default(self):
        # Empty per-user dict; nothing saved means the importer falls
        # back to its own natural defaults.
        assert settings_mod.get_import_defaults_by_media_type() == {}

    def test_import_defaults_by_media_type_roundtrip(self, isolated_settings):
        value = {
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
        }
        settings_mod.set_import_defaults_by_media_type(value)
        assert settings_mod.get_import_defaults_by_media_type() == value

        raw = json.loads(isolated_settings.read_text())
        assert raw["import_defaults_by_media_type"] == value

    def test_import_defaults_by_media_type_persists_across_reset(self, isolated_settings):
        value = {"audio": {"embedder": "clap"}}
        settings_mod.set_import_defaults_by_media_type(value)
        settings_mod.reset()
        assert settings_mod.get_import_defaults_by_media_type() == value


class TestConcurrentWrites:
    """Cross-process safety for per-user settings writes.

    H28 fix: each setter does a read-modify-write under a cross-process
    file lock, so a stale in-process cache can no longer clobber
    concurrent updates to other top-level keys.
    """

    def test_concurrent_write_preserves_other_processes_key(self, isolated_settings):
        # Populate our cache with a known value.
        settings_mod.set_volume(0.5)

        # Simulate another process writing a DIFFERENT key directly to
        # disk after our cache was populated.
        user_path = isolated_settings._user
        existing = json.loads(user_path.read_text())
        existing["theme"] = "dark"
        user_path.write_text(json.dumps(existing))

        # Our next write should re-read disk first and merge.
        settings_mod.set_inclusion(3)

        raw = json.loads(user_path.read_text())
        assert raw["volume"] == 0.5
        assert raw["theme"] == "dark"  # would be lost without the RMW fix
        assert raw["inclusion"] == 3

    def test_atomic_write_uses_unique_tmp_filenames(self, isolated_settings):
        """Tmp files must include PID + uuid so two writers can't truncate
        each other's in-flight temp file."""
        import re

        settings_mod.set_volume(0.5)
        # ``_atomic_write`` removes the tmp via rename, but the filename
        # shape is what we care about. Confirm by inspecting the helper.
        from vtsearch.settings import _atomic_write

        # Drive it once and check the tmp name pattern used by inspecting
        # the directory contents during a captured write.
        captured: list[str] = []
        real_replace = __import__("os").replace

        def spy_replace(src, dst):
            captured.append(str(src))
            real_replace(src, dst)

        import os as _os

        _os.replace = spy_replace
        try:
            _atomic_write(isolated_settings._user, {"volume": 0.5})
        finally:
            _os.replace = real_replace

        assert captured, "expected at least one tmp file rename"
        tmp_name = captured[-1].rsplit("/", 1)[-1]
        # user_settings.json.<pid>.<uuid32>.tmp
        assert re.match(r"^user_settings\.json\.\d+\.[0-9a-f]{32}\.tmp$", tmp_name), tmp_name

    def test_concurrent_threads_no_deadlock(self, isolated_settings):
        """Two threads concurrently setting different keys must both complete.

        Catches lock-ordering regressions (settings_lock vs file_lock).
        """
        import threading

        errors: list[Exception] = []
        ready = threading.Event()
        ready_count = [0]
        ready_lock = threading.Lock()

        def writer(fn, value):
            try:
                with ready_lock:
                    ready_count[0] += 1
                    if ready_count[0] == 2:
                        ready.set()
                ready.wait(timeout=5)
                fn(value)
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(settings_mod.set_volume, 0.5)),
            threading.Thread(target=writer, args=(settings_mod.set_inclusion, 3)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"writers raised: {errors}"
        for t in threads:
            assert not t.is_alive(), "thread hung (likely deadlock)"

        assert settings_mod.get_volume() == pytest.approx(0.5)
        assert settings_mod.get_inclusion() == 3

    def test_mutate_user_rmw_preserves_concurrent_writes(self, isolated_settings):
        """``mutate_user`` re-reads disk before applying its mutator, so
        nested-dict updates (e.g. achievement counters) merge with
        whatever a sibling process wrote since our cache was populated."""
        from vtsearch.settings import mutate_user

        # Seed.
        mutate_user(lambda c: c.update({"theme": "light", "achievement_state": {"counters": {"votes_cast": 0}}}))

        # Simulate another process bumping votes_cast and changing theme
        # on disk while we hold a stale in-memory cache.
        user_path = isolated_settings._user
        disk = json.loads(user_path.read_text())
        disk["theme"] = "dark"
        disk["achievement_state"]["counters"]["votes_cast"] = 100
        user_path.write_text(json.dumps(disk))

        # Our increment should read the disk value (100), bump to 101,
        # and preserve the disk's theme update.
        def _bump(cache):
            cache["achievement_state"]["counters"]["votes_cast"] += 1

        mutate_user(_bump)

        final = json.loads(user_path.read_text())
        assert final["theme"] == "dark"
        assert final["achievement_state"]["counters"]["votes_cast"] == 101

    def test_add_autorun_detector_rmw(self, isolated_settings):
        """Concurrent ``add_autorun_detector`` calls must not lose entries.

        ``autorun_detectors`` is per-user now, so the cross-process race plays
        out on the (default) user's settings file: our cache says
        ``["ours-pre"]`` but disk also has ``"sibling"`` because a sibling
        process added it. The atomic per-user RMW must merge, not clobber.
        """
        # Force-load the per-user cache (so add_autorun_detector below
        # doesn't trigger a fresh load).
        settings_mod.add_autorun_detector("ours-pre")

        # Sibling process directly adds an entry on disk (the per-user file).
        user_path = isolated_settings._user
        disk = json.loads(user_path.read_text())
        disk["autorun_detectors"] = list(disk.get("autorun_detectors", [])) + ["sibling"]
        user_path.write_text(json.dumps(disk))

        # We add another; should merge with disk, not clobber.
        settings_mod.add_autorun_detector("ours-post")

        final = json.loads(user_path.read_text())
        assert "sibling" in final["autorun_detectors"]
        assert "ours-pre" in final["autorun_detectors"]
        assert "ours-post" in final["autorun_detectors"]
