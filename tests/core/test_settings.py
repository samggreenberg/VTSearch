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

import app as app_module  # noqa: F401 — triggers conftest media init
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
        # The hardware-derived default must NOT be persisted to disk — only
        # explicit ``set_*`` calls write keys. (The fixture pre-writes a few
        # path keys; we just check our key is absent.)
        raw = isolated_settings.read_text() if isolated_settings.exists() else ""
        assert "max_concurrent_dataset_downloads" not in raw

    def test_max_concurrent_embeddings_default_from_hardware(self, isolated_settings):
        """CPU-only boxes default to 1; CUDA boxes default to min(2, gpu_count)."""
        from vtsearch.embedding.loader import _detect_cuda_devices

        gpus = _detect_cuda_devices()
        expected = max(1, min(2, gpus)) if gpus > 0 else 1
        assert settings_mod.get_max_concurrent_dataset_embeddings() == expected

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
            settings_mod.set_panel_pct_left({"audio": "invalid"})  # pyright: ignore[reportArgumentType]

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
        # Theme defaults to ``"system"`` — the frontend resolves this to
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
