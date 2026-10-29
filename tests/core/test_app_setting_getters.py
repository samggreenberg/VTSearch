"""The app must answer every library setting from ``vtsearch.settings`` (#4997).

Library code reads its knobs with :func:`vtscore.config.get_setting`, and
``vtsearch.seam_wiring.register_app_setting_getters()`` installs one getter per knob
at startup.  A knob the shim forgets is not a loud failure: ``get_setting``
quietly falls through to the ``CoreConfig`` field default, so the app would
ignore that setting.  These tests are the thing that notices.
"""

from __future__ import annotations

from pathlib import Path

from vtsearch import settings
from vtscore.config import CoreConfig, core_config, get_setting, redirect_settings_path


class TestStartupWiring:
    def test_app_registers_a_getter_for_every_setting(self):
        assert set(core_config._setting_getters) == core_config._SETTING_NAMES

    def test_app_registers_the_settings_path_redirect(self):
        assert core_config._settings_path_redirect is not None


class TestReadsTheLiveSettings:
    def test_each_read_reaches_the_current_settings(self, isolated_settings):
        settings.set_calibrate_count(5)
        assert get_setting("calibrate_count") == 5
        settings.set_calibrate_count(3)
        assert get_setting("calibrate_count") == 3

    def test_a_patched_settings_getter_is_seen(self, monkeypatch):
        monkeypatch.setattr(settings, "get_detectors_dir", lambda: Path("/patched/detectors"))
        assert get_setting("detectors_dir") == Path("/patched/detectors")

    def test_server_settings_reach_the_library(self, isolated_settings):
        settings.set_browse_signpost_vocab({"image": ["cat", "dog"]})
        assert get_setting("signpost_vocab") == {"image": ["cat", "dog"]}
        assert get_setting("saved_datasets_dir") == settings.get_saved_datasets_dir()

    def test_from_settings_assembles_the_same_values(self, isolated_settings):
        settings.set_autopilot_goal_diversity(12)
        assembled = CoreConfig.from_settings()
        assert assembled.autopilot_goal_diversity == 12
        assert assembled.detectors_dir == settings.get_detectors_dir()

    def test_the_redirect_goes_to_set_settings_path(self, monkeypatch):
        seen: list[object] = []
        monkeypatch.setattr(settings, "set_settings_path", seen.append)
        redirect_settings_path("/tmp/run-settings.json")
        assert seen == ["/tmp/run-settings.json"]
