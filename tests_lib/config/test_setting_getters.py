"""The per-knob setting seam: :func:`vtscore.config.get_setting` and its hooks (#4997).

Library code reads one knob at a time instead of snapshotting a whole
``CoreConfig`` per lookup.  A knob answers from the host's getter, else the
library consumer's registered builder, else the ``CoreConfig`` field default -
and a required field with none of those raises rather than guessing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from vtscore.config import (
    CoreConfig,
    DATA_DIR,
    get_setting,
    redirect_settings_path,
    register_setting_getter,
    register_settings_path_redirect,
)
from vtscore.config import core_config


def _config(**overrides) -> CoreConfig:
    base = CoreConfig(
        saved_datasets_dir=Path("/srv/datasets"),
        detectors_dir=Path("/srv/detectors"),
        max_concurrent_dataset_downloads=2,
        max_concurrent_dataset_embeddings=1,
        autofind_detectors=("dogs",),
        dataset_max_age_days=None,
        calibrate_count=4,
        calibration_fraction=None,
        enrich_descriptions=False,
        autopilot_goal_diversity=8,
        data_dir=DATA_DIR,
    )
    return dataclasses.replace(base, **overrides)


@pytest.fixture
def bare(monkeypatch):
    """Nothing registered: no getters, no builder, no redirect."""
    monkeypatch.setattr(core_config, "_setting_getters", {})
    monkeypatch.setattr(core_config, "_core_config_builder", None)
    monkeypatch.setattr(core_config, "_settings_path_redirect", None)


class TestResolution:
    def test_a_host_getter_wins_over_the_builder(self, bare):
        core_config.register_core_config_builder(lambda _path=None: _config(calibrate_count=4))
        register_setting_getter("calibrate_count", lambda: 7)
        assert get_setting("calibrate_count") == 7

    def test_a_knob_with_no_getter_reads_the_builder(self, bare):
        core_config.register_core_config_builder(lambda _path=None: _config(detectors_dir=Path("/x/detectors")))
        register_setting_getter("calibrate_count", lambda: 7)
        assert get_setting("detectors_dir") == Path("/x/detectors")

    def test_the_getter_is_called_on_every_read(self, bare):
        """No snapshot: a setting changed between two reads is seen by the second."""
        values = iter([1, 2])
        register_setting_getter("autopilot_goal_diversity", lambda: next(values))
        assert get_setting("autopilot_goal_diversity") == 1
        assert get_setting("autopilot_goal_diversity") == 2

    def test_a_defaulted_knob_with_nothing_registered_reads_its_field_default(self, bare):
        assert get_setting("labelset_source_push_interval") == core_config.DEFAULT_LABELSET_SOURCE_PUSH_INTERVAL
        assert get_setting("default_labelset_source") is None
        assert get_setting("signpost_vocab") == {}

    def test_a_mutable_default_is_fresh_on_every_read(self, bare):
        first = get_setting("signpost_captioner")
        first["image"] = True
        assert get_setting("signpost_captioner") == {}

    def test_a_required_knob_with_nothing_registered_raises(self, bare):
        with pytest.raises(RuntimeError, match="'detectors_dir'"):
            get_setting("detectors_dir")


class TestNames:
    def test_every_non_retired_field_is_a_setting(self):
        fields = {f.name for f in dataclasses.fields(CoreConfig)}
        assert core_config._SETTING_NAMES == fields - {"hide_ingest_eta", "inclusion"}

    @pytest.mark.parametrize("name", ["nonsense", "hide_ingest_eta", "inclusion"])
    def test_get_refuses_a_name_that_is_not_a_setting(self, name):
        with pytest.raises(ValueError, match="not a vtscore setting"):
            get_setting(name)

    def test_register_refuses_a_name_that_is_not_a_setting(self, bare):
        with pytest.raises(ValueError, match="not a vtscore setting"):
            register_setting_getter("detector_dir", lambda: Path("/typo"))
        assert core_config._setting_getters == {}


class TestRedirect:
    def test_the_redirect_hook_receives_the_path(self, bare):
        seen: list[object] = []
        register_settings_path_redirect(seen.append)
        redirect_settings_path("/tmp/run.json")
        assert seen == ["/tmp/run.json"]

    def test_without_a_hook_the_builder_is_handed_the_path(self, bare):
        seen: list[object] = []

        def _build(settings_path=None):
            seen.append(settings_path)
            return _config()

        core_config.register_core_config_builder(_build)
        redirect_settings_path("/tmp/run.json")
        assert seen == ["/tmp/run.json"]

    def test_with_neither_there_is_nothing_to_redirect(self, bare):
        redirect_settings_path("/tmp/run.json")


class TestFromSettings:
    def test_a_lone_builder_answers_outright(self, bare):
        built = _config()
        core_config.register_core_config_builder(lambda _path=None: built)
        assert CoreConfig.from_settings() is built

    def test_with_getters_it_assembles_what_get_setting_reads(self, bare):
        core_config.register_core_config_builder(lambda _path=None: _config(calibrate_count=4))
        register_setting_getter("calibrate_count", lambda: 9)
        assembled = CoreConfig.from_settings()
        assert assembled.calibrate_count == 9
        assert assembled.detectors_dir == Path("/srv/detectors")
        assert assembled.inclusion is None

    def test_a_settings_path_is_redirected_before_any_knob_is_read(self, bare):
        events: list[str] = []
        values = dataclasses.asdict(_config())
        register_settings_path_redirect(lambda path: events.append(f"redirect {path}"))
        for name in core_config._SETTING_NAMES:
            register_setting_getter(name, lambda name=name: events.append(name) or values[name])

        assembled = CoreConfig.from_settings("/tmp/run.json")

        assert events[0] == "redirect /tmp/run.json"
        assert sorted(events[1:]) == sorted(core_config._SETTING_NAMES)
        assert assembled.detectors_dir == Path("/srv/detectors")

    def test_with_nothing_registered_it_raises(self, bare):
        with pytest.raises(RuntimeError, match="No value for the setting"):
            CoreConfig.from_settings()
