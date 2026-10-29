"""Inclusion is retired as a user preference, and pinned to 0 on the library surface (#4269).

The balance is the operating point (#4413).  Inclusion survives only as the
internal unit the threshold machinery measures cuts in, so the public names
that used to set it keep importing - an out-of-tree caller is not broken on
import - but accept only ``0``, with a ``DeprecationWarning``.  Any other value
is refused rather than ignored: a knob that silently stops moving the line is
worse than an error.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest

from vtscore import state
from vtscore.config import CoreConfig


def _config(**overrides) -> CoreConfig:
    return dataclasses.replace(CoreConfig.from_settings(), **overrides)


class TestTheStateAccessors:
    def test_get_inclusion_is_always_zero(self):
        with pytest.warns(DeprecationWarning, match="get_inclusion"):
            assert state.get_inclusion() == 0

    def test_set_inclusion_accepts_zero_with_a_warning(self):
        with pytest.warns(DeprecationWarning, match="set_inclusion"):
            state.set_inclusion(0)

    @pytest.mark.parametrize("value", [3, -1, 10])
    def test_set_inclusion_refuses_any_other_value(self, value):
        with pytest.raises(ValueError, match="set_beta"):
            state.set_inclusion(value)

    def test_the_old_recompute_hook_accepts_only_zero(self):
        from vtscore.state.core import recompute_detector_thresholds_for_inclusion

        with pytest.warns(DeprecationWarning):
            recompute_detector_thresholds_for_inclusion(0)
        with pytest.raises(ValueError, match="set_beta"):
            recompute_detector_thresholds_for_inclusion(4)

    def test_a_detector_context_has_no_inclusion(self):
        assert not hasattr(state.DetectorContext("det-no-inclusion"), "inclusion")


class TestTheSettingsSeam:
    def test_an_inclusion_persister_is_accepted_and_never_fires(self):
        fired: list = []
        with pytest.warns(DeprecationWarning, match="never be called"):
            state.register_setting_persister("inclusion", fired.append)
        assert "inclusion" not in state._setting_persisters
        assert "inclusion" not in state.KNOWN_SETTING_KEYS
        with pytest.warns(DeprecationWarning):
            state.set_inclusion(0)
        assert fired == []

    def test_an_unknown_key_is_still_refused(self):
        with pytest.raises(ValueError, match="Unknown setting key"):
            state.register_setting_persister("not-a-setting", lambda _v: None)


class TestCoreConfig:
    def test_unset_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            config = _config()
            dataclasses.replace(config, calibrate_count=3)
        assert config.inclusion is None

    def test_zero_is_accepted_with_a_warning(self):
        with pytest.warns(DeprecationWarning, match="CoreConfig"):
            _config(inclusion=0)

    def test_any_other_value_is_refused(self):
        with pytest.raises(ValueError, match="set_beta"):
            _config(inclusion=2)


class TestTheTrainingEntryPoints:
    """Refused before any training work, so no fixture data is needed."""

    def test_train_and_score(self):
        from vtscore.detectors.training import train_and_score

        with pytest.raises(ValueError, match="set_beta"):
            train_and_score({}, {}, {}, inclusion_value=3)
        with pytest.raises(ValueError, match="set_beta"):
            train_and_score({}, {}, {}, 3)

    def test_labelset_train_and_score(self):
        from vtscore.detectors.labelset_training import labelset_train_and_score

        with pytest.raises(ValueError, match="set_beta"):
            labelset_train_and_score(None, None, media_type="audio", clips_dict={}, inclusion_value=-2)  # type: ignore[arg-type]

    def test_the_learned_sort(self):
        from vtscore.detectors.learned_sort import build_learned_sort_signature, run_learned_sort

        with pytest.raises(ValueError, match="set_beta"):
            build_learned_sort_signature(
                det_ctx=None,
                ds_ctx=None,
                snap={},
                labelset=None,
                good={},
                bad={},
                region_boxes_snapshot={},
                calibrate_count_value=2,
                calibration_fraction_value=None,
                inclusion_value=1,
            )
        with pytest.raises(ValueError, match="set_beta"):
            run_learned_sort(
                det_ctx=None,
                ds_ctx=None,
                snap={},
                labelset=None,
                det_media_type="audio",
                good={},
                bad={},
                region_boxes_snapshot={},
                calibrate_count_value=2,
                calibration_fraction_value=None,
                inclusion_value=1,
            )
