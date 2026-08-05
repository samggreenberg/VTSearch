"""The two training entry points must agree on the discarded x-cal placeholder (#2841).

When the mix-in schedule gives the cross-calibration cut zero weight, both
training paths skip the fold trainings and leave a placeholder behind.  That
placeholder is *normally* discarded by the pure-GMM blend - but
``blend_gmm_threshold`` falls back to it when the GMM fit is non-finite, and the
two paths used to leave **different** values there:

* ``train_and_threshold`` (the Find path) left ``NO_GOOD_THRESHOLD``, so a
  degenerate GMM admitted nothing at all;
* ``_train_and_score_xy`` (the vote / labelset path) left ``0.5``, so the same
  collection admitted everything scoring at or above a half.

Same collection, same votes, opposite behaviour, decided by which entry point
happened to run.  These tests pin the two together.
"""

from __future__ import annotations

import numpy as np
import pytest

import vtscore.state as core_state
import vtscore.training.thresholds as thresholds_mod
from vtscore.detectors.training import _train_and_score_xy, train_and_threshold
from vtscore.training.thresholds import NO_GOOD_THRESHOLD

DIM = 8
ID_BASE = 7000


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)


@pytest.fixture
def degenerate_gmm(monkeypatch):
    """Safe-thresholds on, and every GMM fit comes back non-finite.

    This is the regime that makes the placeholder observable: a real fit hides
    it, because the blend then returns the GMM cut and never consults the
    discarded value.
    """
    monkeypatch.setattr(core_state, "get_safe_thresholds", lambda: True)
    monkeypatch.setattr(thresholds_mod, "fit_gmm_threshold", lambda scores: (float("nan"), None))


def _tiny_labelset(rng: np.random.Generator):
    """Four votes - below the production ramp floor, so x-cal is discarded."""
    good = _unit(rng.standard_normal(DIM))
    bad = _unit(rng.standard_normal(DIM))
    X_list = [
        _unit(good + 0.05 * rng.standard_normal(DIM)),
        _unit(good + 0.05 * rng.standard_normal(DIM)),
        _unit(bad + 0.05 * rng.standard_normal(DIM)),
        _unit(bad + 0.05 * rng.standard_normal(DIM)),
    ]
    return X_list, [1.0, 1.0, 0.0, 0.0]


def _snap(rng: np.random.Generator, n: int = 12) -> dict[int, dict]:
    return {
        cid: {
            "id": cid,
            "media_type": "image",
            "embedder": "siglip",
            "embeddings": {"siglip": _unit(rng.standard_normal(DIM))},
        }
        for cid in range(ID_BASE + 1, ID_BASE + n + 1)
    }


class TestPlaceholderAgreement:
    def test_both_paths_admit_nothing_when_the_gmm_degenerates(self, degenerate_gmm):
        rng = np.random.default_rng(2841)
        X_list, y_list = _tiny_labelset(rng)
        snap = _snap(rng)

        _model, find_threshold = train_and_threshold(X_list, y_list, snap=snap, embedder_name="siglip")
        _rows, vote_threshold, _m = _train_and_score_xy(
            X_list,
            y_list,
            snap,
            inclusion_value=0,
            safe_thresholds=True,
            calibrate_count=2,
            calibration_fraction=0.5,
            det_ctx=None,
        )

        assert find_threshold == vote_threshold, (
            "the Find path and the vote path disagreed on the discarded placeholder, "
            "so the same votes admit different media depending on which one ran"
        )
        assert find_threshold == NO_GOOD_THRESHOLD, (
            "a threshold that was never computed must admit nothing, not half the collection"
        )

    def test_a_healthy_gmm_hides_the_placeholder_entirely(self, monkeypatch):
        """Sanity check on the fixture: with a real fit, neither path returns the
        placeholder - which is why the divergence went unnoticed for so long."""
        monkeypatch.setattr(core_state, "get_safe_thresholds", lambda: True)
        rng = np.random.default_rng(2842)
        X_list, y_list = _tiny_labelset(rng)
        snap = _snap(rng)

        _model, find_threshold = train_and_threshold(X_list, y_list, snap=snap, embedder_name="siglip")
        assert find_threshold != NO_GOOD_THRESHOLD
        assert 0.0 <= find_threshold <= 1.0
