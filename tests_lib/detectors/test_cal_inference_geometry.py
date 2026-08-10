"""Calibrating in the scorer's geometry: bag-aware ``stratified_fold_orderings``.

The box-pool path trains a HAC negative on a single whole-image CLS row, but inference
scores an image by ``max`` over every tree node. ``max`` over N rows is an upward-biased
order statistic, so a cut fitted to single rows sits below every inference score and the
detector labels everything positive (measured on vg_s/car: FNR 0.0, FPR 1.0, cost 1.0).

``groups`` + ``score_rows_by_group`` close that gap by collapsing each voted image to one
max-pooled score over the rows the scorer will actually use. These tests pin the grouped
split semantics, the score collapse, and that the ungrouped path is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.threshold_rules import _group_index, calibrated_threshold, stratified_fold_orderings


def _linear_trainer(w_index: int = 0):
    """Trainer whose model reads off one feature, so scores are exactly predictable."""

    def trainer(x, y, seed):
        def predict(z):
            return np.asarray(z, dtype=np.float64)[:, w_index]

        return predict

    return trainer


class TestGroupIndex:
    def test_preserves_first_seen_order_and_collects_rows(self):
        y = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
        gids, rows, labels = _group_index(["a", "a", "b", "c", "b"], y)
        assert gids == ["a", "b", "c"]
        assert rows == [[0, 1], [2, 4], [3]]
        assert labels == [1.0, 0.0, 1.0]


class TestUngroupedIsUnchanged:
    def test_groups_none_matches_the_historical_path(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((20, 3))
        y = np.array([1.0] * 10 + [0.0] * 10)
        a = stratified_fold_orderings(x, y, _linear_trainer(), 7)
        b = stratified_fold_orderings(x, y, _linear_trainer(), 7, groups=None)
        assert a == b
        assert a  # non-empty, i.e. the comparison is meaningful


class TestGroupedCollapse:
    def test_each_calibration_group_yields_exactly_one_score(self):
        # 4 positive images x 2 rows, 4 negative images x 2 rows.
        x = np.zeros((16, 1))
        y = np.array([1.0] * 8 + [0.0] * 8)
        groups = [("g", i) for i in range(4) for _ in range(2)] + [("b", i) for i in range(4) for _ in range(2)]
        out = stratified_fold_orderings(x, y, _linear_trainer(), 3, groups=groups, calibrate_count=2)
        assert out
        for scores, labels in out:
            assert len(scores) == len(labels)
            assert len(scores) <= 8  # 8 groups total, never 16 rows

    def test_score_is_the_max_over_the_supplied_inference_rows(self):
        x = np.zeros((8, 1))
        y = np.array([1.0] * 4 + [0.0] * 4)
        groups = [("g", i) for i in range(4)] + [("b", i) for i in range(4)]
        # Every group scores over a stack whose max is a distinctive value.
        srbg = {g: np.array([[0.0], [float(j + 1) * 10.0]]) for j, g in enumerate(groups)}
        out = stratified_fold_orderings(
            x, y, _linear_trainer(), 1, groups=groups, score_rows_by_group=srbg, calibrate_count=1
        )
        assert out
        scores = out[0][0]
        assert all(s in {10.0 * (j + 1) for j in range(8)} for s in scores)
        assert all(s >= 10.0 for s in scores)  # never the 0.0 row, i.e. a max not a mean

    def test_inference_rows_beat_training_rows_when_they_differ(self):
        """The whole point: a bag trained on one low row must calibrate on its real max."""
        x = np.zeros((8, 1))  # training rows all score 0.0
        y = np.array([1.0] * 4 + [0.0] * 4)
        groups = [("g", i) for i in range(4)] + [("b", i) for i in range(4)]
        srbg = {g: np.array([[0.0], [5.0]]) for g in groups}  # inference max is 5.0
        without = stratified_fold_orderings(x, y, _linear_trainer(), 1, groups=groups, calibrate_count=1)
        with_rows = stratified_fold_orderings(
            x, y, _linear_trainer(), 1, groups=groups, score_rows_by_group=srbg, calibrate_count=1
        )
        assert without and with_rows
        assert set(without[0][0]) == {0.0}
        assert set(with_rows[0][0]) == {5.0}

    def test_rows_of_one_group_never_straddle_a_split(self):
        """A group must be wholly train or wholly calibrate, else the fold leaks."""
        x = np.arange(24, dtype=float).reshape(12, 2)
        y = np.array([1.0] * 6 + [0.0] * 6)
        groups = [("g", i) for i in range(3) for _ in range(2)] + [("b", i) for i in range(3) for _ in range(2)]
        seen = {}

        def trainer(xt, yt, seed):
            seen["train_rows"] = {float(v) for v in xt[:, 0]}
            return lambda z: np.asarray(z, dtype=np.float64)[:, 0]

        stratified_fold_orderings(x, y, trainer, 5, groups=groups, calibrate_count=1)
        # Each group's two rows have first-column values 2k and 2k+2 apart; assert that for
        # every group either both rows trained or neither did.
        for gi in range(6):
            rows = {float(x[2 * gi, 0]), float(x[2 * gi + 1, 0])}
            assert rows <= seen["train_rows"] or not (rows & seen["train_rows"])

    def test_too_few_groups_degrades_to_empty(self):
        """Row count can be ample while group count is not; the guard is on groups."""
        x = np.zeros((20, 1))
        y = np.array([1.0] * 10 + [0.0] * 10)
        groups = [("g", 0)] * 10 + [("b", 0)] * 10  # only 2 groups
        assert stratified_fold_orderings(x, y, _linear_trainer(), 0, groups=groups) == []


class TestCalibratedThresholdPassThrough:
    def test_conformal_accepts_groups(self):
        x = np.zeros((8, 1))
        y = np.array([1.0] * 4 + [0.0] * 4)
        groups = [("g", i) for i in range(4)] + [("b", i) for i in range(4)]
        srbg = {g: np.array([[float(i)]]) for i, g in enumerate(groups)}
        thr = calibrated_threshold(
            x, y, _linear_trainer(), 0, rule="conformal", groups=groups, score_rows_by_group=srbg
        )
        assert np.isfinite(thr)

    def test_argmin_refuses_groups_rather_than_ignoring_them(self):
        x = np.zeros((8, 1))
        y = np.array([1.0] * 4 + [0.0] * 4)
        groups = [("g", i) for i in range(4)] + [("b", i) for i in range(4)]
        with pytest.raises(ValueError, match="no bag-aware path"):
            calibrated_threshold(x, y, _linear_trainer(), 0, rule="argmin", groups=groups)
