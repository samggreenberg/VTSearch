"""Oracle ceilings for accuracy / balanced accuracy (``--show-oracle`` on every metric).

Covers the three layers the feature spans: the ``max_*`` sweeps in ``error_metrics``, the
``oracle_*`` columns ``region_curve`` writes onto every row, and the ``_ORACLE_FIELD``
table that decides which plots grow a dashed companion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from vtscore.eval.error_metrics import (
    max_accuracy,
    max_balanced_accuracy,
    weighted_error,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sod"))


class TestMaxAccuracy:
    def test_separable_reaches_one(self):
        assert max_accuracy([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0

    def test_endpoint_wins_under_imbalance(self):
        # 1 positive in 100, scored worst of all: no cut beats predicting all-negative,
        # which is 99/100 correct. Without the explicit endpoint this would report 0.98.
        scores = [0.0] + [float(i + 1) for i in range(99)]
        labels = [1] + [0] * 99
        assert max_accuracy(scores, labels) == 0.99

    def test_is_a_true_ceiling(self):
        rng = np.random.default_rng(11)
        scores = rng.random(200)
        labels = (rng.random(200) < 0.3).astype(int)
        ceiling = max_accuracy(scores, labels)
        n = len(labels)
        for t in [*np.unique(scores).tolist(), float(scores.max()) + 1.0]:
            err = weighted_error(scores, [float(v) for v in labels], float(t), 0)
            n_pos = int(sum(labels))
            n_neg = n - n_pos
            acc = ((1 - err["fnr"]) * n_pos + (1 - err["fpr"]) * n_neg) / n
            assert acc <= ceiling + 1e-6

    def test_all_tied_scores_collapse_to_the_single_realizable_cut(self):
        # Every score identical -> the only achievable prediction is "all positive"
        # (or all negative). A position-wise sweep would wrongly report 1.0.
        assert max_accuracy([0.5] * 10, [1] * 4 + [0] * 6) == 0.6

    def test_nan_when_a_class_is_absent(self):
        assert np.isnan(max_accuracy([0.1, 0.9], [1, 1]))
        assert np.isnan(max_accuracy([0.1, 0.9], [0, 0]))

    def test_nan_on_empty(self):
        assert np.isnan(max_accuracy([], []))


class TestMaxBalancedAccuracy:
    def test_separable_reaches_one(self):
        assert max_balanced_accuracy([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0

    def test_never_below_the_degenerate_half(self):
        # Perfectly anti-correlated scores: no cut helps, so the predict-all-negative
        # endpoint (TPR 0, TNR 1) floors it at 0.5.
        assert max_balanced_accuracy([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.5

    def test_immune_to_prevalence(self):
        # Same separation quality, wildly different prevalence -> same value, unlike
        # plain accuracy which is dominated by the majority class.
        few = max_balanced_accuracy([0.9] + [0.1] * 99, [1] + [0] * 99)
        many = max_balanced_accuracy([0.9] * 50 + [0.1] * 50, [1] * 50 + [0] * 50)
        assert few == many == 1.0

    def test_matches_one_minus_half_min_cost(self):
        # At inclusion 0, cost == fpr + fnr, so the two optima coincide. Compared against
        # a tie-free score vector, where the position sweep and the cut sweep agree.
        rng = np.random.default_rng(7)
        scores = rng.random(150)
        labels = (rng.random(150) < 0.4).astype(int)
        best = min(
            weighted_error(scores, [float(v) for v in labels], float(t), 0)["cost"]
            for t in [*np.unique(scores).tolist(), float(scores.max()) + 1.0]
        )
        assert max_balanced_accuracy(scores, labels) == pytest.approx(1.0 - best / 2.0, abs=1e-6)

    def test_all_tied_scores_collapse_to_the_single_realizable_cut(self):
        assert max_balanced_accuracy([0.5] * 10, [1] * 4 + [0] * 6) == 0.5

    def test_nan_when_a_class_is_absent(self):
        assert np.isnan(max_balanced_accuracy([0.1, 0.9], [1, 1]))


class TestOracleExtraColumns:
    def test_row_carries_the_new_oracle_fields(self):
        from vtscore.eval.region_curve import _oracle_extra

        extra = _oracle_extra([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
        assert extra["oracle_accuracy"] == 1.0
        assert extra["oracle_balanced_accuracy"] == 1.0
        assert extra["oracle_f1"] == 1.0
        assert extra["auroc"] == 1.0

    def test_degenerate_metrics_get_no_oracle_column(self):
        from vtscore.eval.region_curve import _oracle_extra

        extra = _oracle_extra([0.9, 0.1], [1, 0])
        for degenerate in ("oracle_fpr", "oracle_fnr", "oracle_precision", "oracle_recall", "oracle_auroc"):
            assert degenerate not in extra


class TestOracleFieldTable:
    def test_covers_exactly_the_non_degenerate_metrics(self):
        import plots

        assert set(plots._ORACLE_FIELD) == {"cost", "f1", "accuracy", "balanced_accuracy"}

    def test_every_oracle_field_names_a_known_metric(self):
        import plots

        for metric in plots._ORACLE_FIELD:
            assert metric in plots._METRICS

    def test_train_test_metrics_are_all_plottable(self):
        import plots
        import plots_train_test

        for metric in plots_train_test.TRAIN_TEST_METRICS:
            assert metric in plots._METRICS
