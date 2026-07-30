"""Unit tests for the conformal inclusion-threshold rule (issue #2693).

The rule replaced the min-cost argmin search, whose threshold provably could
not move with inclusion whenever the calibration folds ranked their held-out
votes perfectly (see docs/experiments/inclusion-knob/REPORT.md).  The
properties pinned here are the ones the replacement was chosen for:

* **Monotone by construction**: the threshold is non-increasing in inclusion,
  so included sets are nested - "cut off at Inclusion 1, verify up to
  Inclusion 4" is well-defined.
* **Portable semantics**: inclusion ``+k`` is a false-negative budget
  ``0.25 * 2^-k`` - the alpha-quantile of held-out positive scores - so the
  same knob position means the same miss-tolerance on every detector.
* **Resolution**: distinct knob positions produce distinct thresholds
  whenever the calibration scores have spread (the argmin's failure mode).
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.training.thresholds import (
    CONFORMAL_BASE_BUDGET,
    CONFORMAL_QPOS_MAX,
    NO_GOOD_THRESHOLD,
    compute_fold_orderings,
    conformal_threshold,
    threshold_from_fold_orderings,
)


def _spread_calibration(n_pos: int = 40, n_neg: int = 40, seed: int = 42) -> tuple[list[float], list[float]]:
    """Overlapping but separated score distributions with real spread."""
    rng = np.random.default_rng(seed)
    pos = np.clip(rng.normal(0.75, 0.12, n_pos), 0.0, 1.0)
    neg = np.clip(rng.normal(0.25, 0.12, n_neg), 0.0, 1.0)
    scores = np.concatenate([pos, neg]).tolist()
    labels = [1.0] * n_pos + [0.0] * n_neg
    return scores, labels


class TestDegenerateInputs:
    def test_empty_scores_returns_default(self):
        assert conformal_threshold([], []) == 0.5

    def test_single_class_returns_default(self):
        assert conformal_threshold([0.2, 0.8], [1.0, 1.0]) == 0.5
        assert conformal_threshold([0.2, 0.8], [0.0, 0.0]) == 0.5

    def test_result_is_realizable(self):
        """The rule never abstains: the threshold sits within the score range."""
        scores, labels = _spread_calibration()
        for incl in range(-10, 11):
            t = conformal_threshold(scores, labels, incl)
            assert min(scores) <= t <= max(scores)


class TestPositiveInclusionSemantics:
    def test_fn_budget_is_a_cap(self):
        """The threshold never exceeds the ``0.25 * 2^-k`` quantile of
        positive scores, so at most ~that fraction of matches is missed."""
        scores, labels = _spread_calibration()
        pos = np.array([s for s, lb in zip(scores, labels) if lb == 1.0])
        for k in (1, 3, 5, 10):
            cap = float(np.quantile(pos, CONFORMAL_BASE_BUDGET * 2.0**-k))
            assert conformal_threshold(scores, labels, k) <= cap + 1e-12

    def test_fn_budget_binds_under_overlap(self):
        """When class overlap forces the false-positive guard above the
        budget quantile, +k lands exactly on the ``0.25 * 2^-k`` quantile
        of positive scores - the portable miss-budget semantics."""
        rng = np.random.default_rng(11)
        pos = np.clip(rng.normal(0.6, 0.15, 40), 0.0, 1.0)
        neg = np.clip(rng.normal(0.4, 0.15, 40), 0.0, 1.0)
        scores = np.concatenate([pos, neg]).tolist()
        labels = [1.0] * 40 + [0.0] * 40
        for k in (1, 3, 5):
            expected = float(np.quantile(pos, CONFORMAL_BASE_BUDGET * 2.0**-k))
            assert conformal_threshold(scores, labels, k) == pytest.approx(expected)

    def test_budget_unspent_when_classes_separate(self):
        """Regression: with a clean margin between the classes, the default
        cut must NOT sacrifice 25% of matches just because the budget allows
        it - every calibration positive stays at or above the k=0 cut."""
        scores, labels = _spread_calibration()
        pos = [s for s, lb in zip(scores, labels) if lb == 1.0]
        t = conformal_threshold(scores, labels, 0)
        assert all(s >= t for s in pos)

    def test_high_inclusion_keeps_nearly_all_positives(self):
        """At +10 the budget is ~0.02%: essentially every calibration
        positive must land at or above the cut."""
        scores, labels = _spread_calibration()
        pos = [s for s, lb in zip(scores, labels) if lb == 1.0]
        t = conformal_threshold(scores, labels, 10)
        assert sum(1 for s in pos if s >= t) >= len(pos) - 1


class TestNegativeInclusionSemantics:
    def test_negative_side_guarded_by_fp_budget(self):
        """At k <= 0 the threshold is at least the negative-score quantile
        that caps the false-positive rate at ``0.25 * 2^k``."""
        scores, labels = _spread_calibration()
        neg = np.array([s for s, lb in zip(scores, labels) if lb != 1.0])
        for k in (0, -2, -5, -10):
            beta = CONFORMAL_BASE_BUDGET * 2.0**k
            guard = float(np.quantile(neg, 1.0 - beta))
            assert conformal_threshold(scores, labels, k) >= guard - 1e-12

    def test_minus_ten_walks_to_top_quartile_of_positives(self):
        """With negatives far below, -10 lands on the QPOS_MAX positive
        quantile: only the most confident matches remain."""
        rng = np.random.default_rng(7)
        pos = np.clip(rng.normal(0.8, 0.1, 50), 0.0, 1.0)
        neg = np.clip(rng.normal(0.05, 0.02, 50), 0.0, 1.0)
        scores = np.concatenate([pos, neg]).tolist()
        labels = [1.0] * 50 + [0.0] * 50
        expected = float(np.quantile(pos, CONFORMAL_QPOS_MAX))
        assert conformal_threshold(scores, labels, -10) == pytest.approx(expected)


class TestMonotonicity:
    def test_threshold_monotone_non_increasing_in_inclusion(self):
        scores, labels = _spread_calibration()
        thresholds = [conformal_threshold(scores, labels, k) for k in range(-10, 11)]
        for lo, hi in zip(thresholds, thresholds[1:]):
            assert hi <= lo + 1e-12

    def test_included_sets_are_nested(self):
        """Everything included at inclusion k stays included at k + 1."""
        scores, labels = _spread_calibration()
        rng = np.random.default_rng(3)
        pool = rng.uniform(0.0, 1.0, 500)
        prev: set[int] = set()
        for k in range(-10, 11):
            t = conformal_threshold(scores, labels, k)
            included = {i for i, s in enumerate(pool) if s >= t}
            assert prev <= included
            prev = included

    def test_knob_has_resolution(self):
        """Spread-out calibration scores give the knob many distinct stops
        (the argmin collapsed to 1-4 across all 21 positions)."""
        scores, labels = _spread_calibration()
        thresholds = {round(conformal_threshold(scores, labels, k), 12) for k in range(-10, 11)}
        assert len(thresholds) >= 10

    def test_saturated_scores_still_monotone(self):
        """Exact 0.0/1.0 sigmoid saturation degrades resolution (quantiles
        tie) but must never break monotonicity or error out."""
        scores = [1.0, 1.0, 1.0, 0.0, 0.0]
        labels = [1.0, 1.0, 1.0, 0.0, 0.0]
        thresholds = [conformal_threshold(scores, labels, k) for k in range(-10, 11)]
        for lo, hi in zip(thresholds, thresholds[1:]):
            assert hi <= lo


class TestFoldPooling:
    def test_empty_orderings_returns_no_good(self):
        assert threshold_from_fold_orderings([], 0) == NO_GOOD_THRESHOLD

    def test_pools_across_folds(self):
        """The fold aggregate is the conformal rule on the *pooled* scores,
        not a mean of per-fold thresholds."""
        fold_a = ([0.9, 0.7, 0.3, 0.1], [1.0, 1.0, 0.0, 0.0])
        fold_b = ([0.8, 0.6, 0.4, 0.2], [1.0, 1.0, 0.0, 0.0])
        pooled_scores = fold_a[0] + fold_b[0]
        pooled_labels = fold_a[1] + fold_b[1]
        for k in (-5, 0, 5):
            assert threshold_from_fold_orderings([fold_a, fold_b], k) == conformal_threshold(
                pooled_scores, pooled_labels, k
            )


class TestEndToEndAcceptance:
    """The issue #2693 acceptance criterion, on the real production path:
    train real fold models, pool their held-out scores, sweep the knob, and
    check the included set actually grows."""

    def test_included_set_grows_with_inclusion(self):
        import torch

        from vtscore.training.mlp import train_model

        rng = np.random.default_rng(42)
        dim, n_votes = 16, 24
        # Moderately overlapping clusters so pool scores have real spread.
        X_votes = [
            (rng.standard_normal(dim) + (0.5 if i < n_votes // 2 else -0.5)).astype(np.float32) for i in range(n_votes)
        ]
        y_votes = [1.0 if i < n_votes // 2 else 0.0 for i in range(n_votes)]

        orderings, fallback = compute_fold_orderings(
            X_votes,
            y_votes,
            dim,
            rng=np.random.RandomState(42),
            calibrate_count=2,
            calibration_fraction=0.5,
        )
        assert fallback is None

        X_pool = np.concatenate(
            [
                rng.standard_normal((100, dim)) + 0.5,
                rng.standard_normal((100, dim)) - 0.5,
            ]
        ).astype(np.float32)
        X = torch.tensor(np.array(X_votes), dtype=torch.float32)
        y = torch.tensor(y_votes, dtype=torch.float32).unsqueeze(1)
        model = train_model(X, y, dim)
        with torch.no_grad():
            device = next(model.parameters()).device
            pool_scores = torch.sigmoid(model(torch.tensor(X_pool).to(device))).squeeze(1).cpu().numpy()

        sizes = []
        for k in range(-10, 11):
            t = threshold_from_fold_orderings(orderings, k)
            sizes.append(int(np.sum(pool_scores >= t)))

        # Monotone non-decreasing across the sweep...
        for lo, hi in zip(sizes, sizes[1:]):
            assert hi >= lo
        # ...and strictly more items at +10 than -10 (the knob moves).
        assert sizes[-1] > sizes[0]
