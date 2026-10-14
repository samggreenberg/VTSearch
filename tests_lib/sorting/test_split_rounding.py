"""#4864's split-rounding arm: each class's calibration holdout rounded to nearest, with no coin.

The app dithers the split sizes (#3286, pinned in ``test_split_dither.py``), so
over a band of Goods counts the labels line is a labelset's coin flip.  The arm,
``vtscore.eval.arms_split_rounding.split_rounding("nearest")``, sizes each class
on its own count.  Pinned here: what the arm removes (the shipped holdout is a
coin inside the band); that under the arm both fold paths hold out the nearest
count of each class, whatever the labelset; that the labels line then starts at
one count of Goods on each space; and that the swap is undone on the way out.
"""

import numpy as np
import pytest

from vtscore.config.runtime import DEFAULT_CALIBRATE_COUNT
from vtscore.eval.arms_split_rounding import SPLIT_ROUNDINGS, nearest_holdout, split_rounding
from vtscore.training.thresholds import HEAD_COUNT_GOODS, PRODUCTION_SPLIT_BY_SPACE, compute_fold_orderings, conformal

SINGLE = PRODUCTION_SPLIT_BY_SPACE["single_vector"]
PATCH = PRODUCTION_SPLIT_BY_SPACE["patch"]


def _labelset(rng, n_pos, n_neg, dim=8):
    X = rng.standard_normal((n_pos + n_neg, dim)).astype(np.float32)
    return list(X), [1.0] * n_pos + [0.0] * n_neg


def _held_goods(n_pos, n_neg, fraction, dither):
    sizes = conformal._split_train_counts(n_pos + n_neg, n_pos, n_neg, fraction, dither)
    assert sizes is not None
    return n_pos - sizes[0]


class TestNearestHoldout:
    @pytest.mark.parametrize("fraction", [SINGLE, PATCH])
    def test_nearest_whole_vote_kept_on_both_sides(self, fraction):
        held = [nearest_holdout(c, fraction) for c in range(2, 60)]
        for c, h in zip(range(2, 60), held, strict=True):
            assert 1 <= h <= c - 1
            assert abs(h - fraction * c) <= 0.5 or h in (1, c - 1)
        assert held == sorted(held), "a class that grows must never hold out fewer"

    def test_halves_round_up(self):
        assert nearest_holdout(5, 0.5) == 3
        assert nearest_holdout(15, 0.3) == 5

    @pytest.mark.parametrize(("fraction", "first"), [(SINGLE, 9), (PATCH, 5)])
    def test_the_labels_line_starts_at_one_count_of_goods(self, fraction, first):
        """The module's claim: 9 Goods on a single-vector space, 5 on a patch space, at the app's folds."""
        stands = [DEFAULT_CALIBRATE_COUNT * nearest_holdout(g, fraction) >= HEAD_COUNT_GOODS for g in range(2, 80)]
        assert stands.index(True) + 2 == first
        assert all(stands[first - 2 :]), "once it stands it never falls back"


class TestWhatTheArmRemoves:
    def test_the_shipped_holdout_is_a_coin_inside_the_band(self):
        """At 8 Goods the app's folds hold out 2 or 3 of them, by the labelset's dither (#4864)."""
        held = {_held_goods(8, b, SINGLE, np.random.RandomState(s)) for b in (10, 20, 30) for s in range(200)}
        assert held == {2, 3}

    def test_the_shipped_holdout_is_settled_past_the_band(self):
        held = {_held_goods(12, b, SINGLE, np.random.RandomState(s)) for b in (10, 20, 30) for s in range(200)}
        assert min(held) * DEFAULT_CALIBRATE_COUNT >= HEAD_COUNT_GOODS


class TestTheArmOnTheFolds:
    def test_row_wise_folds_hold_out_the_nearest_count_of_each_class(self):
        rng = np.random.RandomState(1)
        for _ in range(3):  # three labelsets of the same counts: no coin, so one answer
            X, y = _labelset(rng, 8, 13)
            with split_rounding("nearest"):
                orderings, fallback = compute_fold_orderings(
                    X, y, 8, rng=np.random.RandomState(42), calibration_fraction=SINGLE, hidden_dim=0
                )
            assert fallback is None
            for _scores, labels in orderings:
                assert sum(lab == 1.0 for lab in labels) == nearest_holdout(8, SINGLE) == 2
                assert sum(lab == 0.0 for lab in labels) == nearest_holdout(13, SINGLE) == 4

    def test_grouped_folds_hold_out_the_nearest_count_of_each_class(self):
        """Bags of one to three rows: the grouped path sizes by bag, and the arm reaches it too."""
        rng = np.random.RandomState(2)
        X, y, groups = [], [], []
        for g, label in enumerate([1.0] * 9 + [0.0] * 11):
            for _ in range(1 + g % 3):
                X.append(rng.standard_normal(8).astype(np.float32))
                y.append(label)
                groups.append(("bag", g))
        with split_rounding("nearest"):
            orderings, fallback = compute_fold_orderings(
                X, y, 8, rng=np.random.RandomState(42), calibration_fraction=PATCH, hidden_dim=0, groups=groups
            )
        assert fallback is None
        for _scores, labels in orderings:
            assert sum(lab == 1.0 for lab in labels) == nearest_holdout(9, PATCH) == 5
            assert sum(lab == 0.0 for lab in labels) == nearest_holdout(11, PATCH) == 6

    def test_a_class_too_small_to_split_still_returns_the_sentinel(self):
        rng = np.random.RandomState(3)
        X, y = _labelset(rng, 1, 6)
        with split_rounding("nearest"):
            orderings, fallback = compute_fold_orderings(X, y, 8, rng=np.random.RandomState(42), hidden_dim=0)
        assert orderings == [] and fallback == 0.5


class TestTheSwap:
    def test_dither_is_the_app(self):
        shipped = conformal._split_train_counts
        with split_rounding("dither"):
            assert conformal._split_train_counts is shipped

    def test_the_swap_is_undone_even_on_an_error(self):
        shipped = conformal._split_train_counts
        with pytest.raises(RuntimeError), split_rounding("nearest"):
            assert conformal._split_train_counts is not shipped
            raise RuntimeError
        assert conformal._split_train_counts is shipped

    def test_an_unknown_rule_is_refused(self):
        with pytest.raises(ValueError, match="split_rounding"), split_rounding("round"):
            pass
        assert "dither" in SPLIT_ROUNDINGS
