"""The calibration split since #4864: each class's holdout rounded to its nearest whole share, with no coin.

Before #4864 the app dithered the split sizes (#3286; that rule is now the
eval arm ``vtscore.eval.arms_split_rounding.split_rounding("dither")``, pinned
in ``test_split_dither.py``), so over a band of Goods counts the labels line
was a labelset's coin flip, drawn again at every vote.  Pinned here: the
nearest holdout itself; that both fold paths hold out each class's nearest
count, whatever the labelset; that the labels line then starts at one count of
Goods on each space and never falls back; that the dither arm puts the coin
back for a run; and that the swap is undone on the way out.
"""

import numpy as np
import pytest

from vtscore.config.runtime import DEFAULT_CALIBRATE_COUNT
from vtscore.eval.arms_split_rounding import SPLIT_ROUNDINGS, split_rounding
from vtscore.training.thresholds import (
    HEAD_COUNT_GOODS,
    NO_GOOD_THRESHOLD,
    PRODUCTION_SPLIT_BY_SPACE,
    compute_fold_orderings,
    conformal,
    nearest_holdout,
)

SINGLE = PRODUCTION_SPLIT_BY_SPACE["single_vector"]
PATCH = PRODUCTION_SPLIT_BY_SPACE["patch"]


def _labelset(rng, n_pos, n_neg, dim=8):
    X = rng.standard_normal((n_pos + n_neg, dim)).astype(np.float32)
    return list(X), [1.0] * n_pos + [0.0] * n_neg


def _held_goods(n_pos, n_neg, fraction, labelset):
    sizes = conformal._split_train_counts(n_pos + n_neg, n_pos, n_neg, fraction, labelset)
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

    def test_a_class_too_small_to_split_holds_out_none(self):
        assert [nearest_holdout(c, SINGLE) for c in (0, 1)] == [0, 0]

    @pytest.mark.parametrize(("fraction", "first"), [(SINGLE, 9), (PATCH, 5)])
    def test_the_labels_line_starts_at_one_count_of_goods(self, fraction, first):
        """9 Goods on a single-vector space, 5 on a patch space, at the app's folds."""
        stands = [DEFAULT_CALIBRATE_COUNT * nearest_holdout(g, fraction) >= HEAD_COUNT_GOODS for g in range(2, 80)]
        assert stands.index(True) + 2 == first
        assert all(stands[first - 2 :]), "once it stands it never falls back"


class TestTheShippedSplit:
    def test_no_coin_the_labelset_does_not_move_the_sizes(self):
        """At 8 Goods every labelset holds out 2 of them, whatever its vectors or its Bads."""
        rng = np.random.RandomState(0)
        held = set()
        for b in (10, 20, 30):
            for _ in range(50):
                X, y = _labelset(rng, 8, b)
                held.add(_held_goods(8, b, SINGLE, (np.array(X), np.array(y))))
        assert held == {nearest_holdout(8, SINGLE)} == {2}

    def test_each_class_is_sized_on_its_own_count(self):
        """The Goods' holdout does not read the Bads: no whole-labelset stage before the classes."""
        for g in range(2, 30):
            assert {_held_goods(g, b, SINGLE, None) for b in range(2, 120)} == {nearest_holdout(g, SINGLE)}

    def test_a_degenerate_fraction_is_too_small_to_split(self):
        """A share that would leave fewer than two votes to train gets the caller's sentinel."""
        assert conformal._split_train_counts(4, 2, 2, 0.99, None) is None
        assert conformal._split_train_counts(4, 2, 2, SINGLE, None) == (1, 1)

    def test_row_wise_folds_hold_out_the_nearest_count_of_each_class(self):
        rng = np.random.RandomState(1)
        for _ in range(3):  # three labelsets of the same counts: no coin, so one answer
            X, y = _labelset(rng, 8, 13)
            orderings, fallback = compute_fold_orderings(
                X, y, 8, rng=np.random.RandomState(42), calibration_fraction=SINGLE, hidden_dim=0
            )
            assert fallback is None
            for _scores, labels in orderings:
                assert sum(lab == 1.0 for lab in labels) == nearest_holdout(8, SINGLE) == 2
                assert sum(lab == 0.0 for lab in labels) == nearest_holdout(13, SINGLE) == 4

    def test_grouped_folds_hold_out_the_nearest_count_of_each_class(self):
        """Bags of one to three rows: the grouped path sizes by bag."""
        rng = np.random.RandomState(2)
        X, y, groups = [], [], []
        for g, label in enumerate([1.0] * 9 + [0.0] * 11):
            for _ in range(1 + g % 3):
                X.append(rng.standard_normal(8).astype(np.float32))
                y.append(label)
                groups.append(("bag", g))
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
        orderings, fallback = compute_fold_orderings(X, y, 8, rng=np.random.RandomState(42), hidden_dim=0)
        assert orderings == [] and fallback == 0.5

    def test_the_degenerate_fraction_sentinel_on_the_folds(self):
        rng = np.random.RandomState(4)
        X, y = _labelset(rng, 2, 2)
        orderings, fallback = compute_fold_orderings(
            X, y, 8, rng=np.random.RandomState(42), calibration_fraction=0.99, hidden_dim=0
        )
        assert orderings == [] and fallback == NO_GOOD_THRESHOLD


class TestTheDitherArm:
    def test_the_dither_puts_the_coin_back_inside_the_band(self):
        """Under the arm, at 8 Goods the folds hold out 2 or 3 of them, by the labelset's coin."""
        rng = np.random.RandomState(5)
        held = set()
        with split_rounding("dither"):
            for b in (10, 20, 30):
                for _ in range(60):
                    X, y = _labelset(rng, 8, b)
                    held.add(_held_goods(8, b, SINGLE, (np.array(X), np.array(y))))
        assert held == {2, 3}

    def test_the_dither_is_settled_past_the_band(self):
        rng = np.random.RandomState(6)
        with split_rounding("dither"):
            held = {
                _held_goods(12, b, SINGLE, (np.array(X), np.array(y)))
                for b in (10, 20, 30)
                for X, y in (_labelset(rng, 12, b) for _ in range(60))
            }
        assert min(held) * DEFAULT_CALIBRATE_COUNT >= HEAD_COUNT_GOODS

    def test_the_dither_reaches_the_folds(self):
        rng = np.random.RandomState(7)
        held = set()
        with split_rounding("dither"):
            for _ in range(40):
                X, y = _labelset(rng, 8, 13)
                orderings, _ = compute_fold_orderings(
                    X, y, 8, rng=np.random.RandomState(42), calibration_fraction=SINGLE, hidden_dim=0
                )
                held.add(sum(lab == 1.0 for lab in orderings[0][1]))
        assert held == {2, 3}

    def test_the_dither_needs_the_labelset(self):
        with split_rounding("dither"), pytest.raises(ValueError, match="labelset"):
            conformal._split_train_counts(21, 8, 13, SINGLE, None)


class TestTheSwap:
    def test_nearest_is_the_app(self):
        shipped = conformal._split_train_counts
        with split_rounding("nearest"):
            assert conformal._split_train_counts is shipped

    def test_the_swap_is_undone_even_on_an_error(self):
        shipped = conformal._split_train_counts
        with pytest.raises(RuntimeError), split_rounding("dither"):
            assert conformal._split_train_counts is not shipped
            raise RuntimeError
        assert conformal._split_train_counts is shipped

    def test_an_unknown_rule_is_refused(self):
        with pytest.raises(ValueError, match="split_rounding"), split_rounding("round"):
            pass
        assert SPLIT_ROUNDINGS[0] == "nearest"
