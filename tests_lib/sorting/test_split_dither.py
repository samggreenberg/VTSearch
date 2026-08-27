"""The Train/Calibrate split sizes are dithered, not rounded (issue #3286).

``round`` is round-half-to-even, so at the shipped ``calibration_fraction`` of
0.5 the odd vote's destination alternated with the vote count - Train at
``n % 4 == 1``, Calibrate at ``n % 4 == 3``.  Every run of the eval casts one
vote per step, so that seesaw was phase-locked across every trajectory and
survived averaging as a visible 4-vote ripple on the learning curves.

These tests pin the three properties the fix rests on: the count is unbiased,
it is *stable* for a given labelset (the threshold must stay a pure function of
the votes, or the calibration cache would serve stale orderings), and it is
*decoherent* between two different labelsets that happen to be the same size,
which is what makes the ripple average away.
"""

import numpy as np
import pytest

from vtscore.training.thresholds import (
    NO_GOOD_THRESHOLD,
    _dithered_count,
    _split_dither_rng,
    compute_fold_orderings,
)


def _labelset(rng, n, dim=8, n_pos=None):
    """A random ``(X_list, y_list)`` of *n* votes with both classes present."""
    n_pos = max(2, n // 3) if n_pos is None else n_pos
    X = rng.standard_normal((n, dim)).astype(np.float32)
    y = [1.0] * n_pos + [0.0] * (n - n_pos)
    return list(X), y


class TestDitheredCount:
    def test_whole_numbers_pass_through(self):
        rng = np.random.RandomState(0)
        assert [_dithered_count(float(k), rng) for k in range(6)] == [0, 1, 2, 3, 4, 5]

    def test_whole_number_consumes_no_randomness(self):
        """An exact count must not advance the RNG, or it would shift later draws."""
        rng = np.random.RandomState(0)
        _dithered_count(7.0, rng)
        after_exact = rng.random_sample()
        assert after_exact == np.random.RandomState(0).random_sample()

    def test_only_the_two_neighbours_are_reachable(self):
        seen = {_dithered_count(10.5, np.random.RandomState(s)) for s in range(200)}
        assert seen == {10, 11}

    def test_unbiased_at_a_tie(self):
        draws = [_dithered_count(10.5, np.random.RandomState(s)) for s in range(2000)]
        assert np.mean(draws) == pytest.approx(10.5, abs=0.05)

    def test_unbiased_off_a_tie(self):
        """P(round up) tracks the fractional part, so a 0.8 remainder rounds up ~80%."""
        draws = [_dithered_count(10.8, np.random.RandomState(s)) for s in range(2000)]
        assert np.mean(draws) == pytest.approx(10.8, abs=0.05)


class TestSplitDitherRng:
    def test_same_labelset_gives_the_same_draw(self):
        """The threshold stays a pure function of the votes - the cache relies on it."""
        rng = np.random.RandomState(1)
        X_list, y_list = _labelset(rng, 11)
        X, y = np.array(X_list), np.array(y_list)
        draws = {_dithered_count(5.5, _split_dither_rng(X, y)) for _ in range(25)}
        assert len(draws) == 1

    def test_different_embeddings_give_different_draws(self):
        """Same shape, same labels, different vectors - the tie must not resolve alike."""
        rng = np.random.RandomState(2)
        y = np.array([1.0] * 4 + [0.0] * 7)
        ups = sum(
            _dithered_count(5.5, _split_dither_rng(rng.standard_normal((11, 8)).astype(np.float32), y))
            for _ in range(300)
        )
        # Each draw is 5 or 6; ~half should round up.  The pre-#3286 code was a
        # constant here, which is the failure this guards.
        assert 300 * 5 + 90 < ups < 300 * 5 + 210

    def test_the_draw_moves_as_a_session_accumulates_votes(self):
        """A digest over a fixed prefix would freeze; this must keep moving."""
        rng = np.random.RandomState(3)
        X = rng.standard_normal((80, 8)).astype(np.float32)
        odd_deviations = []
        for n in range(9, 60, 2):  # odd n only: the tie cases
            y = np.array([1.0] * (n // 3) + [0.0] * (n - n // 3))
            odd_deviations.append(_dithered_count(n * 0.5, _split_dither_rng(X[:n], y)) - n // 2)
        assert set(odd_deviations) == {0, 1}, "the tie must break both ways within one session"


class TestNoPeriodFourStructure:
    """The regression test: no vote count may resolve its tie the same way twice."""

    def test_mean_split_has_no_mod_four_signature(self):
        rng = np.random.RandomState(4)
        # For each vote count, many independent labelsets - the eval's many runs.
        by_n = {}
        for n in range(9, 41):
            ups = [
                _dithered_count(n * 0.5, _split_dither_rng(rng.standard_normal((n, 8)).astype(np.float32), np.array(y)))
                - n // 2
                for y in ([1.0] * (n // 3) + [0.0] * (n - n // 3),) * 60
            ]
            by_n[n] = float(np.mean(ups))
        # Under the old rule this was exactly 1.0 at n%4==1 and 0.0 at n%4==3
        # (a full-amplitude square wave).  Dithered, every odd n sits near 0.5.
        odd_means = [v for n, v in by_n.items() if n % 2 == 1]
        assert min(odd_means) > 0.25, f"a tie is still resolving one way: {by_n}"
        assert max(odd_means) < 0.75, f"a tie is still resolving one way: {by_n}"
        # Even counts are exact - no draw, no spread.
        assert all(v == 0.0 for n, v in by_n.items() if n % 2 == 0)


class TestFoldOrderingsContract:
    def test_repeated_calls_are_identical(self):
        """Reproducibility: the same labelset must yield byte-identical folds."""
        rng = np.random.RandomState(5)
        X_list, y_list = _labelset(rng, 13)
        first, _ = compute_fold_orderings(X_list, y_list, 8, rng=np.random.RandomState(42), hidden_dim=0)
        second, _ = compute_fold_orderings(X_list, y_list, 8, rng=np.random.RandomState(42), hidden_dim=0)
        assert first == second

    def test_degenerate_fraction_still_returns_the_sentinel(self):
        """n=4 at fraction 0.99 leaves too few training rows either way it rounds."""
        rng = np.random.RandomState(6)
        X_list, y_list = _labelset(rng, 4, n_pos=2)
        orderings, fallback = compute_fold_orderings(
            X_list, y_list, 8, rng=np.random.RandomState(42), calibration_fraction=0.99, hidden_dim=0
        )
        assert orderings == []
        assert fallback == NO_GOOD_THRESHOLD

    def test_tiny_fraction_still_keeps_one_calibration_item(self):
        rng = np.random.RandomState(7)
        X_list, y_list = _labelset(rng, 10)
        orderings, fallback = compute_fold_orderings(
            X_list, y_list, 8, rng=np.random.RandomState(42), calibration_fraction=0.01, hidden_dim=0
        )
        assert fallback is None
        assert orderings
