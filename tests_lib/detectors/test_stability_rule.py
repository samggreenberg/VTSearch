"""The Stable indicator's rule, pinned where it lives (issue #3831).

:mod:`vtscore.detectors.stability` is called by both the app's
``_compute_stable_status`` and the harness's ``stable_status``, so this is the
one place the rule's behaviour is asserted; the two callers' tests only check
that they pass their inputs through.

The two scenarios that motivated the rewrite, both reproduced in the real
harness on a synthetic "dice" pool before the rule was changed:

* A category the embedding cannot separate (dice labeled by whether they rolled
  at least 5, embedded by their number of sides) flips a whole side-group
  across the cut on nearly every retrain.  The old rule counted every one of
  those and held Autopilot in ``hard`` for 127 of 150 clicks.
* On a nearly-labeled haystack the old rate divided by the unlabeled remainder,
  so one flip among the last fifty items read as a 2% spike.
"""

from __future__ import annotations

import pytest

from vtscore.detectors.stability import (
    MIN_PER_CLASS,
    STABLE_BAND_STD_FRACTION,
    STABLE_MAX_THRESHOLD,
    STABLE_MIN_ENTRIES,
    STABLE_RATE_THRESHOLD,
    STABLE_WINDOW,
    ScoredSnapshot,
    ambiguity_band,
    count_flips,
    stability_entry,
    stable_status_from_entries,
)


def _entry(flips: int, confident: int | None = None, *, pool: int = 1000, unlabeled: int = 500) -> dict:
    return {
        "num_flips": flips,
        "num_confident_flips": flips if confident is None else confident,
        "num_unlabeled": unlabeled,
        "num_pool": pool,
    }


class TestConstants:
    def test_values(self):
        assert MIN_PER_CLASS == 5
        assert STABLE_WINDOW == 10
        assert STABLE_MIN_ENTRIES == 5
        assert STABLE_RATE_THRESHOLD == 0.005
        assert STABLE_MAX_THRESHOLD == 0.01
        assert STABLE_BAND_STD_FRACTION == 0.25


class TestAmbiguityBand:
    def test_is_a_fraction_of_the_score_spread(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        import numpy as np

        assert ambiguity_band(scores) == pytest.approx(STABLE_BAND_STD_FRACTION * np.std(scores))

    def test_empty_and_constant_pools_have_no_band(self):
        assert ambiguity_band([]) == 0.0
        assert ambiguity_band([0.7, 0.7, 0.7]) == pytest.approx(0.0, abs=1e-12)

    def test_snapshot_carries_its_own_band(self):
        snap = ScoredSnapshot.from_scores({1: 0.1, 2: 0.9}, 0.5)
        assert snap.band == pytest.approx(ambiguity_band([0.1, 0.9]))
        assert snap.predicted(1) == 0 and snap.predicted(2) == 1
        assert snap.confident(1) and snap.confident(2)


class TestCountFlips:
    def test_boundary_wobble_is_a_flip_but_not_a_confident_one(self):
        """An item just above the cut that lands just below it next retrain."""
        spread = {10 + i: v for i, v in enumerate([0.1, 0.2, 0.8, 0.9])}  # gives both snapshots a real band
        prev = ScoredSnapshot.from_scores({1: 0.51, **spread}, 0.5)
        cur = ScoredSnapshot.from_scores({1: 0.49, **spread}, 0.5)
        assert count_flips(prev, cur) == (1, 0)

    def test_a_clear_reclassification_is_confident(self):
        spread = {10 + i: v for i, v in enumerate([0.1, 0.2, 0.8, 0.9])}
        prev = ScoredSnapshot.from_scores({1: 0.9, **spread}, 0.5)
        cur = ScoredSnapshot.from_scores({1: 0.1, **spread}, 0.5)
        assert count_flips(prev, cur) == (1, 1)

    def test_confidence_needs_both_sides_clear(self):
        """Clear under one detector and marginal under the other is wobble."""
        spread = {10 + i: v for i, v in enumerate([0.1, 0.2, 0.8, 0.9])}
        prev = ScoredSnapshot.from_scores({1: 0.9, **spread}, 0.5)
        cur = ScoredSnapshot.from_scores({1: 0.49, **spread}, 0.5)
        assert count_flips(prev, cur) == (1, 0)

    def test_moving_cut_counts_against_each_snapshots_own_threshold(self):
        spread = {10 + i: v for i, v in enumerate([0.1, 0.2, 0.8, 0.9])}
        prev = ScoredSnapshot.from_scores({1: 0.6, **spread}, 0.5)  # in
        cur = ScoredSnapshot.from_scores({1: 0.6, **spread}, 0.7)  # same score, now out
        assert count_flips(prev, cur)[0] == 1

    def test_only_items_in_both_snapshots_are_compared(self):
        prev = ScoredSnapshot.from_scores({1: 0.9, 2: 0.9, 3: 0.1}, 0.5)
        cur = ScoredSnapshot.from_scores({2: 0.1, 3: 0.1}, 0.5)  # 1 was labeled in between
        assert count_flips(prev, cur) == (1, 1)

    def test_entry_reports_the_pool_beside_the_compared_count(self):
        prev = ScoredSnapshot.from_scores({1: 0.9, 2: 0.1}, 0.5)
        cur = ScoredSnapshot.from_scores({1: 0.9, 2: 0.1}, 0.5)
        assert stability_entry(prev, cur, 40) == {
            "num_flips": 0,
            "num_confident_flips": 0,
            "num_unlabeled": 2,
            "num_pool": 40,
        }


class TestStableStatus:
    def test_red_below_the_class_quorum(self):
        assert stable_status_from_entries([_entry(0)] * 6, MIN_PER_CLASS - 1, 9)["status"] == "red"
        assert stable_status_from_entries([_entry(0)] * 6, 9, MIN_PER_CLASS - 1)["status"] == "red"

    def test_yellow_without_enough_history(self):
        result = stable_status_from_entries([_entry(0)] * (STABLE_MIN_ENTRIES - 1), 9, 9)
        assert result["status"] == "yellow"
        assert "plateau" not in result

    def test_green_once_confident_flips_settle(self):
        result = stable_status_from_entries([_entry(0)] * 6, 9, 9)
        assert result["status"] == "green"
        assert result["plateau"] is False
        assert result["avg_flip_rate"] == 0.0

    def test_yellow_while_confident_flips_continue(self):
        assert stable_status_from_entries([_entry(50)] * 6, 9, 9)["status"] == "yellow"

    def test_a_single_confident_spike_blocks_green(self):
        entries = [_entry(0)] * 5 + [_entry(20)]  # 2% of the pool
        assert stable_status_from_entries(entries, 9, 9)["status"] == "yellow"

    def test_irreducible_wobble_is_green_and_flagged_as_a_plateau(self):
        """The dice pool: a large, steady flip rate with nothing confident in it."""
        result = stable_status_from_entries([_entry(80, confident=0)] * 8, 9, 9)
        assert result["status"] == "green"
        assert result["plateau"] is True
        assert result["avg_flip_rate"] == pytest.approx(0.08)
        assert "irreducible" in result["reason"]

    def test_a_falling_raw_rate_is_still_settling(self):
        """Wobble that halves across the window means the boundary is still
        sharpening - more labels are still buying something."""
        entries = [_entry(80, confident=0)] * 5 + [_entry(20, confident=0)] * 5
        result = stable_status_from_entries(entries, 9, 9)
        assert result["status"] == "yellow"
        assert "settling" in result["reason"]

    def test_a_raw_rate_that_fell_below_the_settled_threshold_is_green(self):
        """Falling all the way to nothing is convergence, not a plateau."""
        entries = [_entry(80, confident=0)] * 5 + [_entry(0)] * 5
        result = stable_status_from_entries(entries, 9, 9)
        assert result["status"] == "green"
        assert result["plateau"] is False

    def test_rates_are_over_the_pool_not_the_unlabeled_remainder(self):
        """One flip among the last four unlabeled items of a 1000-item pool."""
        entries = [_entry(0, unlabeled=4)] * 5 + [_entry(1, unlabeled=4)]
        assert stable_status_from_entries(entries, 9, 9)["status"] == "green"

    def test_only_the_window_is_read(self):
        entries = [_entry(500)] * 10 + [_entry(0)] * STABLE_WINDOW
        assert stable_status_from_entries(entries, 9, 9)["status"] == "green"

    def test_an_empty_pool_reads_as_no_flips(self):
        entries = [_entry(3, pool=0, unlabeled=0)] * 6
        assert stable_status_from_entries(entries, 9, 9)["status"] == "green"
