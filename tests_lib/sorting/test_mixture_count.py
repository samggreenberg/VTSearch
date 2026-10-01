"""The line with no votes: the smaller of the schedule's count and the mixture's (#4389).

The owner's ruling on #4383 for the unchecked line - AutoRun, the CLI, a cold
Find and every session before its first check - priced as ``min-fixed-gmm``
in ``docs/experiments/2026-09-30-line-estimate-4383/REPORT.md``: never worse
than today's count on the shortfall below the floor, better by 0.03-0.12 F1
on small and sparse corpora, and the same on the large ones where the
mixture over-returns and the count caps it.  Pinned here: what the mixture
counts on a planted two-population ranking, that it only ever lowers the
count, that a finished check outranks it, and that it is fitted once per
ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.training.thresholds import (
    FLOOR_UNCHECKED,
    LineRanking,
    SpotCheck,
    check_schedule,
    floor_count,
    floor_line,
    floor_state,
    mixture_count,
    mixture_posterior,
)

N_HIGH, N_LOW = 60, 940


def _two_populations(seed: int = 3, n_high: int = N_HIGH, n_low: int = N_LOW) -> tuple[LineRanking, dict[int, bool]]:
    """*n_high* items scored around 0.85 and *n_low* around 0.2, with 8 Good and 8 Bad votes among them.

    Ids ``1..n_high`` are the high population, the rest the low; the votes
    are the first 8 of each, so ``n_high - 8`` high items are unvoted (52 by
    default).
    """
    rng = np.random.default_rng(seed)
    high = np.clip(rng.normal(0.85, 0.04, n_high), 0.0, 1.0)
    low = np.clip(rng.normal(0.20, 0.08, n_low), 0.0, 1.0)
    scores = np.concatenate([high, low])
    labels = {**dict.fromkeys(range(1, 9), True), **dict.fromkeys(range(n_high + 1, n_high + 9), False)}
    return LineRanking.from_scores(list(range(1, n_high + n_low + 1)), scores, set(labels)), labels


class TestWhatTheMixtureCounts:
    def test_it_counts_as_deep_as_the_set_stays_at_the_floor(self):
        """The ``gmm`` rule: the deepest top *k* whose mean posterior meets the floor.

        52 unvoted high items at posterior ~1: at 90% the set is about them;
        at 50% it runs on into the low population until they are half of it
        (~104); at 10% until they are a tenth (~520).
        """
        ranking, labels = _two_populations()
        counts = {p: mixture_count(ranking, p, labels) or -1 for p in (0.1, 0.5, 0.9)}
        assert 48 <= counts[0.9] <= 62, counts
        assert 90 <= counts[0.5] <= 118, counts
        assert 450 <= counts[0.1] <= 590, counts

    def test_a_higher_floor_counts_fewer_and_a_lower_one_more(self):
        ranking, labels = _two_populations()
        counts = [mixture_count(ranking, p, labels) or -1 for p in (0.1, 0.5, 0.9)]
        assert counts[0] >= counts[1] >= counts[2] >= 1

    def test_votes_cast_since_the_fit_leave_the_count(self):
        ranking, labels = _two_populations()
        base = mixture_count(ranking, 0.5, labels)
        later = mixture_count(ranking, 0.5, labels, also_voted={9, 10, 11, 12})
        assert base is not None and later is not None and later <= base

    def test_nothing_to_count_is_none(self):
        ranking, labels = _two_populations()
        assert mixture_count(None, 0.5, labels) is None
        assert mixture_count(LineRanking.from_scores([], []), 0.5, {}) is None
        assert mixture_count(ranking, 0.5, labels, also_voted=range(1, N_HIGH + N_LOW + 1)) is None

    def test_no_labels_still_counts_off_the_unanchored_fit(self):
        """A labelset that resolves nothing, or a planted ranking: the fit anchors on nothing and still answers."""
        ranking, _labels = _two_populations()
        count = mixture_count(ranking, 0.5, {})
        assert count is not None and 1 <= count <= N_HIGH + N_LOW

    def test_it_is_fitted_once_per_ranking(self, monkeypatch):
        """One fit a ranking, on the first caller's anchors; every floor and every later vote reads it."""
        import vtscore.training.thresholds.gmm as gmm

        ranking, labels = _two_populations()
        calls: list[int] = []
        real = gmm.anchored_gmm_fit

        def _counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(gmm, "anchored_gmm_fit", _counted)
        first = mixture_count(ranking, 0.5, labels)
        assert mixture_count(ranking, 0.5, labels) == first and len(calls) == 1
        mixture_count(ranking, 0.9, labels)
        mixture_count(ranking, 0.5, {**labels, 9: True}, also_voted={9})
        assert len(calls) == 1 and len(ranking.mixture) == 1
        # A ranking the fit failed on answers None thereafter, without retrying.
        bare = LineRanking.from_scores([1, 2, 3], [0.9, 0.5, 0.1])
        monkeypatch.setattr(gmm, "anchored_gmm_fit", lambda *_a, **_k: (None, "gmm_failed:test"))
        assert mixture_count(bare, 0.5, {}) is None and bare.mixture == [None]
        assert mixture_count(bare, 0.5, {}) is None


#: Twenty scores shaped like a learned sort over the app's 20-item test
#: corpus after two Good and two Bad votes (#4419): two near-duplicates at the
#: top, a tight middle, two at the bottom.  The anchored fit lands its high
#: component on the top two alone (``var_hi`` ~3e-7 against a 0.32 spread).
COLLAPSING_SCORES = [
    0.66, 0.659, 0.515, 0.512, 0.511, 0.51, 0.509, 0.506, 0.501, 0.495,
    0.492, 0.491, 0.49, 0.489, 0.488, 0.486, 0.486, 0.476, 0.343, 0.34,
]  # fmt: skip


def _collapsing() -> tuple[LineRanking, dict[int, bool]]:
    labels = {1: True, 2: True, 19: False, 20: False}
    return LineRanking.from_scores(list(range(1, 21)), COLLAPSING_SCORES, set(labels)), labels


class TestACollapsedFitIsNoEstimate:
    """#4419: a component that lands on a few near-duplicate scores says nothing about the rest."""

    def test_the_posterior_and_the_count_are_none(self):
        ranking, labels = _collapsing()
        assert mixture_posterior(ranking, labels) is None
        assert mixture_count(ranking, 0.5, labels) is None
        assert ranking.mixture == [None], "memoised as no fit, not refitted"

    def test_so_the_unchecked_line_keeps_the_schedules_count(self):
        ranking, labels = _collapsing()
        unvoted = ranking.unvoted_ids().size
        assert floor_count(0.5, None, proposal=mixture_count(ranking, 0.5, labels)) == check_schedule(0.5).candidate
        assert floor_line(ranking, 0.5, None, proposal=None) == ranking.threshold_for(min(32, unvoted))

    def test_a_sound_fit_still_counts(self):
        ranking, labels = _two_populations()
        assert mixture_posterior(ranking, labels) is not None
        assert mixture_count(ranking, 0.5, labels) is not None


class TestTheCountItLowers:
    def test_the_unchecked_count_is_the_smaller_of_the_two(self):
        assert floor_count(0.5, None) == 32
        assert floor_count(0.5, None, proposal=10) == 10
        assert floor_count(0.5, None, proposal=100) == 32
        assert floor_count(0.1, None, proposal=100) == 100
        assert floor_count(0.5, None, proposal=0) == 1, "at least one, as the walk's first band is"

    def test_the_line_and_the_state_follow_it(self):
        ranking, labels = _two_populations()
        proposal = mixture_count(ranking, 0.5, labels)
        assert proposal is not None
        assert floor_line(ranking, 0.5, proposal=proposal) == ranking.threshold_for(min(32, proposal))
        state = floor_state(0.5, None, ranking, proposal=proposal)
        assert state.status == FLOOR_UNCHECKED and state.count == min(32, proposal)
        assert state.schedule == check_schedule(0.5), "the walk still starts at the schedule"

    def test_a_finished_check_outranks_it(self):
        ranking, labels = _two_populations()
        check = SpotCheck.start(ranking.unvoted_ids().tolist(), 0.5, seed=0)
        while check.running:
            check.record({cid: cid <= N_HIGH for cid in check.pending})
        assert check.finished
        assert floor_count(0.5, check, proposal=3) == check.k
        assert floor_state(0.5, check, ranking, proposal=3).count == check.k

    @pytest.mark.parametrize("min_precision", [0.1, 0.5, 0.9])
    def test_on_a_large_corpus_the_count_never_exceeds_the_schedule(self, min_precision):
        """Where the mixture over-returns (#4383: 5-66x on large sparse corpora) the schedule caps it."""
        ranking, labels = _two_populations()
        count = floor_count(min_precision, None, proposal=mixture_count(ranking, min_precision, labels))
        assert 1 <= count <= check_schedule(min_precision).candidate
