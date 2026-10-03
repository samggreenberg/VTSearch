"""The vote-anchored mixture the unchecked line reads (#4389, #4413, #4419).

Before any check the balance's line keeps the smaller of its cap and the
mixture's F-beta argmax (:func:`~vtscore.training.thresholds.fbeta_count`);
the walk reads recall against the mixture's count of positives.  Pinned here:
that the mixture is fitted once per ranking, what it answers with nothing to
count, and that a collapsed fit is no estimate.  What the argmax counts is
pinned in ``test_balance.py``.
"""

from __future__ import annotations

import numpy as np

from vtscore.training.thresholds import (
    LineRanking,
    balance_count,
    balance_line,
    balance_schedule,
    fbeta_count,
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


class TestTheFit:
    def test_votes_cast_since_the_fit_leave_the_count(self):
        ranking, labels = _two_populations()
        base = mixture_posterior(ranking, labels)
        later = mixture_posterior(ranking, labels, also_voted={9, 10, 11, 12})
        assert base is not None and later is not None and later.size == base.size - 4

    def test_nothing_to_count_is_none(self):
        ranking, labels = _two_populations()
        assert fbeta_count(None, 1.0, labels) is None
        assert fbeta_count(LineRanking.from_scores([], []), 1.0, {}) is None
        assert fbeta_count(ranking, 1.0, labels, also_voted=range(1, N_HIGH + N_LOW + 1)) is None

    def test_no_labels_still_counts_off_the_unanchored_fit(self):
        """A labelset that resolves nothing, or a planted ranking: the fit anchors on nothing and still answers."""
        ranking, _labels = _two_populations()
        count = fbeta_count(ranking, 1.0, {})
        assert count is not None and 1 <= count <= N_HIGH + N_LOW

    def test_it_is_fitted_once_per_ranking(self, monkeypatch):
        """One fit a ranking, on the first caller's anchors; every balance and every later vote reads it."""
        import vtscore.training.thresholds.gmm as gmm

        ranking, labels = _two_populations()
        calls: list[int] = []
        real = gmm.anchored_gmm_fit

        def _counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(gmm, "anchored_gmm_fit", _counted)
        first = fbeta_count(ranking, 1.0, labels)
        assert fbeta_count(ranking, 1.0, labels) == first and len(calls) == 1
        fbeta_count(ranking, 2.0, labels)
        fbeta_count(ranking, 1.0, {**labels, 9: True}, also_voted={9})
        assert len(calls) == 1 and len(ranking.mixture) == 1
        # A ranking the fit failed on answers None thereafter, without retrying.
        bare = LineRanking.from_scores([1, 2, 3], [0.9, 0.5, 0.1])
        monkeypatch.setattr(gmm, "anchored_gmm_fit", lambda *_a, **_k: (None, "gmm_failed:test"))
        assert fbeta_count(bare, 1.0, {}) is None and bare.mixture == [None]
        assert fbeta_count(bare, 1.0, {}) is None


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
        assert fbeta_count(ranking, 1.0, labels) is None
        assert ranking.mixture == [None], "memoised as no fit, not refitted"

    def test_so_the_unchecked_line_keeps_the_cap(self):
        ranking, labels = _collapsing()
        unvoted = ranking.unvoted_ids().size
        assert balance_count(1.0, None, proposal=fbeta_count(ranking, 1.0, labels)) == balance_schedule(1.0).candidate
        assert balance_line(ranking, 1.0, None, proposal=None) == ranking.threshold_for(min(32, unvoted))

    def test_a_sound_fit_still_counts(self):
        ranking, labels = _two_populations()
        assert mixture_posterior(ranking, labels) is not None
        assert fbeta_count(ranking, 1.0, labels) is not None
