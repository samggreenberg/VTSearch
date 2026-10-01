"""The balance as a setting and a per-detector preference (#4413, step 2).

``beta`` is per detector and seeded from the user's setting on first read,
as the floor is; ``line_preference`` says which of the two draws the line (the
balance by default since the switch's last step; the floor is deprecated).
Under the balance a change of beta re-cuts every loaded detector at its own
beta without a retrain; under the floor it moves nothing.  The detector's
balance state is the wire shape beside ``threshold``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests_lib.sorting.test_mixture_count import _two_populations
from vtscore.config import DEFAULT_BETA, DEFAULT_LINE_PREFERENCE
from vtscore.state import get_beta, get_line_preference, set_beta, set_line_preference
from vtscore.state.core import (
    DetectorContext,
    detector_balance_positives,
    detector_balance_proposal,
    detector_balance_state,
    human_voted_ids,
    recompute_detector_thresholds,
    recut_detector_threshold,
    register_detector_context,
    set_thread_detector_context,
)
from vtscore.training.thresholds import (
    BALANCE_CHECKED,
    FLOOR_UNCHECKED,
    LineRanking,
    SpotCheck,
    balance_count,
    fbeta_count,
    mixture_positives,
)


def _ctx(detector_id: str = "det-balance") -> DetectorContext:
    """A detector holding a two-population ranking with its votes cast, and no fitted cut."""
    ranking, labels = _two_populations(n_high=28)  # 20 unvoted positives: the F1 peak is under the cap
    ctx = DetectorContext(detector_id)
    ctx.line_ranking = ranking
    ctx.good_votes.update({cid: None for cid, good in labels.items() if good})
    ctx.bad_votes.update({cid: None for cid, good in labels.items() if not good})
    ctx.threshold = -999.0
    return ctx


class TestTheSetting:
    def test_a_detector_seeds_its_balance_from_the_user_setting(self):
        ctx = DetectorContext("det-seed-beta")
        set_thread_detector_context(ctx)
        assert get_beta() == DEFAULT_BETA == 1.0
        assert ctx.beta_seeded and ctx.beta == DEFAULT_BETA
        assert get_line_preference() == DEFAULT_LINE_PREFERENCE == "balance"

    @pytest.mark.parametrize("bad", [0.0, 0.2, 5.0, -1.0])
    def test_a_balance_outside_the_range_is_refused(self, bad):
        set_thread_detector_context(DetectorContext("det-bad-beta"))
        with pytest.raises(ValueError, match="beta must be in"):
            set_beta(bad)

    def test_an_unknown_preference_is_refused(self):
        with pytest.raises(ValueError, match="line_preference must be one of"):
            set_line_preference("inclusion")

    def test_under_the_floor_a_balance_change_moves_no_line(self, floor_preference):
        ctx = _ctx("det-beta-under-floor")
        set_thread_detector_context(ctx)
        register_detector_context(ctx)
        set_beta(2.0)
        assert ctx.beta == 2.0 and ctx.beta_seeded
        assert ctx.threshold == -999.0, "the floor draws the line; beta is stored for when the balance does"


class TestTheLine:
    def test_the_acquisition_cut_under_a_balance_is_the_argmax_rank(self):
        """#4409: the app's acquisition cut is the library's, at half the F-beta argmax's depth; no ranking, no change."""
        from vtscore.state.core import detector_acquisition_threshold, detector_line_labels
        from vtscore.training.thresholds import acquisition_threshold

        ctx = _ctx("det-acq")
        want = acquisition_threshold(ctx.line_ranking, 1.0, detector_line_labels(ctx), human_voted_ids(ctx))
        assert want is not None
        assert detector_acquisition_threshold(ctx, None, beta=1.0) == want
        assert detector_acquisition_threshold(ctx, None, beta=0.5) >= want, "a precision-leaning balance samples higher"
        bare = DetectorContext("det-acq-bare")
        bare.threshold = 0.42
        assert detector_acquisition_threshold(bare, None, beta=1.0) == 0.42, "no ranking: the two jobs coincide"

    def test_a_balance_serves_no_inclusion_so_smart_recuts_at_its_own(self):
        """Under a balance the line keeps a set, not an inclusion (#4243, #4413): Smart re-cuts at Inclusion 0."""
        from vtscore.state.core import detector_line_inclusion
        from vtscore.training.thresholds import PRECISION_FLOOR_FALLBACK_INCLUSION

        ctx = _ctx("det-line-inclusion")
        assert detector_line_inclusion(ctx, None) == PRECISION_FLOOR_FALLBACK_INCLUSION
        assert detector_line_inclusion(ctx, None, 1.0) is None
        assert detector_line_inclusion(ctx, 0.5, None) is None

    def test_the_recut_under_a_balance_keeps_the_mixtures_argmax_under_the_cap(self):
        ctx = _ctx()
        voted = human_voted_ids(ctx)
        labels = {cid: cid in ctx.good_votes for cid in voted}
        proposal = fbeta_count(ctx.line_ranking, 1.0, labels, voted)
        assert proposal is not None and detector_balance_proposal(ctx, 1.0) == proposal
        assert 1 <= proposal <= 32
        assert recut_detector_threshold(ctx, beta=1.0) == ctx.line_ranking.threshold_for(proposal, voted)
        # Recall-leaning: the cap is 128, and the argmax runs deeper into the low population.
        assert (detector_balance_proposal(ctx, 2.0) or 0) >= proposal
        assert recut_detector_threshold(ctx, beta=2.0) == ctx.line_ranking.threshold_for(
            balance_count(2.0, None, detector_balance_proposal(ctx, 2.0)), voted
        )

    def test_the_positives_the_walk_reads_recall_against(self):
        ctx = _ctx()
        n_pos = detector_balance_positives(ctx)
        assert n_pos == mixture_positives(
            ctx.line_ranking, {c: c in ctx.good_votes for c in human_voted_ids(ctx)}, human_voted_ids(ctx)
        )
        assert n_pos is not None and 10 <= n_pos <= 40, n_pos

    def test_a_finished_balance_walk_moves_the_line_to_its_peak(self):
        ctx = _ctx()
        voted = human_voted_ids(ctx)
        unvoted = ctx.line_ranking.unvoted_ids(voted).tolist()
        n_pos = detector_balance_positives(ctx)
        assert n_pos is not None
        check = SpotCheck.start_balance(unvoted, 1.0, n_pos, seed=0)
        positives = set(range(1, 29))  # the high population's ids
        while check.running:
            check.record({cid: cid in positives for cid in check.pending})
        check.fingerprint = ctx.line_ranking.fingerprint(check.k, voted)
        ctx.precision_check = check
        assert recut_detector_threshold(ctx, beta=1.0) == ctx.line_ranking.threshold_for(check.k, voted)
        state = detector_balance_state(ctx, 1.0)
        assert state["status"] == BALANCE_CHECKED and state["count"] == check.k
        assert state["precision"]["stale"] is False and state["recall"]["lo"] <= state["recall"]["hi"]
        assert detector_balance_state(ctx, 2.0)["status"] == FLOOR_UNCHECKED, "a result belongs to its balance"

    def test_the_state_before_a_check(self):
        ctx = _ctx()
        state = detector_balance_state(ctx, 1.0)
        assert state["status"] == FLOOR_UNCHECKED and state["beta"] == 1.0
        assert state["count"] == balance_count(1.0, None, detector_balance_proposal(ctx, 1.0))
        assert state["precision"] is None and state["recall"] is None and state["fbeta"] is None
        assert state["schedule"]["candidate"] == 32
        bare = DetectorContext("det-bare-balance")
        assert detector_balance_state(bare, 2.0)["count"] == 128

    def test_each_detector_is_recut_at_its_own_balance(self):
        mine = _ctx("det-own-beta")
        mine.beta, mine.beta_seeded = 2.0, True
        register_detector_context(mine)
        unseeded = _ctx("det-unseeded-beta")
        register_detector_context(unseeded)

        recompute_detector_thresholds(None, beta=1.0)

        assert mine.threshold == recut_detector_threshold(mine, beta=2.0)
        assert unseeded.threshold == recut_detector_threshold(unseeded, beta=1.0)
        assert mine.threshold != -999.0 and unseeded.threshold != -999.0

    def test_with_no_ranking_a_balance_falls_through_to_the_inclusion_fallbacks(self):
        bare = DetectorContext("det-bare-recut")
        assert recut_detector_threshold(bare, 4, beta=1.0) is None
        with pytest.raises(ValueError, match="needs an inclusion"):
            recut_detector_threshold(bare)


def test_the_planted_ranking_is_what_the_tests_assume():
    ranking, labels = _two_populations(n_high=28)
    assert isinstance(ranking, LineRanking) and len(labels) == 16
    assert np.all(ranking.scores[:-1] >= ranking.scores[1:])
