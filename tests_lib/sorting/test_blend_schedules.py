"""Mix-in schedules for the safe-threshold blend (issue #2841).

``prod`` no longer ships - #2841 replaced it with a schedule per voting mode -
but it stays pinned bit-identical to the hard-coded ``clip((n - 6) / 14, 0, 1)``
it replaced, because every number in the study's report is a delta against it.
If ``prod`` drifts, the report stops meaning anything.
"""

from __future__ import annotations

import math

import pytest

from vtscore.training.blend_schedules import (
    PRODUCTION_SCHEDULE,
    PRODUCTION_SCHEDULE_BY_MODE,
    SAFE_BLEND_SCHEDULES,
    BlendContext,
    get_schedule,
    production_schedule_for,
    schedule_names,
)
from vtscore.training.thresholds import (
    NO_GOOD_THRESHOLD,
    GmmFit1D,
    blend_gmm_threshold,
    calculate_gmm_threshold,
    calculate_safe_threshold,
    fit_gmm_threshold,
    safe_blend_weight,
)


def _ctx(n: int, good: int | None = None) -> BlendContext:
    """A context with *n* votes, *good* of them positive (default: half)."""
    g = n // 2 if good is None else good
    return BlendContext(n_labels=n, n_good=g, n_bad=n - g)


class TestProductionFidelity:
    """``prod`` is the study's baseline; nothing may move it."""

    @pytest.mark.parametrize("n", range(0, 80))
    def test_prod_reproduces_the_historical_ramp(self, n):
        # Named explicitly: the default is no longer `prod`, so an implicit
        # lookup here would quietly test whatever ships instead.
        historical = max(0.0, min(1.0, (n - 6) / 14))
        assert safe_blend_weight(_ctx(n), "prod") == historical

    def test_the_mode_agnostic_default_is_the_universally_robust_schedule(self):
        """#2841 ships a schedule per voting mode; the mode-agnostic fallback is
        the one arm that improved *both* modes under *every* cost weighting, so a
        caller that cannot say which mode it is in still improves on the old
        ramp."""
        assert PRODUCTION_SCHEDULE == "cap50"
        assert get_schedule(None) is get_schedule("cap50")

    def test_each_voting_mode_gets_the_schedule_the_study_chose(self):
        assert production_schedule_for(region_voting=True) == "slow_cap50"
        assert production_schedule_for(region_voting=False) == "cap50"
        assert production_schedule_for(region_voting=None) == PRODUCTION_SCHEDULE

    def test_every_shipped_schedule_beats_the_old_ramp_on_its_own_mode(self):
        """Guards against a typo silently shipping a schedule the study rejected."""
        for name in {*PRODUCTION_SCHEDULE_BY_MODE.values(), PRODUCTION_SCHEDULE}:
            assert name in SAFE_BLEND_SCHEDULES
            assert name != "prod", "prod is the measurement baseline, not a ship target"

    def test_blend_matches_a_hand_computed_average(self):
        # 13 labels → weight 0.5 → the plain midpoint of the two cuts.
        assert blend_gmm_threshold(0.9, 0.3, _ctx(13)) == pytest.approx(0.6)

    def test_calculate_safe_threshold_equals_blend_of_its_parts(self):
        scores = [0.1, 0.15, 0.2, 0.75, 0.8, 0.9]
        expected = blend_gmm_threshold(0.9, calculate_gmm_threshold(scores), _ctx(13))
        assert calculate_safe_threshold(0.9, scores, _ctx(13)) == expected


class TestRegistry:
    def test_every_schedule_is_registered_under_its_own_name(self):
        for name, sched in SAFE_BLEND_SCHEDULES.items():
            assert sched.name == name

    def test_unknown_name_raises_with_the_known_names_listed(self):
        with pytest.raises(ValueError, match="unknown safe-threshold schedule"):
            get_schedule("no_such_schedule")

    @pytest.mark.parametrize("name", schedule_names())
    def test_weights_stay_in_the_unit_interval(self, name):
        sched = get_schedule(name)
        for n in range(0, 120):
            w = sched.weight(_ctx(n))
            assert 0.0 <= w <= 1.0

    @pytest.mark.parametrize("name", schedule_names())
    def test_weight_never_decreases_with_more_labels(self, name):
        """More evidence must never mean *less* trust in the learned cut."""
        sched = get_schedule(name)
        weights = [sched.weight(_ctx(n)) for n in range(0, 120)]
        assert weights == sorted(weights)

    @pytest.mark.parametrize("name", schedule_names())
    def test_blend_is_finite_and_bracketed_by_its_inputs(self, name):
        sched = get_schedule(name)
        fit = GmmFit1D(w_lo=0.7, mu_lo=0.2, var_lo=0.01, w_hi=0.3, mu_hi=0.8, var_hi=0.01)
        for n in (0, 6, 13, 20, 50):
            for xcal in (0.05, 0.4, 0.95):
                out = sched.combine(xcal, 0.5, _ctx(n), fit)
                assert math.isfinite(out)
                assert min(xcal, 0.2) <= out <= max(xcal, 0.8)


class TestControls:
    def test_pure_gmm_never_consults_the_learned_cut(self):
        for n in (0, 6, 20, 500):
            assert blend_gmm_threshold(0.9, 0.3, _ctx(n), schedule="pure_gmm") == pytest.approx(0.3)

    def test_pure_xcal_never_blends(self):
        for n in (0, 6, 20, 500):
            assert blend_gmm_threshold(0.9, 0.3, _ctx(n), schedule="pure_xcal") == pytest.approx(0.9)


class TestStatisticFamily:
    """Schedules that ramp on the rarer class, not the total (#2790's starvation)."""

    def test_rare_class_schedule_ignores_a_lopsided_total(self):
        starved = BlendContext(n_labels=40, n_good=1, n_bad=39)
        balanced = BlendContext(n_labels=40, n_good=20, n_bad=20)
        assert safe_blend_weight(starved, "rare") < safe_blend_weight(balanced, "rare")
        # The production schedule cannot tell these apart at all - that is the
        # whole point of the family.
        assert safe_blend_weight(starved, "prod") == safe_blend_weight(balanced, "prod")

    def test_positive_count_schedule_tracks_positives_only(self):
        few_pos = BlendContext(n_labels=40, n_good=2, n_bad=38)
        many_pos = BlendContext(n_labels=40, n_good=30, n_bad=10)
        assert safe_blend_weight(few_pos, "pos") < safe_blend_weight(many_pos, "pos")

    def test_bare_int_context_leaves_the_class_split_degenerate(self):
        """An ``int`` carries no breakdown, so class-reading schedules see zero
        positives rather than a fabricated split."""
        assert safe_blend_weight(40, "pos") == 0.0
        assert safe_blend_weight(40, "prod") == 1.0


class TestCapFamily:
    def test_capped_schedules_keep_a_permanent_gmm_share(self):
        for n in (20, 100, 10_000):
            assert safe_blend_weight(_ctx(n), "cap80") == pytest.approx(0.8)
            assert safe_blend_weight(_ctx(n), "cap50") == pytest.approx(0.5)

    def test_slow_cap50_is_slow_early_and_capped_late(self):
        """The synthesis the region long run implies: `slow`'s gentler ramp kept
        (it won the early window) but capped (it collapsed once it reached pure
        x-cal at 40 labels)."""
        for n in (7, 13, 20):
            assert safe_blend_weight(_ctx(n), "slow_cap50") == safe_blend_weight(_ctx(n), "slow")
        for n in (40, 100, 10_000):
            assert safe_blend_weight(_ctx(n), "slow_cap50") == pytest.approx(0.5)
            assert safe_blend_weight(_ctx(n), "slow") == pytest.approx(1.0)


class TestCorridorFamily:
    """The clamp family: a no-op on sane cuts, a bound on wild ones."""

    FIT = GmmFit1D(w_lo=0.7, mu_lo=0.2, var_lo=0.01, w_hi=0.3, mu_hi=0.8, var_hi=0.01)

    def test_sane_xcal_passes_through_untouched(self):
        out = blend_gmm_threshold(0.45, 0.5, _ctx(30), schedule="corridor", fit=self.FIT)
        assert out == pytest.approx(0.45)

    def test_wild_xcal_is_clamped_to_the_component_means(self):
        assert blend_gmm_threshold(0.01, 0.5, _ctx(30), schedule="corridor", fit=self.FIT) == pytest.approx(0.2)
        assert blend_gmm_threshold(0.99, 0.5, _ctx(30), schedule="corridor", fit=self.FIT) == pytest.approx(0.8)

    def test_the_admit_nothing_sentinel_is_clamped_back_into_range(self):
        """#2788's cold-start failure: a cut above the score range admits
        nothing at all.  The corridor is the targeted defence against it."""
        out = blend_gmm_threshold(NO_GOOD_THRESHOLD, 0.5, _ctx(30), schedule="corridor", fit=self.FIT)
        assert out == pytest.approx(0.8)

    def test_ramped_corridor_is_a_point_at_the_floor_and_open_at_the_top(self):
        at_floor = blend_gmm_threshold(0.01, 0.5, _ctx(6), schedule="corridor_ramp", fit=self.FIT)
        assert at_floor == pytest.approx(0.5)  # zero-width corridor == pure GMM
        wide_open = blend_gmm_threshold(0.01, 0.5, _ctx(20), schedule="corridor_ramp", fit=self.FIT)
        assert wide_open == pytest.approx(0.01)  # unbounded past the ramp

    def test_falls_back_to_a_plain_blend_without_a_fit(self):
        """The median/degenerate fallbacks have no component means, so the
        corridor must not silently become a no-op cut."""
        out = blend_gmm_threshold(0.01, 0.5, _ctx(30), schedule="corridor", fit=None)
        assert out == pytest.approx(0.01)  # weight 1 → x-cal, not an unbounded clamp

    def test_component_means_are_ordered_before_clamping(self):
        """A fit whose ``lo``/``hi`` arrive inverted must still yield a valid
        interval rather than an empty one."""
        inverted = GmmFit1D(w_lo=0.5, mu_lo=0.8, var_lo=0.01, w_hi=0.5, mu_hi=0.2, var_hi=0.01)
        out = blend_gmm_threshold(0.01, 0.5, _ctx(30), schedule="corridor", fit=inverted)
        assert out == pytest.approx(0.2)


class TestFitGmmThreshold:
    """The cut and the fit behind it must come from one EM pass."""

    def test_returns_the_same_cut_as_the_production_helper(self):
        scores = [0.1, 0.12, 0.15, 0.2, 0.75, 0.8, 0.85, 0.9]
        cut, fit = fit_gmm_threshold(scores)
        assert cut == pytest.approx(calculate_gmm_threshold(scores))
        assert fit is not None
        assert cut == pytest.approx(fit.midpoint())

    def test_degenerate_input_yields_no_fit(self):
        cut, fit = fit_gmm_threshold([0.5])
        assert cut == 0.5
        assert fit is None


class TestBlendContext:
    def test_counts_rows_when_there_are_no_bags(self):
        ctx = BlendContext.from_labels([1.0, 0.0, 0.0])
        assert (ctx.n_labels, ctx.n_good, ctx.n_bad) == (3, 1, 2)

    def test_collapses_flooded_bags_to_one_vote_each(self):
        """Region flooding turns one Bad vote into many rows; the schedule must
        still see one vote, or a single flooded Bad would race it to the top."""
        y = [1.0, 0.0, 0.0, 0.0, 0.0]
        groups = ["good-img", "bad-img", "bad-img", "bad-img", "bad-img"]
        ctx = BlendContext.from_labels(y, groups)
        assert (ctx.n_labels, ctx.n_good, ctx.n_bad) == (2, 1, 1)

    def test_rare_is_the_smaller_class(self):
        assert BlendContext(20, 3, 17).n_rare == 3
        assert BlendContext(20, 17, 3).n_rare == 3
