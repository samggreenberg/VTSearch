"""The cut-rule family and its decomposition (issue #2836).

These tests are the derivation itself, written as assertions.  The central
claims - that the prior-free crossing carries no priors, that it collapses to
the midpoint exactly when the variances agree, and that today's crossing sits
above it by the prior-odds term - are all closed-form, so they are checked
against closed form rather than against a golden number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.eval.calibration_metrics import inclusion_weights, operating_cost
from vtscore.eval.cut_rules import (
    ALL_RULES,
    EVT_FIT_RULES,
    EVT_RULES,
    TAIL_ALPHA_GRID,
    TAIL_ALPHA_PREREGISTERED,
    TAIL_RULES,
    decomposition_cuts,
    gaussian_cuts,
    sim_oracle_cut,
    supervised_cut,
    tail_alpha_rule,
    tail_cuts,
)
from vtscore.training.evt_mixture import GumbelNormalFit1D
from vtscore.training.thresholds import GmmFit1D, _weighted_gaussian_crossing


def _fit(w_lo=0.9, mu_lo=0.2, var_lo=0.01, w_hi=0.1, mu_hi=0.8, var_hi=0.01) -> GmmFit1D:
    return GmmFit1D(w_lo=w_lo, mu_lo=mu_lo, var_lo=var_lo, w_hi=w_hi, mu_hi=mu_hi, var_hi=var_hi)


def _root(x: float | None) -> float:
    """A crossing narrowed to non-``None`` (its no-root return).

    Every fit used below is well-separated and does have a root, so a ``None``
    here is a real failure, not a case the assertion should tiptoe around.
    """
    assert x is not None, "expected this fit to have a crossing"
    return x


def _log_density_gap(fit: GmmFit1D, x: float, lam: float) -> float:
    """``log(w_lo*N_lo(x)) - log(lam*w_hi*N_hi(x))`` - zero at the crossing."""

    def logn(mu: float, var: float) -> float:
        return -0.5 * (math.log(2 * math.pi * var) + (x - mu) ** 2 / var)

    return (math.log(fit.w_lo) + logn(fit.mu_lo, fit.var_lo)) - (math.log(lam * fit.w_hi) + logn(fit.mu_hi, fit.var_hi))


class TestCrossingFamily:
    def test_lam_one_is_unchanged_by_the_generalisation(self):
        """The default keeps solving the original equal-density equation."""
        fit = _fit()
        direct = _weighted_gaussian_crossing(fit.w_lo, fit.mu_lo, fit.var_lo, fit.w_hi, fit.mu_hi, fit.var_hi)
        assert fit.crossing() == direct

    @pytest.mark.parametrize("lam", [0.25, 1.0, 4.0, 40.0])
    def test_returned_root_solves_the_equation(self, lam):
        fit = _fit(var_lo=0.02, var_hi=0.005)
        x = fit.crossing(lam=lam)
        assert x is not None
        assert _log_density_gap(fit, x, lam) == pytest.approx(0.0, abs=1e-9)

    def test_larger_lam_moves_the_cut_down(self):
        """``lam`` scales the Good side up, so the boundary retreats toward Bad."""
        fit = _fit()
        cuts = [_root(fit.crossing(lam=lam)) for lam in (0.5, 1.0, 2.0, 8.0)]
        assert cuts == sorted(cuts, reverse=True)

    def test_non_positive_lam_is_rejected(self):
        assert _fit().crossing(lam=0.0) is None
        assert _fit().crossing(lam=-1.0) is None


class TestRateOptimalCut:
    """The #2836 claim: the scored loss is a rate loss, so the cut is prior-free."""

    def test_equal_variance_rate_crossing_is_exactly_the_midpoint(self):
        """With equal variances and equal cost weights the two rules coincide.

        This is why the historical midpoint is not an arbitrary heuristic: it is
        an (unbiased, if the variances match) estimate of the rate-optimal cut.
        """
        for w_lo in (0.5, 0.9, 0.99, 0.999):
            fit = _fit(w_lo=w_lo, w_hi=1.0 - w_lo, var_lo=0.01, var_hi=0.01)
            assert fit.rate_crossing(1.0, 1.0) == pytest.approx(fit.midpoint(), abs=1e-12)

    def test_prior_free_crossing_does_not_depend_on_the_mixture_weights(self):
        """Its defining property, and the one the count-optimal cut lacks.

        Varying the weights over three orders of magnitude - what changing the
        pool's prevalence does - must leave the rate-optimal cut where it is,
        while the equal-density crossing walks steadily upward.
        """
        cuts_rate, cuts_count = [], []
        for w_lo in (0.5, 0.9, 0.99, 0.999):
            fit = _fit(w_lo=w_lo, w_hi=1.0 - w_lo, var_lo=0.02, var_hi=0.004)
            cuts_rate.append(_root(fit.rate_crossing(1.0, 1.0)))
            cuts_count.append(_root(fit.crossing()))
        assert max(cuts_rate) - min(cuts_rate) == pytest.approx(0.0, abs=1e-12)
        assert cuts_count == sorted(cuts_count)
        assert cuts_count[-1] > cuts_count[0] + 0.01

    def test_crossing_exceeds_the_rate_optimum_by_the_prior_odds_term(self):
        """Equal-variance case: the bias is exactly ``var*ln(w_lo/w_hi)/dmu``."""
        w_lo, var = 0.97, 0.01
        fit = _fit(w_lo=w_lo, w_hi=1.0 - w_lo, var_lo=var, var_hi=var)
        expected = var * math.log(fit.w_lo / fit.w_hi) / (fit.mu_hi - fit.mu_lo)
        assert _root(fit.crossing()) - fit.midpoint() == pytest.approx(expected, rel=1e-9)
        assert fit.equal_var_offset() == pytest.approx(expected, rel=1e-9)

    def test_cost_weights_tilt_the_cut_the_right_way(self):
        """Caring more about misses (``fnr_weight`` up) must lower the cut."""
        fit = _fit(var_lo=0.02, var_hi=0.004)
        base = _root(fit.rate_crossing(1.0, 1.0))
        assert _root(fit.rate_crossing(1.0, 4.0)) < base
        assert _root(fit.rate_crossing(4.0, 1.0)) > base

    def test_inclusion_weights_are_the_production_definition(self):
        """The harness must price inclusion the way the shipped rule reads it."""
        from vtscore.training.thresholds import inclusion_cost_weights

        for k in range(-10, 11):
            assert inclusion_weights(k) == inclusion_cost_weights(k)

    def test_rate_crossing_matches_inclusion_weights(self):
        """Inclusion 0 is (1, 1), so the knob's neutral setting *is* prior-free."""
        wf, wn = inclusion_weights(0)
        fit = _fit(var_lo=0.02, var_hi=0.004)
        assert fit.rate_crossing(wf, wn) == pytest.approx(fit.rate_crossing(1.0, 1.0))

    def test_rate_optimum_beats_the_count_optimum_on_the_rate_loss(self):
        """End to end: on a sample from the fitted model, the rate cut wins.

        Two well-separated Gaussians with a 95/5 mixture - the shape a sparse
        category actually produces - sampled densely enough that the empirical
        rate loss is a faithful stand-in for the population one.
        """
        rng = np.random.default_rng(2836)
        n_neg, n_pos = 19_000, 1_000
        neg = rng.normal(0.25, math.sqrt(0.02), n_neg)
        pos = rng.normal(0.75, math.sqrt(0.004), n_pos)
        scores = np.concatenate([neg, pos])
        labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
        fit = GmmFit1D(w_lo=0.95, mu_lo=0.25, var_lo=0.02, w_hi=0.05, mu_hi=0.75, var_hi=0.004)

        cost_rate = operating_cost(scores, labels, _root(fit.rate_crossing(1.0, 1.0)), 1.0, 1.0)[0]
        cost_count = operating_cost(scores, labels, _root(fit.crossing()), 1.0, 1.0)[0]
        assert cost_rate < cost_count


class TestSupervisedAndOracleCuts:
    def test_supervised_cut_recovers_the_analytic_class_crossing(self):
        rng = np.random.default_rng(11)
        neg = rng.normal(0.3, 0.1, 40_000)
        pos = rng.normal(0.7, 0.05, 40_000)
        scores = np.concatenate([neg, pos])
        labels = np.concatenate([np.zeros(40_000), np.ones(40_000)])

        cut, stats = supervised_cut(scores, labels, 1.0, 1.0)
        truth = GmmFit1D(w_lo=1.0, mu_lo=0.3, var_lo=0.01, w_hi=1.0, mu_hi=0.7, var_hi=0.0025)
        assert cut == pytest.approx(truth.crossing(), abs=5e-3)
        assert stats["s_mu_neg"] == pytest.approx(0.3, abs=5e-3)
        assert stats["s_prevalence"] == pytest.approx(0.5, abs=1e-6)

    def test_supervised_cut_is_nan_without_both_classes(self):
        scores = np.linspace(0.0, 1.0, 50)
        cut, _stats = supervised_cut(scores, np.zeros(50), 1.0, 1.0)
        assert math.isnan(cut)

    def test_sim_oracle_matches_a_brute_force_minimiser(self):
        rng = np.random.default_rng(5)
        scores = rng.random(400)
        labels = (rng.random(400) < 0.3).astype(float)
        cut = sim_oracle_cut(scores, labels, 1.0, 2.0)
        best = min(
            (operating_cost(scores, labels, c, 1.0, 2.0)[0] for c in np.unique(scores)),
        )
        assert operating_cost(scores, labels, cut, 1.0, 2.0)[0] == pytest.approx(best)


class TestDecomposition:
    def _sample(self, rng, n=6000, prevalence=0.08):
        """A sparse but cleanly separated pool - every rule has a root here.

        Overlapping modes are covered by ``test_missing_rules_are_nan_...``; this
        fixture exists to check the chain's arithmetic, which needs all four cuts
        to exist.
        """
        labels = (rng.random(n) < prevalence).astype(float)
        scores = np.where(
            labels == 1.0,
            rng.normal(0.80, 0.06, n),
            rng.normal(0.25, 0.08, n),
        ).clip(0.0, 1.0)
        return scores, labels

    def test_every_rule_is_reported(self):
        rng = np.random.default_rng(7)
        scores, labels = self._sample(rng)
        cuts, params, _reasons = decomposition_cuts(scores, labels, 1.0, 1.0)
        assert set(cuts) == set(ALL_RULES)
        assert params["gmm_ok"] == 1
        assert params["sim_n"] == pytest.approx(len(scores))

    def test_chain_telescopes(self):
        """The four terms must sum to the total error of today's cut."""
        rng = np.random.default_rng(8)
        scores, labels = self._sample(rng)
        cuts, _params, _reasons = decomposition_cuts(scores, labels, 1.0, 1.0)
        chain = ["cross", "priorfree", "supervised", "sim_oracle"]
        assert all(math.isfinite(cuts[name]) for name in chain), cuts
        terms = [cuts[a] - cuts[b] for a, b in zip(chain[:-1], chain[1:], strict=True)]
        assert sum(terms) == pytest.approx(cuts["cross"] - cuts["sim_oracle"], abs=1e-12)

    def test_prior_free_sits_below_the_count_optimal_cut(self):
        """The predicted direction: the repaired rule moves toward the negatives."""
        rng = np.random.default_rng(9)
        scores, labels = self._sample(rng, prevalence=0.02)
        cuts, _params, _reasons = decomposition_cuts(scores, labels, 1.0, 1.0)
        assert cuts["priorfree"] < cuts["cross"]

    def test_oracle_tail_level_is_a_probability(self):
        rng = np.random.default_rng(10)
        scores, labels = self._sample(rng)
        _cuts, params, _reasons = decomposition_cuts(scores, labels, 1.0, 1.0)
        assert 0.0 <= params["oracle_lo_sf_gauss"] <= 1.0

    def test_missing_rules_are_nan_not_silently_the_midpoint(self):
        """A rule with no root must not be scored under the midpoint's value.

        Conflating "no crossing exists" with "fall back to the midpoint" inside
        the measurement would make two variants secretly identical and inflate
        the midpoint's apparent agreement with everything else.
        """
        # Reversed means: no Bad-then-Good boundary exists on this fit.
        fit = GmmFit1D(w_lo=0.9, mu_lo=0.8, var_lo=0.01, w_hi=0.1, mu_hi=0.2, var_hi=0.01)
        cuts = gaussian_cuts(fit, 1.0, 1.0)
        assert math.isnan(cuts["cross"])
        assert math.isnan(cuts["priorfree"])
        assert not math.isnan(cuts["mid"])

    def test_every_evt_rule_carries_a_reason(self):
        """A rule that declines must say why, whether or not the fit existed (#2846)."""
        rng = np.random.default_rng(11)
        scores, labels = self._sample(rng)
        _cuts, _params, reasons = decomposition_cuts(scores, labels, 1.0, 1.0)
        assert set(reasons) == set(EVT_FIT_RULES)
        assert all(r for r in reasons.values()), reasons

    def test_an_unfittable_sample_reports_a_fit_reason_not_a_crossing_one(self):
        """ "The fit failed" and "the fit had no crossing" are different diagnoses."""
        scores = np.array([0.5, 0.5, 0.5])
        labels = np.array([0.0, 0.0, 1.0])
        _cuts, params, reasons = decomposition_cuts(scores, labels, 1.0, 1.0)
        assert params["evt_ok"] == 0
        assert all(r.startswith("fit_") for r in reasons.values()), reasons

    def test_the_two_gumbel_families_are_reported_separately(self):
        """``gumbel_*`` keeps #2836's orientation guard; ``gumbel_any_*`` drops it.

        If these ever collapse into the same numbers the re-measurement cannot
        attribute a difference to the repair, which is the entire point of
        carrying both.
        """
        assert {"gumbel_priorfree", "gumbel_any_priorfree"} <= set(EVT_RULES)
        assert set(EVT_RULES) <= set(ALL_RULES)


class TestTailAlphaRule:
    """#2881's one-constant rule: the fitted Bad component's own upper quantile.

    Closed form in both orientations, so these check the arithmetic against the
    definition (``lo_survival`` at the returned cut must be ``alpha``) rather than
    against a golden number - which also means the two directions cannot drift
    apart without a failure here.
    """

    def _evt(self, *, swapped=False) -> GumbelNormalFit1D:
        """A well-separated fit, with the Gumbel on the low or the high mode."""
        if swapped:
            # Normal below, Gumbel above -> `lo_quantile` must read the Normal.
            return GumbelNormalFit1D(w_gumbel=0.1, loc=2.0, scale=0.4, w_normal=0.9, mu=-2.0, var=0.5, mean_loglik=0.0)
        return GumbelNormalFit1D(w_gumbel=0.9, loc=-2.0, scale=0.4, w_normal=0.1, mu=2.0, var=0.5, mean_loglik=0.0)

    @pytest.mark.parametrize("swapped", [False, True])
    @pytest.mark.parametrize("alpha", [0.01, 0.04, 0.158, 0.4, 0.9])
    def test_quantile_inverts_the_survival_function(self, alpha, swapped):
        fit = self._evt(swapped=swapped)
        x = fit.lo_quantile(alpha)
        assert x is not None
        assert fit.lo_survival(x) == pytest.approx(alpha, abs=1e-12)

    def test_gumbel_branch_matches_the_closed_form(self):
        """``x = loc - scale*ln(-ln(1-alpha))``; at 0.158 that is ``loc + 1.761*scale``."""
        fit = self._evt()
        x = fit.lo_quantile(TAIL_ALPHA_PREREGISTERED)
        assert x is not None
        offset = (x - fit.loc) / fit.scale
        assert offset == pytest.approx(-math.log(-math.log1p(-TAIL_ALPHA_PREREGISTERED)), rel=1e-12)
        assert offset == pytest.approx(1.7604, abs=1e-4)

    def test_swapped_fit_reads_the_normal_not_the_gumbel(self):
        """The one thing this rule can still get wrong, and the reason it branches.

        A quantile taken off the Gumbel when the Normal is the low component
        would be a plausible number from the wrong component - silent, and on the
        wrong end of the axis.
        """
        fit = self._evt(swapped=True)
        x = fit.lo_quantile(TAIL_ALPHA_PREREGISTERED)
        assert x is not None
        assert fit.mu < x < fit.loc  # the Normal's upper tail, below the Gumbel's mode
        # A Normal's median is its mean, which the Gumbel branch would miss by
        # ``loc + 0.3665*scale`` - a plausible number from the wrong component.
        assert fit.lo_quantile(0.5) == pytest.approx(fit.mu, abs=1e-12)

    def test_higher_alpha_cuts_lower(self):
        """More Bad mass left above the cut means a cut further down the axis."""
        fit = self._evt()
        raw = [fit.lo_quantile(a) for a in TAIL_ALPHA_GRID]
        assert all(c is not None for c in raw)
        cuts = [c for c in raw if c is not None]
        assert cuts == sorted(cuts, reverse=True)

    def test_alpha_outside_the_unit_interval_is_declined(self):
        fit = self._evt()
        assert fit.lo_quantile(0.0) is None
        assert fit.lo_quantile(1.0) is None
        assert fit.lo_quantile(-0.1) is None

    def test_the_rule_needs_no_crossing_to_exist(self):
        """The whole reason this family is worth a run after #2846.

        On a fit whose modes have collapsed onto each other every crossing rule
        declines, and the crossing family degrades to the midpoint.  A tail
        quantile is still perfectly well defined there.
        """
        fit = GumbelNormalFit1D(w_gumbel=0.9, loc=0.0, scale=0.4, w_normal=0.1, mu=0.02, var=4.0, mean_loglik=0.0)
        assert fit.crossing_state(allow_swapped=True)[0] != "ok"
        cuts, reasons = tail_cuts(fit)
        assert set(reasons.values()) == {"ok"}
        assert all(math.isfinite(c) for c in cuts.values()), cuts

    def test_rule_names_are_stable_and_sorted_by_alpha(self):
        assert tail_alpha_rule(0.158) == "tail_a158"
        assert tail_alpha_rule(0.04) == "tail_a040"
        assert TAIL_RULES == tuple(tail_alpha_rule(a) for a in TAIL_ALPHA_GRID)
        assert list(TAIL_ALPHA_GRID) == sorted(TAIL_ALPHA_GRID)
        assert TAIL_ALPHA_PREREGISTERED in TAIL_ALPHA_GRID

    def test_tail_cuts_are_reported_in_score_space(self):
        """The fit is on the logit axis; the cuts a rule ships are scores."""
        fit = self._evt()
        cuts, _reasons = tail_cuts(fit)
        assert all(0.0 < c < 1.0 for c in cuts.values()), cuts


class TestRuleWiring:
    """A rule that exists but is never emitted produces a missing row, not an error.

    #2884 made an *unclassified* variant loud in the analyzer.  This is the other
    half, one layer down: a rule defined in :mod:`vtscore.eval.cut_rules` but not
    wired into the harness never reaches the analyzer at all, so there is nothing
    for it to classify.  Both lists are spelled out by hand in
    ``voting_iterations`` (which is deliberately import-light), so these
    assertions are what keeps the duplication honest.
    """

    def test_every_rule_is_emitted_as_a_pooled_variant(self):
        from vtscore.eval.voting_iterations import _SAFE_GMM_VARIANTS

        names = {name for name, _f, _r in _SAFE_GMM_VARIANTS}
        missing = sorted(f"pooled_{r}" for r in ALL_RULES if f"pooled_{r}" not in names)
        assert not missing, f"defined in cut_rules but never emitted by the harness: {missing}"

    def test_every_rule_has_a_diagnostic_column(self):
        from vtscore.eval.voting_iterations import _CUT_DIAGNOSTIC_COLUMNS

        missing = sorted(f"tau_{r}" for r in ALL_RULES if f"tau_{r}" not in _CUT_DIAGNOSTIC_COLUMNS)
        assert not missing, f"cut emitted with no column to write it to: {missing}"

    def test_pooled_variant_rules_all_exist(self):
        """And the converse: a variant naming a rule that was renamed or removed."""
        from vtscore.eval.voting_iterations import _SAFE_GMM_VARIANTS

        unknown = sorted({rule for _n, _f, rule in _SAFE_GMM_VARIANTS if rule and rule not in ALL_RULES})
        assert not unknown, f"harness emits variants for rules that do not exist: {unknown}"
