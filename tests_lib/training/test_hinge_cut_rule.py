"""The #3557 sign-dependent ("hinge") cut rules keep - or, for the literal one, break - nesting.

#2865 measured ``cross_tilt`` beating the shipped ``mid_tilt`` below inclusion 0
and losing above it, and the obvious rule that follows is a hinge: ``cross_tilt``
for ``k < 0``, ``mid_tilt`` for ``k >= 0``.  The Inclusion knob's contract is that
its admitted sets are **nested** - everything included at ``k`` stays included at
``k + 1`` - and a hinge is exactly the shape that can break it at the seam,
because nothing orders ``cross_tilt``'s inclusion-0 quantile against the
midpoint's.

These tests pin both halves of that:

* the guarded ``hinge`` (``max(q_cross, q_mid_tilt)`` below zero) and the
  continuous ``hinge_cont`` are non-increasing over the whole knob, across the
  seam, on every fit a property sweep throws at them - the proof in
  :meth:`FoldAnchoredCut._quantile_at`, checked;
* the literal ``hinge_raw`` **does** break nesting on a plausible fit (5%
  prevalence, Good component 25x wider than Bad), so the guard is load-bearing
  and not decoration.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.training.thresholds import (
    FOLD_ANCHOR_HINGE_RULES,
    FOLD_LEVEL_CUT_RULES,
    FoldAnchoredCut,
    GmmFit1D,
    gmm_cut_from_fit,
)

#: A dense knob grid with the seam approached from both sides.  The integer
#: stops alone cannot see a violation that lives in ``(-1, 0)``, and the
#: acquisition cut is taken at ``k + offset`` for fractional offsets too.
_KS = sorted({*np.round(np.arange(-13.0, 13.0001, 0.05), 6).tolist(), -1e-9, 0.0, 1e-9})

#: The counterexample: rare Good (w_hi = 0.05) spread 25x wider than Bad.  The
#: prior-keeping crossing sits far *below* the midpoint here, because the wide
#: Good density overtakes the narrow Bad one a few Bad-sigmas above its mean.
_WIDE_GOOD = GmmFit1D(w_lo=0.95, mu_lo=0.1, var_lo=0.002, w_hi=0.05, mu_hi=0.7, var_hi=0.05)


def _sample(fit: GmmFit1D, n: int, rng: np.random.Generator) -> np.ndarray:
    good = rng.random(n) < fit.w_hi
    x = np.where(
        good,
        rng.normal(fit.mu_hi, math.sqrt(fit.var_hi), n),
        rng.normal(fit.mu_lo, math.sqrt(fit.var_lo), n),
    )
    return np.sort(x)


def _cut(fits: tuple[GmmFit1D, ...], rule: str, seed: int = 0, n: int = 4000) -> FoldAnchoredCut:
    rng = np.random.default_rng(seed)
    hays = tuple(_sample(f, n, rng) for f in fits)
    final = np.sort(np.concatenate(hays))
    return FoldAnchoredCut(fits=fits, fold_haystacks=hays, final_haystack=final, n_anchored=len(fits), cut_rule=rule)


def _random_fit(rng: np.random.Generator) -> GmmFit1D:
    """A fit anywhere in the space real folds visit, and past it on both sides.

    Prevalence from 0.5% to 70% (so the prior odds change sign), variance ratio
    from 1/30 to 30 (so either component can be the wide one), and separations
    from barely-apart to cleanly split.
    """
    w_hi = float(rng.uniform(0.005, 0.7))
    mu_lo = float(rng.uniform(-0.5, 0.5))
    gap = float(rng.uniform(0.02, 1.5))
    var_lo = float(10 ** rng.uniform(-4, -1))
    var_hi = var_lo * float(10 ** rng.uniform(-1.5, 1.5))
    return GmmFit1D(w_lo=1.0 - w_hi, mu_lo=mu_lo, var_lo=var_lo, w_hi=w_hi, mu_hi=mu_lo + gap, var_hi=var_hi)


def _non_increasing(values: list[float]) -> bool:
    return all(b <= a for a, b in zip(values, values[1:], strict=False))


class TestTheGuardedHingeIsNested:
    @pytest.mark.parametrize("rule", ["hinge", "hinge_cont"])
    @pytest.mark.parametrize("seed", range(40))
    def test_quantile_and_threshold_are_non_increasing_across_the_whole_knob(self, rule, seed):
        """The proof in ``_quantile_at``, checked on 40 random two-fold estimators.

        Both the combined quantile (what the rule did) and the realized
        threshold (what the user gets) must be non-increasing everywhere,
        including across ``k = 0`` from either side.
        """
        rng = np.random.default_rng(1000 + seed)
        cut = _cut((_random_fit(rng), _random_fit(rng)), rule, seed=seed, n=1500)
        qs = [cut.quantile_at(k) for k in _KS]
        thrs = [cut.threshold_at(k) for k in _KS]
        assert _non_increasing(qs), f"{rule} quantile rises somewhere on seed {seed}"
        assert _non_increasing(thrs), f"{rule} threshold rises somewhere on seed {seed}"

    @pytest.mark.parametrize("rule", FOLD_ANCHOR_HINGE_RULES)
    def test_every_hinge_is_mid_tilt_bit_for_bit_at_and_above_zero(self, rule):
        rng = np.random.default_rng(7)
        fits = (_random_fit(rng), _random_fit(rng))
        hinge, incumbent = _cut(fits, rule), _cut(fits, "mid_tilt")
        for k in [0, 1e-9, 0.5, 1, 3, 10]:
            assert hinge.threshold_at(k) == incumbent.threshold_at(k)
            assert hinge.quantile_at(k) == incumbent.quantile_at(k)

    @pytest.mark.parametrize("seed", range(20))
    def test_the_guarded_hinge_is_never_more_inclusive_than_the_incumbent_below_zero(self, seed):
        """The guard makes ``hinge`` a one-sided move: below zero - where the
        knob is asking for fewer false alarms - it can only admit *less* than
        ``mid_tilt``, never more."""
        rng = np.random.default_rng(2000 + seed)
        fits = (_random_fit(rng), _random_fit(rng))
        hinge, incumbent = _cut(fits, "hinge", seed=seed, n=1500), _cut(fits, "mid_tilt", seed=seed, n=1500)
        for k in [-10, -4, -1, -0.25, -1e-9]:
            assert hinge.quantile_at(k) >= incumbent.quantile_at(k)
            assert hinge.threshold_at(k) >= incumbent.threshold_at(k)

    def test_hinge_cont_is_continuous_at_the_seam(self):
        cut = _cut((_WIDE_GOOD,), "hinge_cont")
        assert cut.quantile_at(-1e-9) == pytest.approx(cut.quantile_at(0.0), abs=1e-6)

    def test_where_cross_is_the_stricter_rule_the_guard_is_inert(self):
        """On the common shape - Bad dominant, similar widths - ``cross_tilt`` is
        already stricter than ``mid_tilt`` below zero, so the guarded and literal
        hinges coincide there and the guard costs nothing."""
        fit = GmmFit1D(w_lo=0.9, mu_lo=0.2, var_lo=0.01, w_hi=0.1, mu_hi=0.6, var_hi=0.012)
        guarded, raw = _cut((fit,), "hinge"), _cut((fit,), "hinge_raw")
        diag = guarded.seam_diagnostics()
        assert diag["seam_q_cross0"] > diag["seam_q_mid"]
        for k in [-6, -3, -1, -0.25]:
            assert guarded.threshold_at(k) == raw.threshold_at(k)


class TestTheLiteralHingeBreaksNesting:
    def test_the_counterexample_fit_puts_the_cross_cut_below_the_midpoint(self):
        cross0, _kind = gmm_cut_from_fit(_WIDE_GOOD, "cross_tilt", 1.0, 1.0)
        assert cross0 < _WIDE_GOOD.midpoint()
        diag = _cut((_WIDE_GOOD,), "hinge_raw").seam_diagnostics()
        assert diag["seam_q_cross0"] < diag["seam_q_mid"]
        # Rare Good (positive prior odds, in bits) and a Good component far wider.
        assert diag["fit_log2_prior_odds"] > 4
        assert diag["fit_log2_var_ratio"] > 4

    def test_hinge_raw_admits_more_at_k_minus_one_than_at_zero(self):
        """The contract violation, as a user would meet it: dragging the slider
        from 0 to -1 - asking for *fewer* false alarms - lowers the threshold."""
        raw = _cut((_WIDE_GOOD,), "hinge_raw")
        assert raw.threshold_at(-1) < raw.threshold_at(0)
        thrs = [raw.threshold_at(k) for k in _KS]
        assert not _non_increasing(thrs)

    def test_the_guard_repairs_the_same_fit(self):
        guarded = _cut((_WIDE_GOOD,), "hinge")
        assert guarded.threshold_at(-1) >= guarded.threshold_at(0)
        assert _non_increasing([guarded.threshold_at(k) for k in _KS])


class TestSeamDiagnostics:
    def test_the_seam_columns_are_the_quantities_the_rules_compose(self):
        rng = np.random.default_rng(3)
        fits = (_random_fit(rng), _random_fit(rng))
        cut = _cut(fits, "mid_tilt")
        diag = cut.seam_diagnostics()
        assert diag["seam_q_mid"] == cut.quantile_at(0)
        assert diag["seam_q_rate0"] == _cut(fits, "rate").quantile_at(0)
        assert diag["seam_q_cross0"] == _cut(fits, "cross_tilt").quantile_at(0)
        expected_odds = np.mean([math.log2(f.w_lo / f.w_hi) for f in fits])
        assert diag["fit_log2_prior_odds"] == pytest.approx(expected_odds)

    def test_a_degenerate_fold_reports_nan_rather_than_a_clipped_number(self):
        fit = GmmFit1D(w_lo=1.0, mu_lo=0.1, var_lo=0.01, w_hi=0.0, mu_hi=0.5, var_hi=0.01)
        diag = _cut((_WIDE_GOOD,), "mid_tilt")
        degenerate = FoldAnchoredCut(
            fits=(fit,),
            fold_haystacks=diag.fold_haystacks,
            final_haystack=diag.final_haystack,
            n_anchored=0,
        ).seam_diagnostics()
        assert math.isnan(degenerate["fit_log2_prior_odds"])

    def test_fold_level_rules_are_all_rejected_by_the_per_fit_cut(self):
        for rule in FOLD_LEVEL_CUT_RULES:
            with pytest.raises(ValueError):
                gmm_cut_from_fit(_WIDE_GOOD, rule)
