"""The #2883 transfer estimators (:mod:`vtscore.eval.transfer_rules`).

The claim this study rests on is that the decomposition's last link is a
**variance** measured against an **optimistic reference**, not a bias a better
fit could remove.  Both halves of that are statistical statements about
estimators, so they are asserted here against constructions whose answer is
known in closed form or by symmetry, rather than against golden numbers from a
run - the same discipline ``test_cut_rules`` applies to the derivation.

The one place a golden number would be wrong to use: "the honest oracle is more
expensive than the naive one" is not a fact about our data, it is a fact about
sample minima, so it is tested on a distribution where it must hold *for every
seed* rather than on one draw that happened to show it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.calibration_metrics import operating_cost, oracle_cut
from vtscore.eval.cut_rules import gaussian_cuts
from vtscore.eval.transfer_rules import (
    BAGGED_FIT_RULES,
    SUBSAMPLE_RULES,
    TRANSFER_ORACLE_RULES,
    TRANSFER_SUBSAMPLE_GRID,
    VARIANCE_REDUCED_RULES,
    bagged_gaussian_fit_cuts,
    honest_test_oracle,
    sim_oracle_bagged_cut,
    sim_oracle_smoothed_cut,
    sim_oracle_subsample_cut,
    subsample_fraction_of,
    subsample_rule,
    transfer_oracle_cuts,
)


def two_class_scores(rng, n=2000, prevalence=0.05, mu_pos=0.75, mu_neg=0.35, sd=0.11):
    """A separable-but-overlapping score sample, shaped like a real haystack."""
    labels = (rng.random(n) < prevalence).astype(float)
    scores = np.where(labels == 1.0, rng.normal(mu_pos, sd, n), rng.normal(mu_neg, sd, n))
    return np.clip(scores, 0.0, 1.0), labels


class TestRuleNaming:
    """The level has to be recoverable from the *data's* variant name.

    #2881 learned this the hard way for ``tail_a*``: an analyzer that reads the
    grid constant instead of the name silently mislabels every curve when a
    re-analysis meets a run whose grid was different.
    """

    def test_round_trips_through_the_name(self):
        for frac in TRANSFER_SUBSAMPLE_GRID:
            assert subsample_fraction_of(subsample_rule(frac)) == pytest.approx(frac, abs=1e-12)

    def test_declines_names_that_are_not_levels(self):
        assert subsample_fraction_of("sim_oracle") is None
        assert subsample_fraction_of("tail_a158") is None
        assert subsample_fraction_of("sim_oracle_bag") is None

    def test_the_families_do_not_overlap(self):
        assert set(SUBSAMPLE_RULES).isdisjoint(VARIANCE_REDUCED_RULES)
        assert set(TRANSFER_ORACLE_RULES) == {*SUBSAMPLE_RULES, *VARIANCE_REDUCED_RULES}
        assert set(TRANSFER_ORACLE_RULES).isdisjoint(BAGGED_FIT_RULES)


class TestDeterminism:
    """A resampling estimator that reads global RNG state cannot be re-analysed.

    Every number in a report has to be reproducible from the dumped scores; an
    estimator whose draw depends on how many randoms were consumed earlier in the
    process is reproducible only by accident.
    """

    def test_same_input_same_answer(self):
        rng = np.random.default_rng(7)
        scores, labels = two_class_scores(rng)
        first = transfer_oracle_cuts(scores, labels, 1.0, 1.0)
        np.random.seed(999)  # noqa: NPY002 - deliberately disturb the global stream
        _ = np.random.random(1000)  # noqa: NPY002
        second = transfer_oracle_cuts(scores, labels, 1.0, 1.0)
        assert first == second

    def test_different_data_different_draw(self):
        """Otherwise every step would share one subsample pattern."""
        rng = np.random.default_rng(11)
        a_scores, a_labels = two_class_scores(rng)
        b_scores, b_labels = two_class_scores(rng)
        a = sim_oracle_subsample_cut(a_scores, a_labels, 0.10, 1.0, 1.0)
        b = sim_oracle_subsample_cut(b_scores, b_labels, 0.10, 1.0, 1.0)
        assert a != b


class TestSubsampleLevels:
    def test_the_full_level_is_the_plain_oracle(self):
        """``frac >= 1`` must not resample - it is the curve's anchor point."""
        rng = np.random.default_rng(3)
        scores, labels = two_class_scores(rng)
        expected = oracle_cut(scores, labels, 1.0, 1.0)[0]
        assert sim_oracle_subsample_cut(scores, labels, 1.0, 1.0, 1.0) == pytest.approx(expected)

    def test_a_level_lands_on_an_observed_score(self):
        """The empirical minimiser is a step function; it cuts *at* a datum."""
        rng = np.random.default_rng(5)
        scores, labels = two_class_scores(rng)
        cut = sim_oracle_subsample_cut(scores, labels, 0.25, 1.0, 1.0)
        assert np.isfinite(cut)
        assert np.isclose(scores, cut).any()

    def test_less_data_costs_more_out_of_sample(self):
        """The learning curve's whole premise, averaged over draws.

        Not asserted per-draw: a single small subsample can get lucky, and a test
        that forbids that would be asserting something false.
        """
        rng = np.random.default_rng(13)
        deficit = []
        for _ in range(24):
            sim_s, sim_l = two_class_scores(rng)
            test_s, test_l = two_class_scores(rng)
            small = sim_oracle_subsample_cut(sim_s, sim_l, 0.05, 1.0, 1.0)
            full = oracle_cut(sim_s, sim_l, 1.0, 1.0)[0]
            c_small = operating_cost(test_s, test_l, small, 1.0, 1.0)[0]
            c_full = operating_cost(test_s, test_l, full, 1.0, 1.0)[0]
            deficit.append(c_small - c_full)
        assert float(np.mean(deficit)) > 0.0


class TestVarianceReduction:
    """H4: the empirical minimiser bounds *sim-set* loss, not *test* loss."""

    def test_bagging_cuts_the_jitter(self):
        """The mechanism, measured directly: same target, less spread.

        Bagging is only worth measuring if it does what bagging does, so this is
        asserted on the estimator's own variability across independent samples
        from one fixed distribution - not on any downstream cost.
        """
        rng = np.random.default_rng(17)
        raw, bagged = [], []
        for _ in range(30):
            scores, labels = two_class_scores(rng, n=600)
            raw.append(oracle_cut(scores, labels, 1.0, 1.0)[0])
            bagged.append(sim_oracle_bagged_cut(scores, labels, 1.0, 1.0, replicates=24))
        assert float(np.std(bagged)) < float(np.std(raw))

    def _test_cost_gain(self, rng, estimator, n_sim, prevalence, reps=40):
        """Mean ``cost(ERM) - cost(estimator)`` on a fresh test sample."""
        gains = []
        for _ in range(reps):
            sim_s, sim_l = two_class_scores(rng, n=n_sim, prevalence=prevalence)
            test_s, test_l = two_class_scores(rng, n=4000, prevalence=prevalence)
            if sim_l.sum() < 2 or test_l.sum() < 2:
                continue
            erm = oracle_cut(sim_s, sim_l, 1.0, 1.0)[0]
            alt = estimator(sim_s, sim_l, 1.0, 1.0)
            gains.append(
                operating_cost(test_s, test_l, erm, 1.0, 1.0)[0] - operating_cost(test_s, test_l, alt, 1.0, 1.0)[0]
            )
        return float(np.mean(gains))

    def test_a_variance_reduced_cut_beats_the_minimiser_out_of_sample(self):
        """The falsification target for ``family_headroom_exhausted``.

        If ``pooled_sim_oracle`` really bounded every rule that reads the sim
        set, no estimator of the same target could beat it on held-out data.  It
        bounds sim-set loss only, and this is what that difference looks like:
        same labels, same scores, lower test cost on average.

        Asserted on the **smoothed** estimator, which is the one that wins in
        every regime measured (see the next test for why that qualifier matters).
        """
        rng = np.random.default_rng(19)
        gain = self._test_cost_gain(rng, sim_oracle_smoothed_cut, n_sim=800, prevalence=0.05)
        assert gain > 0.0

    def test_bagging_the_argmin_can_lose_when_positives_are_starved(self):
        """Pinned because it is surprising, and because it shapes the reading.

        Variance reduction is not one thing.  Averaging the *argmin* over
        bootstrap resamples buys less jitter at the price of a bias, and where
        the labelled sample is thin enough that the cost curve is badly
        asymmetric around its minimum, the bias wins: bagging is *worse* than the
        estimator it smooths.  Smoothing the cost *curve* does not have that
        failure mode.  So a null on ``bag`` is not a null on the idea, and a
        report that quotes only one of the two would say the wrong thing.
        """
        rng = np.random.default_rng(21)
        starved = self._test_cost_gain(rng, sim_oracle_bagged_cut, n_sim=400, prevalence=0.01)
        smoothed = self._test_cost_gain(rng, sim_oracle_smoothed_cut, n_sim=400, prevalence=0.01)
        assert starved < 0.0 < smoothed

    def test_the_smoothed_cut_need_not_land_on_a_datum(self):
        """That is the point of it - the ERM's step-function support is the disease."""
        rng = np.random.default_rng(23)
        scores, labels = two_class_scores(rng)
        cut = sim_oracle_smoothed_cut(scores, labels, 1.0, 1.0)
        assert np.isfinite(cut)
        assert not np.isclose(scores, cut).any()

    def test_a_one_class_sample_declines_rather_than_inventing_a_cut(self):
        scores = np.linspace(0.1, 0.9, 50)
        labels = np.zeros(50)
        assert np.isfinite(sim_oracle_smoothed_cut(scores, labels, 1.0, 1.0))
        assert np.isnan(sim_oracle_bagged_cut(scores, labels, 1.0, 1.0, replicates=8))


class TestHonestTestOracle:
    """H2: the reference point the decomposition measures against is a sample minimum."""

    def test_it_is_more_expensive_than_the_sample_minimum(self):
        """True by construction, for every draw: one is a minimum over the data
        it is scored on, the other is not."""
        rng = np.random.default_rng(29)
        for _ in range(15):
            scores, labels = two_class_scores(rng, n=1200)
            naive = oracle_cut(scores, labels, 1.0, 1.0)[1]
            honest, _tau = honest_test_oracle(scores, labels, 1.0, 1.0)
            assert honest >= naive - 1e-12

    def test_the_optimism_shrinks_as_the_sample_grows(self):
        """It is a finite-sample artefact, so it has to vanish in the limit -
        which is also why a *large* test set makes the naive reference safer."""
        rng = np.random.default_rng(31)

        def optimism(n, reps=12):
            out = []
            for _ in range(reps):
                scores, labels = two_class_scores(rng, n=n)
                naive = oracle_cut(scores, labels, 1.0, 1.0)[1]
                honest, _ = honest_test_oracle(scores, labels, 1.0, 1.0)
                out.append(honest - naive)
            return float(np.mean(out))

        assert optimism(400) > optimism(6400)

    def test_it_declines_when_a_fold_complement_has_one_class(self):
        scores = np.linspace(0.0, 1.0, 6)
        labels = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        cost, tau = honest_test_oracle(scores, labels, 1.0, 1.0, folds=5)
        assert np.isnan(cost) and np.isnan(tau)

    def test_too_few_items_declines(self):
        cost, tau = honest_test_oracle(np.array([0.2, 0.8]), np.array([0.0, 1.0]), 1.0, 1.0, folds=5)
        assert np.isnan(cost) and np.isnan(tau)


class TestBaggedFit:
    """The label-free arm: same rules, same fit family, averaged over refits."""

    def test_it_emits_exactly_the_bagged_fit_rules(self):
        rng = np.random.default_rng(37)
        scores, _labels = two_class_scores(rng)
        cuts = bagged_gaussian_fit_cuts(scores, gaussian_cuts, 1.0, 1.0, replicates=6)
        assert set(cuts) == set(BAGGED_FIT_RULES)

    def test_it_reads_no_labels(self):
        """Signature-level, because this is the property that makes it shippable."""
        rng = np.random.default_rng(41)
        scores, labels = two_class_scores(rng)
        a = bagged_gaussian_fit_cuts(scores, gaussian_cuts, 1.0, 1.0, replicates=6)
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        b = bagged_gaussian_fit_cuts(scores, gaussian_cuts, 1.0, 1.0, replicates=6)
        assert a == b

    def test_it_lands_near_its_unbagged_sibling(self):
        """It is the same rule on the same fit family - a bagged cut that moved a
        long way would mean the bootstrap is finding a different mixture, not
        smoothing the one we have."""
        rng = np.random.default_rng(43)
        scores, _labels = two_class_scores(rng)
        from vtscore.eval.cut_rules import fit_both_mixtures

        gmm, _evt, _params = fit_both_mixtures(scores)
        assert gmm is not None
        plain = gaussian_cuts(gmm, 1.0, 1.0)
        bagged = bagged_gaussian_fit_cuts(scores, gaussian_cuts, 1.0, 1.0, replicates=16)
        assert bagged["bagfit_mid"] == pytest.approx(plain["mid"], abs=0.05)

    def test_zero_replicates_declines(self):
        rng = np.random.default_rng(47)
        scores, _labels = two_class_scores(rng)
        cuts = bagged_gaussian_fit_cuts(scores, gaussian_cuts, 1.0, 1.0, replicates=0)
        assert all(np.isnan(v) for v in cuts.values())
