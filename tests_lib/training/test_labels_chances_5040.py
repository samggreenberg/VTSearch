"""The labels line's chances with the Bad component fitted on the corpus, and the counted bound loosened (#5040).

#4937 found the line's chances (the 3-part fit) underrate the positives at the line, partly because the labels' Bad
component - a normal at the held-out Bads' mean with the Goods' spread, fixed - puts too much negative weight there.
Since #5040 the app fits that component's mean on the corpus being decided, the held-out Bads counted as labelled
points of it, and holds the 2-part estimate to 4x the counted bound; ``labels`` and 2 are the line before.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.eval.live_threshold_rules import labels_chances
from vtscore.training.thresholds import class_score_model, fit_labels_line
from vtscore.training.thresholds import corpus_model
from vtscore.training.thresholds import labels_line as LL
from vtscore.training.thresholds.class_model import _finite_unit, _logit, _norm_pdf
from vtscore.training.thresholds.labels_line import (
    PREVALENCE_MAX,
    PREVALENCE_MIN,
    corpus_posteriors,
)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _session():
    """A corpus whose negatives have a shoulder near the line, and labels picked the way a session picks them.

    The Goods are the corpus's 15 easiest positives; the Bads were picked near the line, above the shoulder.
    """
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 0.6, 50)
    corpus = np.r_[rng.normal(-3.0, 1.0, 10_000), rng.normal(-0.6, 0.5, 300), pos]
    truth = np.r_[np.zeros(10_300), np.ones(50)]
    goods = np.sort(pos)[-15:] + rng.normal(0.0, 0.05, 15)
    bads = rng.normal(0.0, 0.35, 30)
    orderings = [(_sig(np.r_[goods, bads]), np.r_[np.ones(15), np.zeros(30)])]
    return orderings, _sig(corpus), truth


def _shipped_posteriors(model, unvoted_scores, *, iterations: int = 300, floor: float = LL.MIN_LOGIT_SIGMA):
    """``corpus_posteriors`` as it was before #5040, line for line: the reference the app's default must match."""
    x = _logit(_finite_unit(unvoted_scores))
    mu0 = float(np.median(x))
    s0 = max(1.4826 * float(np.median(np.abs(x - mu0))), floor)
    w1, wb = 0.01, 0.05
    r1 = np.zeros_like(x)
    for _ in range(iterations):
        a1 = w1 * _norm_pdf(x, model.mu_pos, model.sigma)
        ab = wb * _norm_pdf(x, model.mu_neg, model.sigma)
        a0 = max(1.0 - w1 - wb, 1e-9) * _norm_pdf(x, mu0, s0)
        tot = a1 + ab + a0
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = np.nan_to_num(a1 / tot, nan=0.0)
            rb = np.nan_to_num(ab / tot, nan=0.0)
        r0 = np.clip(1.0 - r1 - rb, 0.0, 1.0)
        nw1 = min(max(float(r1.mean()), PREVALENCE_MIN), PREVALENCE_MAX)
        nwb = min(max(float(rb.mean()), PREVALENCE_MIN), 0.9)
        s_r0 = float(r0.sum())
        if s_r0 > 1e-12:
            mu0 = min(float((r0 * x).sum()) / s_r0, model.mu_neg)
            s0 = max(math.sqrt(max(float((r0 * (x - mu0) ** 2).sum()) / s_r0, 0.0)), floor)
        done = abs(nw1 - w1) < 1e-8 and abs(nwb - wb) < 1e-7
        w1, wb = nw1, nwb
        if done:
            break
    return r1


def _fbeta(scores, truth, thr: float, beta: float) -> float:
    kept = scores >= thr
    tp = float((kept & (truth == 1)).sum())
    b2 = beta * beta
    return (1 + b2) * tp / (b2 * float(truth.sum()) + float(kept.sum()))


class TestTheClassModel:
    def test_it_records_the_bads_own_spread(self):
        orderings, _, _ = _session()
        model = class_score_model(orderings)
        assert model is not None and model.neg_sd is not None
        bads = _logit(orderings[0][0][15:])
        assert model.neg_sd == pytest.approx(float(np.std(bads, ddof=1)))

    def test_one_bad_has_no_spread_of_its_own(self):
        model = class_score_model([(_sig([1.0, 2.0, -1.0]), [1.0, 1.0, 0.0])])
        assert model is not None and model.neg_sd is None


class TestTheAppsChances:
    def test_the_app_fits_the_bad_components_mean_and_holds_the_bound_at_4x(self):
        assert (corpus_model.BAD_COMPONENT, LL._BOUND_MARGIN) == ("corpus_mean", 4.0)

    def test_the_labels_arm_is_the_fit_before_5040_bit_for_bit(self):
        orderings, corpus, _ = _session()
        line = fit_labels_line(orderings, corpus, None, {})
        assert line is not None and line.spread is not None
        fitted = line.model.floored(line.spread)
        for floor in (LL.MIN_LOGIT_SIGMA, line.spread):
            np.testing.assert_array_equal(
                corpus_posteriors(fitted, corpus, floor=floor, bad="labels"),
                _shipped_posteriors(fitted, corpus, floor=floor),
            )
            np.testing.assert_array_equal(
                corpus_posteriors(fitted, corpus, floor=floor, bad="corpus_mean"),
                corpus_posteriors(fitted, corpus, floor=floor),
            )

    def test_an_unknown_bad_component_is_refused(self):
        orderings, corpus, _ = _session()
        model = class_score_model(orderings)
        assert model is not None
        with pytest.raises(ValueError, match="bad must be one of"):
            corpus_posteriors(model, corpus, bad="kde")


class TestTheCorpusFittedBadComponent:
    def test_it_gives_the_positives_near_the_line_more_of_their_chance(self):
        orderings, corpus, truth = _session()
        line = fit_labels_line(orderings, corpus, None, {})
        assert line is not None and line.spread is not None
        fitted = line.model.floored(line.spread)
        labels = corpus_posteriors(fitted, corpus, floor=line.spread, bad="labels")
        fitted_bad = corpus_posteriors(fitted, corpus, floor=line.spread, bad="corpus")
        assert fitted_bad[truth == 1].sum() > labels[truth == 1].sum()
        assert fitted_bad.sum() < truth.sum(), "still no more chance than there are positives"

    def test_the_mean_alone_keeps_the_models_spread(self):
        orderings, corpus, truth = _session()
        line = fit_labels_line(orderings, corpus, None, {})
        assert line is not None and line.spread is not None
        fitted = line.model.floored(line.spread)
        mean = corpus_posteriors(fitted, corpus, floor=line.spread, bad="corpus_mean")
        labels = corpus_posteriors(fitted, corpus, floor=line.spread, bad="labels")
        assert mean[truth == 1].sum() > labels[truth == 1].sum()
        assert not np.array_equal(mean, corpus_posteriors(fitted, corpus, floor=line.spread, bad="corpus"))

    @pytest.mark.parametrize("beta", [0.25, 1.0, 4.0])
    def test_the_line_keeps_more_positives_at_every_preset(self, beta):
        orderings, corpus, truth = _session()
        with labels_chances("labels", 2.0):
            shipped = fit_labels_line(orderings, corpus, None, {})
        with labels_chances("corpus", 2.0):
            arm = fit_labels_line(orderings, corpus, None, {})
        assert shipped is not None and arm is not None
        assert _fbeta(corpus, truth, arm.threshold(beta), beta) >= _fbeta(corpus, truth, shipped.threshold(beta), beta)

    @pytest.mark.parametrize("beta", [0.25, 1.0, 4.0])
    def test_find_on_a_corpus_with_no_positives_still_keeps_next_to_nothing(self, beta):
        """#4452's requirement, at the margin the arm loosens the bound to as well."""
        orderings, _, _ = _session()
        empty = _sig(np.random.default_rng(3).normal(-3.0, 1.0, 2000))
        for bad, margin in (("corpus", 2.0), ("corpus", 4.0), (None, None)):
            with labels_chances(bad, margin):
                line = fit_labels_line(orderings, _session()[1], None, {})
                assert line is not None
                find = line.on_corpus(empty)
                assert int((empty >= find.threshold(beta)).sum()) <= 2


class TestTheArmKnob:
    def test_it_swaps_both_pieces_and_puts_them_back(self):
        app = ("corpus_mean", 4.0)
        assert (corpus_model.BAD_COMPONENT, LL._BOUND_MARGIN) == app
        with labels_chances("labels", 2.0):
            assert (corpus_model.BAD_COMPONENT, LL._BOUND_MARGIN) == ("labels", 2.0)
        assert (corpus_model.BAD_COMPONENT, LL._BOUND_MARGIN) == app
        with labels_chances():
            assert (corpus_model.BAD_COMPONENT, LL._BOUND_MARGIN) == app

    def test_it_puts_them_back_after_an_error(self):
        with pytest.raises(RuntimeError), labels_chances("corpus", math.inf):
            raise RuntimeError
        assert (corpus_model.BAD_COMPONENT, LL._BOUND_MARGIN) == ("corpus_mean", 4.0)

    def test_a_looser_bound_estimates_at_least_as_many_positives(self):
        orderings, corpus, _ = _session()
        shares = []
        for margin in (2.0, 4.0, math.inf):
            with labels_chances(bound_margin=margin):
                line = fit_labels_line(orderings, corpus, None, {})
                assert line is not None and line.unvoted_share is not None
                shares.append(line.unvoted_share)
        assert shares[0] <= shares[1] <= shares[2]

    @pytest.mark.parametrize(
        ("bad", "margin"),
        [("kde", None), (None, 0.5), (None, True), (None, "4")],
        ids=["bad", "margin<1", "bool", "str"],
    )
    def test_a_value_it_cannot_run_is_refused(self, bad, margin):
        with pytest.raises(ValueError, match="labels_chances"), labels_chances(bad, margin):
            pass
