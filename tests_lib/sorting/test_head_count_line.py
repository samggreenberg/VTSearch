"""The head's count line (#4786): while the class model is thin, the line counts what stands out.

At the hand-over the labels line's cut on a class model fitted to a handful of
held-out votes went wrong both ways (one face, or thousands).  Until the model holds
``HEAD_COUNT_GOODS`` Goods, however many Bads (#4798), the line at every balance keeps
instead what the Goods' centroid's count rule finds standing out in the head's own
scores.  ``HEAD_COUNT_BADS`` is the knob's Bads half, unbounded in the app.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.detectors.centroid_head import centroid_cut
from vtscore.eval.live_threshold_rules import head_count_bads, head_count_precision, head_count_recall
from vtscore.training.thresholds import (
    HEAD_COUNT_BADS,
    HEAD_COUNT_GOODS,
    corpus_cut,
    fit_labels_line,
    head_count_cut,
    head_count_serves,
)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _orderings(n_pos: int, n_neg: int, seed: int = 0):
    """One fold's held-out scores: Goods around logit 2, Bads around logit -1, spread wide (a thin model)."""
    rng = np.random.default_rng(seed)
    return [
        (list(_sig(rng.normal(2, 1.5, n_pos))) + list(_sig(rng.normal(-1, 1.5, n_neg))), [1.0] * n_pos + [0.0] * n_neg)
    ]


def _corpus(seed: int = 1):
    """A person's few photos standing out of a bulk of 5,000 faces."""
    rng = np.random.default_rng(seed)
    return np.r_[_sig(rng.normal(-3, 1.0, 5000)), _sig(rng.normal(3, 0.3, 4))]


def _kept(scores, threshold: float) -> int:
    return int(np.count_nonzero(np.asarray(scores) >= threshold))


class TestTheCut:
    def test_it_keeps_what_the_centroids_count_rule_counts_on_the_logits(self):
        """The count before #5039 at every balance; since #5039 at beta 1 and below (above it, ``TestAboveBetaOne``)."""
        s = np.sort(_corpus())[::-1]
        x = np.log(s) - np.log1p(-s)
        for beta, knobs in ((0.25, (None, None)), (1.0, (None, None)), (4.0, ("off", "off"))):
            with head_count_recall(*knobs):
                thr = head_count_cut(s, beta)
            k = int(np.count_nonzero(x >= centroid_cut(x, beta=beta)))
            assert _kept(s, thr) == k
            # Between the last kept and the first dropped, never on an item.
            assert s[k] < thr < s[k - 1]

    def test_a_recall_lean_keeps_at_least_the_standouts(self):
        s = np.sort(_corpus())[::-1]
        assert _kept(s, head_count_cut(s, 4.0)) >= 4

    def test_an_empty_corpus_keeps_nothing(self):
        assert head_count_cut(np.array([]), 4.0) > 1.0


class TestTheGate:
    def test_few_bads_and_few_goods(self):
        line = fit_labels_line(_orderings(4, 4), _corpus())
        assert line is not None and line.model.n_neg == 4
        assert head_count_serves(line.model)

    def test_many_bads_do_not_close_it(self):
        """#4798: with few Goods the class model cannot place the positives, however many Bads it holds."""
        assert HEAD_COUNT_BADS == math.inf
        line = fit_labels_line(_orderings(HEAD_COUNT_GOODS - 1, 400), _corpus())
        assert line is not None and line.model.n_neg == 400
        assert head_count_serves(line.model)

    def test_a_bads_gate_is_the_knobs(self):
        """``head_count_bads(32)`` is #4786's gate: the class model took over at 32 held-out Bads."""
        with head_count_bads(32):
            line = fit_labels_line(_orderings(4, 32), _corpus())
            assert line is not None and not head_count_serves(line.model)
            line = fit_labels_line(_orderings(4, 31), _corpus())
            assert line is not None and head_count_serves(line.model)

    def test_and_at_the_goods(self):
        line = fit_labels_line(_orderings(HEAD_COUNT_GOODS, 4), _corpus())
        assert line is not None and not head_count_serves(line.model)
        line = fit_labels_line(_orderings(HEAD_COUNT_GOODS - 1, 4), _corpus())
        assert line is not None and head_count_serves(line.model)


class TestTheLine:
    def test_with_a_thin_model_the_line_is_the_heads_count_at_every_balance(self):
        line = fit_labels_line(_orderings(4, 4), _corpus())
        assert line is not None and line.unvoted_scores is not None
        for beta in (0.25, 1.0, 4.0):
            assert line.threshold(beta) == head_count_cut(line.unvoted_scores, beta, line.model.pos_logits)

    def test_a_recall_lean_keeps_at_least_a_precision_lean(self):
        """The count is beta ** 0.708 times what stands out, so the line keeps more as the balance leans to recall."""
        for seed in range(12):
            line = fit_labels_line(_orderings(4, 4, seed=seed), _corpus(seed + 1))
            assert line is not None and line.unvoted_scores is not None
            kept = [_kept(line.unvoted_scores, line.threshold(b)) for b in (0.25, 0.5, 1.0, 1.5, 2.0, 4.0)]
            assert kept == sorted(kept), kept

    def test_a_model_with_the_labels_cuts_on_them(self):
        line = fit_labels_line(_orderings(HEAD_COUNT_GOODS, 40), _corpus())
        assert line is not None and line.unvoted_scores is not None and line.unvoted_posteriors is not None
        total = float(line.unvoted_share or 0.0) * line.unvoted_scores.size
        for beta in (0.25, 1.0, 4.0):
            assert line.threshold(beta) == corpus_cut(line.unvoted_posteriors, line.unvoted_scores, total, beta)

    def test_find_on_another_corpus_draws_it_there(self):
        line = fit_labels_line(_orderings(4, 4), _corpus(1))
        assert line is not None
        other = _corpus(2)
        on = line.on_corpus(other)
        assert on.unvoted_scores is not None
        assert on.threshold(4.0) == head_count_cut(on.unvoted_scores, 4.0, on.model.pos_logits)

    def test_gate_zero_is_the_line_before_4786(self):
        line = fit_labels_line(_orderings(4, 4), _corpus())
        assert line is not None and line.unvoted_scores is not None and line.unvoted_posteriors is not None
        total = float(line.unvoted_share or 0.0) * line.unvoted_scores.size
        with head_count_bads(0):
            assert line.threshold(4.0) == corpus_cut(line.unvoted_posteriors, line.unvoted_scores, total, 4.0)
        assert line.threshold(4.0) == head_count_cut(line.unvoted_scores, 4.0, line.model.pos_logits)


class TestTheKnob:
    def test_none_is_the_apps_and_the_swap_is_undone(self):
        from vtscore.training.thresholds import labels_line

        with head_count_bads(None):
            assert labels_line.HEAD_COUNT_BADS == HEAD_COUNT_BADS
        with head_count_bads(7):
            assert labels_line.HEAD_COUNT_BADS == 7
        assert labels_line.HEAD_COUNT_BADS == HEAD_COUNT_BADS

    @pytest.mark.parametrize("bad", [-1, "many", True, 2.5])
    def test_it_refuses_what_is_not_a_count(self, bad):
        with pytest.raises(ValueError):
            with head_count_bads(bad):
                pass


class TestACorpusWithNoPositives:
    def test_it_keeps_a_handful_whatever_the_corpus_size(self):
        """A Gaussian bulk alone: the widest drop in its top still keeps a few, times beta ** 0.708 (#4786's cost)."""
        for n in (200, 5_000, 20_000):
            kept = []
            for seed in range(20):
                s = np.sort(_sig(np.random.default_rng(seed).normal(-3, 1.0, n)))[::-1]
                kept.append(_kept(s, head_count_cut(s, 4.0)))
            assert 1 <= int(np.median(kept)) <= 12
            assert max(kept) <= 30


def _weak_head(seed: int = 0, n: int = 11_000, n_pos: int = 50):
    """A weak head's corpus, best first, and its truth: 50 positives spread through the bulk's upper tail."""
    rng = np.random.default_rng(seed)
    s = np.r_[_sig(rng.normal(-3, 0.5, n)), _sig(rng.normal(-1.6, 0.5, n_pos))]
    y = np.r_[np.zeros(n), np.ones(n_pos)]
    order = np.argsort(-s, kind="stable")
    return s[order], y[order]


def _logits(s):
    s = np.asarray(s, dtype=np.float64)
    return np.log(s) - np.log1p(-s)


def _f4(y, kept: int) -> float:
    return 17 * float(y[:kept].sum()) / (16 * float(y.sum()) + kept)


#: The arm the closed loop priced: the count at 1.3x at beta 4, the reach from 3 bulk spreads up.
E13 = math.log(1.3) / math.log(4.0)


class TestAboveBetaOne:
    """#5039: above beta 1 the count grows as beta ** HEAD_COUNT_RECALL_EXPONENT, and the line reaches down the
    head's excess curve where a held-out Good ranks below what stands out."""

    def test_off_is_the_count_before_5039(self):
        s, _ = _weak_head()
        x = _logits(s)
        deep = x[300:301]
        with head_count_recall("off", "off"):
            for beta in (0.25, 1.0, 4.0):
                assert _kept(s, head_count_cut(s, beta, deep)) == int(np.count_nonzero(x >= centroid_cut(x, beta=beta)))

    def test_neither_piece_moves_a_precision_or_even_balance(self):
        s, _ = _weak_head()
        deep = _logits(s)[300:301]
        for beta in (0.25, 0.5, 1.0):
            with head_count_recall("off", "off"):
                count = head_count_cut(s, beta, deep)
            with head_count_recall(E13, 3.0):
                assert head_count_cut(s, beta, deep) == count

    def test_the_count_grows_as_the_exponent_says(self):
        """At beta 4 the count is 4 ** e times what stands out: the #4603 rule at the beta whose 0.708 power that is."""
        from vtscore.training.thresholds.gmm import TEXT_SORT_COUNT_EXPONENT

        s = np.sort(_corpus())[::-1]
        x = _logits(s)
        with head_count_recall(E13, "off"):
            k = _kept(s, head_count_cut(s, 4.0))
        b_count = 4.0 ** (E13 / TEXT_SORT_COUNT_EXPONENT)
        assert k == int(np.count_nonzero(x >= centroid_cut(x, beta=b_count)))
        with head_count_recall("off", "off"):
            assert k <= _kept(s, head_count_cut(s, 4.0))

    def test_a_good_below_the_count_reaches_down_a_weak_heads_tail(self):
        """The positives sit in the bulk's tail and a held-out Good ranks below the count: beta 4 keeps deeper."""
        gains = []
        for seed in range(5):
            s, y = _weak_head(seed)
            deep = _logits(s)[np.flatnonzero(y)[25:26]]  # a held-out Good at the positives' median rank
            with head_count_recall(E13, "off"):
                count = _kept(s, head_count_cut(s, 4.0, deep))
            with head_count_recall(E13, 3.0):
                reached = _kept(s, head_count_cut(s, 4.0, deep))
            assert reached > count
            gains.append(_f4(y, reached) - _f4(y, count))
        assert np.mean(gains) > 0.05

    def test_no_reach_without_a_good_below_the_count(self):
        """Every held-out Good among what stands out (a person's faces), or none given: the count stands."""
        s = np.sort(_corpus())[::-1]
        x = _logits(s)
        with head_count_recall(E13, "off"):
            count = head_count_cut(s, 4.0)
        with head_count_recall(E13, 3.0):
            assert head_count_cut(s, 4.0, x[:2]) == count
            assert head_count_cut(s, 4.0, None) == count
            assert head_count_cut(s, 4.0, np.empty(0)) == count

    def test_a_recall_lean_keeps_at_least_a_precision_lean(self):
        with head_count_recall(E13, 3.0):
            for seed in range(8):
                s, y = _weak_head(seed)
                deep = _logits(s)[np.flatnonzero(y)[25:26]]
                kept = [_kept(s, head_count_cut(s, b, deep)) for b in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)]
                assert kept == sorted(kept), kept

    def test_the_reach_is_the_excess_curves_fbeta_peak(self):
        from scipy.special import ndtr

        from vtscore.training.thresholds.labels_line import _excess_depth

        x = _logits(_weak_head()[0])
        med = np.median(x)
        sd = 1.4826 * np.median(np.abs(x - med))
        z = (x - med) / sd
        k = np.arange(1, x.size + 1)
        excess = k - x.size * ndtr(-z)
        total = excess[z >= 3.0].max()
        f = 17 * np.clip(excess, 0, total) / (16 * total + k)
        f[z < 3.0] = -1
        assert _excess_depth(x, 4.0, 3.0) == int(np.argmax(f)) + 1

    def test_too_few_scores_or_no_spread_reach_no_deeper(self):
        from vtscore.training.thresholds.labels_line import _excess_depth

        assert _excess_depth(np.linspace(3, -3, 40), 4.0, 3.0) == 0
        assert _excess_depth(np.zeros(500), 4.0, 3.0) == 0

    def test_a_corpus_with_no_positives_keeps_few_without_a_deep_good(self):
        """A Gaussian bulk alone and no held-out Good below what stands out: the count at 1.3x keeps fewer than today's."""
        for n in (200, 5_000, 20_000):
            for seed in range(10):
                s = np.sort(_sig(np.random.default_rng(seed).normal(-3, 1.0, n)))[::-1]
                with head_count_recall("off", "off"):
                    today = _kept(s, head_count_cut(s, 4.0, _logits(s)[:1]))
                with head_count_recall(E13, 3.0):
                    assert _kept(s, head_count_cut(s, 4.0, _logits(s)[:1])) <= today


class TestTheLineReadsTheHeldOutGoods:
    def test_the_folds_model_carries_its_goods_and_the_in_sample_stand_in_does_not(self):
        line = fit_labels_line(_orderings(4, 4), _corpus())
        assert line is not None and line.model.pos_logits is not None
        assert line.model.pos_logits.size == 4 and np.all(np.diff(line.model.pos_logits) >= 0)
        # Folds that rank the Goods no higher than the Bads: the in-sample stand-in draws the line.
        flipped = [(list(_sig([-2.0, -2.5, 1.0, 1.5])), [1.0, 1.0, 0.0, 0.0])]
        c = _corpus()
        ids = list(range(c.size))
        stand_in = fit_labels_line(flipped, c, ids, {ids[-1]: True, ids[-2]: True, 0: False, 1: False})
        assert stand_in is not None and stand_in.model.pos_logits is None

    def test_the_thin_line_cuts_with_them(self):
        line = fit_labels_line(_orderings(4, 4), _corpus())
        assert line is not None and line.unvoted_scores is not None
        with head_count_recall(E13, 3.0):
            for beta in (0.25, 1.0, 4.0):
                assert line.threshold(beta) == head_count_cut(line.unvoted_scores, beta, line.model.pos_logits)


class TestTheShippedLine:
    def test_the_app_counts_1_3x_at_beta_4_and_reaches_from_2_75(self):
        """#5039's closed loop shipped the count at 1.3x what stands out at beta 4 and the reach from 2.75 spreads."""
        from vtscore.training.thresholds import labels_line

        assert labels_line.HEAD_COUNT_RECALL_EXPONENT == pytest.approx(math.log(1.3) / math.log(4.0), abs=1e-5)
        assert labels_line.HEAD_COUNT_RECALL_Z == 2.75
        # A person's faces: fewer than before #5039 at beta 4, the same at beta 1.
        faces = np.sort(_corpus())[::-1]
        with head_count_recall("off", "off"):
            before = {b: _kept(faces, head_count_cut(faces, b, _logits(faces)[:2])) for b in (1.0, 4.0)}
        assert _kept(faces, head_count_cut(faces, 1.0, _logits(faces)[:2])) == before[1.0]
        assert _kept(faces, head_count_cut(faces, 4.0, _logits(faces)[:2])) < before[4.0]
        # A weak head with a held-out Good below what stands out: deeper than before.
        s, y = _weak_head()
        deep = _logits(s)[np.flatnonzero(y)[25:26]]
        with head_count_recall("off", "off"):
            count = _kept(s, head_count_cut(s, 4.0, deep))
        assert _kept(s, head_count_cut(s, 4.0, deep)) > count


class TestTheRecallKnob:
    def test_none_is_the_apps_and_the_swap_is_undone(self):
        from vtscore.training.thresholds import labels_line

        shipped = labels_line.HEAD_COUNT_RECALL_EXPONENT, labels_line.HEAD_COUNT_RECALL_Z
        with head_count_recall(None, None):
            assert (labels_line.HEAD_COUNT_RECALL_EXPONENT, labels_line.HEAD_COUNT_RECALL_Z) == shipped
        with head_count_recall(0.2, 2.5):
            assert (labels_line.HEAD_COUNT_RECALL_EXPONENT, labels_line.HEAD_COUNT_RECALL_Z) == (0.2, 2.5)
        with head_count_recall("off", "off"):
            assert labels_line.HEAD_COUNT_RECALL_EXPONENT is None and labels_line.HEAD_COUNT_RECALL_Z is None
        assert (labels_line.HEAD_COUNT_RECALL_EXPONENT, labels_line.HEAD_COUNT_RECALL_Z) == shipped

    @pytest.mark.parametrize("bad", ["deep", True, math.inf, math.nan])
    def test_it_refuses_what_is_not_a_number(self, bad):
        with pytest.raises(ValueError):
            with head_count_recall(bad, None):
                pass
        with pytest.raises(ValueError):
            with head_count_recall(None, bad):
                pass


def _heavy_head(seed: int = 0, n: int = 11_000, n_pos: int = 50):
    """A Region-like head (#5084): a bulk with an exponential upper tail, and 50 positives a little above it."""
    rng = np.random.default_rng(seed)
    s = np.r_[_sig(rng.laplace(-3, 0.5, n)), _sig(rng.normal(0.5, 0.5, n_pos))]
    y = np.r_[np.zeros(n), np.ones(n_pos)]
    order = np.argsort(-s, kind="stable")
    return s[order], y[order]


def _f1(y, kept: int) -> float:
    return 2 * float(y[:kept].sum()) / (float(y.sum()) + kept)


class TestThePrecisionKnob:
    """#5084: at beta 1 and below the count keeps ``scale * beta ** exponent`` times what stands out."""

    def test_off_and_4603s_values_are_todays_count(self):
        from vtscore.training.thresholds.gmm import TEXT_SORT_COUNT_EXPONENT

        for seed in range(3):
            for s in (_heavy_head(seed)[0], _weak_head(seed)[0], np.sort(_corpus(seed + 1))[::-1]):
                for beta in (0.25, 0.5, 1.0):
                    with head_count_precision("off", "off"):
                        today = head_count_cut(s, beta)
                    with head_count_precision(1.0, TEXT_SORT_COUNT_EXPONENT):
                        assert head_count_cut(s, beta) == today

    def test_a_smaller_scale_keeps_nearer_a_heavy_tails_positives(self):
        """The count of what stands out reads the bulk's tail as positives; half of it keeps fewer and scores more."""
        for seed in range(3):
            s, y = _heavy_head(seed)
            with head_count_precision("off", "off"):
                today = _kept(s, head_count_cut(s, 1.0))
            with head_count_precision(0.5, None):
                halved = _kept(s, head_count_cut(s, 1.0))
            assert halved < today
            assert _f1(y, halved) > _f1(y, today)

    def test_the_exponent_sets_how_fast_the_count_falls_below_beta_1(self):
        s, _ = _heavy_head()
        with head_count_precision(1.0, 0.708):
            gentle = _kept(s, head_count_cut(s, 0.25))
        with head_count_precision(1.0, 1.5):
            steep = _kept(s, head_count_cut(s, 0.25))
        assert steep < gentle

    def test_it_never_keeps_fewer_than_the_tops_widest_drop(self):
        """A person's faces stand out of the bulk by the widest drop: the smallest scale still keeps them."""
        faces = np.sort(_corpus())[::-1]
        with head_count_precision(0.05, 1.0):
            for beta in (0.25, 1.0):
                assert _kept(faces, head_count_cut(faces, beta)) >= 4

    def test_a_recall_lean_is_untouched(self):
        s, _ = _heavy_head()
        deep = _logits(s)[300:301]
        today = head_count_cut(s, 4.0, deep)
        with head_count_precision(0.3, 1.0):
            assert head_count_cut(s, 4.0, deep) == today

    def test_none_is_the_apps_and_the_swap_is_undone(self):
        from vtscore.training.thresholds import labels_line

        shipped = labels_line.HEAD_COUNT_PRECISION_SCALE, labels_line.HEAD_COUNT_PRECISION_EXPONENT
        with head_count_precision(None, None):
            assert (labels_line.HEAD_COUNT_PRECISION_SCALE, labels_line.HEAD_COUNT_PRECISION_EXPONENT) == shipped
        with head_count_precision(0.7, 1.0):
            assert (labels_line.HEAD_COUNT_PRECISION_SCALE, labels_line.HEAD_COUNT_PRECISION_EXPONENT) == (0.7, 1.0)
        with head_count_precision("off", "off"):
            assert labels_line.HEAD_COUNT_PRECISION_SCALE is None and labels_line.HEAD_COUNT_PRECISION_EXPONENT is None
        assert (labels_line.HEAD_COUNT_PRECISION_SCALE, labels_line.HEAD_COUNT_PRECISION_EXPONENT) == shipped

    @pytest.mark.parametrize("bad", ["deep", True, math.inf, math.nan, 0.0, -1.0])
    def test_it_refuses_what_is_not_a_positive_number(self, bad):
        with pytest.raises(ValueError):
            with head_count_precision(bad, None):
                pass
        with pytest.raises(ValueError):
            with head_count_precision(None, bad):
                pass
