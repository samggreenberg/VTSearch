"""The head's count line (#4786): while the class model holds few Bads, a recall-leaning line counts what stands out.

At beta 4 the labels line's cut on a class model fitted to a handful of held-out
Bads reached deep into its tail.  Until the model holds ``HEAD_COUNT_BADS`` Bads, a
balance above beta 1 keeps instead what the Goods' centroid's count rule finds
standing out in the head's own scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.detectors.centroid_head import centroid_cut
from vtscore.eval.live_threshold_rules import head_count_bads
from vtscore.training.thresholds import (
    HEAD_COUNT_BADS,
    HEAD_COUNT_MIN_BETA,
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
        s = np.sort(_corpus())[::-1]
        x = np.log(s) - np.log1p(-s)
        for beta in (0.25, 1.0, 4.0):
            k = int(np.count_nonzero(x >= centroid_cut(x, beta=beta)))
            thr = head_count_cut(s, beta)
            assert _kept(s, thr) == k
            # Between the last kept and the first dropped, never on an item.
            assert s[k] < thr < s[k - 1]

    def test_a_recall_lean_keeps_at_least_the_standouts(self):
        s = np.sort(_corpus())[::-1]
        assert _kept(s, head_count_cut(s, 4.0)) >= 4

    def test_an_empty_corpus_keeps_nothing(self):
        assert head_count_cut(np.array([]), 4.0) > 1.0


class TestTheGate:
    def test_few_bads_and_a_recall_lean(self):
        line = fit_labels_line(_orderings(4, 4), _corpus())
        assert line is not None and line.model.n_neg == 4
        assert head_count_serves(line.model, 4.0)
        assert not head_count_serves(line.model, HEAD_COUNT_MIN_BETA)
        assert not head_count_serves(line.model, 0.25)

    def test_the_class_model_takes_over_at_the_bads(self):
        line = fit_labels_line(_orderings(4, HEAD_COUNT_BADS), _corpus())
        assert line is not None and not head_count_serves(line.model, 4.0)
        line = fit_labels_line(_orderings(4, HEAD_COUNT_BADS - 1), _corpus())
        assert line is not None and head_count_serves(line.model, 4.0)


class TestTheLine:
    def test_at_beta_4_with_few_bads_the_line_is_the_heads_count(self):
        line = fit_labels_line(_orderings(4, 4), _corpus())
        assert line is not None and line.unvoted_scores is not None
        assert line.threshold(4.0) == head_count_cut(line.unvoted_scores, 4.0)

    def test_at_beta_1_and_below_it_is_the_class_models_cut(self):
        line = fit_labels_line(_orderings(4, 4), _corpus())
        assert line is not None and line.unvoted_scores is not None and line.unvoted_posteriors is not None
        total = float(line.unvoted_share or 0.0) * line.unvoted_scores.size
        for beta in (0.25, 1.0):
            assert line.threshold(beta) == corpus_cut(line.unvoted_posteriors, line.unvoted_scores, total, beta)

    def test_find_on_another_corpus_draws_it_there(self):
        line = fit_labels_line(_orderings(4, 4), _corpus(1))
        assert line is not None
        other = _corpus(2)
        on = line.on_corpus(other)
        assert on.unvoted_scores is not None
        assert on.threshold(4.0) == head_count_cut(on.unvoted_scores, 4.0)

    def test_gate_zero_is_the_line_before_4786(self):
        line = fit_labels_line(_orderings(4, 4), _corpus())
        assert line is not None and line.unvoted_scores is not None and line.unvoted_posteriors is not None
        total = float(line.unvoted_share or 0.0) * line.unvoted_scores.size
        with head_count_bads(0):
            assert line.threshold(4.0) == corpus_cut(line.unvoted_posteriors, line.unvoted_scores, total, 4.0)
        assert line.threshold(4.0) == head_count_cut(line.unvoted_scores, 4.0)


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
