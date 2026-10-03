"""The line drawn from the labels alone (#4452).

The owner's ruling: an exported labelset must be enough to run Find later, and
Find on a corpus with no positives must not return a fixed count of wrong
images.  The line is the class model the calibration folds' held-out scores of
the votes imply, cut where the expected F-beta peaks at the prevalence that
model estimates on the corpus being decided.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.training.thresholds import (
    LabelsLine,
    class_score_model,
    corpus_prevalence,
    estimate_positives,
    fit_labels_line,
    labels_line_threshold,
)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _orderings(seed: int = 0, n_pos: int = 10, n_neg: int = 30):
    """Two folds' held-out scores: Goods around logit 2, Bads around logit -1."""
    rng = np.random.default_rng(seed)
    return [
        (list(_sig(rng.normal(2, 1, n_pos))) + list(_sig(rng.normal(-1, 1, n_neg))), [1.0] * n_pos + [0.0] * n_neg)
        for _ in range(2)
    ]


def _corpus(seed: int = 1, n_neg: int = 10_000, n_pos: int = 44):
    rng = np.random.default_rng(seed)
    return np.r_[_sig(rng.normal(-4, 1.2, n_neg)), _sig(rng.normal(2, 1, n_pos))]


class TestTheClassModel:
    def test_it_pools_the_folds_and_shares_one_spread(self):
        m = class_score_model(_orderings())
        assert m is not None and m.n_pos == 20 and m.n_neg == 60
        assert m.mu_pos > m.mu_neg and m.sigma > 0

    @pytest.mark.parametrize(
        "orderings",
        [None, [], [([0.9, 0.8], [1.0, 1.0])], [([0.1, 0.2], [0.0, 0.0])], [([0.1, 0.9], [1.0, 0.0])]],
        ids=["none", "empty", "no-bads", "no-goods", "ranks-the-wrong-way"],
    )
    def test_folds_that_cannot_support_a_line_give_none(self, orderings):
        assert class_score_model(orderings) is None
        assert fit_labels_line(orderings, [0.5, 0.6]) is None

    def test_one_good_is_enough_through_the_labels_in_sample_scores(self):
        """Folds that held nothing out (one Good): the head's scores of the labelled items stand in (#4452)."""
        scores = _corpus()
        ids = list(range(scores.size))
        top = int(np.argmax(scores))
        lows = [int(i) for i in np.argsort(scores)[:40]]
        labels = {top: True, **dict.fromkeys(lows, False)}
        line = fit_labels_line([], scores, ids, labels)
        assert line is not None and line.model.n_pos == 1 and line.model.n_neg == 40
        assert fit_labels_line([], scores, ids, {top: True}) is None, "no Bad: no model"
        kept = int((scores >= line.threshold(1.0)).sum())
        assert kept < 200, "never the thousands the population fallback kept"

    def test_perfect_separation_keeps_a_minimum_spread(self):
        m = class_score_model([([0.99, 0.99, 0.01, 0.01], [1.0, 1.0, 0.0, 0.0])])
        assert m is not None and m.sigma >= 0.25

    def test_any_numeric_dtype_is_read(self):
        """The eval harness's folds hold float32 arrays; an isinstance(float) filter dropped them all (#4452)."""
        clean = class_score_model(_orderings())
        f32 = [(np.asarray(s, dtype=np.float32), np.asarray(y, dtype=np.float32)) for s, y in _orderings()]
        model = class_score_model(f32)
        assert model is not None and clean is not None
        assert model.n_pos == clean.n_pos and model.mu_pos == pytest.approx(clean.mu_pos, abs=1e-5)

    def test_unscorable_sentinels_are_ignored(self):
        clean = class_score_model(_orderings())
        noisy = [(list(s) + [-1.0, float("nan")], list(y) + [1.0, 0.0]) for s, y in _orderings()]
        assert class_score_model(noisy) == clean


class TestTheThreshold:
    def test_a_rarer_target_and_a_precision_lean_both_raise_the_cut(self):
        m = class_score_model(_orderings())
        assert m is not None
        rare = {b: labels_line_threshold(m, 0.0044, b) for b in (0.5, 1.0, 2.0)}
        common = {b: labels_line_threshold(m, 0.05, b) for b in (0.5, 1.0, 2.0)}
        assert rare[0.5] > rare[1.0] > rare[2.0]
        assert all(rare[b] > common[b] for b in rare)

    def test_it_reads_nothing_but_the_model_the_prevalence_and_beta(self):
        m = class_score_model(_orderings())
        assert m is not None
        assert labels_line_threshold(m, 0.01, 1.0) == labels_line_threshold(m, 0.01, 1.0)
        line = LabelsLine(m, 0.01)
        assert line.threshold(1.0) == labels_line_threshold(m, 0.01, 1.0)


class TestThePrevalence:
    def test_the_em_finds_a_rare_target_and_an_empty_corpus(self):
        m = class_score_model(_orderings())
        assert m is not None
        found = estimate_positives(m, _corpus(), 0)
        assert 5 < found < 60, "near the 44 planted, on a biased-high negative model"
        assert estimate_positives(m, _sig(np.random.default_rng(2).normal(-4, 1.2, 200)), 0) < 0.5
        assert estimate_positives(m, [], 3) == 3.0, "votes alone when nothing is unvoted"

    def test_votes_in_the_corpus_count_as_they_are(self):
        m = class_score_model(_orderings())
        assert m is not None
        scores = _corpus()
        ids = list(range(scores.size))
        # A Good vote on an item the model would never call positive is one more positive, counted as it is.
        low = int(np.argmin(scores))
        with_vote = corpus_prevalence(m, scores, ids, {low: True})
        without = corpus_prevalence(m, scores, ids, {})
        assert with_vote is not None and without is not None
        assert with_vote == pytest.approx(without + 1 / scores.size, rel=0.05)
        assert corpus_prevalence(m, [], [], {}) is None


class TestTheCorpusSide:
    def test_a_weak_head_and_high_bads_do_not_run_the_prevalence_away(self):
        """keyboard@small, the first pricing cells: two Goods barely above Bads picked near the line, and the old
        EM (negatives modelled by the Bads) estimated 34% for a 0.44% target.  The bulk is fitted on the corpus."""
        rng = np.random.default_rng(5)
        orderings = [(list(_sig([0.5, 0.7])) + list(_sig(rng.normal(0.0, 0.8, 20))), [1.0] * 2 + [0.0] * 20)]
        corpus = np.r_[_sig(rng.normal(-1.5, 1.4, 5800)), _sig(rng.normal(0.4, 0.8, 25))]
        line = fit_labels_line(orderings, corpus, None, {})
        assert line is not None and line.prevalence < 0.05
        assert int((corpus >= line.threshold(1.0)).sum()) < 1000

    def test_the_negatives_are_the_corpus_bulk(self):
        line = fit_labels_line(_orderings(), _corpus(), None, {})
        assert line is not None and line.negatives is not None
        assert line.negatives.mu == pytest.approx(-4.0, abs=0.3) and line.negatives.sigma == pytest.approx(1.2, abs=0.3)


class TestFindOnACorpusWithNoPositives:
    def test_the_line_keeps_next_to_nothing_where_the_count_kept_thirty(self):
        """#4452's case: 200 images, no positives; the count line kept 26-32 at beta <= 1."""
        orderings = _orderings()
        train = fit_labels_line(orderings, _corpus(), range(_corpus().size), {})
        assert train is not None
        empty = _sig(np.random.default_rng(3).normal(-4, 1.2, 200))
        find = train.on_corpus(empty)
        assert find.model == train.model, "the labels' model travels; only the prevalence is re-estimated"
        assert find.prevalence < train.prevalence
        for beta in (0.5, 1.0, 2.0):
            assert int((empty >= find.threshold(beta)).sum()) <= 2

    def test_a_find_corpus_like_the_train_one_gets_the_same_line(self):
        """The owner's working assumption: a Find corpus has the Train corpus's properties."""
        train = fit_labels_line(_orderings(), _corpus(seed=1), None, {})
        assert train is not None
        other = _corpus(seed=7)
        find = train.on_corpus(other)
        assert find.prevalence == pytest.approx(train.prevalence, rel=0.35)
        # The cut is counted on each corpus, so compare what it keeps, not the score it lands on.
        kept_train = int((_corpus(seed=1) >= train.threshold(1.0)).sum())
        kept_find = int((other >= find.threshold(1.0)).sum())
        assert kept_find == pytest.approx(kept_train, rel=0.35, abs=5)
