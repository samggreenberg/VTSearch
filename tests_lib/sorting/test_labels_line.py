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
    RELATIVE_SIGMA_FLOOR,
    ClassScoreModel,
    LabelsLine,
    class_score_model,
    corpus_prevalence,
    corpus_sigma_floor,
    estimate_positives,
    fit_labels_line,
    labels_line_threshold,
)
from vtscore.training.thresholds.labels_line import (
    MIN_LOGIT_SIGMA,
    WEAK_CHECK_COOLDOWN,
    WEAK_CHECK_MIN_VOTES,
    WEAK_SEPARATION_D,
    _line_on,
    weak_check_due,
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


class TestTheCutFollowsTheBalance:
    def test_a_precision_lean_keeps_less_and_a_recall_lean_more(self):
        """#4452: the shipped counted cut kept ~the same set at every beta (54/57/63 at 1/2, 1, 2); the cut from the
        3-part posteriors and the 2-part total moves with beta, as the oracle's does."""
        rng = np.random.default_rng(11)
        corpus = np.r_[
            _sig(rng.normal(-4, 1.0, 5000)), _sig(rng.normal(-1.0, 1.0, 800)), _sig(rng.normal(1.5, 1.2, 40))
        ]
        line = fit_labels_line(_orderings(), corpus, None, {})
        assert line is not None and line.unvoted_posteriors is not None
        kept = [int((corpus >= line.threshold(b)).sum()) for b in (0.5, 1.0, 2.0)]
        assert kept[0] <= kept[1] <= kept[2] and kept[0] < kept[2], kept


# ---------------------------------------------------------------------------
# The spread floor follows the corpus (#4492)
# ---------------------------------------------------------------------------


def _compressed_session(scale: float, seed: int = 0):
    """An early head's corpus: every logit score within ~0.3 of the bulk, positives just above it.

    Returns (class model as the folds would give it, corpus scores, labels), with every logit multiplied by
    *scale* - the same ranking and the same labels on a wider or narrower score scale.
    """
    from scipy.special import expit

    rng = np.random.default_rng(seed)
    n_neg, n_pos = 10_000, 45
    x_neg = rng.normal(-0.12, 0.09, n_neg)
    x_pos = rng.normal(0.30, 0.10, n_pos)
    x = np.r_[x_neg, x_pos] * scale
    y = np.r_[np.zeros(n_neg, int), np.ones(n_pos, int)]
    # The folds' held-out votes: a few Goods near the positives, a few Bads near the line.
    goods = rng.normal(0.36, 0.06, 4) * scale
    bads = rng.normal(0.05, 0.06, 4) * scale
    ordering = (expit(np.r_[goods, bads]), np.r_[np.ones(4), np.zeros(4)])
    return class_score_model([ordering]), expit(x), y


def test_a_compressed_corpus_is_read_on_its_own_scale():
    """#4492: an early head squeezes the corpus; an absolute 0.25 floor read no positives, the corpus's own reads them."""
    model, scores, y = _compressed_session(1.0)
    assert model is not None and model.sigma_raw is not None and model.sigma_raw < MIN_LOGIT_SIGMA
    line = _line_on(model, scores, None, {})
    assert line is not None
    kept = scores >= line.threshold(1.0)
    est = line.prevalence * scores.size
    assert 15 <= est <= 135, f"the positives are estimated near the truth (45), not at zero: {est:.1f}"
    assert kept.sum() >= 10 and (kept & (y == 1)).sum() >= 0.5 * kept.sum(), "a real returned set, mostly right"


def test_the_line_does_not_depend_on_the_logit_scale():
    """#4492: the same ranking and labels on a wider score scale keep the same set - a floor in logit units did not."""
    kept = []
    for scale in (1.0, 3.0):
        model, scores, _ = _compressed_session(scale)
        assert model is not None
        line = _line_on(model, scores, None, {})
        assert line is not None
        kept.append(int((scores >= line.threshold(1.0)).sum()))
    assert abs(kept[0] - kept[1]) <= max(2, 0.1 * kept[0]), kept


def test_a_model_floors_afresh_on_each_corpus():
    """The labels' model keeps its raw spread; each corpus floors it, so on_corpus does not inherit a Train floor."""
    model, scores, _ = _compressed_session(1.0)
    assert model is not None and model.sigma_raw is not None
    line = _line_on(model, scores, None, {})
    assert line is not None
    assert line.model.sigma_raw == model.sigma_raw, "the line keeps the labels' unfloored model"
    assert model.floored(0.05).sigma == max(model.sigma_raw, 0.05)
    assert model.floored(0.5).sigma == 0.5
    wide = scores**0.25  # another corpus, another spread
    assert corpus_sigma_floor(wide) != corpus_sigma_floor(scores)
    assert line.on_corpus(wide) is not None


def test_a_saved_model_without_a_raw_spread_still_loads():
    """Models saved before #4492 carry no sigma_raw; their sigma is taken as given."""
    old = {"mu_pos": 0.4, "mu_neg": -0.6, "sigma": 0.25, "n_pos": 5, "n_neg": 20}
    m = ClassScoreModel(**old)
    assert m.sigma_raw is None and m.floored(0.1).sigma == 0.25
    assert ClassScoreModel(**m.as_dict()) == m


def test_the_corpus_floor_is_relative_and_bounded():
    from scipy.special import expit

    x = np.random.default_rng(1).normal(0.0, 0.2, 5000)
    f = corpus_sigma_floor(expit(x))
    assert f == pytest.approx(RELATIVE_SIGMA_FLOOR * 0.2, rel=0.08)
    assert corpus_sigma_floor(np.full(10, 0.5)) > 0, "a corpus of one score still has a floor"
    assert corpus_sigma_floor(np.array([])) == MIN_LOGIT_SIGMA


def test_separation_is_read_in_the_spread_the_line_was_cut_with():
    """#4496: d' = (Good mean - Bad mean) / spread, the spread the class model's own floored on the corpus."""
    model, scores, _ = _compressed_session(1.0)
    assert model is not None and model.sigma_raw is not None
    line = _line_on(model, scores, None, {})
    assert line is not None and line.spread is not None
    assert line.spread == max(model.sigma_raw, corpus_sigma_floor(scores))
    assert line.separation == pytest.approx((model.mu_pos - model.mu_neg) / line.spread)
    # Free of the logit scale, as the line is.
    wide, wide_scores, _ = _compressed_session(3.0)
    assert wide is not None
    wide_line = _line_on(wide, wide_scores, None, {})
    assert wide_line is not None
    assert wide_line.separation == pytest.approx(line.separation, rel=0.05)


def test_a_line_built_by_hand_reads_separation_in_the_models_spread():
    model = ClassScoreModel(mu_pos=1.0, mu_neg=-0.5, sigma=0.5, n_pos=3, n_neg=3)
    assert LabelsLine(model, 0.01).separation == pytest.approx(3.0)
    assert LabelsLine(ClassScoreModel(0.2, 0.0, 0.0, 1, 1), 0.01).separation == float("inf")
    assert 1.0 < WEAK_SEPARATION_D < 3.0


class TestWeakCheckDue:
    """#4496: the rule the app's Autopilot and the harness's default arm both check by."""

    def test_due_only_on_weak_separation_after_the_first_votes(self):
        assert weak_check_due(1.0, WEAK_CHECK_MIN_VOTES, None)
        assert not weak_check_due(1.0, WEAK_CHECK_MIN_VOTES - 1, None), "too few votes"
        assert not weak_check_due(WEAK_SEPARATION_D, 40, None), "at the threshold is not weak"
        assert not weak_check_due(None, 40, None), "no labels line yet"
        assert not weak_check_due(float("nan"), 40, None)

    def test_it_comes_back_a_cooldown_after_the_last_check_ended(self):
        ended = 30
        assert not weak_check_due(1.0, ended + WEAK_CHECK_COOLDOWN - 1, ended)
        assert weak_check_due(1.0, ended + WEAK_CHECK_COOLDOWN, ended)
        assert not weak_check_due(3.0, ended + 100, ended), "not while separation is strong"

    def test_no_cooldown_makes_it_due_once(self):
        assert weak_check_due(1.0, 20, None, cooldown=None)
        assert not weak_check_due(1.0, 500, 20, cooldown=None)
