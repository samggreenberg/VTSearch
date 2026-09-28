"""Planted-answer tests for the precision-floor cut (#4224; the estimator #4220 measured).

Scores are drawn from two Gaussians with a known posterior - negatives N(0, 1),
positives N(3, 1) - and votes are picked by score only (mostly near the top,
the rest anywhere), never by label, as Autopilot picks them.  The calibration
folds' haystacks are the pool's scores with a little per-fold noise, standing in
for fold models that rank almost as the final model does.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.training.thresholds import (
    MIN_CALIBRATION_POSITIVES,
    PrecisionFloorStatus,
    em_prior_shift,
    fold_rank_evidence,
    precision_floor_cut,
    precision_lower_bound_curve,
)

N = 3000


def _draw(rng: np.random.Generator, n: int, prevalence: float, gap: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    y = (rng.random(n) < prevalence).astype(np.int64)
    s = np.where(y == 1, rng.normal(gap, 1.0, n), rng.normal(0.0, 1.0, n))
    return s, y


def _session(
    seed: int,
    *,
    pool_prevalence: float = 0.03,
    corpus_prevalence: float | None = None,
    n_votes: int = 80,
    gap: float = 3.0,
    same_corpus: bool = False,
):
    """``(corpus_scores, corpus_labels, pool_scores, fold_orderings, fold_haystacks)`` for one planted session.

    *same_corpus* makes the corpus the voted pool itself - the app's case when
    the cut decides the dataset being labelled - rather than a fresh draw.
    """
    rng = np.random.default_rng(seed)
    pool_s, pool_y = _draw(rng, N, pool_prevalence, gap)
    corpus_s, corpus_y = _draw(rng, N, corpus_prevalence or pool_prevalence, gap)
    if same_corpus:
        corpus_s, corpus_y = pool_s, pool_y
    top = np.argsort(-pool_s)[:150]
    weights = np.full(N, 0.3 / N)
    weights[top] += 0.7 / len(top)
    picked = rng.choice(N, size=n_votes, replace=False, p=weights / weights.sum())
    half = n_votes // 2
    orderings = [
        (pool_s[picked[:half]].tolist(), pool_y[picked[:half]].tolist()),
        (pool_s[picked[half:]].tolist(), pool_y[picked[half:]].tolist()),
    ]
    haystacks = [pool_s + rng.normal(0.0, 0.05, N), pool_s + rng.normal(0.0, 0.05, N)]
    return corpus_s, corpus_y, pool_s, orderings, haystacks


def _achieved(cut, corpus_s: np.ndarray, corpus_y: np.ndarray) -> float:
    admitted = corpus_s >= cut.threshold
    return float(corpus_y[admitted].mean())


class TestTheThreeStates:
    def test_the_lower_bound_keeps_the_promises_a_point_estimate_breaks(self):
        """A cut at the point estimate misses the floor about half the time; the lower bound rarely does.

        The #4220 reading, planted: the corpus is the voted pool, cut at X = 50%
        over 20 sessions, once at the shipped 10th-percentile bound and once at
        the bootstrap median (a point estimate).  The bound breaks far fewer
        promises, and the ones it breaks are shallow.
        """
        broken = {10.0: 0, 50.0: 0}
        worst = 1.0
        for seed in range(20):
            corpus_s, corpus_y, pool_s, orderings, haystacks = _session(seed, same_corpus=True)
            for level in broken:
                cut = precision_floor_cut(
                    0.5, corpus_s, pool_s, orderings, haystacks, min_positives=5, lower_percentile=level
                )
                assert cut.status is PrecisionFloorStatus.PROMISED, (seed, level)
                assert cut.estimated_precision is not None
                assert cut.n_returned > 0 and cut.estimated_precision >= 0.5
                achieved = _achieved(cut, corpus_s, corpus_y)
                broken[level] += achieved < 0.5
                if level == 10.0:
                    worst = min(worst, achieved)
        assert broken[10.0] <= 4, broken
        assert broken[10.0] < broken[50.0], broken
        assert worst >= 0.4, f"a broken promise fell to {worst:.2f}"

    def test_too_few_calibration_positives_promise_nothing(self):
        """The gate: below it the estimate breaks most promises (#4220), so none is made."""
        corpus_s, _y, pool_s, orderings, haystacks = _session(0)
        n_pos = int(sum(sum(labels) for _s, labels in orderings))
        cut = precision_floor_cut(0.5, corpus_s, pool_s, orderings, haystacks, min_positives=n_pos + 1)
        assert cut.status is PrecisionFloorStatus.INSUFFICIENT_EVIDENCE
        assert cut.threshold is None and cut.n_returned == 0
        assert cut.calibration_positives == n_pos

    def test_the_default_gate_is_what_4220_measured(self):
        assert MIN_CALIBRATION_POSITIVES == 10

    def test_one_class_of_votes_promises_nothing(self):
        corpus_s, _y, pool_s, orderings, haystacks = _session(0)
        all_bad = [(scores, [0] * len(scores)) for scores, _labels in orderings]
        cut = precision_floor_cut(0.5, corpus_s, pool_s, all_bad, haystacks, min_positives=0)
        assert cut.status is PrecisionFloorStatus.INSUFFICIENT_EVIDENCE

    def test_an_unseparated_rare_class_is_unreachable_and_says_what_is(self):
        """Enough evidence, no cut clears the floor: say so, and report the best bound any cut reaches."""
        corpus_s, _y, pool_s, orderings, haystacks = _session(1, pool_prevalence=0.05, gap=0.3, n_votes=400)
        cut = precision_floor_cut(0.75, corpus_s, pool_s, orderings, haystacks, min_positives=5)
        assert cut.status is PrecisionFloorStatus.UNREACHABLE
        assert cut.threshold is None
        assert cut.estimated_precision is not None and cut.estimated_precision < 0.75


class TestTheCut:
    def test_a_higher_floor_returns_a_subset(self):
        """Nesting: the largest top-k clearing a higher floor is never larger."""
        corpus_s, _y, pool_s, orderings, haystacks = _session(2)
        cuts = [
            precision_floor_cut(x, corpus_s, pool_s, orderings, haystacks, min_positives=5) for x in (0.25, 0.5, 0.75)
        ]
        promised = [c for c in cuts if c.status is PrecisionFloorStatus.PROMISED]
        assert len(promised) >= 2
        thresholds = [c.threshold for c in promised if c.threshold is not None]
        assert len(thresholds) == len(promised)
        assert all(b >= a for a, b in zip(thresholds, thresholds[1:], strict=False))

    def test_one_set_of_votes_yields_one_cut(self):
        """The bootstrap is seeded by default: a line that moved between identical requests would read as a bug."""
        corpus_s, _y, pool_s, orderings, haystacks = _session(3)
        a = precision_floor_cut(0.5, corpus_s, pool_s, orderings, haystacks, min_positives=5)
        b = precision_floor_cut(0.5, corpus_s, pool_s, orderings, haystacks, min_positives=5)
        assert a == b

    def test_em_promises_less_on_a_poorer_corpus(self):
        """Label shift: a session calibrated on a rich pool, run over a poorer corpus.

        The posterior carries the pool's prior, so without EM the curve
        over-states precision on the poorer corpus and the cut returns more.
        """
        corpus_s, corpus_y, pool_s, orderings, haystacks = _session(4, pool_prevalence=0.10, corpus_prevalence=0.01)
        plain = precision_floor_cut(0.5, corpus_s, pool_s, orderings, haystacks, em=False, min_positives=5)
        shifted = precision_floor_cut(0.5, corpus_s, pool_s, orderings, haystacks, em=True, min_positives=5)
        assert plain.status is PrecisionFloorStatus.PROMISED
        returned_with_em = shifted.n_returned if shifted.status is PrecisionFloorStatus.PROMISED else 0
        assert returned_with_em < plain.n_returned

    @pytest.mark.parametrize("coordinate", ["percentile", "tail"])
    @pytest.mark.parametrize("fit", ["logistic", "isotonic"])
    def test_every_option_runs_on_a_separable_session(self, coordinate, fit):
        corpus_s, _y, pool_s, orderings, haystacks = _session(5)
        cut = precision_floor_cut(
            0.5, corpus_s, pool_s, orderings, haystacks, coordinate=coordinate, fit=fit, min_positives=5
        )
        assert cut.status in (PrecisionFloorStatus.PROMISED, PrecisionFloorStatus.UNREACHABLE)

    def test_the_curve_is_the_one_the_cut_reads(self):
        corpus_s, _y, pool_s, orderings, haystacks = _session(6)
        curve = precision_lower_bound_curve(corpus_s, pool_s, orderings, haystacks)
        assert curve is not None
        scores, bound = curve
        cut = precision_floor_cut(0.5, corpus_s, pool_s, orderings, haystacks, min_positives=5)
        assert np.all(np.diff(scores) <= 0)
        i = int(np.flatnonzero(bound >= 0.5).max())
        assert cut.threshold == scores[i] and cut.estimated_precision == bound[i]


class TestPieces:
    def test_em_on_the_pool_itself_changes_nothing(self):
        """The fixed point is immediate when the corpus is the pool the prior came from."""
        p = np.random.default_rng(7).beta(0.5, 5.0, 2000)
        assert np.allclose(em_prior_shift(p, float(p.mean())), p, atol=1e-6)

    def test_em_moves_the_posterior_toward_a_rarer_corpus(self):
        p = np.random.default_rng(8).beta(0.5, 5.0, 2000)
        shifted = em_prior_shift(p, prior_fitted=float(p.mean()) * 4)
        assert shifted.mean() < p.mean()

    def test_each_fold_is_ranked_in_its_own_haystack(self):
        """A score's percentile is read against its own fold model's scale, not a shared one."""
        orderings = [([0.5], [1]), ([5.0], [0])]
        haystacks = [np.linspace(0, 1, 101), np.linspace(0, 10, 101)]
        x, y = fold_rank_evidence(orderings, haystacks)
        assert x.tolist() == pytest.approx([0.5, 0.5], abs=0.01)
        assert y.tolist() == [1.0, 0.0]

    def test_misaligned_folds_are_refused(self):
        with pytest.raises(ValueError, match="fold orderings"):
            fold_rank_evidence([([0.5], [1])], [])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"floor": 0.0}, "floor"),
            ({"floor": 1.5}, "floor"),
            ({"fit": "svm"}, "fit"),
            ({"coordinate": "z"}, "coordinate"),
        ],
    )
    def test_bad_options_are_refused(self, kwargs, match):
        corpus_s, _y, pool_s, orderings, haystacks = _session(0)
        args = {"floor": 0.5, **kwargs}
        floor = args.pop("floor")
        with pytest.raises(ValueError, match=match):
            precision_floor_cut(floor, corpus_s, pool_s, orderings, haystacks, **args)
