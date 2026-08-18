"""Media that cannot be scored are excluded from every threshold fit.

Issue #3180. A head that emits a non-finite logit for a media - a corrupt
vector, a destabilised fit - has that media recorded at
:data:`~vtscore.utils.scores.NON_FINITE_SCORE_SENTINEL` (``-1.0``). The
sentinel is deliberately outside the ``[0, 1]`` sigmoid range so ``score >=
threshold`` is always ``False`` for it, and that contract held right up until
the sentinels were handed to the threshold estimators *as observations*.
Fitting a mixture on a population with a spike a full unit below the range
pulls the cut under zero, at which point every real score clears it and the
detector calls the entire dataset a hit - which is exactly what a CLI
autodetect run reported (threshold ``-0.375``, every image a positive).

These pin the population every estimator is allowed to see: the fold-anchored
fit (haystacks *and* anchors), the pooled conformal orderings, and the schedule
blend. The end-to-end case is the load-bearing one - a haystack with a handful
of unscorable media must produce the *same* threshold as the clean haystack,
not merely a non-negative one.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.detectors.training import _score_all_media, train_and_threshold
from vtscore.training.blend_schedules import BlendContext
from vtscore.training.thresholds import (
    NO_GOOD_THRESHOLD,
    calculate_safe_threshold,
    fit_fold_anchored_cut,
    scored_ordering,
    threshold_from_fold_orderings,
)
from vtscore.utils.scores import NON_FINITE_SCORE_SENTINEL, scored_mask, scored_only

DIM = 32


class TestScoredHelpers:
    """``scored_mask`` / ``scored_only`` name what counts as an observation."""

    def test_sentinel_and_out_of_range_dropped(self):
        scores = [0.0, 0.25, NON_FINITE_SCORE_SENTINEL, 1.0, 2.0, -0.5]
        assert scored_mask(scores).tolist() == [True, True, False, True, False, False]
        assert scored_only(scores).tolist() == [0.0, 0.25, 1.0]

    def test_non_finite_dropped(self):
        assert scored_only([float("nan"), 0.4, float("inf")]).tolist() == [0.4]

    def test_all_scorable_passes_through(self):
        scores = [0.1, 0.9]
        assert scored_only(scores).tolist() == scores

    def test_ordering_drops_the_label_with_the_score(self):
        scores, labels = scored_ordering(([0.2, NON_FINITE_SCORE_SENTINEL, 0.8], [1.0, 1.0, 0.0]))
        assert scores == [0.2, 0.8]
        assert labels == [1.0, 0.0]


class TestBlendIgnoresUnscorable:
    """The schedule blend fits its GMM on the scorable population only."""

    def test_sentinels_do_not_move_the_blend(self):
        rng = np.random.default_rng(0)
        clean = np.concatenate([rng.normal(0.1, 0.02, 200), rng.normal(0.9, 0.02, 200)]).tolist()
        poisoned = clean + [NON_FINITE_SCORE_SENTINEL] * 20
        ctx = BlendContext.from_labels([1.0] * 10 + [0.0] * 10, None)

        assert calculate_safe_threshold(0.6, poisoned, ctx) == calculate_safe_threshold(0.6, clean, ctx)

    def test_no_scorable_population_keeps_the_labelled_cut(self):
        ctx = BlendContext.from_labels([1.0] * 10 + [0.0] * 10, None)
        all_sentinel = [NON_FINITE_SCORE_SENTINEL] * 50

        # Not 0.5 (fit_gmm_threshold's "too few scores" stand-in) and not the
        # sentinel: with no population there is nothing for the GMM to stand in
        # for, so the cross-calibration cut ships alone.
        assert calculate_safe_threshold(0.62, all_sentinel, ctx) == 0.62


class TestConformalPoolIgnoresUnscorable:
    """A held-out item the fold model could not score is not an anchor."""

    def test_sentinel_anchors_dropped_from_the_pool(self):
        clean = [([0.2, 0.3, 0.8, 0.9], [0.0, 0.0, 1.0, 1.0])]
        poisoned = [([0.2, 0.3, 0.8, 0.9, NON_FINITE_SCORE_SENTINEL], [0.0, 0.0, 1.0, 1.0, 1.0])]

        assert threshold_from_fold_orderings(poisoned, 0) == threshold_from_fold_orderings(clean, 0)

    def test_all_sentinel_pool_admits_nothing(self):
        orderings = [([NON_FINITE_SCORE_SENTINEL, NON_FINITE_SCORE_SENTINEL], [1.0, 0.0])]

        assert threshold_from_fold_orderings(orderings, 0) == NO_GOOD_THRESHOLD


class TestFoldAnchoredFitIgnoresUnscorable:
    """Neither the fold haystacks nor the anchors may carry a sentinel."""

    @staticmethod
    def _haystack(rng: np.random.Generator) -> np.ndarray:
        return np.concatenate([rng.normal(0.1, 0.03, 300), rng.normal(0.9, 0.03, 60)])

    def test_cut_matches_the_clean_fit(self):
        rng = np.random.default_rng(7)
        hay = self._haystack(rng)
        final = self._haystack(np.random.default_rng(8))
        ordering = ([0.12, 0.15, 0.88, 0.91], [0.0, 0.0, 1.0, 1.0])

        clean = fit_fold_anchored_cut([hay], [ordering], final)
        poisoned = fit_fold_anchored_cut(
            [np.concatenate([hay, np.full(15, NON_FINITE_SCORE_SENTINEL)])],
            [(list(ordering[0]) + [NON_FINITE_SCORE_SENTINEL], list(ordering[1]) + [1.0])],
            np.concatenate([final, np.full(15, NON_FINITE_SCORE_SENTINEL)]),
        )

        assert clean is not None and poisoned is not None
        assert poisoned.threshold_at(0) == pytest.approx(clean.threshold_at(0))
        assert poisoned.threshold_at(0) >= 0.0

    def test_all_unscorable_yields_no_fit(self):
        sentinels = np.full(40, NON_FINITE_SCORE_SENTINEL)
        ordering = ([0.2, 0.9], [0.0, 1.0])

        assert fit_fold_anchored_cut([sentinels], [ordering], sentinels) is None


def _snap(vecs: np.ndarray, broken: set[int]) -> dict[int, dict]:
    return {
        i: {
            "media_type": "image",
            "embedder": "e",
            "embeddings": {"e": (np.full(DIM, np.nan, np.float32) if i in broken else vecs[i])},
        }
        for i in range(len(vecs))
    }


class TestTrainedThresholdSurvivesBrokenMedia:
    """End-to-end: the #3180 regression, at the size that triggered it."""

    @staticmethod
    def _fixture() -> tuple[np.ndarray, list[np.ndarray], list[float]]:
        rng = np.random.default_rng(3)
        pos = rng.standard_normal((30, DIM)).astype(np.float32) + 1.0
        neg = rng.standard_normal((570, DIM)).astype(np.float32) - 1.0
        vecs = np.concatenate([pos, neg])
        X = [vecs[i] for i in range(10)] + [vecs[i] for i in range(30, 40)]
        y = [1.0] * 10 + [0.0] * 10
        return vecs, X, y

    def test_broken_media_do_not_drag_the_threshold_negative(self):
        vecs, X, y = self._fixture()
        broken = set(range(100, 112))  # 2% of the haystack, all in the Bad mass

        _clean_model, clean_threshold = train_and_threshold(X, y, snap=_snap(vecs, set()))
        model, threshold = train_and_threshold(X, y, snap=_snap(vecs, broken))

        # Before the fix this was -0.4975 against a clean +0.5785. The two are
        # not bit-identical - dropping 12 media really does move the empirical
        # quantile the cut is realized on - but they now differ by order
        # statistics, not by a sign.
        assert threshold >= 0.0
        assert threshold == pytest.approx(clean_threshold, abs=0.05)

        # And the sentinel keeps its own contract: an unscorable media is never
        # a hit, and the run reports a shortlist rather than the whole dataset
        # (before the fix all 588 scorable media cleared the negative cut).
        snap = _snap(vecs, broken)
        _ids, scores, _best = _score_all_media(model, snap, None)
        hits = [s for s in scores if s >= threshold]
        assert len(hits) < len(scores) // 2
        assert all(s > NON_FINITE_SCORE_SENTINEL for s in hits)
