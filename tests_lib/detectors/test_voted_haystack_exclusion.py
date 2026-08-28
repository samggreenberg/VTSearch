"""Voted media are excluded from the calibrated threshold's haystacks.

Issue #3308. Every model in the fold-anchored chain was trained on some or
all of the votes, so a voted item's score under it is optimistically shifted
- and the calibration votes additionally sat in the fold haystack twice, once
as free points and once as anchors. The estimator therefore drops all voted
media from the fold haystacks *and* from the final model's realization
sample, keeping every distribution in the quantile transfer over the one
population the threshold actually decides: the unlabeled remainder.

The load-bearing invariant is the first test: passing ``voted_ids`` over the
full snap must produce **exactly** the threshold of a snap that never
contained the voted media, because on the fold-anchored path the excluded
scores are the only difference between the two runs.
"""

from __future__ import annotations

import numpy as np

import vtscore.training.thresholds as thresholds_mod
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.labelset_training import labeled_media_ids
from vtscore.detectors.training import train_and_score, train_and_threshold

DIM = 32


def _snap(vecs: np.ndarray, ids: "list[int] | None" = None) -> dict[int, dict]:
    keep = range(len(vecs)) if ids is None else ids
    return {
        i: {
            "media_type": "image",
            "embedder": "e",
            "md5": f"md5-{i}",
            "filename": f"clip_{i}.png",
            "embeddings": {"e": vecs[i]},
        }
        for i in keep
    }


def _fixture(n_pos: int = 40, n_neg: int = 560) -> tuple[np.ndarray, list[int], list[int]]:
    """A separable corpus plus the voted ids (positives first, then negatives)."""
    rng = np.random.default_rng(11)
    pos = rng.standard_normal((n_pos, DIM)).astype(np.float32) + 1.0
    neg = rng.standard_normal((n_neg, DIM)).astype(np.float32) - 1.0
    vecs = np.concatenate([pos, neg])
    good_ids = list(range(8))
    bad_ids = list(range(n_pos, n_pos + 8))
    return vecs, good_ids, bad_ids


def _xy(vecs: np.ndarray, good_ids: list[int], bad_ids: list[int]):
    X = [vecs[i] for i in (*good_ids, *bad_ids)]
    y = [1.0] * len(good_ids) + [0.0] * len(bad_ids)
    return X, y


def _spy_fit_sizes(monkeypatch) -> list[tuple[int, ...]]:
    """Capture ``(len(fold_hay_0), ..., len(final))`` of every estimator fit."""
    captured: list[tuple[int, ...]] = []
    real = thresholds_mod.fit_fold_anchored_cut

    def spy(fold_haystacks, orderings, final_scores, **kwargs):
        captured.append((*[len(h) for h in fold_haystacks], len(final_scores)))
        return real(fold_haystacks, orderings, final_scores, **kwargs)

    monkeypatch.setattr(thresholds_mod, "fit_fold_anchored_cut", spy)
    return captured


class TestExclusionEqualsRemoval:
    """``voted_ids`` over the full snap == the same snap without those media."""

    def test_threshold_matches_a_snap_without_the_votes(self):
        vecs, good_ids, bad_ids = _fixture()
        X, y = _xy(vecs, good_ids, bad_ids)
        voted = set(good_ids) | set(bad_ids)
        rest = [i for i in range(len(vecs)) if i not in voted]

        _m1, with_exclusion = train_and_threshold(X, y, snap=_snap(vecs), voted_ids=voted)
        _m2, on_remainder = train_and_threshold(X, y, snap=_snap(vecs, rest))

        # Same votes, same folds, same models; the haystacks are the only
        # input left, and after the exclusion they are identical - so the
        # thresholds are too, bit for bit.
        assert with_exclusion == on_remainder

    def test_omitting_voted_ids_keeps_the_full_haystack(self, monkeypatch):
        vecs, good_ids, bad_ids = _fixture()
        X, y = _xy(vecs, good_ids, bad_ids)

        captured = _spy_fit_sizes(monkeypatch)
        train_and_threshold(X, y, snap=_snap(vecs))
        train_and_threshold(X, y, snap=_snap(vecs), voted_ids=set(good_ids) | set(bad_ids))

        full, excluded = captured
        n, n_voted = len(vecs), len(good_ids) + len(bad_ids)
        assert all(size == n for size in full)
        assert all(size == n - n_voted for size in excluded)


class TestTrainAndScorePassesVotes:
    """The vote-driven pipeline wires its vote ids through automatically."""

    def test_vote_ids_are_dropped_from_every_haystack(self, monkeypatch):
        vecs, good_ids, bad_ids = _fixture()
        snap = _snap(vecs)

        captured = _spy_fit_sizes(monkeypatch)
        results, threshold, model = train_and_score(snap, dict.fromkeys(good_ids), dict.fromkeys(bad_ids))

        assert model is not None
        assert np.isfinite(threshold)
        # Scoring still covers the whole corpus - only the fit population shrinks.
        assert len(results) == len(vecs)
        n_expected = len(vecs) - len(good_ids) - len(bad_ids)
        assert captured and all(size == n_expected for size in captured[-1])


class TestRemainderFloor:
    """Below ``EXCLUSION_MIN_REMAINDER`` leftover scores the exclusion switches
    off entirely - a drained remainder is too coarse and too selection-biased
    to be a population estimate, so the full haystack measures better there."""

    def test_small_remainder_keeps_the_full_haystack(self, monkeypatch):
        # 60 media, 16 votes: the remainder (44) is under the floor (50).
        vecs, good_ids, bad_ids = _fixture(n_pos=20, n_neg=40)

        captured = _spy_fit_sizes(monkeypatch)
        train_and_score(_snap(vecs), dict.fromkeys(good_ids), dict.fromkeys(bad_ids))

        assert len(vecs) - len(good_ids) - len(bad_ids) < thresholds_mod.EXCLUSION_MIN_REMAINDER
        assert captured and all(size == len(vecs) for size in captured[-1])

    def test_everything_voted_keeps_the_full_haystack(self):
        rng = np.random.default_rng(5)
        pos = rng.standard_normal((6, DIM)).astype(np.float32) + 1.0
        neg = rng.standard_normal((6, DIM)).astype(np.float32) - 1.0
        vecs = np.concatenate([pos, neg])
        good_ids, bad_ids = list(range(6)), list(range(6, 12))

        _results, threshold, model = train_and_score(_snap(vecs), dict.fromkeys(good_ids), dict.fromkeys(bad_ids))

        assert model is not None
        assert np.isfinite(threshold)


class TestLabeledMediaIds:
    """The labelset pipelines name their in-dataset labels by md5/origin match."""

    def test_matched_and_unmatched_elements(self):
        vecs, _good, _bad = _fixture(n_pos=4, n_neg=8)
        snap = _snap(vecs)
        labelset = LabelSet(
            [
                LabeledElement(md5="md5-2", label="good"),
                LabeledElement(md5="md5-7", label="bad"),
                LabeledElement(md5="not-in-dataset", label="good"),
                LabeledElement(md5="md5-3", label="skip"),
            ]
        )

        assert labeled_media_ids(labelset, snap) == {2, 7}
        assert labeled_media_ids(labelset, None) == set()
