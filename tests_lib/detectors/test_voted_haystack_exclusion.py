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

import math

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
        # 60 media, 16 votes: the remainder (44) is under the floor (60).
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


class TestSharedExclusionPolicy:
    """``apply_vote_exclusion`` / ``drop_voted`` are the one definition of the
    #3308 population convention - the app and the eval harness both route
    through them, so the harness's default arm cannot drift from production."""

    def test_resolve_floor_is_three_state(self):
        assert thresholds_mod.resolve_exclusion_floor(None) == float(thresholds_mod.EXCLUSION_MIN_REMAINDER)
        assert thresholds_mod.resolve_exclusion_floor(0) == 0.0
        assert thresholds_mod.resolve_exclusion_floor(math.inf) == math.inf

    def test_drop_voted_filters_by_this_arrays_own_ids(self):
        # Deliberately unsorted ids: the harness's region scorer returns rows in
        # score order, so a positional mask computed elsewhere would be wrong.
        scores = [0.9, 0.1, 0.5, 0.3]
        ids = [7, 2, 5, 1]
        assert thresholds_mod.drop_voted(scores, ids, {2, 1}).tolist() == [0.9, 0.5]

    def test_applies_above_the_floor(self):
        scores = list(np.linspace(0.0, 1.0, 100))
        ids = list(range(100))
        kept, applied = thresholds_mod.apply_vote_exclusion(scores, ids, {0, 1, 2}, min_remainder=10)
        assert applied and kept.size == 97

    def test_declines_below_the_floor_and_returns_the_whole_haystack(self):
        scores = list(np.linspace(0.0, 1.0, 100))
        ids = list(range(100))
        kept, applied = thresholds_mod.apply_vote_exclusion(scores, ids, set(range(50)), min_remainder=60)
        assert not applied and kept.size == 100

    def test_inf_floor_is_the_pre_3308_baseline(self):
        scores, ids = [0.1, 0.2, 0.3], [1, 2, 3]
        kept, applied = thresholds_mod.apply_vote_exclusion(scores, ids, {1}, min_remainder=math.inf)
        assert not applied and kept.tolist() == scores

    def test_zero_floor_still_refuses_an_empty_haystack(self):
        # Everything voted: a remainder of nothing is not a population estimate,
        # so even an unconditional floor declines rather than fitting on air.
        scores, ids = [0.1, 0.2], [1, 2]
        kept, applied = thresholds_mod.apply_vote_exclusion(scores, ids, {1, 2}, min_remainder=0)
        assert not applied and kept.tolist() == scores

    def test_nothing_voted_is_never_an_exclusion(self):
        scores, ids = [0.1, 0.2], [1, 2]
        for voted in (None, set()):
            kept, applied = thresholds_mod.apply_vote_exclusion(scores, ids, voted, min_remainder=0)
            assert not applied and kept.tolist() == scores


class TestHarnessArmKnob:
    """#3312: the eval harness can sweep the floor, and its *default* is the app's."""

    def test_simulate_exposes_the_knob_defaulting_to_none(self):
        import inspect

        from vtscore.eval.voting_iterations import simulate_voting_iterations

        param = inspect.signature(simulate_voting_iterations).parameters["exclusion_min_remainder"]
        assert param.default is None, "the default arm must resolve through the app's own floor"

    def test_off_arm_reproduces_the_pre_3308_haystack(self, monkeypatch):
        """``inf`` puts every haystack back on the full population."""
        vecs, good_ids, bad_ids = _fixture()
        X, y = _xy(vecs, good_ids, bad_ids)
        voted = set(good_ids) | set(bad_ids)

        captured = _spy_fit_sizes(monkeypatch)
        # The app has no floor override, so drive the policy directly: this is
        # the exact call `_safe_threshold_for_step` makes for the `off` arm.
        train_and_threshold(X, y, snap=_snap(vecs), voted_ids=voted)
        excluded_sizes = captured[-1]

        ids = list(range(len(vecs)))
        scores = list(np.linspace(0.0, 1.0, len(vecs)))
        off, off_applied = thresholds_mod.apply_vote_exclusion(scores, ids, voted, min_remainder=math.inf)
        on, on_applied = thresholds_mod.apply_vote_exclusion(scores, ids, voted, min_remainder=None)

        assert not off_applied and off.size == len(vecs)
        assert on_applied and on.size == len(vecs) - len(voted)
        assert all(size == on.size for size in excluded_sizes)


class TestArmSemanticsEndToEnd:
    """#3312: a full simulated run under each arm, which is what the GRID
    submits.  These pin the two things the study's validity rests on - the
    default arm IS production, and the arms are actually distinguishable."""

    @staticmethod
    def _medias(dim: int = 16, n_per_cat: int = 60) -> dict[int, dict]:
        rng = np.random.RandomState(0)
        out: dict[int, dict] = {}
        mid = 1
        for cat, centre in (("alpha", 0.3), ("beta", -0.3)):
            for _ in range(n_per_cat):
                vec = rng.normal(centre, 1.0, dim).astype(np.float32)
                out[mid] = {"id": mid, "embeddings": {"emb": vec}, "category": cat}
                mid += 1
        return out

    @staticmethod
    def _run(floor):
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        return simulate_voting_iterations(
            TestArmSemanticsEndToEnd._medias(),
            "alpha",
            seed=42,
            sim_fraction=0.5,
            exclusion_min_remainder=floor,
        )

    @staticmethod
    def _cuts(rows) -> list[float]:
        return [round(r["acq_threshold"], 9) for r in rows]

    def test_default_arm_is_the_shipped_floor(self):
        """The load-bearing one: `None` must be byte-identical to the app's floor.

        If this ever fails, every arm in the #3312 grid is being measured
        against a baseline no user runs - the exact failure the eval/app sync
        gate exists to prevent, here asserted on behaviour rather than on a
        source digest.
        """
        assert self._cuts(self._run(None)) == self._cuts(self._run(float(thresholds_mod.EXCLUSION_MIN_REMAINDER)))

    def test_unconditional_exclusion_is_distinguishable_from_the_floor(self):
        # This environment's haystack (60) never clears the floor, so `always`
        # is the only arm that can fire - which is precisely why it is the arm
        # that measured harmful here, and why the floor exists.
        assert self._cuts(self._run(0.0)) != self._cuts(self._run(None))

    def test_a_floor_above_the_haystack_reproduces_the_off_arm(self):
        assert self._cuts(self._run(250.0)) == self._cuts(self._run(math.inf))

    def test_rows_carry_the_haystack_and_remainder(self):
        rows = self._run(None)
        assert rows, "the fixture must train at least one step"
        first = rows[0]
        assert first["n_haystack"] == 60
        # The remainder shrinks by one per vote and is what the floor reads.
        remainders = [r["n_remainder"] for r in rows]
        assert remainders == sorted(remainders, reverse=True)
        assert max(remainders) < first["n_haystack"]
