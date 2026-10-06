"""A detector's line wired up (#4245, #4272, #4362, #4413): the estimator's seams, the re-cut, a retrain.

``test_precision_floor.py`` pins the #4220 estimator itself,
``test_spot_check.py`` and ``test_balance.py`` the spot check and the balance
that draw the line, and ``test_balance_wiring.py`` the balance as a setting.
This file pins what the wiring adds around them:

* that the fold held-out rows map back to their training rows
  (``holdout_sink``), which the eval harness's precision frames read;
* that the estimator's curve is fitted once and re-cut per floor, with the
  same answer as a fresh ``precision_floor_cut``;
* ``reporting_line``, the library's own account of the estimator's line, kept
  as public API off the app's path;
* the detector-state seams the balance leaves to the inclusion: a re-cut at an
  inclusion, the human votes, and a retrain parking the ranking and drawing
  the balance's line (and, since #4362, no estimate);
* the retired #4245 calibration filter and the estimate's parking: every
  public name survives, answers as documented, and warns.

Planted sessions come from ``test_precision_floor``: two Gaussians, votes
picked by score only, fold haystacks the pool with a little noise - squashed
through a sigmoid here, because the wiring (like the fold-anchored cut) keeps
only real sigmoid outputs and drops anything outside ``[0, 1]`` as unscorable.
The estimator reads ranks, so the squash changes no answer.
"""

from __future__ import annotations


import numpy as np
import pytest

from tests_lib.sorting.test_precision_floor import _session
from vtscore.datasets.vote_provenance import calibrates_precision
from vtscore.state.core import (
    DetectorContext,
    detector_balance_state,
    detector_precision_floor,
    human_voted_ids,
    recut_detector_threshold,
)
from vtscore.training.thresholds import (
    PRECISION_FLOOR_FALLBACK_INCLUSION,
    LineRanking,
    PrecisionFloorEstimate,
    PrecisionFloorStatus,
    SpotCheck,
    calibration_folds_cached,
    compute_fold_orderings,
    eligible_fold_orderings,
    fit_fold_anchored_cut,
    fit_precision_floor_curve,
    line_inclusion,
    precision_floor_cut,
    reporting_line,
    walk_positives,
)

LEARNED_HARD: dict[str, object] = {"flow": "autopilot", "phase": "hard", "select_mode": "hard", "sort_kind": "learned"}


def _sigmoid(a) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(a, dtype=np.float64)))


def _scored_session(seed: int, **kwargs):
    """``test_precision_floor._session`` on the sigmoid scale a detector's scores live on."""
    corpus_s, corpus_y, pool_s, orderings, haystacks = _session(seed, **kwargs)
    return (
        _sigmoid(corpus_s),
        corpus_y,
        _sigmoid(pool_s),
        [(_sigmoid(sc).tolist(), list(lb)) for sc, lb in orderings],
        [_sigmoid(h) for h in haystacks],
    )


class TestTheRetiredFilterStillAnswers:
    """``calibrates_precision`` (#4245) is deprecated (#4362) but keeps its rule, with a warning."""

    @pytest.mark.parametrize(
        "provenance",
        [
            LEARNED_HARD,
            {"flow": "autopilot", "phase": "good", "select_mode": "top", "sort_kind": "learned"},
            {"flow": "list_review", "select_mode": "top", "sort_kind": "learned"},
            {"flow": "list_review", "select_mode": "hard", "sort_kind": "learned", "rank_at_vote": 4},
        ],
    )
    def test_a_draw_off_the_learned_ranking_calibrates(self, provenance):
        with pytest.deprecated_call(match="calibrates_precision"):
            assert calibrates_precision(provenance)

    @pytest.mark.parametrize(
        "provenance",
        [
            None,
            {},
            # Autopilot's opening walks the text sort: #4222's broken promises.
            {"flow": "autopilot", "phase": "good", "select_mode": "top", "sort_kind": "text"},
            {"flow": "autopilot", "phase": "bad", "select_mode": "hard", "sort_kind": "load"},
            # The atlas picks the node on coverage: not score-only (owner, 2026-09-28).
            {"flow": "autopilot", "phase": "new", "select_mode": "new", "sort_kind": "learned"},
            # Flows that are not a draw, or that select on the label.
            {"flow": "find_verify", "select_mode": "top", "sort_kind": "learned"},
            {"flow": "labelset_review", "select_mode": "top", "sort_kind": "learned"},
            {"flow": "bulk", "select_mode": "top", "sort_kind": "learned"},
            {"flow": "undo", "select_mode": "hard", "sort_kind": "learned"},
            {"flow": "import"},
            {"flow": "seed_example"},
            # A spot check's pick (#4272): uniform within the candidate, not a
            # draw off the ranking's head or its cutoff band.
            {"flow": "check"},
            # A learned-sort vote recorded before select_mode existed.
            {"flow": "list_review", "sort_kind": "learned"},
        ],
    )
    def test_everything_else_does_not_calibrate(self, provenance):
        with pytest.deprecated_call(match="calibrates_precision"):
            assert not calibrates_precision(provenance)


class TestHeldOutRowsMapBackToVotes:
    def _xy(self, n: int = 24, dim: int = 6):
        rng = np.random.default_rng(42)
        y = [1.0 if i % 3 == 0 else 0.0 for i in range(n)]
        X = [rng.standard_normal(dim).astype(np.float32) + (2.0 if lab else 0.0) for lab in y]
        return X, y

    def test_row_wise_holdouts_line_up_with_each_ordering(self):
        X, y = self._xy()
        holdouts: list = []
        orderings, fallback = compute_fold_orderings(X, y, 6, calibrate_count=3, holdout_sink=holdouts)
        assert fallback is None
        assert len(holdouts) == len(orderings) == 3
        for (scores, labels), rows in zip(orderings, holdouts, strict=True):
            assert len(rows) == len(scores)
            assert labels == [y[r] for r in rows]

    def test_the_sink_is_read_only(self):
        """Asking for the held-out rows changes no split, fit or ordering."""
        X, y = self._xy()
        plain, _ = compute_fold_orderings(X, y, 6, calibrate_count=2)
        sunk, _ = compute_fold_orderings(X, y, 6, calibrate_count=2, holdout_sink=[])
        assert plain == sunk

    def test_grouped_holdouts_name_one_row_of_each_bag(self):
        X, y = self._xy()
        # Every Bad vote floods three rows; a Good vote is one.
        groups: list = []
        X_rows, y_rows = [], []
        for i, (vec, lab) in enumerate(zip(X, y, strict=True)):
            copies = 1 if lab else 3
            for _ in range(copies):
                X_rows.append(vec)
                y_rows.append(lab)
                groups.append(("g" if lab else "b", i))
        holdouts: list = []
        orderings, fallback = compute_fold_orderings(
            X_rows, y_rows, 6, calibrate_count=2, groups=groups, holdout_sink=holdouts
        )
        assert fallback is None
        for (scores, labels), rows in zip(orderings, holdouts, strict=True):
            assert len(rows) == len(scores)
            assert labels == [y_rows[r] for r in rows]
            assert len({groups[r] for r in rows}) == len(rows), "one row per held-out bag"

    def test_a_cache_hit_hands_back_the_same_rows(self):
        X, y = self._xy()
        ctx = DetectorContext("det-holdouts")
        first: list = []
        calibration_folds_cached(
            X, y, 6, calibrate_count=2, calibration_fraction=0.5, hidden_dim=0, det_ctx=ctx, holdout_sink=first
        )
        again: list = []
        calibration_folds_cached(
            X, y, 6, calibrate_count=2, calibration_fraction=0.5, hidden_dim=0, det_ctx=ctx, holdout_sink=again
        )
        assert first and again == first

    def test_the_retired_eligibility_filter_still_answers_with_a_warning(self):
        """``eligible_fold_orderings`` is deprecated (#4362) but keeps its rule."""
        orderings = [([0.9, 0.1, 0.8], [1.0, 0.0, 1.0]), ([0.7, 0.2], [1.0, 0.0])]
        with pytest.deprecated_call(match="eligible_fold_orderings"):
            kept = eligible_fold_orderings(orderings, [(0, 1, 2), (3, 4)], [True, False, False, True, True])
        assert kept == [([0.9], [1.0]), ([0.7, 0.2], [1.0, 0.0])]
        with pytest.deprecated_call():
            assert eligible_fold_orderings(orderings[:1], [], None) == [([0.9, 0.1, 0.8], [1.0, 0.0, 1.0])]
        with pytest.deprecated_call():
            assert eligible_fold_orderings(orderings[:1], [(0,)], [True]) == [([], [])]


class TestFittedOnceCutAtAnyFloor:
    def test_the_curve_cuts_exactly_as_a_fresh_call(self):
        corpus_s, _y, pool_s, orderings, haystacks = _scored_session(2)
        curve = fit_precision_floor_curve(corpus_s, pool_s, orderings, haystacks, min_positives=5)
        for floor in (0.25, 0.5, 0.75, 0.99):
            assert curve.cut(floor) == precision_floor_cut(
                floor, corpus_s, pool_s, orderings, haystacks, min_positives=5
            ), floor

    def test_the_estimate_fits_on_first_use_and_counts_the_whole_corpus(self):
        corpus_s, _y, _pool, orderings, haystacks = _scored_session(3, same_corpus=True)
        estimate = PrecisionFloorEstimate(corpus_s, orderings, haystacks, min_positives=5)
        assert estimate._curve is None, "nothing is fitted until a floor is asked for"
        verdict = estimate.cut(0.5)
        assert estimate._curve is not None
        assert verdict == precision_floor_cut(0.5, corpus_s, corpus_s, orderings, haystacks, min_positives=5)
        assert verdict.threshold is not None
        assert estimate.count_at(verdict.threshold) == verdict.n_returned

    def test_above_the_sample_cap_the_count_is_still_the_whole_corpus(self):
        _c, _y, pool_s, orderings, haystacks = _scored_session(4)
        rng = np.random.default_rng(42)
        big = np.concatenate([pool_s, _sigmoid(rng.normal(0.0, 1.0, 60_000))])
        estimate = PrecisionFloorEstimate(big, orderings, haystacks, min_positives=5)
        assert estimate.corpus_size == big.size
        assert estimate.count_at(0.5) == int(np.count_nonzero(big >= 0.5))

    def test_positives_are_counted_without_fitting(self):
        _c, _y, pool_s, orderings, haystacks = _scored_session(5)
        estimate = PrecisionFloorEstimate(pool_s, orderings, haystacks)
        assert estimate.calibration_positives == int(sum(sum(lb) for _s, lb in orderings))
        assert estimate._curve is None


def _fitted(seed: int = 6):
    """A planted session's fold-anchored cut and precision-floor estimate, as a retrain parks them."""
    _c, _y, pool_s, orderings, haystacks = _scored_session(seed, same_corpus=True)
    cut = fit_fold_anchored_cut([np.asarray(h) for h in haystacks], orderings, pool_s)
    assert cut is not None
    return cut, PrecisionFloorEstimate(pool_s, orderings, haystacks)


class TestWhichLineAnOperatingPointDraws:
    """``reporting_line``: the estimator's own line, library API off the app's path since #4272.

    The app hands it no estimate any more (the spot check keeps the set); what
    it says at a floor is still the estimator's verdict, for a library caller
    that wants one.
    """

    def test_no_floor_cuts_at_the_inclusion_it_is_given(self):
        """The internal unit: the app gives Inclusion 0, a re-cut another (#4269)."""
        cut, estimate = _fitted()
        line = reporting_line(cut, estimate, inclusion_value=3, min_precision=None)
        assert line.threshold == cut.threshold_at(3)
        assert line.inclusion == 3 and line.floor is None

    def test_a_promised_floor_draws_its_own_line_whatever_inclusion_says(self):
        cut, estimate = _fitted()
        verdict = estimate.cut(0.5)
        assert verdict.status is PrecisionFloorStatus.PROMISED
        for k in (-5, 0, 5):
            line = reporting_line(cut, estimate, inclusion_value=k, min_precision=0.5)
            assert line.threshold == verdict.threshold and line.floor == verdict
            assert line.inclusion is None, "a promised line was not drawn at any inclusion"

    def test_a_floor_that_promises_nothing_falls_back_to_inclusion_zero(self):
        cut, estimate = _fitted()
        line = reporting_line(cut, estimate, inclusion_value=7, min_precision=1.0)
        assert line.floor is not None and line.floor.status is not PrecisionFloorStatus.PROMISED
        assert line.threshold == cut.threshold_at(PRECISION_FLOOR_FALLBACK_INCLUSION)
        assert line.inclusion == PRECISION_FLOOR_FALLBACK_INCLUSION

    def test_no_estimate_is_no_evidence(self):
        cut, _estimate = _fitted()
        line = reporting_line(cut, None, inclusion_value=2, min_precision=0.5)
        assert line.floor is not None
        assert line.floor.status is PrecisionFloorStatus.INSUFFICIENT_EVIDENCE
        assert line.floor.calibration_positives == 0
        assert line.threshold == cut.threshold_at(0)

    def test_no_fitted_cut_leaves_the_line_to_the_caller(self):
        line = reporting_line(None, None, inclusion_value=0, min_precision=0.5)
        assert line.threshold is None

    def test_acquisition_derives_its_origin_from_a_promised_line(self):
        cut, estimate = _fitted()
        line = reporting_line(cut, estimate, inclusion_value=0, min_precision=0.5)
        origin = line_inclusion(line, cut)
        assert origin is not None and line.threshold is not None
        # The origin reproduces the line: the offset is measured from where the cut really sits.
        assert cut.threshold_at(origin) <= line.threshold


def _planted_ranking(n: int = 200, voted: set[int] | None = None) -> LineRanking:
    """A haystack of *n* items on a strict descending ladder, rank == id - 1."""
    return LineRanking.from_scores(list(range(1, n + 1)), np.linspace(0.99, 0.01, n), voted or set())


class TestTheDetectorsLine:
    def _ctx(self, detector_id: str = "det-line", n: int = 200) -> DetectorContext:
        cut, _estimate = _fitted()
        ctx = DetectorContext(detector_id)
        ctx.anchored_cut_cache = cut
        ctx.line_ranking = _planted_ranking(n)
        ctx.threshold = cut.threshold_at(0)
        return ctx

    def test_recut_at_an_inclusion_reads_the_anchored_cut(self):
        ctx = self._ctx()
        assert recut_detector_threshold(ctx, 4) == ctx.anchored_cut_cache.threshold_at(4)

    def test_an_operating_point_needs_one_of_the_two(self):
        with pytest.raises(ValueError, match="operating point"):
            recut_detector_threshold(self._ctx())

    def test_the_retired_estimators_verdict_is_always_no_evidence(self):
        """``detector_precision_floor`` is deprecated (#4362): nothing parks an estimate for it to read."""
        with pytest.deprecated_call(match="detector_precision_floor"):
            verdict = detector_precision_floor(DetectorContext("det-bare"), 0.5)
        assert verdict.status is PrecisionFloorStatus.INSUFFICIENT_EVIDENCE
        assert verdict.calibration_positives == 0 and verdict.threshold is None
        # Even an estimate an old caller planted is no longer read.
        planted = DetectorContext("det-planted")
        planted.precision_floor_cache = _fitted()[1]
        with pytest.deprecated_call():
            assert detector_precision_floor(planted, 0.5).status is PrecisionFloorStatus.INSUFFICIENT_EVIDENCE

    def test_the_human_votes_are_the_verified_ones_in_find_mode(self):
        ctx = self._ctx()
        ctx.good_votes.update({1: None, 2: None})
        ctx.bad_votes[3] = None
        assert human_voted_ids(ctx) == {1, 2, 3}
        ctx.find_mode = True
        ctx.verified_ids[2] = None
        assert human_voted_ids(ctx) == {2}, "every Find item carries a machine label; only the verified are votes"


class TestARetrain:
    """The whole pipeline: ``train_and_score`` parks the fold-anchored cut and the ranking, and no estimate (#4362)."""

    def _clips(self) -> dict[int, dict]:
        rng = np.random.default_rng(7)
        return {
            cid: {
                "embeddings": {"test": (rng.standard_normal(8) + (1.5 if cid < 506 else 0.0)).astype(np.float32)},
                "embedder": "test",
                "media_type": "audio",
                "md5": f"m{cid:031d}",
            }
            for cid in range(500, 530)
        }

    def _train(self, det_ctx_check: SpotCheck | None = None, **kwargs):
        from vtscore.detectors.training import train_and_score

        good = {cid: None for cid in range(500, 506)}
        bad = {cid: None for cid in range(506, 514)}
        ctx = DetectorContext("det-retrain-line", media_type="audio")
        ctx.precision_check = det_ctx_check
        ctx.vote_provenance = {cid: LEARNED_HARD for cid in [*good, *bad]}
        _results, threshold, model = train_and_score(self._clips(), good, bad, det_ctx=ctx, **kwargs)
        assert model is not None and ctx.anchored_cut_cache is not None
        return ctx, threshold

    def test_a_retrain_parks_no_estimate(self):
        """The #4220 estimate lost its last reader in #4360, so a retrain no longer builds it.

        The votes carry learned-sort provenance, which is what used to make them its evidence.
        """
        ctx, _t = self._train()
        assert ctx.precision_floor_cache is None

    def test_a_retrain_parks_the_ranking_and_cuts_at_the_labels_line(self):
        """The line is the labels' line (#4452); the ranking it is read against is parked, voted items marked (#4272)."""
        ctx, threshold = self._train(beta=1.0)
        ranking = ctx.line_ranking
        assert ranking is not None and ranking.size == 30
        assert ranking.voted == frozenset(range(500, 514))
        assert ctx.labels_line is not None, "the fixture's votes separate, so the folds fit a class model"
        assert threshold == pytest.approx(ctx.labels_line.threshold(1.0), abs=1e-12)
        assert recut_detector_threshold(ctx, beta=1.0) == pytest.approx(threshold, abs=1e-12)

    def test_the_balances_state_counts_what_the_labels_line_keeps(self):
        """No count is drawn on the ranking (#4452): the state reports how many unvoted items clear the line."""
        ctx, threshold = self._train(beta=1.0)
        ctx.threshold = threshold  # what the caller of a retrain stores, as the app does
        ranking = ctx.line_ranking
        assert ranking is not None
        unvoted = ranking.unvoted_ids(set())
        kept = sum(1 for cid in unvoted if ranking.score_of(int(cid)) >= threshold)
        assert detector_balance_state(ctx, 1.0)["count"] == kept
        assert ranking.mixture == [], "the count line's mixture is never fitted under the labels' line"

    def test_a_finished_walk_never_moves_the_labels_line(self):
        """Every preset's check is advisory since #4452: a retrain after a walk still cuts at the labels' line."""
        ctx, _t = self._train(beta=2.0)
        ranking = ctx.line_ranking
        assert ranking is not None
        unvoted = tuple(int(i) for i in ranking.unvoted_ids())
        check = SpotCheck.start_balance(unvoted, 2.0, walk_positives(ranking, 2.0, {}), seed=1)
        while check.running:
            check.record({cid: cid < 520 for cid in check.pending})
        check.fingerprint = ranking.fingerprint(check.k, set(check.labels))
        ctx2, threshold = self._train(beta=2.0, det_ctx_check=check)
        assert ctx2.labels_line is not None
        assert threshold == pytest.approx(ctx2.labels_line.threshold(2.0), abs=1e-12)
        assert recut_detector_threshold(ctx2, beta=2.0) == pytest.approx(threshold, abs=1e-12)


class TestTheRetiredTrainingFilter:
    """The #4245 filter's training-side names are deprecated (#4362): kept, warning, and read by nothing."""

    def _xy_snap(self):
        rng = np.random.default_rng(5)
        X = [rng.standard_normal(8).astype(np.float32) + (1.5 if i < 6 else 0.0) for i in range(14)]
        y = [1.0] * 6 + [0.0] * 8
        snap = {
            i: {
                "embeddings": {"test": rng.standard_normal(8).astype(np.float32)},
                "embedder": "test",
                "md5": f"m{i:031d}",
            }
            for i in range(40)
        }
        return X, y, snap

    def test_calibrating_groups_warns_and_changes_nothing(self):
        from vtscore.detectors.training import train_and_threshold

        X, y, snap = self._xy_snap()
        groups = [("g" if lab else "b", i) for i, lab in enumerate(y)]
        plain = DetectorContext("det-no-filter")
        _m, expected = train_and_threshold(X, y, snap=snap, det_ctx=plain, groups=groups)
        filtered = DetectorContext("det-label-file")
        with pytest.deprecated_call(match="calibrating_groups"):
            _m, threshold = train_and_threshold(
                X, y, snap=snap, det_ctx=filtered, groups=groups, calibrating_groups=set()
            )
        assert threshold == expected
        assert plain.precision_floor_cache is None and filtered.precision_floor_cache is None

    def test_the_row_and_bag_helpers_still_answer(self):
        from vtscore.detectors.training import calibration_rows_for, vote_calibrating_groups

        groups = [("g", 1), ("b", 2), ("b", 3)]
        with pytest.deprecated_call(match="calibration_rows_for"):
            assert calibration_rows_for(groups, {("g", 1), ("b", 3)}) == [True, False, True]
        with pytest.deprecated_call():
            assert calibration_rows_for(groups, None) is None
        with pytest.deprecated_call():
            assert calibration_rows_for(None, set()) == []
        text = {"flow": "autopilot", "phase": "good", "select_mode": "top", "sort_kind": "text"}
        with pytest.deprecated_call(match="vote_calibrating_groups"):
            bags = vote_calibrating_groups({1: None, 2: None}, {3: None}, {1: LEARNED_HARD, 2: text, 3: LEARNED_HARD})
        assert bags == {("g", 1), ("b", 3)}

    def test_the_labelset_helper_still_answers(self):
        from vtscore.datasets.labelset import LabeledElement, LabelSet
        from vtscore.datasets.vote_provenance import attach_provenance
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import labelset_calibrating_groups

        learned = LabeledElement(md5="a" * 32, label="good", metadata=attach_provenance({}, LEARNED_HARD))
        unattributed = LabeledElement(md5="b" * 32, label="bad")
        labelset = LabelSet(elements=[learned, unattributed])
        with pytest.deprecated_call(match="labelset_calibrating_groups"):
            assert labelset_calibrating_groups(labelset) == {("g", stable_element_id(learned))}
