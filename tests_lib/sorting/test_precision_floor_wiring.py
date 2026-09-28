"""The precision floor wired to a detector (#4245): evidence, operating point, re-cut, state.

``test_precision_floor.py`` pins the estimator itself.  This file pins what the
wiring adds around it:

* which votes may serve as evidence (``calibrates_precision``), and that the
  fold held-out rows map back to them (``holdout_sink``);
* that the curve is fitted once and re-cut per floor, with the same answer as
  a fresh ``precision_floor_cut``;
* which line an operating point draws (``reporting_line``) - the floor's when
  promised, the Inclusion 0 cut when not, the knob's when no floor is set;
* the detector-state seams: ``recut_detector_threshold`` with a floor,
  per-detector seeding, ``set_min_precision``.

Planted sessions come from ``test_precision_floor``: two Gaussians, votes
picked by score only, fold haystacks the pool with a little noise - squashed
through a sigmoid here, because the wiring (like the fold-anchored cut) keeps
only real sigmoid outputs and drops anything outside ``[0, 1]`` as unscorable.
The estimator reads ranks, so the squash changes no answer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests_lib.sorting.test_precision_floor import _session
from vtscore.datasets.vote_provenance import calibrates_precision
from vtscore.state.core import (
    DetectorContext,
    detector_line_inclusion,
    detector_precision_floor,
    recompute_detector_thresholds,
    recut_detector_threshold,
    register_detector_context,
    set_thread_detector_context,
)
from vtscore.training.thresholds import (
    DEFAULT_MIN_PRECISION,
    NO_PRECISION_FLOOR,
    PRECISION_FLOOR_FALLBACK_INCLUSION,
    PrecisionFloorEstimate,
    PrecisionFloorStatus,
    calibration_folds_cached,
    compute_fold_orderings,
    eligible_fold_orderings,
    fit_fold_anchored_cut,
    fit_precision_floor_curve,
    line_inclusion,
    precision_floor_cut,
    reporting_line,
    resolve_min_precision,
)

LEARNED_HARD = {"flow": "autopilot", "phase": "hard", "select_mode": "hard", "sort_kind": "learned"}


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


class TestWhichVotesCalibrate:
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
            # A learned-sort vote recorded before select_mode existed.
            {"flow": "list_review", "sort_kind": "learned"},
        ],
    )
    def test_everything_else_trains_but_does_not_calibrate(self, provenance):
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

    def test_only_eligible_rows_survive_and_folds_keep_their_places(self):
        orderings = [([0.9, 0.1, 0.8], [1.0, 0.0, 1.0]), ([0.7, 0.2], [1.0, 0.0])]
        holdouts = [(0, 1, 2), (3, 4)]
        eligible = [True, False, False, True, True]
        kept = eligible_fold_orderings(orderings, holdouts, eligible)
        assert kept == [([0.9], [1.0]), ([0.7, 0.2], [1.0, 0.0])]

    def test_no_provenance_keeps_every_vote_and_untraceable_folds_keep_none(self):
        orderings = [([0.9, 0.1], [1.0, 0.0])]
        assert eligible_fold_orderings(orderings, [], None) == [([0.9, 0.1], [1.0, 0.0])]
        # Rows missing, or not lining up with the ordering: nothing can be shown fair.
        assert eligible_fold_orderings(orderings, [], [True, True]) == [([], [])]
        assert eligible_fold_orderings(orderings, [(0,)], [True, True]) == [([], [])]


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
    def test_no_floor_is_the_inclusion_knob(self):
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
        assert origin is not None
        # The origin reproduces the line: the offset is measured from where the cut really sits.
        assert cut.threshold_at(origin) <= line.threshold


class TestResolvingTheArmKnob:
    def test_none_is_the_apps_default(self):
        assert resolve_min_precision(None) == DEFAULT_MIN_PRECISION

    def test_off_is_the_inclusion_arm(self):
        assert resolve_min_precision(NO_PRECISION_FLOOR) is None

    def test_a_number_pins_a_floor(self):
        assert resolve_min_precision(0.75) == 0.75

    @pytest.mark.parametrize("bad", [0.0, 1.5, "half", True])
    def test_anything_else_is_refused(self, bad):
        with pytest.raises(ValueError):
            resolve_min_precision(bad)


class TestTheDetectorsLine:
    def _ctx(self, detector_id: str = "det-floor") -> DetectorContext:
        cut, estimate = _fitted()
        ctx = DetectorContext(detector_id)
        ctx.anchored_cut_cache = cut
        ctx.precision_floor_cache = estimate
        ctx.threshold = cut.threshold_at(0)
        return ctx

    def test_recut_at_a_floor_or_an_inclusion(self):
        ctx = self._ctx()
        assert recut_detector_threshold(ctx, 4) == ctx.anchored_cut_cache.threshold_at(4)
        assert recut_detector_threshold(ctx, 4, min_precision=0.5) == ctx.precision_floor_cache.cut(0.5).threshold
        assert recut_detector_threshold(ctx, 4, min_precision=1.0) == ctx.anchored_cut_cache.threshold_at(0)

    def test_an_operating_point_needs_one_of_the_two(self):
        with pytest.raises(ValueError, match="operating point"):
            recut_detector_threshold(self._ctx())

    def test_the_verdict_without_an_estimate_is_no_evidence(self):
        verdict = detector_precision_floor(DetectorContext("det-bare"), 0.5)
        assert verdict.status is PrecisionFloorStatus.INSUFFICIENT_EVIDENCE

    def test_acquisition_origin_under_each_operating_point(self):
        ctx = self._ctx()
        assert detector_line_inclusion(ctx, 3, None) == 3
        assert detector_line_inclusion(ctx, 3, 1.0) == PRECISION_FLOOR_FALLBACK_INCLUSION
        assert detector_line_inclusion(ctx, 3, 0.5) is None

    def test_each_detector_keeps_its_own_floor(self):
        """As with Inclusion (#3416): a floor change moves only the detector that holds it and the unseeded."""
        mine = self._ctx("det-own-floor")
        mine.min_precision, mine.min_precision_seeded = None, True
        mine.inclusion = 2
        register_detector_context(mine)
        unseeded = self._ctx("det-unseeded-floor")
        register_detector_context(unseeded)

        recompute_detector_thresholds(0, 0.5)

        assert mine.threshold == mine.anchored_cut_cache.threshold_at(2), "no floor: its own inclusion"
        assert unseeded.threshold == unseeded.precision_floor_cache.cut(0.5).threshold


class TestTheSetting:
    def test_a_detector_seeds_its_floor_from_the_user_setting(self):
        from vtscore.state import get_min_precision

        ctx = DetectorContext("det-seed")
        set_thread_detector_context(ctx)
        assert get_min_precision() == DEFAULT_MIN_PRECISION
        assert ctx.min_precision_seeded and ctx.min_precision == DEFAULT_MIN_PRECISION

    @pytest.mark.usefixtures("no_precision_floor")
    def test_a_cleared_user_floor_seeds_none(self):
        from vtscore.state import get_min_precision

        set_thread_detector_context(DetectorContext("det-seed-none"))
        assert get_min_precision() is None

    def test_setting_it_persists_and_recuts_the_active_detector(self):
        from vtscore.state import get_min_precision, register_setting_persister, set_min_precision

        persisted: list = []
        register_setting_persister("min_precision", persisted.append)
        ctx = self._active_ctx()

        set_min_precision(0.5)
        assert persisted == [0.5] and get_min_precision() == 0.5
        assert ctx.threshold == ctx.precision_floor_cache.cut(0.5).threshold

        set_min_precision(None)
        assert persisted == [0.5, None] and get_min_precision() is None
        assert ctx.threshold == ctx.anchored_cut_cache.threshold_at(ctx.inclusion)

    def test_a_floor_outside_the_unit_interval_is_refused(self):
        from vtscore.state import set_min_precision

        for bad in (0.0, -0.1, 1.01):
            with pytest.raises(ValueError, match="floor"):
                set_min_precision(bad)

    def _active_ctx(self) -> DetectorContext:
        ctx = TestTheDetectorsLine()._ctx("det-active-floor")
        ctx.inclusion = 3
        register_detector_context(ctx)
        set_thread_detector_context(ctx)
        return ctx


def test_the_default_floor_is_the_one_the_owner_ruled():
    """Half of what the cut returns should be right (owner, 2026-09-28)."""
    assert DEFAULT_MIN_PRECISION == 0.5
    assert not math.isnan(DEFAULT_MIN_PRECISION)


class TestARetrainParksTheEstimate:
    """The whole pipeline: ``train_and_score`` builds the estimate beside the fold-anchored cut."""

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

    def _train(self, provenance: dict | None, **kwargs):
        from vtscore.detectors.training import train_and_score

        good = {cid: None for cid in range(500, 506)}
        bad = {cid: None for cid in range(506, 514)}
        ctx = DetectorContext("det-retrain-floor", media_type="audio")
        if provenance is not None:
            ctx.vote_provenance = {cid: provenance for cid in [*good, *bad]}
        _results, threshold, model = train_and_score(self._clips(), good, bad, det_ctx=ctx, **kwargs)
        assert model is not None and ctx.anchored_cut_cache is not None
        return ctx, threshold

    def test_votes_with_no_provenance_calibrate_nothing(self):
        ctx, _t = self._train(None)
        assert ctx.precision_floor_cache is not None
        assert ctx.precision_floor_cache.calibration_positives == 0

    def test_learned_sort_votes_are_the_evidence(self):
        ctx, _t = self._train(LEARNED_HARD)
        _key, folds, _holdouts = ctx.calibration_cache
        held_out_positives = int(sum(sum(labels) for _scores, labels in folds.orderings))
        assert held_out_positives > 0
        assert ctx.precision_floor_cache.calibration_positives == held_out_positives

    def test_text_sort_votes_train_but_do_not_calibrate(self):
        text = {"flow": "autopilot", "phase": "good", "select_mode": "top", "sort_kind": "text"}
        ctx, _t = self._train(text)
        assert ctx.precision_floor_cache.calibration_positives == 0

    def test_an_unmet_floor_cuts_where_inclusion_zero_does(self):
        ctx, threshold = self._train(LEARNED_HARD, inclusion_value=6, min_precision=0.5)
        assert ctx.precision_floor_cache.cut(0.5).status is not PrecisionFloorStatus.PROMISED
        assert threshold == ctx.anchored_cut_cache.threshold_at(0)
