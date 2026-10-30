"""The face crops' ball (#5478): the Goods' centroid cut at a cosine floor, from beta 1 up with two Goods.

On FHIBE's faces the centroid of a person's Goods and one cosine floor per balance, moved by how spread out
the Goods sit, scored above the trained head's line over a session from beta 1 up
(``docs/experiments/2026-10-30-face-ball-line-5478``).  These tests pin the rule
(:data:`~vtscore.detectors.centroid_head.FACE_BALL_RULE`), the measurement rules it was priced with, the tier
that hands it out past the quota, the cached head Find reuses, and the harness's mirror of both.
"""

from __future__ import annotations

import numpy as np
import pytest

import vtscore.detectors.labelset_training as lt
from tests_lib.detectors.test_label_quota import DIM, EMBEDDER, _unit, labels  # noqa: F401
from vtscore.detectors.centroid_head import (
    CENTROID_LINE_RULES,
    FACE_BALL_BELOW,
    FACE_BALL_CENTRE,
    FACE_BALL_FLOORS,
    FACE_BALL_RULE,
    FACE_BALL_SLOPE,
    ball_rule,
    centroid_cut,
    centroid_line_rule,
    face_ball_floor,
    face_ball_holds,
    fit_centroid,
    goods_spread,
    is_centroid_head,
    is_centroid_line_rule,
)
from vtscore.detectors.label_quota import TIER_CENTROID, TIER_NONE, TIER_TRAINED
from vtscore.detectors.labelset_heads import face_ball_tier
from vtscore.state.core import DetectorContext


def _cosines(seed: int = 0) -> np.ndarray:
    """A person's cosines to their Goods' centroid: a bulk, a few look-alikes, four own photos."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.02, 0.07, 5000), rng.uniform(0.3, 0.4, 6), rng.uniform(0.5, 0.7, 4)])


class TestTheRule:
    def test_faces_draw_the_ball_and_every_other_type_keeps_the_apps_rule(self):
        assert centroid_line_rule("face") == FACE_BALL_RULE
        assert FACE_BALL_RULE in CENTROID_LINE_RULES
        assert centroid_line_rule("image") != FACE_BALL_RULE

    def test_from_beta_1_with_a_spread_it_is_the_floor_moved_by_the_spread(self):
        cos = _cosines()
        for beta, floor in FACE_BALL_FLOORS.items():
            for spread in (0.55, FACE_BALL_CENTRE, 0.75):
                want = floor + FACE_BALL_SLOPE * (spread - FACE_BALL_CENTRE)
                assert centroid_cut(cos, rule=FACE_BALL_RULE, beta=beta, spread=spread) == pytest.approx(want)

    def test_below_beta_1_or_with_one_good_it_is_the_z55_count(self):
        cos = _cosines()
        for beta, spread in ((0.25, 0.65), (0.5, 0.65), (1.0, None), (4.0, None), (None, 0.65)):
            got = centroid_cut(cos, rule=FACE_BALL_RULE, beta=beta, spread=spread)
            assert got == centroid_cut(cos, rule=FACE_BALL_BELOW, beta=beta)

    def test_the_floor_falls_with_the_balance_in_log_beta(self):
        spread = FACE_BALL_CENTRE
        assert face_ball_floor(1.0, spread) == pytest.approx(FACE_BALL_FLOORS[1.0])
        assert face_ball_floor(4.0, spread) == pytest.approx(FACE_BALL_FLOORS[4.0])
        assert face_ball_floor(2.0, spread) == pytest.approx((FACE_BALL_FLOORS[1.0] + FACE_BALL_FLOORS[4.0]) / 2)
        assert face_ball_floor(16.0, spread) < face_ball_floor(4.0, spread)
        assert face_ball_floor(0.5, spread) == pytest.approx(FACE_BALL_FLOORS[1.0])

    def test_it_holds_from_beta_1_with_two_goods(self):
        assert face_ball_holds(1.0, 2) and face_ball_holds(4.0, 7)
        assert not face_ball_holds(0.25, 5) and not face_ball_holds(1.0, 1) and not face_ball_holds(None, 5)


class TestTheMeasurementBalls:
    def test_a_ball_rule_names_its_numbers(self):
        assert ball_rule("ball0.41") == (0.41, 0.0, 0.0)
        assert ball_rule("ball0.4425_s0.25_c0.6371") == (0.4425, 0.25, 0.6371)
        assert ball_rule("count_gap") is None and ball_rule("ball") is None
        assert is_centroid_line_rule("ball0.41") and is_centroid_line_rule("count_gap")
        assert not is_centroid_line_rule("ball0.41_s0.25")

    def test_a_ball_keeps_what_clears_its_floor(self):
        cos = _cosines()
        assert centroid_cut(cos, rule="ball0.41", beta=1.0) == pytest.approx(0.41)
        assert centroid_cut(cos, rule="ball0.4_s0.5_c0.6", beta=1.0, spread=0.7) == pytest.approx(0.45)
        assert centroid_cut(cos, rule="ball0.4_s0.5_c0.6", beta=1.0, spread=None) == pytest.approx(0.4)

    def test_the_variants_parse_a_ball(self):
        from vtscore.eval.voting_iterations import parse_centroid_line_variants

        assert parse_centroid_line_variants("ball0.41@1,ball0.44_s0.25_c0.63@4") == [
            ("ball0.41", 1.0),
            ("ball0.44_s0.25_c0.63", 4.0),
        ]

    def test_the_spread_is_each_goods_cosine_to_the_others(self):
        e = np.eye(3)
        assert goods_spread([e[0]]) is None
        assert goods_spread([e[0], e[0]]) == pytest.approx(1.0)
        assert goods_spread([e[0], e[1]]) == pytest.approx(0.0)
        # Three orthogonal Goods: each is at 0 to the mean of the other two.
        assert goods_spread(list(e)) == pytest.approx(0.0)


class TestTheFittedHead:
    def test_the_head_keeps_the_faces_at_or_above_the_floor_and_redraws_below_beta_1(self):
        rng = np.random.default_rng(5478)
        proto = _unit(rng.standard_normal(DIM))
        goods = [_unit(proto + 0.3 * rng.standard_normal(DIM)) for _ in range(4)]
        corpus = np.stack([_unit(proto + s * rng.standard_normal(DIM)) for s in [0.3] * 5 + [2.0] * 300])

        def score(probe):
            import torch

            with torch.no_grad():
                return torch.sigmoid(probe(torch.from_numpy(corpus.astype(np.float32)))).squeeze(1).tolist()

        head, _thr, line = fit_centroid(goods, score, rule=FACE_BALL_RULE, beta=1.0)
        assert line.spread is not None and line.spread == pytest.approx(goods_spread(goods))
        floor = face_ball_floor(1.0, line.spread)
        cos = line.cosines
        assert set(np.flatnonzero(cos >= line.cut)) == set(np.flatnonzero(cos >= floor))
        # A balance change below 1 redraws the face count on the same cosines.
        below = line.threshold_at(0.25)
        from vtscore.detectors.centroid_head import _gap_centre

        want = _gap_centre(cos, centroid_cut(cos, rule=FACE_BALL_BELOW, beta=0.25))
        assert below == pytest.approx(1.0 / (1.0 + np.exp(-line.scale * (want - line.cut))))


class TestTheTier:
    @pytest.mark.parametrize(
        ("tier", "media", "beta", "goods", "want"),
        [
            (TIER_TRAINED, "face", 1.0, 3, TIER_CENTROID),
            (TIER_TRAINED, "face", 4.0, 2, TIER_CENTROID),
            (TIER_TRAINED, "face", 0.25, 3, TIER_TRAINED),
            (TIER_TRAINED, "face", 1.0, 1, TIER_TRAINED),
            (TIER_TRAINED, "image", 1.0, 3, TIER_TRAINED),
            (TIER_CENTROID, "face", 0.25, 3, TIER_CENTROID),
            (TIER_NONE, "face", 1.0, 0, TIER_NONE),
        ],
    )
    def test_past_the_quota_faces_get_the_ball_from_beta_1(self, tier, media, beta, goods, want):
        assert face_ball_tier(tier, media, beta, goods) == want


def _faces(snap: dict[int, dict]) -> dict[int, dict]:
    """The fixture's corpus as face crops: the learned sort reads a detector's media type off its corpus."""
    for media in snap.values():
        media["media_type"] = "face"
    return snap


def _face_ctx(beta: float) -> DetectorContext:
    ctx = DetectorContext("", name="ball", media_type="face")
    ctx.beta, ctx.beta_seeded = beta, True
    return ctx


class TestFindGetsTheBall:
    def test_past_the_quota_a_face_detector_at_beta_1_is_the_centroid(self, labels):  # noqa: F811
        snap, make = labels
        snap = _faces(snap)
        ls = make([1, 2, 3], [40, 41, 42, 43])
        ctx = _face_ctx(1.0)
        assert lt.train_from_labelset(ctx, ls, media_type="face", snap=snap) is True
        assert is_centroid_head(ctx.model)
        assert ctx.centroid_line.rule == FACE_BALL_RULE and ctx.centroid_line.spread is not None
        low = _face_ctx(0.25)
        lt.train_from_labelset(low, ls, media_type="face", snap=snap)
        assert ctx.model is not None and not is_centroid_head(low.model)

    def test_a_cold_find_gets_the_ball_too(self, labels):  # noqa: F811
        snap, make = labels
        snap = _faces(snap)
        ctx = _face_ctx(4.0)
        _results, _threshold, model = lt.labelset_train_and_score(
            ctx, make([1, 2, 3], [40, 41, 42, 43]), media_type="face", clips_dict=snap, beta=4.0, label_quota=True
        )
        assert is_centroid_head(model)

    def test_a_balance_change_across_1_asks_for_the_other_head(self, labels):  # noqa: F811
        from vtscore.detectors.learned_sort import update_det_ctx_with_trained_model
        from vtscore.detectors.model_loading import cached_head_is_current

        snap, make = labels
        snap = _faces(snap)
        ls = make([1, 2, 3], [40, 41, 42, 43])
        ctx = _face_ctx(1.0)
        lt.train_from_labelset(ctx, ls, media_type="face", snap=snap)
        assert cached_head_is_current(ctx, ls)
        ctx.beta = 0.25
        assert not cached_head_is_current(ctx, ls), "the ball is not the head below beta 1"
        lt.train_from_labelset(ctx, ls, media_type="face", snap=snap)
        assert not is_centroid_head(ctx.model) and cached_head_is_current(ctx, ls)
        ctx.beta = 1.0
        assert not cached_head_is_current(ctx, ls), "the trained head is not the face detector from beta 1"
        # The Train view's learned sort stores a trained head; Find still refuses it at beta 1.
        _r, threshold, model = lt.labelset_train_and_score(ctx, ls, media_type="face", clips_dict=snap)
        update_det_ctx_with_trained_model(ctx, model, threshold, ls, {}, snap, [], [])
        assert not cached_head_is_current(ctx, ls)


class TestTheHarnessMirrors:
    def test_the_harness_default_is_the_ball_for_faces(self):
        from vtscore.eval.voting_iterations import production_centroid_line

        assert production_centroid_line({1: {"media_type": "face"}}) == FACE_BALL_RULE
