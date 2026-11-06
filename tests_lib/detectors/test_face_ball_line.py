"""The face crops' ball (#5478): the Goods' centroid cut at a cosine floor, from beta 1 up with two Goods.

On FHIBE's faces the centroid of a person's Goods and one cosine floor per balance, moved by how spread out
the Goods sit, scored above the trained head's line over a session from beta 1 up
(``docs/experiments/2026-10-30-face-ball-line-5478``).  The floors are one face space's, read from its entry
in ``face_ball.json`` (#5497), and a space with no entry keeps the line from before the ball.  These tests pin
the rule (:func:`~vtscore.detectors.centroid_head.face_ball_rule`), the measurement rules it was priced with,
the tier that hands it out past the quota, the cached head Find reuses, and the harness's mirror of both.
"""

from __future__ import annotations

import numpy as np
import pytest

import vtscore.detectors.labelset_training as lt
from tests_lib.detectors.test_label_quota import DIM, EMBEDDER, _unit, labels  # noqa: F401
from tests_shared.face_ball import CENTRE, COMPAT, FLOORS, SLOPE, face_ball_space, tag_faces  # noqa: F401
from vtscore.concurrency.notifications import notifications
from vtscore.detectors.centroid_head import (
    CENTROID_LINE_RULES,
    FACE_BALL_BELOW,
    FACE_BALL_RULE,
    ball_rule,
    centroid_cut,
    centroid_line_rule,
    face_ball_compat,
    face_ball_floor,
    face_ball_holds,
    face_ball_rule,
    fit_centroid,
    goods_spread,
    is_centroid_head,
    is_centroid_line_rule,
    is_face_ball_rule,
)
from vtscore.detectors.label_quota import TIER_CENTROID, TIER_NONE, TIER_TRAINED
from vtscore.detectors.labelset_heads import face_ball_tier
from vtscore.state.core import DetectorContext

RULE = face_ball_rule(COMPAT)


def _cosines(seed: int = 0) -> np.ndarray:
    """A person's cosines to their Goods' centroid: a bulk, a few look-alikes, four own photos."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.02, 0.07, 5000), rng.uniform(0.3, 0.4, 6), rng.uniform(0.5, 0.7, 4)])


@pytest.fixture
def toasts():
    """Every notification published while the test runs."""
    got = []
    notifications.subscribe(got.append)
    yield got
    notifications.unsubscribe(got.append)


class TestTheRule:
    def test_the_rule_names_its_face_space(self):
        assert RULE == "face_ball[oss-arcface-mbf:1]"
        assert face_ball_compat(RULE) == COMPAT and is_face_ball_rule(RULE) and is_centroid_line_rule(RULE)
        assert face_ball_compat(FACE_BALL_RULE) is None and not is_face_ball_rule(FACE_BALL_RULE)
        assert FACE_BALL_RULE not in CENTROID_LINE_RULES and not is_centroid_line_rule(FACE_BALL_RULE)
        with pytest.raises(ValueError):
            face_ball_rule("a@b")
        with pytest.raises(ValueError, match="unknown centroid line rule"):
            centroid_cut(_cosines(), rule=FACE_BALL_RULE, beta=1.0, spread=0.6)

    def test_faces_in_a_space_with_a_ball_draw_it(self, face_ball_space):  # noqa: F811
        assert centroid_line_rule("face", COMPAT) == RULE
        assert centroid_line_rule("image", COMPAT) != RULE

    def test_from_beta_1_with_a_spread_it_is_the_floor_moved_by_the_spread(self, face_ball_space):  # noqa: F811
        cos = _cosines()
        for beta, floor in FLOORS.items():
            for spread in (0.55, CENTRE, 0.75):
                want = floor + SLOPE * (spread - CENTRE)
                assert centroid_cut(cos, rule=RULE, beta=beta, spread=spread) == pytest.approx(want)

    def test_below_beta_1_or_with_one_good_it_is_the_z55_count(self, face_ball_space):  # noqa: F811
        cos = _cosines()
        for beta, spread in ((0.25, 0.65), (0.5, 0.65), (1.0, None), (4.0, None), (None, 0.65)):
            got = centroid_cut(cos, rule=RULE, beta=beta, spread=spread)
            assert got == centroid_cut(cos, rule=FACE_BALL_BELOW, beta=beta)

    def test_the_floor_falls_with_the_balance_in_log_beta(self, face_ball_space):  # noqa: F811
        def floor(beta: float) -> float:
            got = face_ball_floor(beta, CENTRE, COMPAT)
            assert got is not None
            return got

        assert floor(1.0) == pytest.approx(FLOORS[1.0])
        assert floor(4.0) == pytest.approx(FLOORS[4.0])
        assert floor(2.0) == pytest.approx((FLOORS[1.0] + FLOORS[4.0]) / 2)
        assert floor(16.0) < floor(4.0)
        assert floor(0.5) == pytest.approx(FLOORS[1.0])

    def test_the_file_gives_the_floors_the_code_gave(self, face_ball_space):  # noqa: F811
        """#5497: the entry reproduces #5478's in-code formula exactly, so no pick on buffalo_s moves."""
        lo, hi = 1.0, 4.0
        for beta in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.5):
            for spread in (0.41, 0.55, CENTRE, 0.6999, 0.83):
                x = (np.log(max(beta, lo)) - np.log(lo)) / (np.log(hi) - np.log(lo))
                shipped = float(FLOORS[lo] + x * (FLOORS[hi] - FLOORS[lo]) + SLOPE * (spread - CENTRE))
                assert face_ball_floor(beta, spread, COMPAT) == shipped

    def test_it_holds_from_beta_1_with_two_goods(self):
        assert face_ball_holds(1.0, 2) and face_ball_holds(4.0, 7)
        assert not face_ball_holds(0.25, 5) and not face_ball_holds(1.0, 1) and not face_ball_holds(None, 5)


class TestAnUncalibratedSpace:
    """#5497: a tag with no entry, or a no-go one, keeps the pre-ball line and says so once."""

    def test_no_entry_keeps_the_pre_ball_line_with_a_warning(self, toasts):
        assert centroid_line_rule("face", COMPAT) == FACE_BALL_BELOW
        assert centroid_line_rule("face", COMPAT) == FACE_BALL_BELOW
        assert centroid_line_rule("face", None) == FACE_BALL_BELOW
        warned = [n for n in toasts if n.level == "warning"]
        assert [n.message for n in warned] == [
            f"The face search line is not calibrated for this face embedder ({COMPAT}).",
            "The face search line is not calibrated for this face embedder.",
        ]
        assert "has no entry" in (warned[0].detail or "")
        assert face_ball_floor(1.0, CENTRE, COMPAT) is None

    def test_another_space_than_the_entrys_is_uncalibrated(self, face_ball_space, toasts):  # noqa: F811
        assert centroid_line_rule("face", "oss-arcface-r50:1") == FACE_BALL_BELOW
        assert toasts and "oss-arcface-r50:1" in toasts[0].message

    def test_a_no_go_entry_keeps_the_pre_ball_line(self, face_ball_space, toasts):  # noqa: F811
        from tests_shared.face_ball import face_ball
        from vtscore.media.face import face_ball as module

        module.write_entry(face_ball("facenet:practice", use=False))
        assert module.face_ball_entry("facenet:practice") is not None
        assert module.face_ball_for("facenet:practice") is None
        assert centroid_line_rule("face", "facenet:practice") == FACE_BALL_BELOW
        assert "found the ball no better" in (toasts[0].detail or "")
        # The other space's entry is kept beside it.
        assert centroid_line_rule("face", COMPAT) == RULE

    def test_a_ball_rule_whose_entry_has_gone_redraws_the_count(self):
        cos = _cosines()
        assert centroid_cut(cos, rule=RULE, beta=1.0, spread=0.65) == centroid_cut(cos, rule=FACE_BALL_BELOW, beta=1.0)

    def test_past_the_quota_the_trained_head_stands(self):
        assert face_ball_tier(TIER_TRAINED, "face", 1.0, 3, COMPAT) == TIER_TRAINED

    def test_find_gets_the_trained_head(self, labels):  # noqa: F811
        snap, make = labels
        snap = tag_faces(snap)
        ctx = _face_ctx(1.0)
        assert lt.train_from_labelset(ctx, make([1, 2, 3], [40, 41, 42, 43]), media_type="face", snap=snap) is True
        assert ctx.face_compat == COMPAT
        assert ctx.model is not None and not is_centroid_head(ctx.model)


class TestTheFile:
    def test_a_malformed_file_turns_the_ball_off_and_says_why(self, face_ball_space, toasts):  # noqa: F811
        from vtscore.media.face import face_ball as module

        module.face_ball_path().write_text("{not json")
        module.reset_for_tests()
        assert module.face_ball_for(COMPAT) is None
        assert centroid_line_rule("face", COMPAT) == FACE_BALL_BELOW
        assert "is not readable" in (toasts[0].detail or "")
        with pytest.raises(ValueError):
            module.write_entry(face_ball_space)

    def test_an_entry_round_trips(self, face_ball_space):  # noqa: F811
        from vtscore.media.face.face_ball import FaceBall

        again = FaceBall.from_json(COMPAT, face_ball_space.to_json())
        assert again.betas == face_ball_space.betas and again.centre == CENTRE and again.use
        with pytest.raises(ValueError):
            FaceBall.from_json(COMPAT, {"use": True, "betas": {}, "centre": 0.6})

    def test_the_space_is_the_medias_one_tag(self):
        from vtscore.media.face.face_ball import face_compat_of

        assert face_compat_of(tag_faces({1: {}, 2: {}})) == COMPAT
        mixed = [{"custom_metadata": {"Face Compat": "a"}}, {"custom_metadata": {"Face Compat": "b"}}]
        assert face_compat_of(mixed) is None
        assert face_compat_of({1: {"media_type": "face"}}) is None and face_compat_of(None) is None


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

        assert parse_centroid_line_variants(f"ball0.41@1,ball0.44_s0.25_c0.63@4,{RULE}@1") == [
            ("ball0.41", 1.0),
            ("ball0.44_s0.25_c0.63", 4.0),
            (RULE, 1.0),
        ]

    def test_the_spread_is_each_goods_cosine_to_the_others(self):
        e = np.eye(3)
        assert goods_spread([e[0]]) is None
        assert goods_spread([e[0], e[0]]) == pytest.approx(1.0)
        assert goods_spread([e[0], e[1]]) == pytest.approx(0.0)
        # Three orthogonal Goods: each is at 0 to the mean of the other two.
        assert goods_spread(list(e)) == pytest.approx(0.0)


class TestTheFittedHead:
    def test_the_head_keeps_the_faces_at_or_above_the_floor_and_redraws_below_beta_1(self, face_ball_space):  # noqa: F811
        rng = np.random.default_rng(5478)
        proto = _unit(rng.standard_normal(DIM))
        goods = [_unit(proto + 0.3 * rng.standard_normal(DIM)) for _ in range(4)]
        corpus = np.stack([_unit(proto + s * rng.standard_normal(DIM)) for s in [0.3] * 5 + [2.0] * 300])

        def score(probe):
            import torch

            with torch.no_grad():
                return torch.sigmoid(probe(torch.from_numpy(corpus.astype(np.float32)))).squeeze(1).tolist()

        head, _thr, line = fit_centroid(goods, score, rule=RULE, beta=1.0)
        assert line.spread is not None and line.spread == pytest.approx(goods_spread(goods))
        floor = face_ball_floor(1.0, line.spread, COMPAT)
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
    def test_past_the_quota_faces_get_the_ball_from_beta_1(self, face_ball_space, tier, media, beta, goods, want):  # noqa: F811
        assert face_ball_tier(tier, media, beta, goods, COMPAT) == want


def _face_ctx(beta: float) -> DetectorContext:
    ctx = DetectorContext("", name="ball", media_type="face")
    ctx.beta, ctx.beta_seeded = beta, True
    return ctx


class TestFindGetsTheBall:
    def test_past_the_quota_a_face_detector_at_beta_1_is_the_centroid(self, face_ball_space, labels):  # noqa: F811
        snap, make = labels
        snap = tag_faces(snap)
        ls = make([1, 2, 3], [40, 41, 42, 43])
        ctx = _face_ctx(1.0)
        assert lt.train_from_labelset(ctx, ls, media_type="face", snap=snap) is True
        assert is_centroid_head(ctx.model)
        assert ctx.centroid_line.rule == RULE and ctx.centroid_line.spread is not None
        low = _face_ctx(0.25)
        lt.train_from_labelset(low, ls, media_type="face", snap=snap)
        assert ctx.model is not None and not is_centroid_head(low.model)

    def test_a_cold_find_gets_the_ball_too(self, face_ball_space, labels):  # noqa: F811
        snap, make = labels
        snap = tag_faces(snap)
        ctx = _face_ctx(4.0)
        _results, _threshold, model = lt.labelset_train_and_score(
            ctx, make([1, 2, 3], [40, 41, 42, 43]), media_type="face", clips_dict=snap, beta=4.0, label_quota=True
        )
        assert is_centroid_head(model)

    def test_a_balance_change_across_1_asks_for_the_other_head(self, face_ball_space, labels):  # noqa: F811
        from vtscore.detectors.learned_sort import update_det_ctx_with_trained_model
        from vtscore.detectors.model_loading import cached_head_is_current

        snap, make = labels
        snap = tag_faces(snap)
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
    def test_the_harness_default_is_the_ball_in_the_medias_space(self, face_ball_space):  # noqa: F811
        from vtscore.eval.voting_iterations import production_centroid_line

        assert production_centroid_line(tag_faces({1: {}})) == RULE
        assert production_centroid_line(tag_faces({1: {}}, "oss-arcface-r50:1")) == FACE_BALL_BELOW
        assert production_centroid_line({1: {"media_type": "face"}}) == FACE_BALL_BELOW
