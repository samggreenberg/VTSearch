"""The label quota and the Goods' centroid (#4643, #4731), in the library tier.

Under 3 Goods or 4 Bads, and under a Good with 16 Bads, a labelset gives the Goods' centroid, not a trained
head: the unit mean of the Goods, everything ranked by its cosine to it, cut
at the two-Gaussian midpoint of those cosines on the scored corpus.  These
tests pin the rule's numbers to Autopilot's, the tiers, the centroid against
the example sort it is, and ``train_from_labelset`` / ``labelset_train_and_score``
/ ``cached_head_is_current`` at each tier.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import vtscore.detectors.labelset_training as lt
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.centroid_head import (
    CENTROID_THRESHOLD,
    centroid_head,
    fit_centroid_head,
    goods_centroid,
    is_centroid_head,
)
from vtscore.detectors.label_quota import (
    BAD_QUOTA,
    DRY_BAD_QUOTA,
    GOOD_QUOTA,
    TIER_CENTROID,
    TIER_NONE,
    TIER_TRAINED,
    label_quota,
    labelset_quota,
    quota_from_groups,
    served_quota,
)
from vtscore.detectors.labelset_elements import stable_element_id
from vtscore.detectors.training import score_rows_with_model, scoring_rows_for_snap
from vtscore.state.core import DetectorContext
from vtscore.training.thresholds import calculate_gmm_threshold

REPO = Path(__file__).resolve().parents[2]
DIM = 8
EMBEDDER = "test_embedder"


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / np.linalg.norm(vec)).astype(np.float32)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


class TestTheQuotaIsAutopilotsQuorum:
    def test_the_quota_is_the_eval_harness_ported_targets(self):
        from vtscore.eval.autopilot_flow import BAD_TARGET, GOOD_TARGET

        assert (GOOD_QUOTA, BAD_QUOTA) == (GOOD_TARGET, BAD_TARGET) == (3, 4)

    def test_the_quota_is_the_apps_good_and_bad_to_start(self):
        """Read off the TypeScript itself: the quota is Autopilot's own quorum."""
        ts = (REPO / "frontend/src/app/services/autopilot-state.service.ts").read_text()
        initial = ts[ts.index("const INITIAL_STATE") :]
        good = re.search(r"goodToStart:\s*(\d+)", initial)
        bad = re.search(r"badToStart:\s*(\d+)", initial)
        assert good is not None and bad is not None, "Autopilot's INITIAL_STATE no longer names its quorum"
        assert (GOOD_QUOTA, BAD_QUOTA) == (int(good.group(1)), int(bad.group(1)))

    def test_the_second_quota_is_the_good_walks_dry_run(self):
        """#4731: a Good walk that ran dry leaves a Good and ``moreDryRun`` Bads, which is the second quota."""
        from vtscore.eval.autopilot_flow import MORE_DRY_RUN

        ts = (REPO / "frontend/src/app/services/autopilot-state.service.ts").read_text()
        dry = re.search(r"moreDryRun:\s*(\d+)", ts[ts.index("const INITIAL_STATE") :])
        assert dry is not None, "Autopilot's INITIAL_STATE no longer names its dry run"
        assert DRY_BAD_QUOTA == MORE_DRY_RUN == int(dry.group(1)) == 16


class TestTheTiers:
    @pytest.mark.parametrize(
        ("n_good", "n_bad", "tier"),
        [
            (0, 0, TIER_NONE),
            (0, 9, TIER_NONE),
            (1, 0, TIER_CENTROID),
            (2, 10, TIER_CENTROID),
            (3, 3, TIER_CENTROID),
            (10, 1, TIER_CENTROID),
            (3, 4, TIER_TRAINED),
            (20, 40, TIER_TRAINED),
            (1, 15, TIER_CENTROID),
            (1, 16, TIER_TRAINED),
            (2, 30, TIER_TRAINED),
            (0, 16, TIER_NONE),
        ],
    )
    def test_the_counts_decide(self, n_good, n_bad, tier):
        assert label_quota(n_good, n_bad).tier == tier

    def test_an_eval_arm_can_drop_the_second_quota(self):
        """``dry_bad_quota=None`` is the rule before #4731, for the harness's pre-#4731 arm."""
        assert label_quota(1, 16, dry_bad_quota=None).tier == TIER_CENTROID
        assert label_quota(3, 4, dry_bad_quota=None).tier == TIER_TRAINED
        assert label_quota(1, 8, dry_bad_quota=8).tier == TIER_TRAINED

    def test_what_is_owed(self):
        q = label_quota(1, 3)
        assert (q.goods_owed, q.bads_owed) == (2, 1)
        assert label_quota(5, 9).goods_owed == label_quota(5, 9).bads_owed == 0
        # A head from the second quota owes nothing either.
        assert (label_quota(1, 16).goods_owed, label_quota(1, 16).bads_owed) == (0, 0)
        assert q.as_dict() == {
            "tier": TIER_CENTROID,
            "n_good": 1,
            "n_bad": 3,
            "goods_owed": 2,
            "bads_owed": 1,
            "good_quota": GOOD_QUOTA,
            "bad_quota": BAD_QUOTA,
            "dry_bad_quota": DRY_BAD_QUOTA,
        }

    def test_a_flooded_bad_is_one_label(self):
        """A Bad on a patch dataset floods many rows; the quota counts bags, not rows."""
        groups = [("g", "a"), ("b", "x"), ("b", "x"), ("b", "x"), ("b", "y")]
        q = quota_from_groups(groups)
        assert (q.n_good, q.n_bad) == (1, 2)

    def test_a_labelset_counts_its_elements(self):
        ls = LabelSet(
            [LabeledElement(md5=f"g{i}", label="good") for i in range(2)] + [LabeledElement(md5="b", label="bad")]
        )
        q = labelset_quota(ls)
        assert (q.n_good, q.n_bad, q.tier) == (2, 1, TIER_CENTROID)
        assert labelset_quota(None).tier == TIER_NONE

    def test_what_was_served_is_read_off_the_model(self):
        """A labelset whose counts meet the quota can still have been served the centroid (an origin did not resolve)."""
        ls = LabelSet(
            [LabeledElement(md5=f"g{i}", label="good") for i in range(3)]
            + [LabeledElement(md5=f"b{i}", label="bad") for i in range(4)]
        )
        head = centroid_head(np.ones(DIM, np.float32) / np.sqrt(DIM), 0.2)
        assert served_quota(head, ls)["tier"] == TIER_CENTROID
        assert served_quota(object(), ls)["tier"] == TIER_TRAINED
        assert served_quota(None, ls)["tier"] == TIER_NONE


# ---------------------------------------------------------------------------
# The centroid
# ---------------------------------------------------------------------------


def _corpus(n: int = 60, seed: int = 4643) -> tuple[dict[int, dict], np.ndarray]:
    """A plain single-vector corpus: a tight cluster around one direction and a broad cloud."""
    rng = np.random.default_rng(seed)
    proto = _unit(rng.standard_normal(DIM))
    snap: dict[int, dict] = {}
    for cid in range(1, n + 1):
        vec = proto + (0.15 if cid <= n // 4 else 1.5) * rng.standard_normal(DIM)
        snap[cid] = {"id": cid, "media_type": "image", "embedder": EMBEDDER, "embeddings": {EMBEDDER: _unit(vec)}}
    return snap, proto


class TestTheCentroid:
    def test_every_good_weighs_the_same_and_the_mean_is_a_unit_vector(self):
        a = np.array([3.0, 0.0], np.float32)
        b = np.array([0.0, 0.5], np.float32)
        c = goods_centroid([a, b])
        assert np.allclose(c, np.array([1.0, 1.0]) / np.sqrt(2.0), atol=1e-6)
        with pytest.raises(ValueError):
            goods_centroid([])

    def test_it_is_a_marked_linear_head_a_portable_consumer_can_read(self):
        head = centroid_head(_unit(np.arange(1, DIM + 1, dtype=np.float32)), 0.3)
        assert is_centroid_head(head)
        assert set(head.state_dict()) == {"0.weight", "0.bias"}
        from vtscore.training.mlp import LINEAR_SVM_HEAD, build_model

        assert not is_centroid_head(build_model(DIM, hidden_dim=LINEAR_SVM_HEAD))

    def test_its_cut_is_the_example_sorts_midpoint_and_it_ranks_by_cosine(self):
        """The planted answer: cosine to the unit mean of the Goods, cut at the GMM midpoint of those cosines."""
        snap, _proto = _corpus()
        goods = [snap[cid]["embeddings"][EMBEDDER] for cid in (1, 2, 3)]
        rows = scoring_rows_for_snap(snap, EMBEDDER)
        head, threshold = fit_centroid_head(goods, lambda h: score_rows_with_model(h, rows)[0])
        assert threshold == CENTROID_THRESHOLD

        centroid = goods_centroid(goods)
        cosines = rows.matrix.astype(np.float64) @ centroid.astype(np.float64)
        cut = calculate_gmm_threshold(cosines.tolist())
        scores = np.asarray(score_rows_with_model(head, rows)[0])
        # The same media on either side of the line, and the same order.
        assert np.array_equal(scores >= threshold, cosines >= cut + 1e-6) or np.array_equal(
            scores >= threshold, cosines >= cut - 1e-6
        )
        assert list(np.argsort(-scores, kind="stable")) == list(np.argsort(-cosines, kind="stable"))
        assert 0 < int((scores >= threshold).sum()) < len(snap)


# ---------------------------------------------------------------------------
# train_from_labelset and labelset_train_and_score at each tier
# ---------------------------------------------------------------------------


@pytest.fixture
def labels(monkeypatch):
    """Label elements naming corpus media, cached as if they resolved in-dataset."""
    snap, _proto = _corpus()

    def _labelset(goods: list[int], bads: list[int]) -> LabelSet:
        return LabelSet(
            [LabeledElement(md5=f"m{c}", label="good", origin_name=f"m{c}") for c in goods]
            + [LabeledElement(md5=f"m{c}", label="bad", origin_name=f"m{c}") for c in bads]
        )

    def _fake_populate(det_ctx, labelset, *, media_type, snap, on_progress=None):
        det_ctx.embedder = EMBEDDER
        for elem in labelset.elements:
            det_ctx.label_embeddings[stable_element_id(elem)] = _snap[int(elem.md5[1:])]["embeddings"][EMBEDDER]

    _snap = snap
    monkeypatch.setattr(lt, "populate_label_embeddings", _fake_populate)
    return snap, _labelset


def _ctx() -> DetectorContext:
    return DetectorContext("", name="quota", media_type="image")


class TestTrainFromLabelset:
    def test_no_good_gives_nothing(self, labels):
        snap, make = labels
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, make([], [40, 41]), media_type="image", snap=snap) is False
        assert ctx.model is None

    @pytest.mark.parametrize(("goods", "bads"), [([1], []), ([1, 2], [40, 41, 42, 43, 44]), ([1, 2, 3], [40, 41, 42])])
    def test_under_the_quota_it_is_the_goods_centroid(self, labels, goods, bads):
        snap, make = labels
        ctx = _ctx()
        ls = make(goods, bads)
        assert lt.train_from_labelset(ctx, ls, media_type="image", snap=snap) is True
        assert is_centroid_head(ctx.model) and ctx.threshold == CENTROID_THRESHOLD
        # Nothing for a balance change to re-cut: the centroid's line is its own.
        assert ctx.calibration_cache is None and ctx.anchored_cut_cache is None and ctx.labels_line is None
        from vtscore.detectors.model_loading import cached_head_is_current

        assert cached_head_is_current(ctx, ls)

    def test_at_the_quota_it_trains_the_head(self, labels):
        snap, make = labels
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, make([1, 2, 3], [40, 41, 42, 43]), media_type="image", snap=snap) is True
        assert ctx.model is not None and not is_centroid_head(ctx.model)

    def test_a_trained_head_cached_under_the_quota_is_not_handed_out(self, labels):
        """The Train view's learned sort trains from any Good and Bad and stamps the same signature (#4643)."""
        from vtscore.detectors.learned_sort import update_det_ctx_with_trained_model
        from vtscore.detectors.model_loading import cached_head_is_current

        snap, make = labels
        ctx = _ctx()
        under = make([1, 2], [40])
        # What the learned sort does: train on the labelset with no quota, then store the head.
        _results, threshold, model = lt.labelset_train_and_score(ctx, under, media_type="image", clips_dict=snap)
        assert model is not None and not is_centroid_head(model)
        update_det_ctx_with_trained_model(ctx, model, threshold, under, {}, snap, [], [])
        assert not cached_head_is_current(ctx, under), "a head on 2 Goods and 1 Bad would be handed out"
        at = make([1, 2, 3], [40, 41, 42, 43])
        lt.train_from_labelset(ctx, at, media_type="image", snap=snap)
        assert cached_head_is_current(ctx, at)


class TestLabelsetTrainAndScore:
    def test_a_cold_find_gets_the_centroid_under_the_quota(self, labels):
        snap, make = labels
        ctx = _ctx()
        results, threshold, model = lt.labelset_train_and_score(
            ctx, make([1, 2], [40]), media_type="image", clips_dict=snap, label_quota=True
        )
        assert is_centroid_head(model) and threshold == CENTROID_THRESHOLD
        assert len(results) == len(snap)
        assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)

    def test_a_cold_find_with_no_good_gets_nothing(self, labels):
        snap, make = labels
        results, _threshold, model = lt.labelset_train_and_score(
            _ctx(), make([], [40, 41]), media_type="image", clips_dict=snap, label_quota=True
        )
        assert (results, model) == ([], None)

    def test_the_learned_sort_still_trains_from_the_first_good_and_bad(self, labels):
        """``label_quota=False`` (the default) is the Train view's sort, not a detector handed out."""
        snap, make = labels
        _results, _threshold, model = lt.labelset_train_and_score(
            _ctx(), make([1], [40]), media_type="image", clips_dict=snap
        )
        assert model is not None and not is_centroid_head(model)


class TestTheBalanceCountsTheCentroidsLine:
    def test_the_state_counts_what_the_midpoint_keeps(self, labels):
        """The centroid's line does not take the balance, so its state counts that line, not a proposal."""
        from vtscore.state.core import detector_balance_state
        from vtscore.training.thresholds import LineRanking

        snap, make = labels
        ctx = _ctx()
        lt.train_from_labelset(ctx, make([1, 2], [40]), media_type="image", snap=snap)
        rows = scoring_rows_for_snap(snap, EMBEDDER)
        scores = score_rows_with_model(ctx.model, rows)[0]
        ctx.line_ranking = LineRanking.from_scores(rows.ids, scores, ())
        state = detector_balance_state(ctx, 1.0)
        assert state is not None
        assert state["count"] == int(sum(s >= ctx.threshold for s in scores))
