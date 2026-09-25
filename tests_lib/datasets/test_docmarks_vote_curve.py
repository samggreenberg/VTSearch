"""#4162: the vote-curve arms replay a template x page matrix correctly."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_DOCMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "docmarks"


@pytest.fixture(scope="module")
def vc():
    sys.path.insert(0, str(_DOCMARKS))
    try:
        yield importlib.import_module("vote_curve")
    finally:
        sys.path.remove(str(_DOCMARKS))


def _stats(inliers: np.ndarray) -> np.ndarray:
    stats = np.zeros(inliers.shape + (9,), dtype=np.float16)
    stats[..., 0] = np.log1p(inliers)
    stats[..., 8] = inliers > 0
    return stats


def _class(vc, tmp_path, inliers, positives, template_ids, tentative=None):
    """Pool pages p0..pN-1; template 0 is the crop (page ``q``, not in the pool)."""
    inliers = np.asarray(inliers, dtype=np.int16)
    n = inliers.shape[1]
    np.savez(
        tmp_path / "c.npz",
        pool_ids=np.array([f"p{i}" for i in range(n)]),
        template_ids=np.array(template_ids),
        stats=_stats(inliers),
        inliers=inliers,
        tentative=np.zeros_like(inliers, dtype=np.int32) if tentative is None else np.asarray(tentative),
        positives=np.array(positives),
    )
    vec = np.zeros((n, 2), dtype=np.float32)
    return vc.ClassData(tmp_path / "c.npz", vec, vec, np.zeros(2, np.float32), np.zeros(2, np.float32))


class TestPrimitives:
    def test_order_by_breaks_ties_by_page_order_after_every_key(self, vc):
        assert vc.order_by(np.array([1, 3, 3, 0]), np.array([0, 1, 5, 0])).tolist() == [2, 1, 0, 3]
        assert vc.order_by(np.array([2, 2, 2])).tolist() == [0, 1, 2]

    def test_shortlist_reorders_only_the_head_and_keeps_stage1_order_on_ties(self, vc):
        stage1 = np.array([3, 0, 1, 2])
        assert vc.shortlist(stage1, 2, np.array([5, 0, 9, 1])).tolist() == [0, 3, 1, 2]
        assert vc.shortlist(stage1, 3, np.zeros(4)).tolist() == [3, 0, 1, 2]

    def test_average_precision_matches_the_reference(self, vc):
        # The ``vc`` fixture has the DocMarks directory on sys.path; load it the same way.
        ev = importlib.import_module("eval_retrieval")

        pos = np.array([False, True, False, True, True])
        order = np.array([1, 0, 3, 2, 4])
        ref = ev.average_precision([f"p{i}" for i in order], {f"p{i}" for i in np.flatnonzero(pos)})
        assert vc.average_precision(order, pos) == pytest.approx(ref)
        assert np.isnan(vc.average_precision(order, np.zeros(5, bool)))


class TestArms:
    # Crop finds p0 only; p0's box template finds p1 and p2; p3 is a negative p1 matches.
    INLIERS = [
        [30, 0, 0, 5, 0],  # crop
        [99, 40, 35, 0, 0],  # p0's box
        [20, 99, 25, 12, 0],  # p1's box
        [0, 10, 99, 0, 0],  # p2's box
    ]
    POS = [True, True, True, False, False]
    TIDS = ["q", "p0", "p1", "p2"]

    def test_exemplar_ignores_votes_and_max_adds_each_good(self, vc, tmp_path):
        cd = _class(vc, tmp_path, self.INLIERS, self.POS, self.TIDS)
        assert vc.rank("a0_exemplar", cd, [0], [3]).tolist()[:2] == [0, 3]
        assert vc.rank("a1_max", cd, [], []).tolist()[:2] == [0, 3]
        assert vc.rank("a1_max", cd, [0], []).tolist()[:3] == [0, 1, 2]

    def test_goods_only_drops_the_crop_once_a_good_exists(self, vc, tmp_path):
        inl = [row[:] for row in self.INLIERS]
        inl[0][4] = 50  # only the crop reaches p4
        cd = _class(vc, tmp_path, inl, self.POS, self.TIDS)
        assert vc.rank("a1_max", cd, [0], []).tolist()[1] == 4
        assert 4 not in vc.rank("a1p_goods_only", cd, [0], []).tolist()[:3]

    def test_repick_keeps_the_crop_until_two_goods_then_takes_the_best_median(self, vc, tmp_path):
        cd = _class(vc, tmp_path, self.INLIERS, self.POS, self.TIDS)
        assert vc.rank("a2_repick", cd, [0], []).tolist() == vc.rank("a0_exemplar", cd, [], []).tolist()
        # Goods p0, p1: crop's median over them is 15, p0's (against p1) 40, p1's (against p0) 20.
        assert vc.rank("a2_repick", cd, [0, 1], []).tolist() == vc.order_by(cd.inliers[1], cd.tentative[1]).tolist()

    def test_closed_loop_counts_goods_and_shared_sequence_scores_the_same_remainder(self, vc, tmp_path):
        cd = _class(vc, tmp_path, self.INLIERS, self.POS, self.TIDS)
        rows = vc.run_class(cd, ["a0_exemplar", "a1_max"], (0, 1, 3))
        closed = {(r["arm"], r["v"]): r for r in rows if r["readout"] == "closed"}
        # a1 votes p0 (Good), then its template brings p1 and p2 to the top.
        assert closed[("a1_max", 3)]["found"] == 3 and closed[("a1_max", 3)]["left"] == 0
        # a0 never learns: p0, then p3 (5 inliers), then page order.
        assert closed[("a0_exemplar", 3)]["found"] == 2
        shared = {(r["arm"], r["v"]): r for r in rows if r["readout"] == "shared"}
        assert shared[("a1_max", 1)]["left"] == shared[("a0_exemplar", 1)]["left"] == 2
        assert shared[("a1_max", 1)]["ap"] == pytest.approx(1.0)
        assert shared[("a0_exemplar", 1)]["ap"] < 1.0
