"""#4458 round 2: the fast rule scorer's adaptive floor and Stage-1 tail, on a synthetic frame."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_FULLMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "fullmarks"


@pytest.fixture(scope="module")
def cv():
    sys.path.insert(0, str(_FULLMARKS))
    try:
        yield importlib.import_module("balance_rules_cv")
    finally:
        sys.path.remove(str(_FULLMARKS))


@pytest.fixture
def frame(tmp_path: Path, cv):
    """Six test pages: four verified (30, 12, 9, 5 inliers), two beyond the shortlist."""
    nan = np.nan
    np.savez_compressed(
        tmp_path / "c__x__v003.npz",
        positive=np.array([1, 0, 1, 0, 1, 0], dtype=bool),
        test=np.ones(6, dtype=bool),
        stage1=np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.1], dtype=np.float32),
        shortlisted=np.array([1, 1, 1, 1, 0, 0], dtype=bool),
        inliers=np.array([30, 12, 9, 5, nan, nan], dtype=np.float32),
        ratio=np.full(6, 0.9, dtype=np.float32),
        reproj=np.full(6, 0.001, dtype=np.float32),
        good_ids=np.array(["g1", "g2"]),
        good_loo_ratio=np.array([0.9, 0.9], dtype=np.float32),
        good_loo_reproj=np.array([0.001, 0.001], dtype=np.float32),
        good_loo_inliers=np.array([20.0, 40.0], dtype=np.float32),  # median 30
        good_loo_stage1=np.array([0.5, 0.95], dtype=np.float32),
        bad_ids=np.array([], dtype=str),
        bad_inliers=np.array([], dtype=np.float32),
        bad_ratio=np.array([], dtype=np.float32),
        bad_reproj=np.array([], dtype=np.float32),
    )
    return cv.load_counts(tmp_path / "c__x__v003.npz", {"ratio_min": 0.75, "reproj_max": 0.004887})


def _f(tp: int, k: int, pos: int, beta: float) -> float:
    b2 = beta * beta
    return (1 + b2) * tp / (b2 * pos + k)


def test_the_adaptive_floor_scales_with_the_goods_own_fits(cv, frame):
    ra = cv.rule_arrays([cv.SHIPPED, cv.SHIPPED._replace(q=0.5)])
    best = frame.best[1.0]
    shipped, adaptive = cv.shares(frame, ra, 1.0)
    # Shipped: >= 8 inliers keeps pages 0, 1, 2 (2 of the 3 positives).
    assert shipped == pytest.approx(_f(2, 3, 3, 1.0) / best)
    # q = 0.5 of the Goods' median (30) makes the floor 15: only page 0.
    assert adaptive == pytest.approx(_f(1, 1, 3, 1.0) / best)


def test_the_tail_adds_unverified_pages_above_the_goods_stage1(cv, frame):
    ra = cv.rule_arrays([cv.SHIPPED._replace(tail=10)])
    # The 10th percentile of (0.5, 0.95) is 0.545: page 4 (0.55, a positive) joins, page 5 (0.1) does not.
    assert frame.tail_n[1] == 1 and frame.tail_tp[1] == 1
    (share,) = cv.shares(frame, ra, 4.0)
    assert share == pytest.approx(_f(3, 4, 3, 4.0) / frame.best[4.0])


def test_folds_are_stable_and_the_family_has_6048_rules(cv):
    assert cv.fold("ucsf/logo_bat_leaf") == cv.fold("ucsf/logo_bat_leaf")
    assert len(cv.family()) == 9 * 4 * 7 * 2 * 2 * 2 * 3
    assert cv.SHIPPED in cv.family()
