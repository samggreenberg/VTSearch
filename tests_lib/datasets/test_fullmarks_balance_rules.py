"""The balance-aware line's rule family, scored offline on frames (#4458)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_FULLMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "fullmarks"


@pytest.fixture(scope="module")
def br():
    sys.path.insert(0, str(_FULLMARKS))
    try:
        yield importlib.import_module("balance_rules")
    finally:
        sys.path.remove(str(_FULLMARKS))


def _frame(br, *, n_votes: int, n_bads: int, ceiling: float = 0.0):
    inliers = np.array([30, 12, 9, 8, 7, -1], dtype=np.float32)
    return br.Frame(
        cid="c",
        v=1,
        test=np.ones(6, dtype=bool),
        positive=np.array([1, 1, 0, 1, 0, 0], dtype=bool),
        order=np.arange(6),
        inliers=inliers,
        tight=np.array([1, 0, 1, 1, 1, 0], dtype=bool),
        n_votes=n_votes,
        n_bads=n_bads,
        ceiling=ceiling,
    )


def test_the_shipped_rule_is_the_apps_line_in_each_state(br):
    s = br.SHIPPED
    # Click 0, the example sort: the plain 8-inlier gate.
    assert br.accept(s, _frame(br, n_votes=0, n_bads=0)).tolist() == [1, 1, 1, 1, 0, 0]
    # Votes but no Bad (H1, #4440): the gate and a tight fit.
    assert br.accept(s, _frame(br, n_votes=2, n_bads=0)).tolist() == [1, 0, 1, 1, 0, 0]
    # After a Bad (#4367): more inliers than the Bads' best, and at least 8; no geometry.
    assert br.accept(s, _frame(br, n_votes=3, n_bads=1, ceiling=9)).tolist() == [1, 1, 0, 0, 0, 0]


def test_the_ceiling_offset_and_no_ceiling(br):
    f = _frame(br, n_votes=3, n_bads=1, ceiling=9)
    looser = br.SHIPPED._replace(d=-2)  # inliers > 7, so the 8-inlier floor binds
    assert br.accept(looser, f).tolist() == [1, 1, 1, 1, 0, 0]
    assert br.accept(br.SHIPPED._replace(d=None), f).tolist() == [1, 1, 1, 1, 0, 0]
    assert br.accept(br.SHIPPED._replace(d=8), f).tolist() == [1, 0, 0, 0, 0, 0]
    assert br.accept(br.SHIPPED._replace(t=12, d=None), f).tolist() == [1, 1, 0, 0, 0, 0]


def test_share_is_one_when_the_rule_is_the_best_cut(br):
    f = _frame(br, n_votes=0, n_bads=0)
    best = br.best_f_beta(f, 1.0)  # order 1,1,0,1,...: depth 4 holds all 3 positives
    assert best == pytest.approx(2 * 3 / (4 + 3))
    assert br.share(br.SHIPPED, f, 1.0, best) == pytest.approx(1.0)


def test_ties_go_to_the_rule_closest_to_shipped(br):
    assert br.distance(br.SHIPPED) == 0
    assert br.distance(br.SHIPPED._replace(t=10)) < br.distance(br.SHIPPED._replace(d=None))
    assert len(br.family()) == 7 * 2 * 2 * 6 * 2
