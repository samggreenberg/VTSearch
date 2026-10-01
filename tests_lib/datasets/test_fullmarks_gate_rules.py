"""#4367: the pre-registered accept rules, scored offline from frames."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_FULLMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "fullmarks"


@pytest.fixture(scope="module")
def gr():
    sys.path.insert(0, str(_FULLMARKS))
    try:
        yield importlib.import_module("gate_rules")
    finally:
        sys.path.remove(str(_FULLMARKS))


def _frame(**kw):
    base = {
        "inliers": np.array([60, 40, 20, 10, np.nan], dtype=np.float32),
        "shortlisted": np.array([True, True, True, True, False]),
        "stage1": np.array([0.9, 0.8, 0.7, 0.6, 0.55], dtype=np.float32),
        "good_loo_inliers": np.array([55.0, 45.0], dtype=np.float32),
        "good_loo_stage1": np.array([0.6, 0.5], dtype=np.float32),
        "bad_inliers": np.array([30.0], dtype=np.float32),
    }
    base.update(kw)
    return base


def test_the_vote_fit_threshold_sits_between_the_goods_and_the_bads(gr):
    assert gr.vote_fit_threshold(np.array([55.0, 45.0]), np.array([30.0])) == 45.0
    assert gr.vote_fit_threshold(np.array([]), np.array([30.0])) == 8.0  # no Good: the shipped gate
    assert gr.vote_fit_threshold(np.array([5.0]), np.array([])) == 8.0  # never below the floor


def test_rules_accept_what_they_promise(gr):
    z = _frame()
    assert gr.accept("R0", z).tolist() == [True, True, True, True, False]
    assert gr.accept("R1", z).tolist() == [True, True, False, False, False]  # above the Bad's 30
    assert gr.accept("R2", z).tolist() == [True, False, False, False, False]  # >= 45
    # R3 also takes the unverified page: its Stage-1 0.55 clears the Goods' 10th percentile (0.51).
    assert gr.accept("R3", z).tolist()[-1] is True


def test_the_bad_ceiling_without_bads_is_the_shipped_gate(gr):
    z = _frame(bad_inliers=np.array([], dtype=np.float32))
    assert gr.accept("R1", z).tolist() == gr.accept("R0", z).tolist()
