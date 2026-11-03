"""#4392: the Document Logo review's readouts on one ranking."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_FULLMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "fullmarks"


@pytest.fixture(scope="module")
def sd():
    sys.path.insert(0, str(_FULLMARKS))
    try:
        yield importlib.import_module("sota_documents")
    finally:
        sys.path.remove(str(_FULLMARKS))


HITS = np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=bool)  # 4 positives in 8


def test_the_gate_set_scores_against_every_remaining_positive(sd):
    precision, recall, f1, k = sd.set_metrics(np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool), HITS)
    assert (k, precision, recall) == (2, 0.5, 0.25)
    assert f1 == pytest.approx(2 * 1 / (2 + 4))
    assert np.isnan(sd.set_metrics(np.zeros(8, dtype=bool), HITS)[0])  # an empty set has no precision


def test_floor_cuts_take_the_deepest_cut_that_still_meets_p(sd):
    out = sd.cut_metrics(HITS)
    # precision by depth: 1, .5, .67, .75, .6, .5, .43, .5
    assert out["k_at_p90"] == 1 and out["recall_at_p90"] == 0.25
    assert out["k_at_p50"] == 8 and out["recall_at_p50"] == 1.0
    assert out["best_f1"] == pytest.approx(2 * 3 / (4 + 4))  # depth 4: 3 hits


def test_no_positive_left_reads_as_missing_not_zero(sd):
    out = sd.cut_metrics(np.zeros(5, dtype=bool))
    assert np.isnan(out["best_f1"]) and np.isnan(out["recall_at_p50"])


def test_the_best_cut_at_each_balance_moves_with_beta(sd):
    # The structural line ignores beta, so every beta is scored on the same sessions (#4413).
    out = sd.cut_metrics(HITS)
    assert out["best_fb1"] == pytest.approx(out["best_f1"])
    assert out["best_fb025"] == pytest.approx(1.0625 * 1 / (0.0625 * 4 + 1))  # precision-leaning: stop at depth 1
    assert out["best_fb4"] == pytest.approx(17 * 4 / (16 * 4 + 8))  # recall-leaning: take all 8
    assert sd.beta_tag(0.25) == "025" and sd.beta_tag(1.0) == "1" and sd.beta_tag(4.0) == "4"


def test_thinning_keeps_a_seeded_nested_fraction_of_positives(sd):
    pool = [f"p{i:02d}" for i in range(20)]
    positive = np.array([i % 2 == 0 for i in range(20)])  # 10 positives
    ids_q, pos_q = sd.thin_pool("c", pool, positive, 0.25)
    ids_h, _pos_h = sd.thin_pool("c", pool, positive, 0.5)
    assert int(pos_q.sum()) == 3  # ceil(0.25 * 10)
    assert len(ids_q) == 13  # the 10 negatives stay
    kept_q = {p for p, y in zip(ids_q, pos_q) if y}
    assert kept_q <= set(ids_h)  # a smaller fraction keeps a subset of a larger one's
    assert sd.thin_pool("c", pool, positive, 0.25)[0] == ids_q  # reproducible
    assert int(sd.thin_pool("c", pool, positive, 0.01)[1].sum()) == 1  # never below one
