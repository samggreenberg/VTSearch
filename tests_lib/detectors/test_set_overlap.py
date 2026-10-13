"""#4816's set overlap: how the head's set and the typed query's split the unvoted media at a trained click."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests_lib.detectors.test_eval_query_tier import _run, _world
from vtscore.eval.arms_set_overlap import OVERLAP_COLUMNS, PARTS, PICK_OVERLAP_COLUMNS, SetOverlap
from vtscore.eval.voting_columns import PICK_COLUMNS


@pytest.fixture(scope="module")
def plain_run():
    return _run()


@pytest.fixture(scope="module")
def overlap_run():
    sink: list[dict] = []
    rows, picks = _run(set_overlap_sink=sink)
    return rows, picks, sink


class TestInTheHarness:
    def test_the_session_is_the_apps(self, plain_run, overlap_run):
        rows, picks = plain_run
        o_rows, o_picks, _sink = overlap_run
        assert [(p["t"], p["picked_id"]) for p in o_picks] == [(p["t"], p["picked_id"]) for p in picks]
        key = ("t", "gmm_variant", "detector_tier", "fbeta", "threshold")
        assert [tuple(r.get(k) for k in key) for r in o_rows] == [tuple(r.get(k) for k in key) for r in rows]

    def test_the_pick_log_carries_the_columns_off_or_on(self, plain_run, overlap_run):
        _rows, picks = plain_run
        assert set(PICK_OVERLAP_COLUMNS) <= set(PICK_COLUMNS)
        assert all(set(p) == set(PICK_COLUMNS) for p in picks)
        assert all(math.isnan(p[c]) for p in picks for c in PICK_OVERLAP_COLUMNS)
        _o_rows, o_picks, _sink = overlap_run
        assert all(set(p) == set(PICK_COLUMNS) for p in o_picks)

    def test_a_row_at_every_click_a_head_scored_the_pool(self, overlap_run):
        _rows, picks, sink = overlap_run
        assert sink, "the run should train a head"
        assert all(set(OVERLAP_COLUMNS) <= set(s) for s in sink)
        # A head trains from the first click with a Good and a Bad (Test gives the query until the quota).
        trained = sorted({p["t"] for p in picks if p["n_good"] >= 1 and p["n_bad"] >= 1})
        assert [s["t"] for s in sink] == trained

    def test_the_parts_add_up(self, overlap_run):
        _rows, _picks, sink = overlap_run
        totals = set()
        for s in sink:
            assert s["n_dq"] <= min(s["n_d"], s["n_q"])
            assert s["n_d"] + s["n_q"] - s["n_dq"] <= s["n_pool"]
            assert s["pos_dq"] <= s["n_dq"] and s["pos_d_only"] <= s["n_d"] - s["n_dq"]
            # Every positive of the voting half is unvoted in one part, or a Good in hand.
            totals.add(sum(s[f"pos_{p}"] for p in PARTS) + s["n_good"])
            assert sum(s[f"good_{p}"] for p in PARTS) == s["n_good"]
            assert sum(s[f"bad_{p}"] for p in PARTS) == s["n_bad"]
        assert len(totals) == 1

    def test_the_query_cut_is_the_text_baselines(self, overlap_run):
        from vtscore.training.thresholds import resolve_line_knobs
        from vtscore.training.thresholds.gmm import text_sort_threshold

        _rows, _picks, sink = overlap_run
        _clips, seed_scores = _world()
        cut = text_sort_threshold(list(seed_scores.values()), beta=resolve_line_knobs(None))
        assert all(s["query_cut"] == pytest.approx(cut) for s in sink)

    def test_each_pick_sits_where_the_previous_click_drew_the_sets(self, overlap_run):
        _rows, picks, sink = overlap_run
        by_t = {s["t"]: s for s in sink}
        _clips, seed_scores = _world()
        read = 0
        for p in picks:
            prev = by_t.get(p["t"] - 1)
            assert p["picked_in_query"] == float(seed_scores[p["picked_id"]] >= sink[0]["query_cut"])
            if prev is None:
                continue
            read += 1
            if abs(p["picked_detector_score"] - prev["head_threshold"]) > 1e-5:
                assert p["picked_in_head"] == float(p["picked_detector_score"] >= prev["head_threshold"])
            assert 0 <= p["picked_part_rank"] <= p["picked_head_rank"] < prev["n_pool"]
        assert read

    def test_off_on_an_example_opening(self):
        from tests_lib.detectors.test_example_opening import _medias
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        sink: list[dict] = []
        simulate_voting_iterations(
            _medias(n_pos=8, n_neg=400),
            target_category="person",
            seed=3,
            dataset_name="stub",
            max_steps=12,
            atlas_min_node_size=8,
            spot_check="off",
            seed_examples=1,
            stratify_target=True,
            set_overlap_sink=sink,
        )
        assert sink == []


class TestSetOverlap:
    """The parts on a hand-made pool."""

    def test_parts_truth_and_posteriors(self):
        from types import SimpleNamespace

        query = {1: 0.9, 2: 0.8, 3: 0.1, 4: 0.05, 5: 0.7, 6: 0.0}
        labels = {1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 1.0, 6: 0.0}
        ov = SetOverlap(query, 1.0, labels)
        ov.query_cut = 0.5
        ov._in_q = frozenset(c for c, s in query.items() if s >= 0.5)
        line = SimpleNamespace(
            unvoted_scores=np.array([0.9, 0.6, 0.2, 0.1]),
            unvoted_posteriors=np.array([0.8, 0.5, 0.1, 0.0]),
            unvoted_share=0.25,
        )
        row = ov.click({1: 0.9, 2: 0.2, 3: 0.6, 4: 0.1}, 0.5, line, [5], [6], {5: 0.95, 6: 0.3})
        assert (row["n_pool"], row["n_d"], row["n_q"], row["n_dq"]) == (4, 2, 2, 1)
        assert [row[f"pos_{p}"] for p in PARTS] == [1, 1, 0, 0]
        assert [row[f"post_{p}"] for p in PARTS] == pytest.approx([0.8, 0.5, 0.1, 0.0])
        assert row["est_positives"] == pytest.approx(1.0)
        assert [row[f"good_{p}"] for p in PARTS] == [1, 0, 0, 0]
        assert [row[f"bad_{p}"] for p in PARTS] == [0, 0, 0, 1]
        assert ov.pick(3) == {
            "picked_in_head": 1.0,
            "picked_in_query": 0.0,
            "picked_head_rank": 1.0,
            "picked_part_rank": 0.0,
            "picked_head_posterior": pytest.approx(0.5),
        }
        unknown = ov.pick(5)
        assert unknown["picked_in_query"] == 1.0 and math.isnan(unknown["picked_in_head"])
