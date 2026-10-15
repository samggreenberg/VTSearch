"""The eval harness's typed-query runs score the withheld half as a typed-query detector's Test does (#4758).

Until the label quota, the app gives a detector made from a typed query that
query's own set, from the first vote.  "The Eval Default Arm IS the App"
(CLAUDE.md): so a default run that opens on a typed query writes a row at every
click, ``detector_tier`` is ``"query"`` until the quota and ``"trained"`` from
it, and the query rows are the text sort's set on the withheld half.  The
Train side never reads Test, so the trajectory is the arm before's
(``query_tier=False``: the Goods' centroid from the first Good).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.detectors.label_quota import TIER_CENTROID, TIER_QUERY, TIER_TRAINED, label_quota
from vtscore.eval.voting_iterations import simulate_voting_iterations

DIM = 12


def _world(n_per_cat: int = 40, seed: int = 0) -> tuple[dict[int, dict], dict[int, float]]:
    """Two categories and a typed query's cosines that rank ``alpha`` near the top, with misses."""
    rng = np.random.default_rng(seed)
    centres = {"alpha": rng.standard_normal(DIM), "beta": rng.standard_normal(DIM)}
    query = centres["alpha"] + 1.2 * rng.standard_normal(DIM)
    query /= np.linalg.norm(query)
    clips: dict[int, dict] = {}
    seed_scores: dict[int, float] = {}
    for i in range(2 * n_per_cat):
        cat = "alpha" if i < n_per_cat // 4 else "beta"
        vec = centres[cat] + 0.9 * rng.standard_normal(DIM)
        vec = (vec / np.linalg.norm(vec)).astype(np.float32)
        clips[i + 1] = {"id": i + 1, "embeddings": {"emb": vec}, "category": cat}
        seed_scores[i + 1] = float(vec @ query)
    return clips, seed_scores


def _run(**kwargs):
    clips, seed_scores = _world()
    picks: list[dict] = []
    rows = simulate_voting_iterations(
        clips,
        "alpha",
        seed=0,
        max_steps=20,
        calibrate_count=2,
        spot_check="off",
        pick_sink=picks,
        seed_scores=seed_scores,
        **kwargs,
    )
    return rows, picks


@pytest.fixture(scope="module")
def typed_run():
    """The quota's hand-over, before #4736's hold: the tiers this class pins."""
    return _run(query_hold=0)


@pytest.fixture(scope="module")
def before_run():
    return _run(query_tier=False)


class TestATypedQueryRun:
    def test_rows_start_at_the_first_vote(self, typed_run):
        rows, _picks = typed_run
        assert [r["t"] for r in rows] == list(range(1, 21))

    def test_each_row_is_the_query_until_the_quota_and_the_head_from_it(self, typed_run):
        rows, _picks = typed_run
        for r in rows:
            trained = label_quota(r["n_good"], r["n_bad"]).tier == TIER_TRAINED
            assert r["detector_tier"] == (TIER_TRAINED if trained else TIER_QUERY), r["t"]
        assert {r["detector_tier"] for r in rows} == {TIER_QUERY, TIER_TRAINED}, "the run crosses the quota"

    def test_the_query_rows_are_one_set_whatever_the_votes(self, typed_run):
        """The typed query's set does not move with the votes: every query row scores the same."""
        rows, _picks = typed_run
        query = [r for r in rows if r["detector_tier"] == TIER_QUERY]
        assert len(query) >= 2
        for col in ("precision", "recall", "fbeta", "n_flagged", "average_precision"):
            assert len({r[col] for r in query}) == 1, col
        assert all(math.isfinite(r["recall"]) for r in query)

    def test_the_trajectory_and_the_trained_rows_are_the_arm_befores(self, typed_run, before_run):
        rows, picks = typed_run
        old, old_picks = before_run
        assert [(p["t"], p["picked_id"]) for p in picks] == [(p["t"], p["picked_id"]) for p in old_picks]
        old_by_t = {r["t"]: r for r in old}
        trained = [r for r in rows if r["detector_tier"] == TIER_TRAINED]
        assert trained
        for r in trained:
            for col in ("precision", "recall", "fbeta", "auroc", "acq_threshold"):
                o = old_by_t[r["t"]][col]
                assert r[col] == o or (math.isnan(r[col]) and math.isnan(o)), (r["t"], col)

    def test_the_arm_before_gives_the_centroid_from_the_first_good(self, before_run):
        rows, picks = before_run
        first_good = min(p["t"] for p in picks if p["picked_label"] == 1)
        assert rows[0]["t"] == first_good
        assert TIER_QUERY not in {r["detector_tier"] for r in rows}
        assert TIER_CENTROID in {r["detector_tier"] for r in rows}


HOLD = 12


@pytest.fixture(scope="module")
def held_run():
    """The app's hold (#4736) at a count this short run crosses."""
    return _run(query_hold=HOLD)


def _base(rows):
    return [r for r in rows if not r.get("gmm_variant")]


class TestTheQueryHold:
    """Past the quota the typed query's set stays the untagged row until the hold; the head's rows ride tagged."""

    def test_the_default_is_the_apps_hold(self):
        import inspect

        from vtscore.detectors.label_quota import QUERY_HOLD_LABELS

        assert inspect.signature(simulate_voting_iterations).parameters["query_hold"].default == QUERY_HOLD_LABELS

    def test_the_picks_are_the_hand_overs_at_the_quota(self, held_run, typed_run):
        _rows, picks = held_run
        _old, old_picks = typed_run
        assert [(p["t"], p["picked_id"]) for p in picks] == [(p["t"], p["picked_id"]) for p in old_picks]

    def test_one_untagged_row_a_click_the_query_until_the_hold(self, held_run):
        rows, _picks = held_run
        base = _base(rows)
        assert [r["t"] for r in base] == list(range(1, 21))
        for r in base:
            held = r["n_good"] + r["n_bad"] < HOLD
            trained = label_quota(r["n_good"], r["n_bad"]).tier == TIER_TRAINED
            assert r["detector_tier"] == (TIER_TRAINED if trained and not held else TIER_QUERY), r["t"]
        held_past_quota = [
            r
            for r in base
            if r["detector_tier"] == TIER_QUERY and label_quota(r["n_good"], r["n_bad"]).tier == TIER_TRAINED
        ]
        assert held_past_quota, "the run meets the quota before the hold"
        assert all(r["app_trained"] == 0 for r in held_past_quota)

    def test_a_held_clicks_head_is_the_tagged_row_and_the_rest_are_the_hand_overs(self, held_run, typed_run):
        from vtscore.eval.voting_iterations import QUERY_HOLD_HEAD_TAG

        rows, _picks = held_run
        old_by_t = {r["t"]: r for r in _base(typed_run[0])}
        tagged = {r["t"]: r for r in rows if r.get("gmm_variant") == QUERY_HOLD_HEAD_TAG}
        assert tagged and max(tagged) < HOLD
        base_by_t = {r["t"]: r for r in _base(rows)}
        for t, old in old_by_t.items():
            new = tagged.get(t, base_by_t[t]) if old["detector_tier"] == TIER_TRAINED else base_by_t[t]
            assert new["detector_tier"] == old["detector_tier"], t
            for col in ("precision", "recall", "fbeta", "n_flagged"):
                assert new[col] == old[col] or (math.isnan(new[col]) and math.isnan(old[col])), (t, col)

    def test_the_held_rows_are_the_query_rows_set(self, held_run):
        rows, _picks = held_run
        query = [r for r in _base(rows) if r["detector_tier"] == TIER_QUERY]
        for col in ("precision", "recall", "fbeta", "n_flagged"):
            assert len({r[col] for r in query}) == 1, col


class TestOtherOpeningsAreUnchanged:
    def test_an_example_opening_gives_the_centroid(self):
        clips, _seed_scores = _world()
        rows = simulate_voting_iterations(
            clips, "alpha", seed=0, max_steps=12, calibrate_count=2, spot_check="off", seed_examples=1
        )
        assert TIER_QUERY not in {r["detector_tier"] for r in rows}

    def test_without_the_quota_there_is_no_query_tier(self):
        rows, _picks = _run(label_quota=False)
        assert {r["detector_tier"] for r in rows} == {TIER_TRAINED}


class TestTheCalibrationPath:
    def test_a_query_row_says_so_in_its_provenance(self):
        rows, _picks = _run(style="whole_image", emit_calibration_metrics=True)
        query = [r for r in rows if r["detector_tier"] == TIER_QUERY]
        assert query
        assert {r["threshold_provenance"] for r in query} == {"query"}
        assert all(r["threshold"] == 0.5 for r in query)


class TestTheQueryTestIsThePlantedAnswer:
    @pytest.mark.parametrize("beta", [0.25, 1.0, 4.0])
    def test_it_keeps_the_text_sorts_set_on_the_withheld_half(self, beta):
        """The typed query's cosines on the withheld half, cut where ``text_sort_cuts`` draws there."""
        from vtscore.eval.labels import media_is_positive
        from vtscore.eval.voting_iterations import _query_test
        from vtscore.training.thresholds import text_sort_cuts

        clips, seed_scores = _world()
        test_ids = list(range(2, 81, 2))
        _step, rows = _query_test(
            seed_scores, clips, test_ids, "alpha", 0, beta=beta, calibration_rows=False, line_beta=beta
        )
        cos = np.asarray([seed_scores[c] for c in test_ids])
        line = text_sort_cuts(list(cos), beta=beta).threshold
        kept = cos >= line - 1e-9
        pos = np.asarray([media_is_positive(clips[c], "alpha") for c in test_ids], dtype=bool)
        assert rows[0]["n_flagged"] == int(kept.sum())
        assert rows[0]["recall"] == pytest.approx(float((kept & pos).sum() / pos.sum()))
