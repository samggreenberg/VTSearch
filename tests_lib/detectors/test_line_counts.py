"""#4786's line counts: what each candidate line keeps on the withheld half at a trained click."""

from __future__ import annotations

import pytest

from vtscore.eval.arms_handover import HANDOVER_CENTROID_TAG
from vtscore.eval.arms_line_counts import LINE_COUNT_COLUMNS


def _fbeta_at(k: int, ranks: str, n_pos: int, beta: float) -> float:
    """F-beta of keeping the top *k* of a ranking whose positives sit at 0-based *ranks*."""
    from vtscore.training.thresholds import fbeta_score

    tp = sum(1 for r in ranks.split() if int(r) < k)
    return fbeta_score(tp, k, n_pos, beta)


class TestInTheHarness:
    """A few-positive target: Test gives the centroid until the quota, the trained head after it."""

    def _run(self, beta: float = 4.0, **kw):
        from tests_lib.detectors.test_example_opening import _medias
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        picks: list[dict] = []
        rows = simulate_voting_iterations(
            _medias(n_pos=8, n_neg=400),
            target_category="person",
            seed=3,
            dataset_name="stub",
            max_steps=16,
            atlas_min_node_size=8,
            spot_check="off",
            seed_examples=1,
            stratify_target=True,
            beta=beta,
            pick_sink=picks,
            **kw,
        )
        return rows, picks

    def test_the_session_is_the_apps_and_every_trained_click_is_counted(self):
        base_rows, base_picks = self._run()
        sink: list[dict] = []
        rows, picks = self._run(line_count_sink=sink)
        assert [p["picked_id"] for p in picks] == [p["picked_id"] for p in base_picks]
        assert [(r["t"], r.get("gmm_variant"), r.get("fbeta")) for r in rows] == [
            (r["t"], r.get("gmm_variant"), r.get("fbeta")) for r in base_rows
        ]
        trained = {r["t"] for r in base_rows if r.get("detector_tier") == "trained" and not r.get("gmm_variant")}
        assert trained, "the run should reach the quota"
        assert {s["t"] for s in sink} == trained
        assert all(set(LINE_COUNT_COLUMNS) <= set(s) for s in sink)

    def test_the_counts_reproduce_what_test_returns(self):
        """Keeping the top ``k_test`` of the recorded ranking scores Test's own row."""
        sink: list[dict] = []
        rows, _ = self._run(line_count_sink=sink)
        base = {r["t"]: r for r in rows if r.get("detector_tier") == "trained" and not r.get("gmm_variant")}
        for s in sink:
            row = base[s["t"]]
            assert s["k_test"] == row["n_flagged"]
            assert s["k_labels_b4"] == s["k_test"]  # the run's beta is 4
            assert s["n_test_pos"] == row["n_test_pos"]
            assert _fbeta_at(s["k_test"], s["test_pos_ranks"], s["n_test_pos"], 4.0) == pytest.approx(row["fbeta"])
            assert s["n_good"] + s["n_bad"] >= 1
            assert 0 <= s["k_head_count"] <= s["n_test"]

    def test_the_centroid_count_is_the_shadows(self):
        """The centroid's line on the withheld half is the one #4776's shadow scores there."""
        sink: list[dict] = []
        rows, _ = self._run(line_count_sink=sink, handover_shadow=True, handover_sink=[])
        shadow = {r["t"]: r for r in rows if r.get("gmm_variant") == HANDOVER_CENTROID_TAG}
        assert sink and set(shadow) == {s["t"] for s in sink}
        for s in sink:
            assert s["k_centroid"] == shadow[s["t"]]["n_flagged"]
            assert s["centroid_fbeta"] == pytest.approx(shadow[s["t"]]["fbeta"])
            assert s["centroid_ap"] == pytest.approx(shadow[s["t"]]["average_precision"])

    def test_off_without_a_balance(self):
        sink: list[dict] = []
        from tests_lib.detectors.test_example_opening import _medias
        from vtscore.eval.voting_iterations import simulate_voting_iterations

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
            beta="off",
            line_count_sink=sink,
        )
        assert sink == []


class TestTheHeadCountLine:
    """#4786 in the harness: while the class model holds few Bads, beta 4's Test keeps the head's count."""

    def test_test_keeps_the_heads_count_while_the_gate_is_open(self):
        from vtscore.training.thresholds import HEAD_COUNT_BADS

        sink: list[dict] = []
        TestInTheHarness()._run(line_count_sink=sink)
        gated = [s for s in sink if s["model_n_neg"] == s["model_n_neg"] and s["model_n_neg"] < HEAD_COUNT_BADS]
        assert gated, "the run should reach a trained click with few Bads"
        assert all(s["k_test"] == s["k_head_count"] for s in gated)

    def test_gate_zero_draws_the_class_models_cut(self):
        from vtscore.eval.live_threshold_rules import head_count_bads

        sink: list[dict] = []
        with head_count_bads(0):
            TestInTheHarness()._run(line_count_sink=sink)
        assert sink and any(s["k_test"] != s["k_head_count"] for s in sink)
