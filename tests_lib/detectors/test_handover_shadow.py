"""#4776's hand-over shadow: the centroid's Test past the label quota, and a rule's evidence."""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.arms_handover import HANDOVER_CENTROID_TAG, HANDOVER_COLUMNS, HandoverJudge, held_out_rank


class TestHeldOutRank:
    def test_counts_the_media_scored_above(self):
        assert held_out_rank(np.array([0.9, 0.5, 0.1]), 0.6) == 2
        assert held_out_rank(np.array([0.9, 0.5, 0.1]), 1.0) == 1
        assert held_out_rank(np.array([]), 0.0) == 1

    def test_a_tie_does_not_count_against_it(self):
        assert held_out_rank(np.array([0.5, 0.5]), 0.5) == 1


def _faces(dim: int = 8) -> dict[int, dict]:
    """Two people and a crowd: person A's four photos sit along one axis, a near-duplicate of B beside them."""
    rng = np.random.default_rng(0)
    medias: dict[int, dict] = {}
    axis = np.eye(dim, dtype=np.float32)
    for i in range(4):
        medias[i] = {"id": i, "category": "a", "embeddings": {"stub": axis[0] + 0.05 * rng.standard_normal(dim)}}
    for i in range(4, 60):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec[0] = -abs(vec[0])
        medias[i] = {"id": i, "category": "other", "embeddings": {"stub": vec}}
    return medias


class TestHandoverJudge:
    def _judge(self, medias: dict[int, dict], sim: list[int], test: list[int]) -> HandoverJudge:
        from vtscore.embedding.media_vectors import media_embedding

        return HandoverJudge(medias, sim, test, "a", media_embedding)

    def test_holds_out_each_good_and_ranks_it_in_the_unvoted_pool(self):
        medias = _faces()
        sim = [0, 1, 2, *range(4, 40)]
        judge = self._judge(medias, sim, [3, *range(40, 60)])
        row = judge.row([0, 1, 2], [4, 5], [c for c in sim if c not in (0, 1, 2, 4, 5)])
        assert row["n_good"] == 3 and row["n_bad"] == 2 and row["n_pool"] == len(sim) - 5
        centroid = [int(r) for r in row["centroid_ranks"].split()]
        head = [int(r) for r in row["head_ranks"].split()]
        assert len(centroid) == len(head) == 3
        # Person A's photos are the corpus's best match to each other.
        assert centroid == [1, 1, 1]
        assert all(1 <= r <= row["n_pool"] + 1 for r in head)

    def test_one_good_has_nothing_to_hold_out(self):
        medias = _faces()
        sim = [0, *range(4, 40)]
        row = self._judge(medias, sim, [1, 2, 3, *range(40, 60)]).row([0], list(range(4, 20)), list(range(20, 40)))
        assert row["centroid_ranks"] == row["head_ranks"] == ""

    def test_the_prior_heads_rank_the_withheld_half(self):
        medias = _faces()
        sim = [0, 1, *range(4, 40)]
        row = self._judge(medias, sim, [2, 3, *range(40, 60)]).row([0, 1], [4, 5, 6], list(range(7, 40)))
        aps = [row[c] for c in HANDOVER_COLUMNS if c.startswith("prior_ap_s")]
        assert aps and all(0.0 <= a <= 1.0 for a in aps)
        # The centroid alone ranks A's withheld photos first, so a head started there, cheaply, does too.
        assert row["prior_ap_s10"] == pytest.approx(1.0)


class TestInTheHarness:
    """A few-positive target: Test gives the centroid until the quota, the trained head after it."""

    def _run(self, **kw):
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
            beta=1.0,
            pick_sink=picks,
            **kw,
        )
        return rows, picks

    def test_the_session_is_the_apps_and_the_shadow_covers_every_trained_click(self):
        base_rows, base_picks = self._run()
        sink: list[dict] = []
        rows, picks = self._run(handover_shadow=True, handover_sink=sink)
        assert [p["picked_id"] for p in picks] == [p["picked_id"] for p in base_picks]
        untagged = [r for r in rows if (r.get("gmm_variant") or "") != HANDOVER_CENTROID_TAG]
        assert [(r["t"], r.get("gmm_variant"), r.get("fbeta")) for r in untagged] == [
            (r["t"], r.get("gmm_variant"), r.get("fbeta")) for r in base_rows
        ]
        trained = {r["t"] for r in base_rows if r.get("detector_tier") == "trained" and not r.get("gmm_variant")}
        assert trained, "the run should reach the quota"
        shadow = {r["t"]: r for r in rows if r.get("gmm_variant") == HANDOVER_CENTROID_TAG}
        assert set(shadow) == trained
        assert all(r["detector_tier"] == "centroid" for r in shadow.values())
        assert {s["t"] for s in sink} == trained
        assert all(
            set(HANDOVER_COLUMNS) - {"seed", "dataset", "category", "calibration_seed", "style"} <= set(s) for s in sink
        )

    def test_the_centroid_runs_on_unbroken_past_the_quota(self):
        """A Bad leaves the centroid where it was: at a hand-over on a Bad, the shadow repeats the last centroid row."""
        rows, picks = self._run(handover_shadow=True, handover_sink=[])
        base = {r["t"]: r for r in rows if r.get("detector_tier") == "centroid" and not r.get("gmm_variant")}
        shadow = {r["t"]: r for r in rows if r.get("gmm_variant") == HANDOVER_CENTROID_TAG}
        first = min(shadow)
        label = {p["t"]: p["picked_label"] for p in picks}
        assert first - 1 in base
        if not label[first]:
            for col in ("fbeta", "average_precision", "n_flagged"):
                assert shadow[first][col] == pytest.approx(base[first - 1][col])

    def test_off_without_the_quota(self):
        sink: list[dict] = []
        rows, _ = self._run(handover_shadow=True, handover_sink=sink, label_quota=False)
        assert not [r for r in rows if r.get("gmm_variant") == HANDOVER_CENTROID_TAG]
        assert sink == []
