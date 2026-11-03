"""Issue #4699's example opening and stratified split.

A face benchmark's class is one person with ~6 photos. The harness's no-text start hands the session
three random positives, which is all of a typical identity's voting half, and a plain random split
leaves some identities nothing withheld. The example opening starts from K photos the way the app's
example sort does, and the stratified split keeps one positive on each side.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.example_opening import (
    EXAMPLE_PHASE,
    check_example_opening,
    choose_examples,
    example_sort,
    positive_split_sizes,
    stratified_split,
)


def _medias(n_pos: int = 6, n_neg: int = 200, seed: int = 0, dim: int = 16) -> dict[int, dict]:
    """A single-vector pool: *n_pos* positives near one centre, the rest noise."""
    rng = np.random.default_rng(seed)
    centre = rng.standard_normal(dim).astype(np.float32)
    medias: dict[int, dict] = {}
    for i in range(n_pos + n_neg):
        positive = i < n_pos
        vec = centre * (1.5 if positive else 0.0) + rng.standard_normal(dim).astype(np.float32) * 0.8
        medias[i] = {"id": i, "category": "person" if positive else "other", "embeddings": {"stub": vec}}
    return medias


class TestPositiveSplitSizes:
    @pytest.mark.parametrize(
        "n,want",
        [(0, (0, 0)), (1, (1, 0)), (2, (1, 1)), (3, (2, 1)), (6, (3, 3)), (7, (4, 3)), (10, (5, 5))],
    )
    def test_half_and_half_with_one_on_each_side(self, n, want):
        assert positive_split_sizes(n, 0.5) == want

    def test_another_fraction(self):
        assert positive_split_sizes(6, 0.8) == (5, 1)
        # 10 * (1 - 0.7) is 2.9999999999999996 in floating point; it still withholds three.
        assert positive_split_sizes(10, 0.7) == (7, 3)

    def test_never_withholds_every_positive(self):
        assert positive_split_sizes(2, 0.01) == (1, 1)


class TestStratifiedSplit:
    def test_partitions_the_pool(self):
        medias = _medias()
        sim, test = stratified_split(medias, "person", 0.5, np.random.RandomState(0))
        assert set(sim).isdisjoint(test)
        assert set(sim) | set(test) == set(medias)

    @pytest.mark.parametrize("n_pos", [2, 3, 6, 9])
    def test_every_seed_keeps_a_positive_on_each_side(self, n_pos):
        medias = _medias(n_pos=n_pos)
        want_sim, want_test = positive_split_sizes(n_pos, 0.5)
        for seed in range(50):
            sim, test = stratified_split(medias, "person", 0.5, np.random.RandomState(seed))
            assert sum(medias[c]["category"] == "person" for c in sim) == want_sim
            assert sum(medias[c]["category"] == "person" for c in test) == want_test

    def test_negatives_split_at_the_fraction(self):
        medias = _medias(n_pos=6, n_neg=201)
        sim, _ = stratified_split(medias, "person", 0.5, np.random.RandomState(1))
        assert sum(medias[c]["category"] == "other" for c in sim) == 100

    def test_fixed_by_the_seed(self):
        medias = _medias()
        a = stratified_split(medias, "person", 0.5, np.random.RandomState(7))
        b = stratified_split(medias, "person", 0.5, np.random.RandomState(7))
        assert a == b


class TestChooseExamples:
    def test_nested_in_the_count(self):
        medias = _medias(n_pos=10)
        sim, _ = stratified_split(medias, "person", 0.5, np.random.RandomState(0))
        one = choose_examples(sim, medias, "person", 1, seed=3)
        four = choose_examples(sim, medias, "person", 4, seed=3)
        assert four[:1] == one

    def test_only_voting_half_positives(self):
        medias = _medias(n_pos=10)
        sim, _ = stratified_split(medias, "person", 0.5, np.random.RandomState(0))
        got = choose_examples(sim, medias, "person", 5, seed=0)
        assert set(got) <= set(sim)
        assert all(medias[c]["category"] == "person" for c in got)

    def test_refuses_more_than_the_voting_half_holds(self):
        medias = _medias(n_pos=6)
        sim, _ = stratified_split(medias, "person", 0.5, np.random.RandomState(0))
        with pytest.raises(ValueError, match="needs 4 examples"):
            choose_examples(sim, medias, "person", 4, seed=0)


class TestExampleSort:
    def test_one_example_ranks_itself_first(self):
        medias = _medias()
        scores = example_sort(medias, [2])
        assert max(scores, key=scores.__getitem__) == 2
        assert scores[2] == pytest.approx(1.0, abs=1e-5)

    def test_several_rank_against_the_centroid_of_unit_vectors(self):
        medias = _medias()
        ex = [0, 1, 2]
        vec = {c: medias[c]["embeddings"]["stub"] for c in medias}
        unit = [vec[c] / np.linalg.norm(vec[c]) for c in ex]
        centre = np.mean(unit, axis=0)
        centre /= np.linalg.norm(centre)
        scores = example_sort(medias, ex)
        for cid in (0, 5, 50):
            v = vec[cid] / np.linalg.norm(vec[cid])
            assert scores[cid] == pytest.approx(float(v @ centre), abs=1e-5)

    def test_refuses_a_media_with_no_vector(self):
        medias = _medias()
        medias[9]["embeddings"] = {}
        with pytest.raises(ValueError, match="9"):
            example_sort(medias, [1])

    def test_refuses_a_patch_dataset(self):
        medias = _medias()
        medias[0]["patch_grid"] = np.zeros((4, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="patch"):
            example_sort(medias, [1])


class TestCheck:
    @pytest.mark.parametrize("bad", [0, -1, 2.5, "1", True])
    def test_refuses_a_count_that_is_not_a_positive_integer(self, bad):
        with pytest.raises(ValueError):
            check_example_opening(
                bad,
                stratify_target=True,
                seed_scores=None,
                startup_schedule=None,
                train_mix=None,
                target_prevalence=None,
            )

    @pytest.mark.parametrize(
        "kw",
        [
            {"seed_scores": {0: 1.0}},
            {"startup_schedule": "g3@top"},
            {"train_mix": "equal"},
            {"target_prevalence": 0.01},
        ],
    )
    def test_refuses_what_it_cannot_combine_with(self, kw):
        base = {"seed_scores": None, "startup_schedule": None, "train_mix": None, "target_prevalence": None}
        with pytest.raises(ValueError):
            check_example_opening(2, stratify_target=True, **{**base, **kw})

    def test_unset_passes_and_stratify_refuses_a_mixed_band_cell(self):
        check_example_opening(
            None,
            stratify_target=False,
            seed_scores={0: 1.0},
            startup_schedule="x",
            train_mix=None,
            target_prevalence=None,
        )
        with pytest.raises(ValueError, match="train_mix"):
            check_example_opening(
                None,
                stratify_target=True,
                seed_scores=None,
                startup_schedule=None,
                train_mix="equal",
                target_prevalence=None,
            )


class TestInTheHarness:
    def _run(self, seed_examples, *, stratify=True, n_pos=8, seed=3, **kw):
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        medias = _medias(n_pos=n_pos)
        picks: list[dict] = []
        rows = simulate_voting_iterations(
            medias,
            target_category="person",
            seed=seed,
            dataset_name="stub",
            max_steps=12,
            atlas_min_node_size=8,
            spot_check="off",
            seed_examples=seed_examples,
            stratify_target=stratify,
            pick_sink=picks,
            **kw,
        )
        return medias, rows, picks

    def test_off_by_default(self):
        _, _, picks = self._run(None, stratify=False)
        assert not any(p["phase"] == EXAMPLE_PHASE for p in picks)

    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_the_first_k_votes_are_the_examples(self, k):
        medias, rows, picks = self._run(k)
        assert rows
        assert [p["phase"] for p in picks[:k]] == [EXAMPLE_PHASE] * k
        assert all(medias[p["picked_id"]]["category"] == "person" for p in picks[:k])
        assert not any(p["phase"] == EXAMPLE_PHASE for p in picks[k:])

    def test_examples_are_nested_across_runs(self):
        _, _, one = self._run(1)
        _, _, four = self._run(4)
        assert one[0]["picked_id"] == four[0]["picked_id"]

    def test_one_example_then_walks_its_sort(self):
        """With fewer than three Goods, the Good phase takes the top of the example sort."""
        medias, _, picks = self._run(1, seed=5)
        sim, _ = stratified_split(medias, "person", 0.5, np.random.RandomState(5))
        ex = choose_examples(sim, medias, "person", 1, seed=5)
        assert picks[0]["picked_id"] == ex[0]
        scores = example_sort(medias, ex)
        unvoted = [c for c in sim if c != ex[0]]
        assert picks[1]["phase"] == "good"
        assert picks[1]["picked_id"] == max(unvoted, key=lambda c: scores[c])

    def test_a_text_sort_and_examples_are_refused(self):
        with pytest.raises(ValueError, match="seed_scores"):
            self._run(1, seed_scores={i: 0.0 for i in range(208)})

    def test_too_few_positives_in_the_voting_half_is_loud(self):
        with pytest.raises(ValueError, match="examples"):
            self._run(4, n_pos=6)
