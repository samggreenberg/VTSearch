"""Issue #3688's one wrong vote: the simulated user answers exactly one click wrongly.

An experiment knob, not app behaviour: off by default, and the run is the honest
one pick for pick up to the flipped click.
"""

from __future__ import annotations

import pytest

from vtscore.eval.voting_iterations import parse_flip_vote


class TestParse:
    def test_off(self):
        assert parse_flip_vote(None) == (None, 0)

    def test_sides(self):
        assert parse_flip_vote("good:3") == (True, 3)
        assert parse_flip_vote("bad:1") == (False, 1)

    @pytest.mark.parametrize("bad", ["good", "good:0", "good:-1", "neutral:2", "good:1.5", ""])
    def test_refuses_what_it_cannot_run(self, bad):
        with pytest.raises(ValueError):
            parse_flip_vote(bad)


class TestInTheHarness:
    def _run(self, flip_vote):
        from tests_lib.eval.test_startup_schedule import _seeded_dataset
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        medias, seed_scores = _seeded_dataset(n_pos=40)
        picks: list = []
        rows = simulate_voting_iterations(
            medias,
            target_category="target",
            seed=3,
            dataset_name="stub",
            max_steps=40,
            seed_scores=seed_scores,
            atlas_min_node_size=8,
            spot_check="off",
            flip_vote=flip_vote,
            pick_sink=picks,
        )
        return picks, rows

    def test_off_by_default_every_vote_is_the_truth(self):
        picks, _ = self._run(None)
        assert all(p["picked_vote"] == p["picked_label"] for p in picks)

    @pytest.mark.parametrize(("spec", "truth", "k"), [("good:1", 1, 1), ("good:3", 1, 3), ("bad:2", 0, 2)])
    def test_exactly_the_kth_of_that_truth_is_flipped(self, spec, truth, k):
        picks, _ = self._run(spec)
        wrong = [i for i, p in enumerate(picks) if p["picked_vote"] != p["picked_label"]]
        assert len(wrong) == 1
        (i,) = wrong
        assert picks[i]["picked_label"] == truth
        assert sum(1 for p in picks[: i + 1] if p["picked_label"] == truth) == k

    def test_the_run_is_the_honest_one_up_to_the_flip(self):
        plain, _ = self._run(None)
        flipped, _ = self._run("good:2")
        i = next(i for i, p in enumerate(flipped) if p["picked_vote"] != p["picked_label"])
        assert [p["picked_id"] for p in plain[: i + 1]] == [p["picked_id"] for p in flipped[: i + 1]]
        assert flipped[i]["n_good"] == plain[i]["n_good"] - 1

    def test_a_bad_spec_is_refused_before_the_run(self):
        with pytest.raises(ValueError):
            self._run("good:0")


class TestHoldOut:
    """#3688's held-out record: out of the withheld half always, out of the voting half on request."""

    def _run(self, **kw):
        from tests_lib.eval.test_startup_schedule import _seeded_dataset
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        medias, seed_scores = _seeded_dataset(n_pos=40)
        picks: list = []
        simulate_voting_iterations(
            medias,
            target_category="target",
            seed=3,
            dataset_name="stub",
            max_steps=40,
            seed_scores=seed_scores,
            atlas_min_node_size=8,
            spot_check="off",
            pick_sink=picks,
            **kw,
        )
        return picks

    def test_withheld_only_leaves_the_picks_alone(self):
        from tests_lib.eval.test_startup_schedule import _seeded_dataset

        plain = self._run()
        # Every fifth media, so both halves lose some: the voting half keeps serving its share.
        held = {cid for cid in _seeded_dataset(n_pos=40)[0] if cid % 5 == 0}
        assert held & {p["picked_id"] for p in plain}
        assert [p["picked_id"] for p in self._run(hold_out=held)] == [p["picked_id"] for p in plain]

    def test_voting_too_never_serves_them(self):
        plain = self._run()
        held = {p["picked_id"] for p in plain[:6]}
        assert not held & {p["picked_id"] for p in self._run(hold_out=held, hold_out_voting=True)}

    def test_voting_needs_ids(self):
        with pytest.raises(ValueError):
            self._run(hold_out_voting=True)
