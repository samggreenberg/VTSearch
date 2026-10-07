"""Issue #4637's More-walk knob: the walk takes the detector's top, and the detector is shown.

An experiment knob, not app behaviour: ``seed`` by default, so the default arm is
the app's own opening pick for pick.
"""

from __future__ import annotations

import pytest

from vtscore.eval.al_strategies import ALContext, _select_phase_faithful
from vtscore.eval.autopilot_flow import app_has_detector


def _ctx(phase: str, *, more_walk: str = "seed", scores: dict[int, float] | None = None) -> ALContext:
    """Pool 10..14: the text sort ranks 14 first, the detector ranks 11 first."""
    pool = [10, 11, 12, 13, 14]
    return ALContext(
        pool_ids=pool,
        embeddings={},
        labeled={1: 1.0, 2: 0.0},
        scores={10: 0.2, 11: 0.9, 12: 0.5, 13: 0.4, 14: 0.1} if scores is None else scores,
        model=object(),
        threshold=0.45,
        atlas=None,
        seed_scores={i: float(i) for i in pool},
        phase=phase,
        more_walk=more_walk,
    )


class TestPick:
    def test_the_default_walk_is_the_text_sorts_top(self):
        assert _select_phase_faithful(_ctx("more"), "more") == 14

    def test_the_detector_walk_is_the_detectors_top(self):
        assert _select_phase_faithful(_ctx("more", more_walk="detector"), "more") == 11

    @pytest.mark.parametrize("phase", ["good", "bad"])
    def test_good_and_bad_stay_on_the_text_sort(self, phase):
        assert _select_phase_faithful(_ctx(phase, more_walk="detector"), phase) == _select_phase_faithful(
            _ctx(phase), phase
        )

    def test_without_detector_scores_it_walks_the_text_sort(self):
        assert _select_phase_faithful(_ctx("more", more_walk="detector", scores={}), "more") == 14


class TestShown:
    def test_more_is_shown_only_under_the_arm(self):
        assert not app_has_detector("more")
        assert app_has_detector("more", more_shown=True)

    @pytest.mark.parametrize("phase", ["good", "bad", "s0"])
    def test_the_arm_shows_nothing_before_the_walk(self, phase):
        assert not app_has_detector(phase, more_shown=True)

    def test_hard_is_shown_either_way(self):
        assert app_has_detector("hard") and app_has_detector("hard", more_shown=True)


class TestInTheHarness:
    def _run(self, more_walk: str, **kw):
        from tests_lib.detectors.test_startup_schedule import _seeded_dataset
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        medias, seed_scores = _seeded_dataset(n_pos=40)
        picks: list = []
        rows = simulate_voting_iterations(
            medias,
            target_category="target",
            seed=3,
            dataset_name="stub",
            max_steps=24,
            seed_scores=seed_scores,
            atlas_min_node_size=8,
            spot_check="off",
            more_walk=more_walk,
            pick_sink=picks,
            **kw,
        )
        return rows, picks

    def test_the_arms_agree_until_the_walk_and_part_there(self):
        _, seed = self._run("seed")
        _, det = self._run("detector")
        first_more = next(i for i, p in enumerate(seed) if p["phase"] == "more")
        assert [p["picked_id"] for p in seed[:first_more]] == [p["picked_id"] for p in det[:first_more]]
        assert det[first_more]["phase"] == "more"
        assert any(a["picked_id"] != b["picked_id"] for a, b in zip(seed[first_more:], det[first_more:]))

    def test_the_detector_walk_is_shown(self):
        rows, _ = self._run("detector")
        more = [r for r in rows if r.get("phase") == "more"]
        assert more and all(r["app_trained"] == 1 for r in more)
        rows, _ = self._run("seed")
        assert all(r["app_trained"] == 0 for r in rows if r.get("phase") == "more")

    @pytest.mark.parametrize(
        ("more_walk", "kw"),
        [
            ("learned", {}),
            ("detector", {"startup_schedule": "g3@top,b4@mid"}),
            ("detector", {"opening_diversity": "0.85/1"}),
        ],
    )
    def test_refuses_what_it_cannot_run(self, more_walk, kw):
        with pytest.raises(ValueError):
            self._run(more_walk, **kw)
