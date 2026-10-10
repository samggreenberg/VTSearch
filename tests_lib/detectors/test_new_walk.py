"""Issue #4671's New-phase knob: the New phase takes the Hard pick instead of the atlas's.

An experiment knob, not app behaviour: ``atlas`` by default, so the default arm is
the app's own New phase pick for pick, and the arm changes nothing outside ``new``.
"""

from __future__ import annotations

import pytest

from vtscore.eval.al_strategies import ALContext, _select_phase_faithful
from vtscore.eval.voting_iterations import NEW_WALKS


class _Atlas:
    """A stand-in atlas whose next under-explored item is always 10, the far end of the ranking."""

    def next_sample(self, scores, threshold):
        del scores, threshold
        return 10


def _ctx(phase: str, *, new_walk: str = "atlas", atlas: object | None = None) -> ALContext:
    """Pool 10..14 scored so the Hard pick (nearest the 0.45 cut by rank) is not the atlas's 10."""
    pool = [10, 11, 12, 13, 14]
    return ALContext(
        pool_ids=pool,
        embeddings={},
        labeled={1: 1.0, 2: 0.0},
        scores={10: 0.2, 11: 0.9, 12: 0.5, 13: 0.4, 14: 0.1},
        model=object(),
        threshold=0.45,
        atlas=_Atlas() if atlas is None else atlas,  # type: ignore[arg-type]
        seed_scores={i: float(i) for i in pool},
        phase=phase,
        new_walk=new_walk,
    )


def test_the_default_new_phase_walks_the_atlas():
    assert _select_phase_faithful(_ctx("new"), "new") == 10


def test_the_arm_takes_the_hard_pick_in_new():
    hard = _select_phase_faithful(_ctx("hard"), "hard")
    assert hard != 10
    assert _select_phase_faithful(_ctx("new", new_walk="hard"), "new") == hard


@pytest.mark.parametrize("phase", ["good", "bad", "more", "hard", "done"])
def test_the_arm_changes_nothing_outside_new(phase):
    assert _select_phase_faithful(_ctx(phase, new_walk="hard"), phase) == _select_phase_faithful(_ctx(phase), phase)


def test_the_knob_values():
    assert NEW_WALKS == ("atlas", "hard")
