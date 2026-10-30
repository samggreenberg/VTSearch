"""The harness's Span target (issue #5473): ``span_green=0`` skips the Diversity step.

``span_green`` is the app's ``autopilot_goal_diversity`` at another value.  At 0,
Span reads green from the first vote, so the phase machine never enters ``new``
and Done waits on Smart and Stable alone.  That is the app's default since #5473;
a positive target (``SPAN_GREEN_ON``, 40, the default before) turns the step back on.
"""

from __future__ import annotations

import pytest

from vtscore.eval.autopilot_flow import SPAN_GREEN_DEFAULT, SPAN_GREEN_ON, span_status

from .test_autopilot_flow import _sim_clips


def _run(**kwargs):
    from vtscore.eval.voting_iterations import simulate_voting_iterations

    clips = _sim_clips()
    seed_scores = {i: float(clips[i]["embeddings"]["e"].mean()) for i in clips}
    return simulate_voting_iterations(
        clips, "cat", seed=1, max_steps=45, atlas_min_node_size=5, seed_scores=seed_scores, **kwargs
    )


def test_a_zero_target_reads_green_on_any_atlas():
    assert all(span_status(level, 100, 0) == "green" for level in (0, 1, 9, 39))
    assert span_status(0, 100, SPAN_GREEN_ON) == "red"
    assert span_status(0, 0, SPAN_GREEN_ON) == "green"


def test_the_default_is_the_diversity_step_off():
    assert SPAN_GREEN_DEFAULT == 0


def test_a_zero_target_never_enters_the_diversity_step():
    rows = [r for r in _run(span_green=0) if r.get("span")]
    assert rows, "the run never read Span"
    assert {r["span"] for r in rows} == {"green"}
    assert "new" not in {r["phase"] for r in rows}


def test_the_default_target_is_the_apps():
    assert [r["phase"] for r in _run(span_green=SPAN_GREEN_DEFAULT)] == [r["phase"] for r in _run()]


def test_a_positive_target_turns_the_step_back_on():
    assert "new" in {r["phase"] for r in _run(span_green=SPAN_GREEN_ON)}


@pytest.mark.parametrize("bad", [-1, 1.5, True, "0"])
def test_a_bad_target_is_refused(bad):
    with pytest.raises(ValueError, match="span_green"):
        _run(span_green=bad)
