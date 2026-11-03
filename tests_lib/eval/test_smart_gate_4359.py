"""#4359's bound on the Smart light: ``smart_gate="never"`` keeps Autopilot in ``hard``.

A harness-only arm.  It answers how much objective a later Smart could buy by
holding the light yellow for the phase decision, so a session never moves on to
``new`` or ``done``.  The light itself is still computed and reported.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.autopilot_flow import SMART_GATES, AutopilotFlow
from vtscore.eval.voting_iterations import simulate_voting_iterations


def test_the_gates():
    assert SMART_GATES == ("app", "never")
    with pytest.raises(ValueError, match="smart_gate must be one of"):
        AutopilotFlow(smart_gate="later")


def _separable(n_per_cat: int = 60, dim: int = 16) -> dict[int, dict]:
    """A pool the head separates cleanly, so the app's lights go green and the session leaves ``hard``."""
    rng = np.random.RandomState(0)
    medias: dict[int, dict] = {}
    for i in range(2 * n_per_cat):
        cat = "alpha" if i < n_per_cat else "beta"
        emb = rng.normal(1.0 if cat == "alpha" else -1.0, 0.5, dim).astype(np.float32)
        medias[i + 1] = {"id": i + 1, "embeddings": {"emb": emb}, "category": cat}
    return medias


def _phases(**kwargs) -> list[str]:
    picks: list[dict] = []
    simulate_voting_iterations(
        _separable(), "alpha", seed=0, max_steps=100, calibrate_count=2, spot_check="off", pick_sink=picks, **kwargs
    )
    return [str(p["phase"]) for p in picks]


def test_never_keeps_the_session_in_hard():
    """Teeth first: on this pool the app's lights go green and the session moves on."""
    assert {"new", "done"} & set(_phases())
    phases = _phases(smart_gate="never")
    assert "hard" in phases
    assert not {"new", "done"} & set(phases), sorted(set(phases))


def test_the_app_gate_is_the_default():
    assert _phases() == _phases(smart_gate="app")
