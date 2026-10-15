"""The stop-rule pricing of #4878: when each rule fires, and the two edges it is judged by.

``scripts/experiments/state_of_app/stop_signal_4878.py`` reads a stop rule's
click off a session's per-click signal, and scores it against a stop at a fixed
click and against its own stops dealt to the sessions at random.  A wrong click
or a wrong edge would rank the rules wrongly in the report, so each is pinned
here on a planted session.

Meta-group: the subject is repo tooling under ``scripts/``, loaded by path with
``_cells_io`` / ``objective`` importable from the calibration directory, as
``test_calibration_stopping.py`` loads ``stopping``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "experiments" / "state_of_app" / "stop_signal_4878.py"
_CALIB = _ROOT / "scripts" / "experiments" / "calibration"


@pytest.fixture(scope="module")
def S() -> Any:
    saved = {name: sys.modules.get(name) for name in ("_cells_io", "objective")}
    saved_path = list(sys.path)
    sys.path.insert(0, str(_CALIB))
    try:
        spec = importlib.util.spec_from_file_location("_stop_signal_4878", _SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = saved_path
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _rising_then_flat() -> np.ndarray:
    """No reading before click 20; 0.2 to 0.8 over clicks 20-59; 0.8 from click 59 on."""
    sig = np.full(150, np.nan)
    sig[19:] = np.r_[np.linspace(0.2, 0.8, 40), np.full(91, 0.8)]
    return sig


class TestWhenARuleFires:
    def test_the_running_best_stops_rising_a_window_after_the_signal_does(self, S):
        assert S.rule_stop(_rising_then_flat(), 20, 20, "best", 10, 0.0) == 69

    def test_the_windowed_mean_waits_for_both_windows_to_be_flat(self, S):
        assert S.rule_stop(_rising_then_flat(), 20, 20, "mean", 10, 0.0) == 78

    def test_a_rule_read_from_a_later_click_fires_there_if_it_already_holds(self, S):
        assert S.rule_stop(_rising_then_flat(), 20, 100, "mean", 10, 0.0) == 100

    def test_a_signal_still_rising_never_fires(self, S):
        sig = np.full(150, np.nan)
        sig[19:] = np.linspace(0.0, 1.0, 131)
        assert S.rule_stop(sig, 20, 20, "best", 10, 0.0) is None

    def test_a_level_rule_fires_at_the_first_click_the_reading_clears_it(self, S):
        sig = np.full(150, np.nan)
        sig[29:] = np.linspace(0.3, 0.9, 121)
        assert S.level_stop(sig, 30, 0.5) == 70
        assert S.level_stop(sig, 100, 0.5) == 100
        assert S.level_stop(sig, 30, 0.95) is None


def _sessions(stops: list[float]) -> pd.DataFrame:
    """Two kinds of session: 'flat' already at 0.5 by click 1, 'climb' rising from 0 to 1 over the 150 clicks."""
    rows = []
    for i, stop in enumerate(stops):
        climb = i % 2 == 1
        fb = np.linspace(0.0, 1.0, 150) if climb else np.full(150, 0.5)
        rows.append(
            {
                "category": f"c{i // 2}@large",
                "seed": 0,
                "stop[r]": stop,
                **{f"fbeta@{c}": float(fb[c - 1]) for c in range(1, 151)},
            }
        )
    return pd.DataFrame(rows)


class TestTheTwoEdges:
    def test_stops_given_to_the_sessions_still_climbing_beat_the_same_stops_dealt_at_random(self, S):
        # Flat sessions stop at 10, climbing ones at 150: the clicks go where they pay.
        good = S.score(_sessions([10, 150] * 4), "stop[r]")
        # The same two stops, the other way round.
        bad = S.score(_sessions([150, 10] * 4), "stop[r]")
        assert good["shuffle_edge"] > 0 > bad["shuffle_edge"]
        assert good["shuffle_edge"] == pytest.approx(-bad["shuffle_edge"])
        assert good["mean_leave"] == bad["mean_leave"] == 80

    def test_a_rule_that_never_fires_leaves_at_the_horizon(self, S):
        s = S.score(_sessions([float("nan")] * 4), "stop[r]")
        assert s["fire_rate"] == 0.0 and s["mean_leave"] == 150 and s["gain_after"] == 0.0

    def test_one_click_for_every_session_has_no_edge_either_way(self, S):
        s = S.score(_sessions([80] * 8), "stop[r]")
        assert s["edge"] == pytest.approx(0.0) and s["shuffle_edge"] == pytest.approx(0.0)
