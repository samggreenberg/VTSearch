"""The Smart indicator's rule, pinned where it lives (issue #3832).

:mod:`vtscore.detectors.cost_trend` is called by both the app's
``_compute_smart_status`` and the harness's ``smart_status``, so this is the one
place the rule's behaviour is asserted; the two callers' tests only check that
they pass their inputs through.

The scenario that motivated the noise gate, reproduced in the real harness on
the synthetic "dice" pool from #3831 (dice labeled by whether they rolled at
least 5, embedded only by their number of sides) before the rule was changed:
after the run reaches ``done`` the error cost is flat on average but jumps step
to step, and a slope fitted through ten such points dips below the flatness
threshold often enough to bounce Autopilot's phase display between Done and
Boundary.  Over six seeds the reproduction spent 20 post-``done`` steps with
Smart non-green before this rule and 7 after, all seven of them a real
step-down in cost (``t`` around -3).

The Monte-Carlo classes below are the general statement of the same thing: what
changed is that the false-yellow rate is now a property of the *test* rather
than of how noisy the pool happens to be.
"""

from __future__ import annotations

import math
import random

import pytest

from vtscore.detectors.cost_trend import (
    MIN_PER_CLASS,
    SMART_FLAT_THRESHOLD,
    SMART_MIN_POINTS,
    SMART_SLOPE_T,
    SMART_WINDOW,
    cost_trend,
    smart_status_from_costs,
)


#: One error-cost window the #3832 dice reproduction really produced, on a
#: category that had long since plateaued: noisy, not falling, and read as
#: "still declining" by the slope test alone.
PLATEAU_WINDOW = [0.861, 0.832, 0.861, 0.861, 0.861, 0.887, 0.755, 0.669, 0.699, 0.857]


def _status(costs, good: int = 9, bad: int = 9) -> str:
    return smart_status_from_costs(costs, good, bad)["status"]


def _flat_window(rng: random.Random, cv: float, mean: float = 0.4, n: int = SMART_WINDOW) -> list[float]:
    """A window with no trend at all, only step-to-step scatter of size *cv*."""
    return [max(0.0, rng.gauss(mean, cv * mean)) for _ in range(n)]


def _declining_window(
    rng: random.Random, per_step: float, cv: float, mean: float = 0.4, n: int = SMART_WINDOW
) -> list[float]:
    """A window genuinely losing *per_step* of the mean cost per step."""
    return [max(0.0, mean + per_step * mean * ((n - 1) / 2 - i) + rng.gauss(0.0, cv * mean)) for i in range(n)]


def _rate(windows) -> float:
    return sum(1 for w in windows if _status(w) == "yellow") / len(windows)


class TestConstants:
    def test_values(self):
        assert MIN_PER_CLASS == 5
        assert SMART_WINDOW == 10
        assert SMART_MIN_POINTS == 3
        assert SMART_FLAT_THRESHOLD == -0.015
        assert SMART_SLOPE_T == 2.0


class TestCostTrend:
    def test_slope_is_relative_to_the_window_mean(self):
        # 0.5 -> 0.4 -> 0.3 -> 0.2: -0.1 per step on a mean of 0.35.
        trend = cost_trend([0.5, 0.4, 0.3, 0.2])
        assert trend.relative_slope == pytest.approx(-0.1 / 0.35)

    def test_a_straight_line_has_no_residual_and_so_no_doubt(self):
        assert cost_trend([0.5, 0.4, 0.3, 0.2]).t_stat == -math.inf
        assert cost_trend([0.2, 0.3, 0.4, 0.5]).t_stat == math.inf

    def test_a_constant_window_has_no_slope_at_all(self):
        trend = cost_trend([0.3] * SMART_WINDOW)
        assert trend.relative_slope == 0.0
        assert trend.t_stat == 0.0

    def test_the_same_slope_is_less_significant_in_a_noisier_window(self):
        """The two windows below fall at the same rate; only one is believed.

        The second is the first plus a zigzag chosen to be orthogonal to the
        trend, so the fitted slope is unchanged to four decimals and every bit
        of the difference is in the scatter around it.
        """
        quiet = [0.500, 0.480, 0.460, 0.440, 0.420, 0.400, 0.380, 0.360, 0.340, 0.320]
        noisy = [0.565, 0.371, 0.536, 0.342, 0.507, 0.313, 0.478, 0.284, 0.449, 0.255]
        assert cost_trend(quiet).relative_slope == pytest.approx(cost_trend(noisy).relative_slope, rel=0.01)
        assert cost_trend(quiet).t_stat < -SMART_SLOPE_T < cost_trend(noisy).t_stat
        assert (_status(quiet), _status(noisy)) == ("yellow", "green")

    def test_only_the_last_window_of_costs_is_read(self):
        # An ancient collapse from 1.0 must not be regressed over forever.
        long_history = [1.0] * 20 + [0.3] * SMART_WINDOW
        assert _status(long_history) == "green"


class TestGates:
    def test_red_below_five_per_class(self):
        assert _status([0.5, 0.4, 0.3], good=MIN_PER_CLASS - 1) == "red"
        assert _status([0.5, 0.4, 0.3], bad=MIN_PER_CLASS - 1) == "red"

    def test_red_carries_no_trend_numbers(self):
        result = smart_status_from_costs([0.5, 0.4, 0.3], MIN_PER_CLASS - 1, 9)
        assert "slope" not in result and "slope_t" not in result

    def test_yellow_without_enough_points(self):
        assert _status([0.5, 0.4]) == "yellow"
        assert _status([]) == "yellow"

    def test_the_trend_numbers_ride_along_once_there_is_a_fit(self):
        result = smart_status_from_costs([0.5, 0.4, 0.3, 0.2], 9, 9)
        assert result["slope"] == pytest.approx(-0.2857, abs=1e-4)
        assert result["slope_t"] == -math.inf


class TestFlatness:
    def test_green_once_the_error_cost_levels_off(self):
        assert _status([0.30] * 4) == "green"

    def test_green_when_the_cost_is_rising(self):
        assert _status([0.2, 0.3, 0.4, 0.5]) == "green"

    def test_yellow_while_the_error_cost_is_still_falling(self):
        assert _status([0.9, 0.7, 0.5, 0.3]) == "yellow"

    def test_a_shallow_decline_is_not_worth_another_click(self):
        # Dead straight, so never in doubt - but only 1% of the mean per step.
        costs = [0.40 - 0.004 * i for i in range(SMART_WINDOW)]
        trend = cost_trend(costs)
        assert trend.t_stat == -math.inf
        assert trend.relative_slope > SMART_FLAT_THRESHOLD
        assert _status(costs) == "green"


class TestNoiseGate:
    """The half of the rule #3832 added: is the decline bigger than the jitter?"""

    def test_a_noisy_plateau_reads_green_even_when_the_slope_dips(self):
        # A window the dice reproduction really produced (seed 1, click 52):
        # flat around 0.85 with the step-to-step jumps a noisy labelset gives,
        # one dip, and a recovery.  The fitted slope is below the flatness
        # threshold, which is exactly what used to turn the light yellow with
        # nothing having changed about the detector.
        trend = cost_trend(PLATEAU_WINDOW)
        assert trend.relative_slope < SMART_FLAT_THRESHOLD
        assert trend.t_stat > -SMART_SLOPE_T
        assert _status(PLATEAU_WINDOW) == "green"

    def test_the_same_dip_in_a_quiet_window_is_believed(self):
        # The same decline without the jitter: now it is a trend, not a dip.
        costs = [0.86 - 0.014 * i for i in range(SMART_WINDOW)]
        trend = cost_trend(costs)
        assert trend.relative_slope < SMART_FLAT_THRESHOLD
        assert trend.relative_slope == pytest.approx(cost_trend(PLATEAU_WINDOW).relative_slope, abs=0.005)
        assert _status(costs) == "yellow"

    def test_the_reason_says_which_green_it_is(self):
        noisy = smart_status_from_costs(PLATEAU_WINDOW, 9, 9)
        settled = smart_status_from_costs([0.43] * SMART_WINDOW, 9, 9)
        assert noisy["status"] == settled["status"] == "green"
        assert "bounces around" in noisy["reason"]
        assert "bounces around" not in settled["reason"]


class TestFalseYellowRateIsNoiseInvariant:
    """A flat series must not turn the light yellow, however noisy it is.

    The pre-#3832 rule compared the slope with a fixed cutoff and nothing else,
    so its false-yellow rate grew with the pool's noise without bound - 8.6% of
    windows at a coefficient of variation of 0.1, 24% at 0.2, 36% at 0.4 on
    these same synthetic series.  Requiring the slope to clear its own standard
    error pins the rate at the test's tail area instead, wherever the noise
    sits.
    """

    @pytest.mark.parametrize("cv", [0.05, 0.1, 0.2, 0.4])
    def test_flat_series_stay_green(self, cv):
        rng = random.Random(20260913)
        windows = [_flat_window(rng, cv) for _ in range(2000)]
        assert _rate(windows) < 0.07

    def test_the_slope_alone_would_have_flapped_on_the_noisiest_of_them(self):
        """The control: the same windows under the old rule, to show it matters."""
        rng = random.Random(20260913)
        windows = [_flat_window(rng, 0.4) for _ in range(2000)]
        old_yellow = sum(1 for w in windows if cost_trend(w).relative_slope < SMART_FLAT_THRESHOLD)
        assert old_yellow / len(windows) > 0.25


class TestGenuineDeclinesSurvive:
    """The gate must not buy its quiet by going blind to real improvement."""

    def test_a_falling_cost_still_holds_the_light_yellow(self):
        rng = random.Random(20260913)
        windows = [_declining_window(rng, 0.05, 0.1) for _ in range(2000)]
        assert _rate(windows) > 0.9

    def test_even_under_heavy_noise_a_steep_decline_is_caught_most_of_the_time(self):
        rng = random.Random(20260913)
        windows = [_declining_window(rng, 0.10, 0.3) for _ in range(2000)]
        assert _rate(windows) > 0.7


class TestAppIndicatorPlumbing:
    """The app's ``/api/labeling-status`` Smart branch reads this same rule.

    ``_smart_status_uncached`` owns only the plumbing - which models are in the
    window, and what they cost against the current labelset - so what is
    asserted here is that the plumbing hands the window over unchanged, plus
    the two "not enough yet" branches that never reach the rule.
    """

    @staticmethod
    def _status(monkeypatch, costs, *, model_steps=None, good=9, bad=9):
        from vtscore.detectors import labeling_progress

        monkeypatch.setattr(
            labeling_progress,
            "_eval_cached_models",
            lambda cache, eval_set, inclusion_value, indices=None: [{"error_cost": c} for c in costs],
        )
        steps = list(range(len(costs))) if model_steps is None else model_steps
        return labeling_progress._smart_status_uncached(None, steps, None, 0, good, bad)

    def test_red_below_five_per_class(self, monkeypatch):
        result = self._status(monkeypatch, [0.3] * SMART_WINDOW, good=MIN_PER_CLASS - 1)
        assert result["status"] == "red"
        assert result["reason"] == smart_status_from_costs([], MIN_PER_CLASS - 1, 9)["reason"]

    def test_yellow_before_three_detectors_exist(self, monkeypatch):
        result = self._status(monkeypatch, [0.3, 0.3], model_steps=[0, 1])
        assert result["status"] == "yellow"
        assert "Sort to train one" in result["reason"]

    def test_the_window_reaches_the_rule_unchanged(self, monkeypatch):
        for costs in ([0.9, 0.7, 0.5, 0.3], [0.3] * SMART_WINDOW, PLATEAU_WINDOW):
            assert self._status(monkeypatch, costs) == smart_status_from_costs(costs, 9, 9)

    def test_a_noisy_plateau_no_longer_turns_the_light_yellow(self, monkeypatch):
        """The #3832 flap, at the level the user sees it."""
        assert self._status(monkeypatch, PLATEAU_WINDOW)["status"] == "green"
