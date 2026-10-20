"""#5134: Smart in the objective's units - the rising rule, its scorers, and the eval arm.

What these pin:

1. :func:`rising_status_from_scores` is :func:`smart_status_from_costs` with the
   sign turned over: the same gates, green once the score stops *rising*.
2. The window scorers are what they say: average precision and F-beta, checked
   against brute force, NaN without both classes.
3. The flow's ``smart_signal`` arm reads its light off the reading it names, and
   the app's (``"cost"``) is untouched by readings handed alongside.
4. In the harness, recording the readings moves nothing (same phases, same
   picks), the reading columns agree with the rule, and an arm's light is the
   rising rule on its reading.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.detectors.cost_trend import (
    SMART_RISE_THRESHOLD,
    SMART_SLOPE_T,
    average_precision,
    best_fbeta,
    fbeta_at_cut,
    rising_status_from_scores,
    smart_status_from_costs,
)
from vtscore.detectors.stability import MIN_PER_CLASS
from vtscore.eval.autopilot_flow import (
    SMART_READING_COLUMNS,
    SMART_READINGS,
    SMART_SIGNALS,
    AutopilotFlow,
)
from vtscore.eval.step_trainers import _heldout_readings
from vtscore.eval.voting_columns import IDENT_COLUMNS, VOTING_COLUMNS
from vtscore.eval.voting_iterations import simulate_voting_iterations


class TestRisingRule:
    def test_red_below_the_class_minimum(self):
        assert rising_status_from_scores([0.5] * 10, MIN_PER_CLASS - 1, 10)["status"] == "red"

    def test_yellow_with_too_few_points(self):
        assert rising_status_from_scores([0.5, 0.6], 10, 10)["status"] == "yellow"

    def test_a_clean_rise_is_yellow(self):
        out = rising_status_from_scores([0.3, 0.5, 0.7, 0.9], 10, 10)
        assert out["status"] == "yellow"
        assert out["slope"] > SMART_RISE_THRESHOLD

    def test_flat_is_green(self):
        assert rising_status_from_scores([0.8] * 10, 10, 10)["status"] == "green"

    def test_a_fall_is_green(self):
        assert rising_status_from_scores([0.9, 0.8, 0.7, 0.6], 10, 10)["status"] == "green"

    def test_a_rise_inside_the_scatter_is_green(self):
        noisy = [0.5, 0.9, 0.4, 0.95, 0.45, 0.9, 0.5, 0.95, 0.5, 0.92]
        out = rising_status_from_scores(noisy, 10, 10)
        assert out["slope"] > SMART_RISE_THRESHOLD and out["slope_t"] < SMART_SLOPE_T
        assert out["status"] == "green" and out["drift_within_noise"]

    def test_mirror_of_the_cost_rule(self):
        """A score window and its negation get the same light from the two rules."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            w = list(1.0 + np.cumsum(rng.normal(rng.normal(0, 0.02), 0.03, size=10)))
            up = rising_status_from_scores(w, 10, 10)["status"]
            # The cost rule normalises by the mean, as this does; mirror around a level that keeps it positive.
            mirrored = [2 * np.mean(w) - v for v in w]
            assert up == smart_status_from_costs(mirrored, 10, 10)["status"]

    def test_non_finite_points_are_dropped(self):
        assert rising_status_from_scores([math.nan, 0.8, 0.8, 0.8], 10, 10)["status"] == "green"
        assert rising_status_from_scores([math.nan, math.nan, 0.8], 10, 10)["status"] == "yellow"


def _brute_ap(s, y):
    order = np.argsort(-s, kind="stable")
    hits = y[order].astype(bool)
    precisions = [hits[: k + 1].mean() for k in range(hits.size) if hits[k]]
    return float(np.mean(precisions))


def _brute_best_f(s, y, beta):
    best = 0.0
    n_pos = y.sum()
    for cut in np.unique(s):
        kept = s >= cut
        tp = (kept & (y > 0.5)).sum()
        best = max(best, (1 + beta**2) * tp / (beta**2 * n_pos + kept.sum()))
    return best


class TestScorers:
    @pytest.mark.parametrize("seed", range(5))
    def test_against_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        y = (rng.random(40) < 0.4).astype(float)
        y[:2] = [1.0, 0.0]
        s = rng.random(40) + 0.5 * y
        assert average_precision(s, y) == pytest.approx(_brute_ap(s, y))
        for beta in (0.25, 1.0, 4.0):
            assert best_fbeta(s, y, beta) == pytest.approx(_brute_best_f(s, y, beta))
            cut = float(np.median(s))
            kept = s >= cut
            tp = (kept & (y > 0.5)).sum()
            want = (1 + beta**2) * tp / (beta**2 * y.sum() + kept.sum())
            assert fbeta_at_cut(s, y, cut, beta) == pytest.approx(want)

    def test_ties_cut_together(self):
        s = np.array([0.9, 0.5, 0.5, 0.1])
        y = np.array([1.0, 1.0, 0.0, 0.0])
        # No cut separates the tied pair: the best set is the top item alone or the top three.
        assert best_fbeta(s, y, 1.0) == pytest.approx(max(2 * 1 / (2 + 1), 2 * 2 / (2 + 3)))

    def test_nan_without_both_classes(self):
        for y in (np.ones(5), np.zeros(5)):
            assert math.isnan(average_precision(np.arange(5.0), y))
            assert math.isnan(best_fbeta(np.arange(5.0), y, 1.0))
            assert math.isnan(fbeta_at_cut(np.arange(5.0), y, 2.0, 1.0))

    def test_heldout_pools_the_folds(self):
        folds = [(np.array([0.9, 0.2]), np.array([1.0, 0.0])), (np.array([0.8, 0.85]), np.array([0.0, 1.0]))]
        out = _heldout_readings(folds, 1.0)
        s = np.array([0.9, 0.2, 0.8, 0.85])
        y = np.array([1.0, 0.0, 0.0, 1.0])
        assert out["heldout_ap"] == pytest.approx(average_precision(s, y))
        assert out["heldout_fbeta"] == pytest.approx(best_fbeta(s, y, 1.0))
        assert math.isnan(_heldout_readings([], 1.0)["heldout_ap"])
        assert math.isnan(_heldout_readings(folds, None)["heldout_fbeta"])


RISING = [0.3, 0.4, 0.5, 0.6, 0.7]
FLAT = [0.7] * 5


def _flow(signal):
    flow = AutopilotFlow(smart_signal=signal, done_check=False, stop_rule="lights")
    return flow


class TestFlowArm:
    def test_unknown_signal_refused(self):
        with pytest.raises(ValueError, match="smart_signal"):
            AutopilotFlow(smart_signal="vibes")

    def test_signals(self):
        assert SMART_SIGNALS[0] == "cost" and set(SMART_SIGNALS[1:]) == set(SMART_READINGS)

    @pytest.mark.parametrize("window, light", [(RISING, "yellow"), (FLAT, "green")])
    def test_window_arm_reads_its_window(self, window, light):
        flow = _flow("ap")
        # A falling cost would hold the app's light yellow; the arm ignores it.
        flow.record_step([0.9, 0.7, 0.5, 0.3], None, windows={"ap": window, "fbeta": [], "fbeta_best": []})
        flow.update(10, 10, 100, None)
        assert flow.smart == light

    def test_history_arm_appends_finite_values(self):
        flow = _flow("line")
        for v in (None, math.nan, 0.5, 0.5, 0.5):
            flow.record_step([0.3] * 4, None, history={"line": v})
        assert flow.smart_reading("line") == [0.5, 0.5, 0.5]
        flow.update(10, 10, 100, None)
        assert flow.smart == "green"
        for v in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6):
            flow.record_step([0.3] * 4, None, history={"line": v})
        assert len(flow.smart_reading("line")) == 10
        flow.update(10, 10, 100, None)
        assert flow.smart == "yellow"

    def test_cost_light_ignores_readings(self):
        flow = _flow("cost")
        flow.record_step([0.9, 0.7, 0.5, 0.3], None, windows={"ap": FLAT})
        flow.update(10, 10, 100, None)
        assert flow.smart == "yellow"  # the cost is falling
        cols = flow.smart_reading_columns
        assert cols["smart_ap_mean"] == pytest.approx(0.7) and cols["smart_ap_last"] == pytest.approx(0.7)
        assert math.isnan(cols["smart_line_slope"])

    def test_columns_nan_below_the_class_minimum(self):
        flow = _flow("cost")
        flow.record_step([0.5] * 4, None, windows={"ap": FLAT})
        flow.update(MIN_PER_CLASS - 1, 10, 100, None)
        assert all(math.isnan(v) for v in flow.smart_reading_columns.values())


def _clips(n=120, dim=16, seed=7):
    rng = np.random.default_rng(seed)
    medias = {}
    for i in range(n):
        cat = "target" if i < n // 3 else "other"
        emb = rng.standard_normal(dim).astype(np.float32) + (1.0 if cat == "target" else -1.0)
        medias[i + 1] = {
            "id": i + 1,
            "embedder": "e5",
            "embeddings": {"e5": (emb / np.linalg.norm(emb)).astype(np.float32)},
            "category": cat,
        }
    return medias


def _run(**kw):
    return simulate_voting_iterations(
        _clips(), "target", seed=3, dataset_name="synthetic", max_steps=40, calibrate_count=1, **kw
    )


@pytest.fixture(scope="module")
def plain():
    return _run()


@pytest.fixture(scope="module")
def read():
    return _run(smart_readings=True)


def _green(r, name):
    """The rising rule's light, re-read off a row's reading columns (red/yellow folded together)."""
    slope, t = r[f"smart_{name}_slope"], r[f"smart_{name}_t"]
    if math.isnan(slope):
        return False
    return slope <= SMART_RISE_THRESHOLD or t < SMART_SLOPE_T


class TestHarness:
    def test_columns_declared(self, plain):
        for col in SMART_READING_COLUMNS:
            assert col in IDENT_COLUMNS and col in VOTING_COLUMNS
        assert all(set(r) == set(VOTING_COLUMNS) for r in plain)

    def test_off_by_default(self, plain):
        assert all(math.isnan(r[c]) for r in plain for c in SMART_READING_COLUMNS)

    def test_reading_moves_nothing(self, plain, read):
        keep = [c for c in VOTING_COLUMNS if c not in SMART_READING_COLUMNS and not c.endswith("_seconds")]
        assert len(plain) == len(read)
        for a, b in zip(plain, read, strict=True):
            for c in keep:
                va, vb = a[c], b[c]
                if isinstance(va, float) and math.isnan(va):
                    assert isinstance(vb, float) and math.isnan(vb), c
                else:
                    assert va == vb, (c, va, vb)

    def test_readings_filled(self, read):
        late = [r for r in read if r["t"] >= 30]
        assert late
        for name in ("ap", "heldout_ap"):
            assert all(not math.isnan(r[f"smart_{name}_mean"]) for r in late), name
            assert all(0.0 <= r[f"smart_{name}_last"] <= 1.0 for r in late), name

    def test_arm_light_is_the_rising_rule(self):
        rows = _run(smart_signal="ap")
        checked = 0
        for r in rows:
            if r["smart"] in ("green", "yellow") and not math.isnan(r["smart_ap_slope"]):
                assert (r["smart"] == "green") == _green(r, "ap"), r["t"]
                checked += 1
        assert checked > 10
