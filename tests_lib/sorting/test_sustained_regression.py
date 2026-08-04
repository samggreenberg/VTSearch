"""Pure-core tests for the sustained wrong-way-trend detector (issue #2825).

The science that matters here is the *discrimination*: a persistent climb in held-out cost
must fire, and a transient deep spike (the #2790 phenomenon, already studied by
``deep_spikes.py``) must NOT — nor may sub-delta drift or plain noise. Everything below runs
on synthetic trajectories, so no sweep data, cache, model or GPU is needed. The tool lives
under ``scripts/experiments/threshold_stability``; the test puts that dir on ``sys.path`` the
same way the runner does.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "threshold_stability"))

from sustained_regression import (  # noqa: E402
    Run,
    analysis_window,
    characterise,
    detect,
    find_best_late_gap,
    find_sustained_rise,
    null_severities,
    permutation_p,
    permuted_copy,
    smooth_median,
)


class _Cfg:
    """Stand-in for the argparse namespace the detector reads."""

    k = 5
    delta = 0.10
    hold = 5
    min_up_frac = 0.5
    smooth = 3
    late_window = 10
    regime = "learned"
    null_mode = "demean"
    null_block = 5
    per_run_null = 99
    null_alpha = 0.05
    top = 5


def _run(costs, **extra):
    steps = []
    for i, c in enumerate(costs):
        s = {"t": i + 1, "cost": c, "calib_mode": "conformal", "n_good": 3, "n_bad": 4}
        for k, v in extra.items():
            s[k] = v[i] if isinstance(v, list) else v
        steps.append(s)
    return Run(key=("k",), meta={"head": "mlp", "class": "car", "seed": 0}, steps=steps)


# --------------------------------------------------------------------------- smoothing


def test_smooth_median_removes_single_step_spike():
    xs = [0.1, 0.1, 0.9, 0.1, 0.1]
    assert smooth_median(xs, 3) == [0.1, 0.1, 0.1, 0.1, 0.1]


def test_smooth_median_keeps_a_sustained_level_change():
    xs = [0.1] * 5 + [0.5] * 5
    out = smooth_median(xs, 3)
    assert out[:4] == [0.1] * 4
    assert out[-4:] == [0.5] * 4


# --------------------------------------------------------------------------- rise finder


def test_sustained_rise_fires_on_a_persistent_climb():
    s = [0.10, 0.12, 0.16, 0.22, 0.28, 0.34, 0.40, 0.40, 0.41, 0.40, 0.42, 0.41]
    hit = find_sustained_rise(s, k=5, delta=0.10, hold=5, min_up_frac=0.5)
    assert hit is not None
    assert hit["rise"] > 0.25
    assert hit["span"] >= 5
    assert hit["held_to_end"]


def test_transient_spike_does_not_fire():
    """The #2790 deep spike: one violent step that snaps straight back."""
    s = [0.08] * 10 + [0.95] + [0.08] * 10
    assert find_sustained_rise(smooth_median(s, 3), k=5, delta=0.10, hold=5, min_up_frac=0.5) is None


def test_two_step_excursion_does_not_fire():
    s = [0.08] * 10 + [0.60, 0.55] + [0.08] * 10
    assert find_sustained_rise(smooth_median(s, 3), k=5, delta=0.10, hold=5, min_up_frac=0.5) is None


def test_sub_delta_drift_does_not_fire():
    s = [0.20 + 0.004 * i for i in range(20)]  # real trend, but only +0.076 total
    assert find_sustained_rise(s, k=5, delta=0.10, hold=5, min_up_frac=0.5) is None


def test_rise_that_is_given_back_does_not_fire():
    """Climbs by 0.3 but returns to the floor immediately: not sustained."""
    s = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10]
    assert find_sustained_rise(s, k=5, delta=0.10, hold=5, min_up_frac=0.5) is None


def test_step_change_that_never_recovers_fires():
    """Not a gradual climb, but the cost steps up and simply stays there: in scope."""
    s = [0.10, 0.45, 0.46, 0.47] + [0.46] * 8
    hit = find_sustained_rise(s, k=5, delta=0.10, hold=5, min_up_frac=0.5)
    assert hit is not None
    assert hit["held_to_end"]


def test_a_run_that_ends_elevated_is_not_penalised_for_having_no_room_left():
    """The rise peaks on the last step — there is nowhere left to recover, so it counts."""
    s = [0.10] * 3 + [0.14, 0.20, 0.27, 0.33, 0.40]
    assert find_sustained_rise(s, k=5, delta=0.10, hold=5, min_up_frac=0.5) is not None


def test_long_flat_then_permanent_step_is_caught_by_the_late_gap_rule():
    run = _run([0.08] * 14 + [0.55] * 14)
    det = detect(run, _Cfg())
    assert det is not None
    assert det["kind"] == "late-gap"


def test_improving_run_does_not_fire():
    s = [0.60 - 0.02 * i for i in range(25)]
    assert find_sustained_rise(s, k=5, delta=0.10, hold=5, min_up_frac=0.5) is None


# --------------------------------------------------------------------------- late-gap finder


def test_late_gap_fires_when_the_end_sits_above_an_earlier_best():
    s = [0.5, 0.3, 0.1] + [0.2] * 5 + [0.35] * 10
    hit = find_best_late_gap(s, k=5, delta=0.10, late_window=10)
    assert hit is not None
    assert hit["best"] == 0.1
    assert round(hit["gap"], 3) == 0.25


def test_late_gap_silent_when_the_run_ends_at_its_best():
    s = [0.5] * 5 + [0.3] * 5 + [0.1] * 10
    assert find_best_late_gap(s, k=5, delta=0.10, late_window=10) is None


def test_late_gap_ignores_a_best_at_the_very_end():
    s = [0.4] * 12 + [0.1] * 10
    assert find_best_late_gap(s, k=5, delta=0.10, late_window=10) is None


# --------------------------------------------------------------------------- regime window


def test_analysis_window_drops_cold_start():
    run = _run([0.5] * 6)
    for i in range(3):
        run.steps[i]["calib_mode"] = "cosine_coldstart"
    assert analysis_window(run, "learned") == [3, 4, 5]
    assert analysis_window(run, "coldstart") == [0, 1, 2]
    assert analysis_window(run, "all") == [0, 1, 2, 3, 4, 5]


def test_detect_ignores_a_cold_start_handoff_jump():
    """A jump that happens exactly at the cosine->learned handoff is out of the window."""
    run = _run([0.05] * 8 + [0.55] * 12)
    for i in range(8):
        run.steps[i]["calib_mode"] = "cosine_coldstart"
    assert detect(run, _Cfg()) is None


# --------------------------------------------------------------------------- characterisation


def test_characterise_splits_ranking_from_calibration():
    costs = [0.10, 0.14, 0.20, 0.26, 0.32, 0.38, 0.40, 0.40, 0.41, 0.40, 0.42, 0.41]
    # oracle flat => the whole rise is calibration regret
    calib = _run(costs, oracle_cost=[0.09] * len(costs), fnr=[c * 0.9 for c in costs], fpr=[0.01] * len(costs))
    det = detect(calib, _Cfg())
    assert det is not None
    ch = characterise(calib, det)
    assert ch["failure"] == "calibration"
    assert ch["driver"] == "fnr"

    # oracle rises with cost => the ranking itself degraded (the scary case)
    rank = _run(costs, oracle_cost=[c - 0.02 for c in costs], fnr=[0.01] * len(costs), fpr=[c * 0.9 for c in costs])
    det2 = detect(rank, _Cfg())
    ch2 = characterise(rank, det2)
    assert ch2["failure"] == "ranking"
    assert ch2["ranking_share"] > 0.9
    assert ch2["driver"] == "fpr"


def test_a_source_without_oracle_cost_is_undecided_not_judged_on_ap():
    """#2825 is a cost=FNR+FPR question; AP is a different metric and must not classify."""
    costs = [0.10, 0.14, 0.20, 0.26, 0.32, 0.38, 0.40, 0.40, 0.41, 0.40, 0.42, 0.41]
    run = _run(costs, average_precision=[0.8 - 0.02 * i for i in range(len(costs))])
    det = detect(run, _Cfg())
    ch = characterise(run, det)
    assert ch["d_ap"] < -0.01  # still reported...
    assert ch["ranking_share"] is None  # ...but it decides nothing
    assert ch["failure"] == "undecided-no-oracle-cost"


def test_detection_is_driven_by_cost_not_by_ap():
    """AP collapsing while cost stays flat is not a #2825 wrong-way run."""
    flat = [0.20] * 24
    run = _run(flat, average_precision=[0.9 - 0.03 * i for i in range(24)])
    assert detect(run, _Cfg()) is None


# --------------------------------------------------------------------------- null control


def test_raw_surrogate_preserves_the_delta_multiset_and_endpoints():
    run = _run([0.1, 0.3, 0.2, 0.6, 0.5, 0.4])
    perm = permuted_copy(run, random.Random(0), mode="raw")
    orig = sorted(round(b - a, 9) for a, b in zip(run.series("cost"), run.series("cost")[1:]))
    got = sorted(round(b - a, 9) for a, b in zip(perm.series("cost"), perm.series("cost")[1:]))
    assert orig == got
    assert perm.series("cost")[0] == run.series("cost")[0]
    # ...which is exactly why `raw` cannot test a trend: the end point is fixed too.
    assert round(perm.series("cost")[-1], 9) == round(run.series("cost")[-1], 9)


def test_demeaned_surrogate_keeps_the_volatility_but_drops_the_drift():
    costs = [0.10 + 0.05 * i for i in range(12)]  # steady climb
    run = _run(costs)
    perm = permuted_copy(run, random.Random(0), mode="demean")
    seq = perm.series("cost")
    assert abs(seq[-1] - seq[0]) < 1e-9  # no net drift left
    spread = max(abs(b - a) for a, b in zip(seq, seq[1:]))
    assert spread < 1e-9  # a perfectly steady climb has zero volatility once demeaned


def test_demeaned_surrogate_does_not_flatten_a_jumpy_run():
    run = _run([0.1, 0.5, 0.15, 0.6, 0.2, 0.55])
    perm = permuted_copy(run, random.Random(3), mode="demean")
    seq = perm.series("cost")
    assert max(abs(b - a) for a, b in zip(seq, seq[1:])) > 0.2


def test_permuted_copy_leaves_other_fields_alone():
    run = _run([0.1, 0.3, 0.2, 0.6, 0.5, 0.4], fnr=[0.5] * 6)
    perm = permuted_copy(run, random.Random(1))
    assert perm.series("fnr") == [0.5] * 6
    assert perm.series("t") == [1, 2, 3, 4, 5, 6]


def test_permutation_p_is_one_sided_with_add_one():
    assert permutation_p(1.0, [0.0] * 99) == 1 / 100
    assert permutation_p(0.0, [1.0] * 99) == 100 / 100
    assert permutation_p(0.5, [1.0] * 49 + [0.0] * 50) == 50 / 100


def test_a_steady_climb_beats_its_own_null_but_a_jumpy_flat_run_does_not():
    cfg = _Cfg()
    rng = random.Random(0)
    climb = _run([0.10 + 0.02 * i for i in range(30)])
    det = detect(climb, cfg)
    assert det is not None
    assert permutation_p(det["severity"], null_severities(climb, cfg, 99, rng)) <= 0.05

    jumpy = _run([0.2, 0.55, 0.18, 0.6, 0.22, 0.58, 0.19, 0.62, 0.21, 0.57] * 3)
    d2 = detect(jumpy, cfg)
    if d2 is not None:
        assert permutation_p(d2["severity"], null_severities(jumpy, cfg, 99, rng)) > 0.05


# --------------------------------------------------------------------------- block surrogate


def test_block_shuffle_keeps_a_spike_next_to_its_recovery():
    """A +0.9 jump and its -0.9 snap-back must not be torn apart by the surrogate."""
    from sustained_regression import _shuffle_blocks

    deltas = [0.0, 0.0, 0.9, -0.9, 0.0, 0.0, 0.0, 0.0]
    for seed in range(20):
        out = _shuffle_blocks(list(deltas), block=4, rng=random.Random(seed))
        i = out.index(0.9)
        assert out[i + 1] == -0.9, out


def test_single_move_shuffle_does_tear_the_pair_apart():
    """Which is exactly why block=1 makes the null far too conservative here."""
    from sustained_regression import _shuffle_blocks

    deltas = [0.0, 0.0, 0.9, -0.9, 0.0, 0.0, 0.0, 0.0]
    torn = 0
    for seed in range(40):
        out = _shuffle_blocks(list(deltas), block=1, rng=random.Random(seed))
        i = out.index(0.9)
        if i + 1 >= len(out) or out[i + 1] != -0.9:
            torn += 1
    assert torn > 20


def test_block_surrogate_still_destroys_a_trend():
    run = _run([0.10 + 0.02 * i for i in range(40)])
    perm = permuted_copy(run, random.Random(0), mode="demean", block=5)
    seq = perm.series("cost")
    assert abs(seq[-1] - seq[0]) < 1e-9


def test_climb_off_a_long_plateau_is_found():
    """The floor must anchor at the END of the plateau, else up_frac sinks and the climb is lost."""
    s = [0.10] * 25 + [0.10 + 0.012 * i for i in range(1, 26)]
    hit = find_sustained_rise(s, k=5, delta=0.10, hold=5, min_up_frac=0.5)
    assert hit is not None
    assert hit["start_idx"] >= 24  # anchored at the end of the plateau, not index 0
    assert hit["up_frac"] > 0.9
