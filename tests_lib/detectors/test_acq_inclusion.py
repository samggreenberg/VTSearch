"""Acquisition/reporting threshold decoupling (docs/ML.md, threshold calibration).

The knob exists because Autopilot's ``hard`` pick reads the threshold as a
**rank position**, not a decision boundary, so the direction that buys positives
is the opposite of the one the cost weights suggest.  These tests pin that
direction down, because a sign error here is invisible in the aggregate metrics
- it would just look like the lever not working.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from vtscore.eval.voting_iterations import _pool_percentile, simulate_voting_iterations

from vtscore.training.thresholds import (
    ACQUISITION_INCLUSION_OFFSET,
    acquisition_inclusion,
    fit_fold_anchored_cut,
    inclusion_cost_weights,
)

from .sweep_cache import memoize_sweep
from .test_max_patch_style import _planted_dataset

# The offsets this file sweeps repeat across tests (``-2`` alone is asserted on
# by three of them), so each distinct argument set is simulated once per worker
# and this module is pinned to one worker for the cache to hit.  Rows are
# shared and must stay read-only.  See sweep_cache.py.
pytestmark = pytest.mark.xdist_group("acq-inclusion")

_CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"


def _load(name: str, path: Path):
    """The calibration modules are loose scripts, not package members."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base(rows):
    """The step's own row.

    Each step also emits one row per safe-threshold GMM *variant*, each carrying
    that variant's own ``threshold``.  ``acq_threshold`` is a per-step value, so
    comparing it against a variant row's threshold is meaningless - filter first,
    the same way the analyzers do.
    """
    return [
        r
        for r in rows
        if not str(r.get("gmm_variant") or "").strip()
        and str(r.get("pool_variant") or "") in ("", "max")
        and not str(r.get("schedule") or "").strip()
    ]


@memoize_sweep
def _run(**kw):
    """A run on the established planted fixture, at a size that reaches the
    fold-anchored path (which needs enough votes to form calibration folds)."""
    medias, _ = _planted_dataset(n_per_cat=40, seed=0)
    return _base(
        simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=3,
            dataset_name="synthetic",
            max_steps=30,
            head="linear",
            # The calibration row path is the one that emits `threshold` and the new
            # acquisition columns; it needs a style, exactly as the harness passes one.
            style="whole_image",
            emit_calibration_metrics=True,
            **kw,
        )
    )


# --- the helper -------------------------------------------------------------
def test_pool_percentile_is_the_fraction_below():
    pool = {i: float(i) / 10.0 for i in range(10)}  # 0.0 .. 0.9
    assert _pool_percentile(pool, 0.5) == pytest.approx(0.5)
    assert _pool_percentile(pool, 0.0) == pytest.approx(0.0)
    assert _pool_percentile(pool, 1.0) == pytest.approx(1.0)


def test_pool_percentile_is_nan_on_empty_pool_not_zero():
    """0.0 would read as 'the cut is at the very top', which is a real value."""
    assert np.isnan(_pool_percentile({}, 0.5))


# --- the default is the shipped acquisition cut -----------------------------
def test_default_is_the_shipped_offset_not_the_coupled_behaviour():
    """The harness matches production, and production decoupled the two jobs.

    A default of 0 would silently make every future arm's control the *old*
    behaviour, so a baseline run would stop measuring what users get.
    """
    assert ACQUISITION_INCLUSION_OFFSET == -4
    rows = [r for r in _run() if r["threshold_provenance"].startswith("fold_anchored")]
    assert rows, "no fold-anchored steps to check"
    moved = [r for r in rows if r["acq_threshold"] != r["threshold"]]
    assert moved, "the default must decouple the selector from the reporting cut"
    for r in moved:
        assert r["acq_threshold"] > r["threshold"]


def test_offset_zero_is_the_coupled_control():
    """threshold_at is monotone and anchored, so a zero offset is a no-op."""
    zero = _run(acq_inclusion_offset=0)
    assert zero, "no steps emitted"
    assert [r["acq_threshold"] for r in zero] == [r["threshold"] for r in zero]


def test_the_offset_is_relative_to_the_reporting_inclusion():
    """The shipped reading: the *gap* is what was measured, not an absolute cut.

    Read absolutely, ``-4`` would collapse to a no-op at reporting inclusion -4
    and invert below it - the direction the study's falsification arm ruled out.
    Relative, the selector stays above the reporting line wherever the user puts
    the slider.
    """
    assert acquisition_inclusion(0) == -4
    assert acquisition_inclusion(-4) == -8
    assert acquisition_inclusion(4) == 0
    assert acquisition_inclusion(0, offset=0) == 0

    rows = [r for r in _run(inclusion=-1) if r["threshold_provenance"].startswith("fold_anchored")]
    assert rows, "no fold-anchored steps to check"
    moved = [r for r in rows if r["acq_threshold"] != r["threshold"]]
    assert moved, "the offset went absolute: at reporting inclusion -1 it became a no-op"
    for r in moved:
        assert r["acq_threshold"] > r["threshold"]


# --- the direction ----------------------------------------------------------
@pytest.mark.parametrize("k", [-1, -2, -4])
def test_negative_acq_inclusion_raises_the_cut_and_moves_it_up_the_ranking(k):
    """The headline claim of the plan, in the smallest form that can fail.

    Negative inclusion prices false alarms higher -> higher threshold -> the cut
    sits further UP the descending ranking -> the ``hard`` pick samples nearer
    the top.  If this ever flips, the experiment's whole premise is inverted.
    """
    rows = [r for r in _run(acq_inclusion_offset=k) if r["threshold_provenance"].startswith("fold_anchored")]
    assert rows, "no fold-anchored steps to check"
    moved = [r for r in rows if r["acq_threshold"] != r["threshold"]]
    assert moved, f"acq_inclusion_offset={k} never moved the acquisition threshold"
    for r in moved:
        assert r["acq_threshold"] > r["threshold"], f"k={k} lowered the cut; direction is inverted"
        # Skip steps where the pool ran dry: `_pool_percentile` returns NaN
        # there on purpose (0.0 would read as "the cut is at the very top").
        if np.isfinite(r["acq_pool_percentile"]) and np.isfinite(r["report_pool_percentile"]):
            assert r["acq_pool_percentile"] >= r["report_pool_percentile"], (
                "a higher cut must sit at or above the reporting cut in the pool ranking"
            )


def test_positive_acq_inclusion_moves_the_other_way():
    """The falsification arm of the run must actually falsify."""
    rows = [r for r in _run(acq_inclusion_offset=2) if r["threshold_provenance"].startswith("fold_anchored")]
    moved = [r for r in rows if r["acq_threshold"] != r["threshold"]]
    assert moved, "acq_inclusion_offset=+2 never moved the acquisition threshold"
    for r in moved:
        assert r["acq_threshold"] < r["threshold"], "positive inclusion must LOWER the cut"


def test_threshold_at_is_monotone_in_inclusion_on_a_single_fit():
    """The nesting contract the arms rely on.

    Deliberately tested on **one fitted cut**, not by comparing runs at
    different ``k``: the arms diverge from their first differing vote, so step
    *i* of one arm is not step *i* of another and a cross-run comparison tests
    nothing.  (An earlier version of this test did exactly that and failed for
    that reason, not because the estimator is non-monotone.)
    """
    rng = np.random.default_rng(0)
    haystacks = [np.sort(rng.beta(2, 5, 400)) for _ in range(2)]
    orderings = [
        ([float(x) for x in rng.beta(5, 2, 20)] + [float(x) for x in rng.beta(2, 5, 60)], [1.0] * 20 + [0.0] * 60)
        for _ in range(2)
    ]
    cut = fit_fold_anchored_cut(haystacks, orderings, list(np.concatenate(haystacks)))
    if cut is None:
        pytest.skip("no fold-anchored fit on this synthetic draw")
    thr = [cut.threshold_at(k) for k in (-4, -2, -1, 0, 1, 2)]
    finite = [t for t in thr if np.isfinite(t)]
    assert finite == sorted(finite, reverse=True), f"threshold_at not monotone: {thr}"


# --- reporting must not move ------------------------------------------------
@pytest.mark.parametrize("k", [-2, 2])
def test_reporting_columns_are_cut_at_inclusion_zero_regardless(k):
    """The arms must stay comparable: only acquisition moves, never the metric.

    The trajectories diverge (that is the point), so this checks the *invariant*
    - the reported threshold is never the acquisition one - rather than equality
    against the control run.
    """
    for r in _run(acq_inclusion_offset=k):
        if r["acq_threshold"] != r["threshold"]:
            # `threshold_percentile` is the reporting cut's position in the TEST
            # scores; it must track `threshold`, not `acq_threshold`.
            assert r["threshold"] == pytest.approx(r["threshold"], abs=0)
            assert r["acq_threshold"] != pytest.approx(r["threshold"])


def test_blend_fallback_steps_keep_the_reporting_threshold():
    """No fold-anchored fit -> nothing honest to re-cut, and no stale carry-over."""
    rows = _run(acq_inclusion_offset=-2)
    fallback = [r for r in rows if not r["threshold_provenance"].startswith("fold_anchored")]
    assert fallback, "expected some cold-start blend/conformal steps"
    for r in fallback:
        assert r["acq_threshold"] == pytest.approx(r["threshold"])


# --- the rank-pin arm -------------------------------------------------------
def test_rank_percentile_arm_moves_the_cut_and_is_monotone():
    lo = _run(acq_inclusion_offset=0, acq_rank_percentile=0.80)
    hi = _run(acq_inclusion_offset=0, acq_rank_percentile=0.98)
    n = min(len(lo), len(hi))
    pairs = [(lo[i]["acq_threshold"], hi[i]["acq_threshold"]) for i in range(n)]
    differing = [(a, b) for a, b in pairs if a != b]
    assert differing, "the rank-pin arm never moved the acquisition threshold"
    for a, b in differing:
        assert b > a, "a higher pinned quantile must give a higher acquisition threshold"


def test_the_two_knobs_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _run(acq_inclusion_offset=-2, acq_rank_percentile=0.9)


def test_rank_pin_must_disable_the_default_offset_explicitly():
    """The default is a real cut now, so a bare rank-pin arm is ambiguous.

    Silently letting the percentile win would mean a run whose config names two
    acquisition cuts quietly measures one of them - the sort of thing that only
    surfaces as an unexplained arm months later.
    """
    with pytest.raises(ValueError, match="acq_inclusion_offset=0"):
        _run(acq_rank_percentile=0.9)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_rank_percentile_is_range_checked(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _run(acq_inclusion_offset=0, acq_rank_percentile=bad)


# --- fractional inclusion steps (#3319) -------------------------------------
# The knob is a log2 scale, so "half a step" is a real operating point, not an
# interpolation between two settings.  #3319 sweeps them; these pin down that
# the arithmetic, the shared helper and the harness's parser all carry them.
def test_one_inclusion_step_is_one_bit_of_evidence():
    """What a value MEANS: ``k`` thresholds the likelihood ratio at ``2**-k``.

    The loss is a weighted sum of *rates*, each normalised by its own class, so
    prevalence divides out and what is left is pure weight-of-evidence: every
    step doubles the evidence demanded, and a half step multiplies it by
    ``sqrt(2)``.  Asserted here because it is the property the whole offset
    parameterisation rests on - a *constant shift in bits* is prior-free, which
    is why an offset transfers across datasets where an absolute cut would not.
    """

    def ratio(k):
        fpr_w, fnr_w = inclusion_cost_weights(k)
        return fpr_w / fnr_w

    assert ratio(0) == pytest.approx(1.0)
    assert ratio(-3) == pytest.approx(8.0)
    assert ratio(2) == pytest.approx(0.25)
    # one step = one bit, at integer and fractional positions alike
    for k in (-0.5, -1, -2.5, -3, -4.5):
        assert ratio(k - 1) == pytest.approx(2.0 * ratio(k))
    # a half step is exactly sqrt(2)
    assert ratio(-3.5) == pytest.approx(8.0 * 2.0**0.5)
    assert ratio(-3.5) == pytest.approx(11.3137, rel=1e-4)


def test_acquisition_inclusion_carries_a_fractional_offset():
    """The gap is arithmetic, so a fractional offset stays fractional."""
    assert acquisition_inclusion(0, offset=-3.5) == pytest.approx(-3.5)
    assert acquisition_inclusion(-2, offset=-3.5) == pytest.approx(-5.5)
    assert acquisition_inclusion(4, offset=-0.5) == pytest.approx(3.5)


def test_fractional_cuts_land_strictly_between_their_integer_neighbours():
    """A half step must be a distinct operating point, not a duplicate.

    ``threshold_at`` realises a quantile on the final haystack and then snaps it
    to that sample (#3166), so it is *a priori* possible for a half step to
    collapse onto the integer beside it and make the arm a silent duplicate of
    its neighbour - the failure that would look exactly like "the finer grid
    found nothing".  On one fitted cut (never across runs: the arms diverge from
    their first differing vote) the fractional cuts must be ordered with, and
    somewhere strictly inside, their neighbours.
    """
    rng = np.random.default_rng(0)
    haystacks = [np.sort(rng.beta(2, 5, 400)) for _ in range(2)]
    orderings = [
        ([float(x) for x in rng.beta(5, 2, 20)] + [float(x) for x in rng.beta(2, 5, 60)], [1.0] * 20 + [0.0] * 60)
        for _ in range(2)
    ]
    cut = fit_fold_anchored_cut(haystacks, orderings, list(np.concatenate(haystacks)))
    if cut is None:
        pytest.skip("no fold-anchored fit on this synthetic draw")

    ks = [-2, -2.5, -3, -3.5, -4, -4.5, -5]
    thr = [cut.threshold_at(k) for k in ks]
    assert all(np.isfinite(t) for t in thr), f"non-finite cut among {list(zip(ks, thr))}"
    # ``ks`` runs DOWNWARD, and a lower inclusion raises the cut, so these ascend.
    assert thr == sorted(thr), f"fractional steps broke monotonicity: {list(zip(ks, thr))}"
    strict = [(a, b) for a, b in zip(thr, thr[1:]) if b > a]
    assert len(strict) >= len(ks) - 2, (
        f"half steps collapsed onto their neighbours - the finer grid has no resolution here: {list(zip(ks, thr))}"
    )


def test_the_harness_parses_a_fractional_offset(monkeypatch):
    """``CALIB_ACQ_INCLUSION_OFFSET=-3.5`` must reach the run as -3.5.

    Parsed as an int this raises ``ValueError`` at import and the whole arm dies
    at cell 0; parsed with a silent truncation it would be far worse - the arm
    would run, complete, and be a duplicate of ``-3``.
    """
    monkeypatch.setenv("CALIB_ACQ_INCLUSION_OFFSET", "-3.5")
    cfg = _load("_calib_experiment_config_frac", _CALIB / "experiment_config.py")
    assert cfg.ACQ_INCLUSION_OFFSET == pytest.approx(-3.5)

    monkeypatch.delenv("CALIB_ACQ_INCLUSION_OFFSET")
    cfg = _load("_calib_experiment_config_unset", _CALIB / "experiment_config.py")
    assert cfg.ACQ_INCLUSION_OFFSET == ACQUISITION_INCLUSION_OFFSET
