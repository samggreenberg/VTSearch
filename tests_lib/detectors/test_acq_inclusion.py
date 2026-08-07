"""Acquisition/reporting threshold decoupling (docs/plans/acquisition-inclusion-decoupling.md).

The knob exists because Autopilot's ``hard`` pick reads the threshold as a
**rank position**, not a decision boundary, so the direction that buys positives
is the opposite of the one the cost weights suggest.  These tests pin that
direction down, because a sign error here is invisible in the aggregate metrics
- it would just look like the lever not working.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.voting_iterations import _pool_percentile, simulate_voting_iterations

from vtscore.training.thresholds import fit_fold_anchored_cut

from .test_max_patch_style import _planted_dataset


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


# --- the default is unchanged production ------------------------------------
def test_default_leaves_the_two_jobs_identical():
    rows = _run()
    assert rows, "no steps emitted"
    for r in rows:
        assert r["acq_threshold"] == pytest.approx(r["threshold"]), (
            "with no acquisition cut configured the selector must see the reporting threshold"
        )


def test_acq_inclusion_zero_reproduces_the_reporting_cut():
    """threshold_at is monotone and anchored, so inclusion 0 is a no-op."""
    base = _run()
    zero = _run(acq_inclusion=0)
    assert [r["threshold"] for r in base] == [r["threshold"] for r in zero]
    assert [r["acq_threshold"] for r in zero] == [r["threshold"] for r in zero]


# --- the direction ----------------------------------------------------------
@pytest.mark.parametrize("k", [-1, -2, -4])
def test_negative_acq_inclusion_raises_the_cut_and_moves_it_up_the_ranking(k):
    """The headline claim of the plan, in the smallest form that can fail.

    Negative inclusion prices false alarms higher -> higher threshold -> the cut
    sits further UP the descending ranking -> the ``hard`` pick samples nearer
    the top.  If this ever flips, the experiment's whole premise is inverted.
    """
    rows = [r for r in _run(acq_inclusion=k) if r["threshold_provenance"].startswith("fold_anchored")]
    assert rows, "no fold-anchored steps to check"
    moved = [r for r in rows if r["acq_threshold"] != r["threshold"]]
    assert moved, f"acq_inclusion={k} never moved the acquisition threshold"
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
    rows = [r for r in _run(acq_inclusion=2) if r["threshold_provenance"].startswith("fold_anchored")]
    moved = [r for r in rows if r["acq_threshold"] != r["threshold"]]
    assert moved, "acq_inclusion=+2 never moved the acquisition threshold"
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
    for r in _run(acq_inclusion=k):
        if r["acq_threshold"] != r["threshold"]:
            # `threshold_percentile` is the reporting cut's position in the TEST
            # scores; it must track `threshold`, not `acq_threshold`.
            assert r["threshold"] == pytest.approx(r["threshold"], abs=0)
            assert r["acq_threshold"] != pytest.approx(r["threshold"])


def test_blend_fallback_steps_keep_the_reporting_threshold():
    """No fold-anchored fit -> nothing honest to re-cut, and no stale carry-over."""
    rows = _run(acq_inclusion=-2)
    fallback = [r for r in rows if not r["threshold_provenance"].startswith("fold_anchored")]
    assert fallback, "expected some cold-start blend/conformal steps"
    for r in fallback:
        assert r["acq_threshold"] == pytest.approx(r["threshold"])


# --- the rank-pin arm -------------------------------------------------------
def test_rank_percentile_arm_moves_the_cut_and_is_monotone():
    lo = _run(acq_rank_percentile=0.80)
    hi = _run(acq_rank_percentile=0.98)
    n = min(len(lo), len(hi))
    pairs = [(lo[i]["acq_threshold"], hi[i]["acq_threshold"]) for i in range(n)]
    differing = [(a, b) for a, b in pairs if a != b]
    assert differing, "the rank-pin arm never moved the acquisition threshold"
    for a, b in differing:
        assert b > a, "a higher pinned quantile must give a higher acquisition threshold"


def test_the_two_knobs_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _run(acq_inclusion=-2, acq_rank_percentile=0.9)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_rank_percentile_is_range_checked(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _run(acq_rank_percentile=bad)
