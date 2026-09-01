"""Integration tests for the #2781 calibration harness (torch; run on a GPU/CPU box).

Exercises the ``emit_calibration_metrics`` path end-to-end on the synthetic
planted patch dataset, and — most importantly — proves the base (max) pooling's
trained threshold is **byte-identical** to production's grouped cross-calibration,
so adding the study's instrumentation did not perturb the base arms.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.patch_styles import resolve_style
from vtscore.eval.voting_columns import CALIBRATION_COLUMNS
from vtscore.eval.voting_iterations import _calibrate_with_details, simulate_voting_iterations
from vtscore.training.mlp import _auto_hidden_dim
from vtscore.training.thresholds import calculate_cross_calibration_threshold

# Reuse the synthetic planted-patch dataset builders from the Max-Patch tests.
from .test_max_patch_style import _planted_dataset


def _grouped_vote_inputs(style_name, n_good=6, n_bad=6, seed=0):
    """Assemble (X_list, y_list, cal_groups, score_rows_by_group, hidden_dim, input_dim).

    Mirrors ``_style_train_and_calibrate``'s vote-to-vector assembly so the two
    threshold entry points get identical inputs.
    """
    from vtscore.detectors.training import _flood_context

    medias, _ = _planted_dataset(n_per_cat=max(n_good, n_bad) + 2, seed=seed)
    style = resolve_style(style_name)
    goods = [m for m in medias.values() if m["category"] == "cat0"][:n_good]
    bads = [m for m in medias.values() if m["category"] == "cat1"][:n_bad]

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    groups: list = []
    score_rows_by_group: dict = {}
    for m in goods:
        X_list.append(np.asarray(style.good_vec(m, None), dtype=np.float32))
        y_list.append(1.0)
        groups.append(("g", m["id"]))
        score_rows_by_group[("g", m["id"])] = style.score_rows(m)
    for m in bads:
        for vec in style.bad_vecs(m):
            X_list.append(np.asarray(vec, dtype=np.float32))
            y_list.append(0.0)
            groups.append(("b", m["id"]))
        score_rows_by_group[("b", m["id"])] = style.score_rows(m)

    n_votes, cal_groups, _ = _flood_context(X_list, y_list, groups)
    hidden_dim = _auto_hidden_dim(n_votes)
    input_dim = int(X_list[0].shape[0])
    return X_list, y_list, cal_groups, score_rows_by_group, hidden_dim, input_dim


def test_base_threshold_is_byte_identical_to_production():
    """The emit-path base (max) threshold equals production cross-calibration."""
    X_list, y_list, cal_groups, srbg, hidden_dim, input_dim = _grouped_vote_inputs("max_patch_pca_hac")
    assert cal_groups is not None  # the tree arm floods, so calibration is grouped

    prod = calculate_cross_calibration_threshold(
        X_list,
        y_list,
        input_dim,
        0,
        rng=np.random.RandomState(42),
        calibrate_count=2,
        calibration_fraction=0.5,
        hidden_dim=hidden_dim,
        groups=cal_groups,
        score_rows_by_group=srbg,
    )
    emit, details = _calibrate_with_details(
        X_list,
        y_list,
        input_dim,
        0,
        calibrate_count=2,
        calibration_fraction=0.5,
        hidden_dim=hidden_dim,
        cal_groups=cal_groups,
        score_rows_by_group=srbg,
    )
    assert emit == prod  # exact float equality
    assert details["provenance"] == "conformal"
    assert details["fold_node_data"] is not None


def test_wholeimage_base_threshold_is_byte_identical():
    """Row-wise (whole-image) path also matches production exactly."""
    X_list, y_list, cal_groups, srbg, hidden_dim, input_dim = _grouped_vote_inputs("whole_image")
    assert cal_groups is None  # whole-image does not flood -> row-wise calibration

    prod = calculate_cross_calibration_threshold(
        X_list,
        y_list,
        input_dim,
        0,
        rng=np.random.RandomState(42),
        calibrate_count=2,
        calibration_fraction=0.5,
        hidden_dim=hidden_dim,
        groups=None,
        score_rows_by_group=None,
    )
    emit, details = _calibrate_with_details(
        X_list,
        y_list,
        input_dim,
        0,
        calibrate_count=2,
        calibration_fraction=0.5,
        hidden_dim=hidden_dim,
        cal_groups=None,
        score_rows_by_group=None,
    )
    assert emit == prod
    assert details["fold_node_data"] is None


def _run_emit(style, seed=0, variants=("topk", "pnorm")):
    """Emit the #2781 calibration frame on the **no-fusion control arm**.

    These tests pin the re-pooling plumbing and the conformal provenance
    ladder, both of which live upstream of the threshold estimator.  Running
    them with ``safe_thresholds=False`` keeps the base row's provenance the
    conformal one and keeps the safe-variant row family out of the frame; the
    shipped fused configuration is covered by ``test_safe_gmm_variant_rows`` and
    ``test_anchored_variant_rows``.
    """
    medias, _ = _planted_dataset(n_per_cat=40, seed=seed)
    sweep: list = []
    rows = simulate_voting_iterations(
        medias,
        target_category="cat0",
        seed=seed,
        dataset_name="planted",
        inclusion=0,
        region_voting=True,
        safe_thresholds=False,
        max_steps=18,
        style=style,
        emit_calibration_metrics=True,
        repool_variants=list(variants),
        inclusion_sweep_ks=[-2, 0, 1, 2],
        sweep_sink=sweep,
    )
    return rows, sweep


def test_tree_arm_emits_base_plus_remedial_rows():
    rows, sweep = _run_emit("max_patch_pca_hac")
    assert rows, "no rows produced"
    variants = {r["pool_variant"] for r in rows}
    assert variants == {"max", "topk", "pnorm"}
    # No duplicate (step, variant) rows.
    from collections import Counter

    per_step = Counter((r["t"], r["pool_variant"]) for r in rows)
    assert all(v == 1 for v in per_step.values())
    # Every step emits the base row; a step with a real conformal cut (enough
    # vote-groups to calibrate) also emits both remedial re-pools.  Early steps
    # that fall back to a sentinel/too-few threshold have no fold node data to
    # re-pool, so they legitimately carry only the base pooling.
    by_step: dict[int, dict[str, str]] = {}
    for r in rows:
        by_step.setdefault(r["t"], {})[r["pool_variant"]] = r["threshold_provenance"]
    saw_full = False
    for variants_at_t in by_step.values():
        assert "max" in variants_at_t
        if variants_at_t["max"] == "conformal":
            assert set(variants_at_t) == {"max", "topk", "pnorm"}
            saw_full = True
        else:
            assert set(variants_at_t) == {"max"}
    assert saw_full, "no step reached a conformal cut with remedial re-pools"


def test_calibration_columns_and_invariants():
    rows, sweep = _run_emit("max_patch_pca_hac")
    for r in rows:
        # every declared column present
        assert set(CALIBRATION_COLUMNS).issubset(r.keys())
        assert r["threshold_provenance"] in {"conformal", "no_good_sentinel", "too_few_default", "gmm_blend"}
        assert r["degenerate"] in (0, 1)
        # the oracle can never cost more than the trained cut -> regret >= 0
        if np.isfinite(r["regret"]):
            assert r["regret"] >= -1e-6
        assert r["oracle_cost"] <= r["cost"] + 1e-6
        # regret decomposition identity (when the calibration oracle exists)
        if np.isfinite(r["rule_inefficiency"]) and np.isfinite(r["calibration_shift"]):
            assert r["rule_inefficiency"] + r["calibration_shift"] == pytest.approx(r["regret"], abs=1e-5)
        # #3116: the honest re-decomposition telescopes the same way, off the
        # cross-fitted reference instead of the sample minimum.
        if np.isfinite(r["rule_inefficiency"]) and np.isfinite(r["calibration_shift_honest"]):
            assert r["rule_inefficiency"] + r["calibration_shift_honest"] == pytest.approx(r["regret_honest"], abs=1e-5)
        # `regret_honest` is the same subtraction off the other reference.
        # Deliberately NOT asserting `oracle_cost <= oracle_cost_honest` per
        # row: the cross-fitted rule applies a *different* cut per fold, so it
        # has freedom the single-threshold sample minimum does not and can beat
        # it on a given draw.  The bracket is a statement about the population
        # optimum, not a per-sample ordering, and asserting it here would be a
        # coin-flip test.
        if np.isfinite(r["oracle_cost_honest"]):
            assert r["regret_honest"] == pytest.approx(r["cost"] - r["oracle_cost_honest"], abs=1e-5)
        assert 0.0 <= r["threshold_percentile"] <= 1.0 or np.isnan(r["threshold_percentile"])
    # tree arm pools over many nodes; base n_pool_rows should exceed 1
    assert max(r["n_pool_rows"] for r in rows if r["pool_variant"] == "max") > 1.0


def test_inclusion_sweep_side_rows():
    rows, sweep = _run_emit("max_patch_pca_hac")
    assert sweep, "no inclusion-sweep rows"
    ks = {r["inclusion_k"] for r in sweep}
    assert ks == {-2, 0, 1, 2}
    for r in sweep:
        assert r["alpha"] == pytest.approx(0.25 * 2.0 ** (-r["inclusion_k"]))
        assert 0.0 <= r["sweep_fnr"] <= 1.0
        assert r["excess_fnr"] == pytest.approx(r["sweep_fnr"] - r["alpha"], abs=1e-6)


def test_wholeimage_arm_has_no_remedial_variants():
    rows, _ = _run_emit("whole_image")
    assert rows
    # whole-image carries no fold_node_data, so only the base pooling is emitted
    assert {r["pool_variant"] for r in rows} == {"max"}
    assert all(r["n_pool_rows"] == pytest.approx(1.0) for r in rows)
