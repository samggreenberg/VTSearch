"""Tests for the safe-threshold GMM variant rows in the eval harness (issue #2799).

With ``safe_thresholds=True`` and ``emit_calibration_metrics=True`` the harness
emits one extra metric row per GMM variant at every trainable step - the #2799
measurement arms.  The load-bearing invariant is that the ``pooled_cross``
variant (pooled fit, equal-density crossing, sigmoid space - i.e. exactly what
production computes) reproduces the base row's blended threshold bit-for-bit,
so the study measures the shipped code and every other variant differs from it
along exactly one named axis.
"""

from __future__ import annotations

import numpy as np

from vtscore.eval.voting_iterations import (
    _CALIBRATION_COLUMNS,
    _SAFE_GMM_VARIANTS,
    simulate_voting_iterations,
)

# Reuse the synthetic planted-patch dataset builders from the Max-Patch tests.
from .test_max_patch_style import _planted_dataset

_VARIANT_NAMES = {name for name, _fit, _cut, _space in _SAFE_GMM_VARIANTS}


def _run_safe(style, seed=0, max_steps=16):
    medias, _ = _planted_dataset(n_per_cat=40, seed=seed)
    return simulate_voting_iterations(
        medias,
        target_category="cat0",
        seed=seed,
        dataset_name="planted",
        inclusion=0,
        region_voting=True,
        safe_thresholds=True,
        max_steps=max_steps,
        style=style,
        emit_calibration_metrics=True,
    )


class TestSafeGmmVariantRows:
    def test_every_step_emits_base_plus_all_variants(self):
        rows = _run_safe("max_patch")
        assert rows, "no rows produced"
        by_step: dict[int, set[str]] = {}
        for r in rows:
            by_step.setdefault(r["t"], set()).add(r["gmm_variant"])
        for variants_at_t in by_step.values():
            assert variants_at_t == {"", *_VARIANT_NAMES}

    def test_columns_and_tags(self):
        rows = _run_safe("max_patch")
        for r in rows:
            assert set(_CALIBRATION_COLUMNS).issubset(r.keys())
        base = [r for r in rows if r["gmm_variant"] == ""]
        variant = [r for r in rows if r["gmm_variant"] != ""]
        assert base and variant
        # The production operating point is the base row; its provenance is the
        # blend tag, and it records the pre-blend conformal cut alongside.
        for r in base:
            assert r["threshold_provenance"] == "gmm_blend"
            assert np.isfinite(r["xcal_threshold"])
        for r in variant:
            assert r["pool_variant"] == "max"
            assert 0.0 <= r["blend_weight"] <= 1.0 or np.isnan(r["blend_weight"])

    def test_pooled_cross_reproduces_production_blend(self):
        rows = _run_safe("max_patch")
        base = {r["t"]: r for r in rows if r["gmm_variant"] == ""}
        pc = {r["t"]: r for r in rows if r["gmm_variant"] == "pooled_cross"}
        assert set(base) == set(pc)
        for t, b in base.items():
            assert pc[t]["threshold"] == b["threshold"], f"step {t}: variant diverged from production"

    def test_xcal_only_is_the_unblended_cut(self):
        rows = _run_safe("max_patch")
        base = {r["t"]: r for r in rows if r["gmm_variant"] == ""}
        xo = {r["t"]: r for r in rows if r["gmm_variant"] == "xcal_only"}
        for t, b in base.items():
            assert xo[t]["threshold"] == b["xcal_threshold"]
            # The control row keeps the pre-blend provenance, not the blend tag.
            assert xo[t]["threshold_provenance"] != "gmm_blend"

    def test_pure_gmm_below_the_ramp(self):
        """Below 6 votes the blend weight is 0, so blended == the GMM cut."""
        rows = _run_safe("max_patch")
        saw_pure = False
        for r in rows:
            if r["gmm_variant"] in ("", "xcal_only"):
                continue
            n_votes = r["n_good"] + r["n_bad"]
            if n_votes <= 6:
                assert r["blend_weight"] == 0.0
                assert r["threshold"] == r["gmm_cut"]
                saw_pure = True
        assert saw_pure, "no sub-ramp step reached - raise max_steps"

    def test_whole_image_collapses_fit_geometries(self):
        """On a single-vector style the pooled and image-level scores coincide,
        so the two fit geometries must produce identical cuts."""
        rows = _run_safe("whole_image", max_steps=12)
        by_step: dict[int, dict[str, dict]] = {}
        for r in rows:
            if r["gmm_variant"]:
                by_step.setdefault(r["t"], {})[r["gmm_variant"]] = r
        assert by_step
        for variants in by_step.values():
            assert variants["image_mid"]["gmm_cut"] == variants["pooled_mid"]["gmm_cut"]
            assert variants["image_cross"]["gmm_cut"] == variants["pooled_cross"]["gmm_cut"]

    def test_no_variant_rows_without_safe_thresholds(self):
        medias, _ = _planted_dataset(n_per_cat=40, seed=0)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=0,
            dataset_name="planted",
            inclusion=0,
            region_voting=True,
            safe_thresholds=False,
            max_steps=10,
            style="max_patch",
            emit_calibration_metrics=True,
        )
        assert rows
        assert all(r["gmm_variant"] == "" for r in rows)
        # Without the blend the recorded xcal threshold is the threshold itself.
        assert all(r["xcal_threshold"] == r["threshold"] for r in rows)
