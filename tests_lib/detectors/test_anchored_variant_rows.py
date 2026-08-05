"""Tests for the anchored-mixture eval variant rows (issue #2852).

With ``safe_thresholds=True``, ``emit_calibration_metrics=True``, a style, and
``anchored_thresholds=True`` the harness emits one extra metric row per
anchored arm at every trainable step: the label-anchored family
(``anchored_w{W}_{rule}``), the fold-anchored "cross-LabeledGMM" family
(``fold_anchored_w{W}_{rule}_{combine}``), and the ``rank_transfer``
attribution arm.  The load-bearing invariants: the rows are step-paired with
the existing ``pooled_mid`` / ``xcal_only`` controls (same test scores, same
columns), the anchored thresholds are raw (no blend), the estimator path taken
is surfaced in ``threshold_provenance``, and everything is deterministic.
"""

from __future__ import annotations

import numpy as np

from vtscore.eval.voting_iterations import _CALIBRATION_COLUMNS, simulate_voting_iterations

# Reuse the synthetic planted-patch dataset builders from the Max-Patch tests.
from .test_max_patch_style import _planted_dataset

_ANCHORED = "anchored_w10_mid"
_FOLD = "fold_anchored_w10_mid_qmean"
_RANK = "rank_transfer"


def _run(seed=0, max_steps=12, fold_arms=True):
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
        style="max_patch",
        emit_calibration_metrics=True,
        anchored_thresholds=True,
        anchored_weights=[10.0],
        anchored_rules=["mid"],
        anchored_fold_combines=["qmean"],
        anchored_fold_arms=fold_arms,
    )


class TestAnchoredVariantRows:
    def test_anchored_arms_emitted_and_paired_per_step(self):
        rows = _run()
        assert rows, "no rows produced"
        by_step: dict[int, set[str]] = {}
        calibratable: dict[int, bool] = {}
        for r in rows:
            by_step.setdefault(r["t"], set()).add(r["gmm_variant"])
            # The fold arms need real calibration folds, which the grouped
            # calibrator only trains with >= 2 votes of each class.
            calibratable[r["t"]] = r["n_good"] >= 2 and r["n_bad"] >= 2
        assert any(calibratable.values()), "run never reached a calibratable step"
        for t, variants_at_t in by_step.items():
            # Paired within the step against the shipped blend and pure x-cal.
            assert {_ANCHORED, "pooled_mid", "xcal_only"} <= variants_at_t
            if calibratable[t]:
                assert {_FOLD, _RANK} <= variants_at_t

    def test_columns_provenance_and_no_blend(self):
        rows = _run()
        anchored = [r for r in rows if r["gmm_variant"] == _ANCHORED]
        fold = [r for r in rows if r["gmm_variant"] == _FOLD]
        rank = [r for r in rows if r["gmm_variant"] == _RANK]
        assert anchored and fold and rank
        for r in (*anchored, *fold, *rank):
            assert set(_CALIBRATION_COLUMNS).issubset(r.keys())
            assert np.isfinite(r["threshold"])
            # The estimator replaces the blend: raw threshold, no ramp weight.
            assert np.isnan(r["blend_weight"])
            assert r["gmm_cut"] == r["threshold"]
            assert r["raw_cut_cost"] == r["cost"]
        for r in anchored:
            assert r["threshold_provenance"] == "anchored" or r["threshold_provenance"].startswith(
                ("unanchored:", "gmm_failed:")
            )
        for r in fold:
            assert r["threshold_provenance"].startswith(("fold_anchored[", "fold_fallback"))
        for r in rank:
            assert r["threshold_provenance"] == "rank_transfer"

    def test_fold_arms_can_be_disabled(self):
        rows = _run(fold_arms=False)
        variants = {r["gmm_variant"] for r in rows}
        assert _ANCHORED in variants
        assert _FOLD not in variants
        assert _RANK not in variants

    def test_deterministic(self):
        thr_a = [
            (r["t"], r["gmm_variant"], r["threshold"])
            for r in _run()
            if r["gmm_variant"].startswith(("anchored", "fold_anchored", "rank"))
        ]
        thr_b = [
            (r["t"], r["gmm_variant"], r["threshold"])
            for r in _run()
            if r["gmm_variant"].startswith(("anchored", "fold_anchored", "rank"))
        ]
        assert thr_a == thr_b

    def test_off_by_default(self):
        medias, _ = _planted_dataset(n_per_cat=40, seed=0)
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=0,
            dataset_name="planted",
            inclusion=0,
            region_voting=True,
            safe_thresholds=True,
            max_steps=8,
            style="max_patch",
            emit_calibration_metrics=True,
        )
        assert not any(r["gmm_variant"].startswith(("anchored", "fold_anchored", "rank")) for r in rows)
