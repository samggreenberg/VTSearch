"""Tests for the calibration fold-count eval arms (issue #2897).

With ``emit_calibration_metrics=True``, a style, and ``fold_count_variants``
the harness trains ``max(calibrate_count, *variants)`` calibration folds per
step and emits one ``folds_k{K}_xcal`` row per K - plus a ``folds_k{K}_blend``
row wherever a safe-threshold fit exists - each carrying that fold count's
regret and its measured wall clock.

The invariants the whole study rests on:

1. **Nesting is exact.** The arm at fold count K reports the threshold a plain
   run at ``calibrate_count=K`` computes for the same votes, byte-for-byte;
   folds are independent stratified draws off one seeded RNG at a size that
   does not depend on the count, so the first K of Kmax *are* the K.
2. **The live path is untouched.** Training the extra folds does not move the
   base row's threshold, cost, or trajectory - the screen is free of its own
   observer effect - and the reported ``xcal_seconds`` is still billed for the
   live fold count only.
3. **The pooled calibration set grows with K**, which is the mechanism the
   benefit half of the question is about.
"""

from __future__ import annotations

import numpy as np

from vtscore.eval.voting_iterations import _CALIBRATION_COLUMNS, simulate_voting_iterations

# Reuse the synthetic planted-patch dataset builder from the Max-Patch tests.
from .test_max_patch_style import _planted_dataset

_COUNTS = [1, 2, 4]


def _run(
    *, fold_counts=None, calibrate_count=2, safe=False, seed=0, max_steps=10, region_voting=True, style="max_patch"
):
    medias, _ = _planted_dataset(n_per_cat=40, seed=seed)
    return simulate_voting_iterations(
        medias,
        target_category="cat0",
        seed=seed,
        dataset_name="planted",
        inclusion=0,
        region_voting=region_voting,
        safe_thresholds=safe,
        calibrate_count=calibrate_count,
        max_steps=max_steps,
        style=style,
        emit_calibration_metrics=True,
        fold_count_variants=fold_counts,
    )


def _base_rows(rows):
    """The un-tagged base rows, keyed by step."""
    return {r["t"]: r for r in rows if r["gmm_variant"] == "" and r["pool_variant"] == "max"}


def _arm_rows(rows, arm):
    return {r["t"]: r for r in rows if r["gmm_variant"] == arm}


class TestFoldCountArms:
    def test_one_arm_per_count_per_calibratable_step(self):
        rows = _run(fold_counts=_COUNTS)
        assert rows, "no rows produced"
        by_step: dict[int, set[str]] = {}
        for r in rows:
            by_step.setdefault(r["t"], set()).add(r["gmm_variant"])
        tagged = [s for s in by_step.values() if any(v.startswith("folds_k") for v in s)]
        assert tagged, "no step ever emitted a fold-count arm"
        for arms in tagged:
            assert {f"folds_k{k}_xcal" for k in _COUNTS} <= arms

    def test_rows_carry_the_full_column_set(self):
        rows = _run(fold_counts=_COUNTS)
        cols = set(_CALIBRATION_COLUMNS)
        for r in rows:
            if r["gmm_variant"].startswith("folds_k"):
                assert cols <= set(r), f"missing columns: {cols - set(r)}"
                assert r["fold_count"] in _COUNTS
                assert np.isfinite(r["fold_seconds"]) and r["fold_seconds"] >= 0.0
                assert r["n_cal_scores"] > 0

    def test_arm_at_live_count_reproduces_the_step_threshold(self):
        """The control that licenses the table: K == calibrate_count is production."""
        rows = _run(fold_counts=_COUNTS, calibrate_count=2)
        base = _base_rows(rows)
        live = _arm_rows(rows, "folds_k2_xcal")
        assert live, "no live-count arm emitted"
        for t, arm in live.items():
            assert arm["threshold"] == base[t]["threshold"]
            assert arm["cost"] == base[t]["cost"]
            assert arm["regret"] == base[t]["regret"]

    def test_prefix_arms_match_plain_runs_at_each_count(self):
        """Nesting: the K-arm of a Kmax run == the base row of a real K run."""
        screened = _run(fold_counts=_COUNTS, calibrate_count=2)
        for k in _COUNTS:
            plain = _base_rows(_run(fold_counts=None, calibrate_count=k))
            arm = _arm_rows(screened, f"folds_k{k}_xcal")
            compared = 0
            for t, row in arm.items():
                if t not in plain:
                    continue
                assert row["threshold"] == plain[t]["threshold"], f"K={k} step {t}"
                compared += 1
            assert compared, f"no comparable steps at K={k}"

    def test_screen_does_not_perturb_the_live_trajectory(self):
        """Training Kmax folds must leave the un-instrumented run byte-identical."""
        plain = _base_rows(_run(fold_counts=None, calibrate_count=2))
        screened = _base_rows(_run(fold_counts=_COUNTS, calibrate_count=2))
        assert set(plain) == set(screened), "the screen changed which steps ran"
        for t, row in plain.items():
            for col in ("threshold", "cost", "regret", "n_good", "n_bad", "acq_threshold"):
                assert row[col] == screened[t][col], f"step {t} column {col} moved"

    def test_reported_xcal_seconds_is_billed_for_the_live_count(self):
        """The base row's timing stays the one a plain run would report."""
        screened = _base_rows(_run(fold_counts=[1, 2, 8], calibrate_count=2))
        arms = _arm_rows(_run(fold_counts=[1, 2, 8], calibrate_count=2), "folds_k8_xcal")
        for t, row in screened.items():
            if t in arms:
                # 8 folds cost strictly more than the 2 the base row is billed for.
                assert row["xcal_seconds"] <= arms[t]["fold_seconds"] + 1e-6

    def test_pooled_calibration_set_grows_with_k(self):
        rows = _run(fold_counts=_COUNTS)
        by_step: dict[int, dict[int, int]] = {}
        for r in rows:
            if r["gmm_variant"].endswith("_xcal"):
                by_step.setdefault(r["t"], {})[int(r["fold_count"])] = int(r["n_cal_scores"])
        assert by_step
        for sizes in by_step.values():
            ordered = [sizes[k] for k in sorted(sizes)]
            assert ordered == sorted(ordered)
            # Independent repeated splits, not a partition: K folds pool K
            # holdouts of the same size, so the pooled set is exactly linear.
            assert sizes[4] == 4 * sizes[1]
            assert sizes[2] == 2 * sizes[1]

    def test_fold_seconds_is_monotone_in_k(self):
        rows = _run(fold_counts=_COUNTS)
        by_step: dict[int, dict[int, float]] = {}
        for r in rows:
            if r["gmm_variant"].endswith("_xcal"):
                by_step.setdefault(r["t"], {})[int(r["fold_count"])] = float(r["fold_seconds"])
        assert by_step
        for secs in by_step.values():
            ordered = [secs[k] for k in sorted(secs)]
            assert ordered == sorted(ordered)


class TestFoldCountBlendArm:
    def test_blend_arm_only_under_safe_thresholds(self):
        off = _run(fold_counts=_COUNTS, safe=False)
        assert not any(r["gmm_variant"].endswith("_blend") for r in off)
        on = _run(fold_counts=_COUNTS, safe=True)
        assert any(r["gmm_variant"].endswith("_blend") for r in on), "no blended fold-count arm"

    def test_blend_arm_shares_the_xcal_cut_and_weight(self):
        rows = _run(fold_counts=_COUNTS, safe=True)
        xcal = _arm_rows(rows, "folds_k4_xcal")
        blend = _arm_rows(rows, "folds_k4_blend")
        assert blend
        for t, b in blend.items():
            # Same pre-blend cut; the blend only mixes in the GMM cut, whose
            # weight depends on the vote counts and so is shared across K.
            assert b["xcal_threshold"] == xcal[t]["xcal_threshold"]
            assert np.isfinite(b["blend_weight"])
            if b["blend_weight"] == 1.0:
                assert b["threshold"] == xcal[t]["threshold"]


class TestFoldCountBinaryVoting:
    """The row-wise (binary-voting) calibration path, the other half of #2897.

    ``whole_image`` without region voting is the boxless arm the study runs on
    Caltech: every bag is one row, so this exercises the row-wise calibrator
    rather than the bag-aware one the region arms take.
    """

    def test_arms_emitted_and_nested_without_region_voting(self):
        kw = {"region_voting": False, "style": "whole_image"}
        screened = _run(fold_counts=_COUNTS, calibrate_count=2, **kw)
        assert any(r["gmm_variant"].startswith("folds_k") for r in screened)
        plain = _base_rows(_run(fold_counts=None, calibrate_count=4, **kw))
        arm = _arm_rows(screened, "folds_k4_xcal")
        compared = 0
        for t, row in arm.items():
            if t in plain:
                assert row["threshold"] == plain[t]["threshold"]
                compared += 1
        assert compared, "no comparable steps"


class TestFoldCountDeterminism:
    def test_two_runs_agree(self):
        a = _run(fold_counts=_COUNTS)
        b = _run(fold_counts=_COUNTS)
        assert len(a) == len(b)
        for ra, rb in zip(a, b, strict=True):
            assert ra["gmm_variant"] == rb["gmm_variant"]
            assert ra["threshold"] == rb["threshold"]
            assert ra["regret"] == rb["regret"]
