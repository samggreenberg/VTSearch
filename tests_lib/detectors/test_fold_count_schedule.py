"""The eval-only adaptive fold-count knob (issue #3314).

``fold_count_schedule="K@N"`` resolves the **live** ``calibrate_count`` per step
from the vote count: ``K`` while ``n_votes < N``, then the run's own count.  The
issue's proposal in one line -- the folds that help most (few votes) are also the
cheapest to fit, so spend them early and decay to production's 2.

This is deliberately unlike ``fold_count_variants``, and the difference is the
whole reason stage B exists.  The variants are **counterfactual**: nested
prefixes of one Kmax calibration, scored inside a trajectory they cannot move.
The schedule is **live**: it sets the threshold the app would have shown, the
threshold sets the acquisition cut, and the cut sets which item Autopilot samples
next -- so a scheduled run collects different votes from its first trained step
and cannot be screened inside another run.

What is pinned here:

1. The spec parses to the function it claims to be, and a malformed one fails
   at parse time rather than forty minutes into a cell.
2. Every row records the count its step actually lived at, so an arm is
   readable from the frame rather than from the directory it was read out of.
3. A schedule that never fires is byte-identical to no schedule -- the
   off-by-default guarantee every other study depends on.
4. A schedule that does fire reaches the SHIPPED cut - the base row is the
   scheduled count's arm, not the other count's - and the two counts do cut
   differently somewhere, so that check is not vacuous.  If they did not, stage
   B would be measuring nothing and the screen would already have answered it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.voting_iterations import parse_fold_count_schedule, simulate_voting_iterations

from .test_max_patch_style import _planted_dataset

pytestmark = pytest.mark.xdist_group("fold-count-schedule")

#: Small enough that the fixture's 10-step horizon crosses it, so one run holds
#: both sides of the schedule.
CUT = 6
K_EARLY = 4


def _run(*, schedule=None, calibrate_count=2, seed=0, max_steps=10, fold_counts=None):
    medias, _ = _planted_dataset(n_per_cat=40, seed=seed)
    return simulate_voting_iterations(
        medias,
        target_category="cat0",
        seed=seed,
        dataset_name="planted",
        inclusion=0,
        region_voting=True,
        safe_thresholds=True,
        calibrate_count=calibrate_count,
        max_steps=max_steps,
        style="max_patch",
        emit_calibration_metrics=True,
        fold_count_schedule=schedule,
        fold_count_variants=fold_counts,
    )


def _arm_rows(rows, arm):
    return {r["t"]: r for r in rows if r["gmm_variant"] == arm}


def _base_rows(rows):
    return {r["t"]: r for r in rows if r["gmm_variant"] == "" and r["pool_variant"] == "max"}


class TestParseFoldCountSchedule:
    def test_off_by_default(self):
        assert parse_fold_count_schedule(None, 2) is None
        assert parse_fold_count_schedule("", 2) is None
        assert parse_fold_count_schedule("   ", 2) is None

    def test_single_step(self):
        f = parse_fold_count_schedule("6@25", 2)
        assert f is not None
        # Strictly below the cut, exactly as the PLAN writes it: `n_votes < N`.
        assert [f(n) for n in (1, 24)] == [6, 6]
        assert [f(n) for n in (25, 26, 1000)] == [2, 2, 2]

    def test_falls_back_to_the_runs_own_count_not_to_two(self):
        """The tail is *base*, so the knob composes with `calibrate_count`.

        Hard-coding 2 would silently overwrite the live count of any run that
        set one, which is exactly the kind of pin that outlives what it pinned.
        """
        f = parse_fold_count_schedule("8@10", 3)
        assert f(5) == 8
        assert f(50) == 3

    def test_segments_are_read_in_cut_order_not_string_order(self):
        """`"4@60,8@25"` and `"8@25,4@60"` must be the same schedule."""
        a = parse_fold_count_schedule("8@25,4@60", 2)
        b = parse_fold_count_schedule("4@60,8@25", 2)
        assert a is not None and b is not None
        for n in (1, 24, 25, 59, 60, 200):
            assert a(n) == b(n), n
        assert [a(n) for n in (1, 24, 25, 59, 60)] == [8, 8, 4, 4, 2]

    @pytest.mark.parametrize("bad", ["6", "6@", "@25", "6@0", "0@25", "six@25", "6@25@3"])
    def test_a_malformed_spec_raises(self, bad):
        with pytest.raises(ValueError):
            parse_fold_count_schedule(bad, 2)


class TestScheduledRun:
    def test_every_row_records_the_count_its_step_lived_at(self):
        rows = _run(schedule=f"{K_EARLY}@{CUT}")
        assert rows
        seen = {}
        for r in rows:
            seen.setdefault(int(r["n_good"] + r["n_bad"]), set()).add(int(r["calibrate_count"]))
        assert seen, "no row carried a fold count"
        for votes, counts in seen.items():
            assert counts == {K_EARLY if votes < CUT else 2}, (votes, counts)

    def test_both_sides_of_the_cut_are_in_the_run(self):
        """A fixture whose horizon never crosses the cut would test nothing."""
        rows = _run(schedule=f"{K_EARLY}@{CUT}")
        counts = {int(r["calibrate_count"]) for r in rows}
        assert counts == {K_EARLY, 2}, counts

    def test_a_schedule_that_never_fires_is_byte_identical_to_no_schedule(self):
        """The off-by-default guarantee, asserted rather than promised.

        `"2@1000"` resolves to the run's own count at every step, so it must
        reproduce the unscheduled run exactly - not approximately.  This is what
        says the knob cannot perturb the 40-odd other studies that share this
        harness.
        """
        plain = _run(schedule=None)
        inert = _run(schedule="2@1000")
        assert len(plain) == len(inert)
        for a, b in zip(plain, inert, strict=True):
            for col in ("t", "gmm_variant", "threshold", "cost", "regret", "n_good", "n_bad", "acq_threshold"):
                assert a[col] == b[col] or (isinstance(a[col], float) and np.isnan(a[col]) and np.isnan(b[col])), (
                    col,
                    a["t"],
                    a["gmm_variant"],
                )

    def test_the_live_cut_is_the_scheduled_counts_cut(self):
        """The assertion that the schedule reached the SHIPPED threshold.

        Run the counterfactual screen alongside it at ``{2, K_EARLY}``.  The
        screen's arms are nested prefixes of one calibration, and the arm at
        ``K == calibrate_count`` reproduces that step's own shipped cut
        byte-for-byte (see ``test_fold_count_variant_rows.py``).  So the base
        row must match the arm the SCHEDULE named at that step, and not the
        other one - which is a statement about which count the live path used,
        not about where a quantile happened to land.
        """
        rows = _run(schedule=f"{K_EARLY}@{CUT}", fold_counts=[2, K_EARLY])
        base = _base_rows(rows)
        arms = {k: _arm_rows(rows, f"folds_k{k}_anchored") for k in (2, K_EARLY)}
        checked = 0
        for t, row in base.items():
            # Only where the shipped rule actually anchored; a step that fell
            # back to the blend has no fold-anchored cut to reproduce.
            if not str(row["threshold_provenance"]).startswith("fold_anchored"):
                continue
            k_live = int(row["calibrate_count"])
            assert t in arms[k_live], (t, k_live)
            assert row["threshold"] == arms[k_live][t]["threshold"], (t, k_live)
            checked += 1
        assert checked, "no step reached a fold-anchored cut; the fixture tests nothing"

    def test_the_two_counts_calibrate_on_different_sets(self):
        """...and the check above is not vacuous: the counts are really different.

        Asserted on the **calibration set**, not on the realized threshold.  K is
        a knob on how many independent held-out splits the conformal quantile is
        read over, so ``n_cal_scores`` scales exactly with it - a statement no
        amount of fixture noise can blur.  The *cut* is a quantile carried onto
        a 40-media-per-category haystack, where neighbouring quantiles routinely
        land on the same order statistic (the caveat
        ``test_fold_count_variant_rows.py`` records for the same reason), so two
        counts colliding there is an artefact of the fixture and not evidence
        that the calibration was reused.
        """
        rows = _run(schedule=f"{K_EARLY}@{CUT}", fold_counts=[2, K_EARLY])
        a, b = _arm_rows(rows, "folds_k2_anchored"), _arm_rows(rows, f"folds_k{K_EARLY}_anchored")
        shared = sorted(set(a) & set(b))
        assert shared, "no step emitted both arms"
        for t in shared:
            # Independent repeated splits, not a partition: K folds pool K
            # holdouts of the same size, so the pooled set is exactly linear.
            assert b[t]["n_cal_scores"] == (K_EARLY // 2) * a[t]["n_cal_scores"], t
