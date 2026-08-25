"""Tests for the calibration fold-count eval arms (issue #2897).

With ``emit_calibration_metrics=True``, a style, and ``fold_count_variants``
the harness trains ``max(calibrate_count, *variants)`` calibration folds per
step and emits one ``folds_k{K}_xcal`` row per K - plus ``folds_k{K}_blend`` and
``folds_k{K}_anchored`` rows wherever a safe-threshold fit exists - each
carrying that fold count's regret and its measured wall clock.

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
import pytest

from vtscore.eval.voting_iterations import _CALIBRATION_COLUMNS, simulate_voting_iterations
from vtscore.training import thresholds as _thresholds

# Reuse the synthetic planted-patch dataset builder from the Max-Patch tests.
from .sweep_cache import memoize_sweep
from .test_max_patch_style import _planted_dataset

# Pinned to one worker so the memoized sweeps below actually hit the cache
# under ``--dist loadgroup``.  See sweep_cache.py.
pytestmark = pytest.mark.xdist_group("fold-count-variant-rows")

_COUNTS = [1, 2, 4]

#: Make-believe price of one calibration fold, for :class:`_StubClock`.  Chosen
#: far above any real wall clock in the run so the reported seconds are a *fold
#: count* in disguise and no real measurement can perturb the arithmetic.
_STUB_FOLD_SECONDS = 1000.0


class _StubClock:
    """A ``time`` stand-in whose every reading is *seconds* later than the last.

    Installed over ``vtscore.training.thresholds.time`` — whose only clock user
    is the per-fold stopwatch, exactly two readings per fold — this prices every
    calibration fold at exactly *seconds*, turning the seconds a row reports
    into a count of the folds it was billed for.  Nothing else in that module
    reads a clock, and the eval module keeps the real one, so the surrounding
    overhead stays honest (and, being dwarfed by the stub rate, clamps to zero).
    """

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._now = 0.0

    def monotonic(self) -> float:
        self._now += self._seconds
        return self._now


def _run_uncached(
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


#: Memoized view of :func:`_run_uncached`.  Six tests sweep the identical
#: ``fold_counts=_COUNTS`` frame and read different columns of it.  Two callers
#: deliberately stay on the uncached function: the determinism test (a cache hit
#: would compare a list against itself) and the stub-clock test (whose rows are
#: computed under a monkeypatched clock and must never enter a shared cache).
_run = memoize_sweep(_run_uncached)


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

    @pytest.mark.parametrize(
        ("region_voting", "style"), [(True, "max_patch"), (False, "whole_image")], ids=["grouped", "rowwise"]
    )
    def test_reported_xcal_seconds_is_billed_for_the_live_count(self, monkeypatch, region_voting, style):
        """The base row's timing stays the one a plain run would report.

        Asserted structurally rather than as a race between two real stopwatches:
        with every fold priced at :data:`_STUB_FOLD_SECONDS` (see
        :class:`_StubClock`), "which folds got billed" is exact arithmetic on the
        reported seconds instead of a hope that six extra fold fits out-measure a
        scheduler stall.  Both fold-timing sinks are covered — the bag-aware
        grouped calibrator and the row-wise one.
        """
        monkeypatch.setattr(_thresholds, "time", _StubClock(_STUB_FOLD_SECONDS))
        # Uncached on purpose: these rows are billed by a stubbed clock, so they
        # must not be memoized where a later test could pick them up.
        rows = _run_uncached(fold_counts=[1, 2, 8], calibrate_count=2, region_voting=region_voting, style=style)
        base = _base_rows(rows)
        live = _arm_rows(rows, "folds_k2_xcal")
        screened = _arm_rows(rows, "folds_k8_xcal")
        assert screened, "no k8 arm emitted"
        extra_folds = 8 - 2
        for t, arm in screened.items():
            # Each arm bills its own K folds, so their difference is the six
            # folds only the screen trained, at the stub rate.
            assert arm["fold_seconds"] - live[t]["fold_seconds"] == pytest.approx(extra_folds * _STUB_FOLD_SECONDS)
            # ...and the base row hands those six back: what remains of its
            # xcal_seconds is the real (tiny) wall clock of everything else in
            # the calibration window.  Billing all 8 would leave -2000 here and
            # billing none would leave +6000, so the band cannot be hit by noise.
            residual = base[t]["xcal_seconds"] + extra_folds * _STUB_FOLD_SECONDS
            assert -1e-6 <= residual < 60.0, f"step {t}: xcal_seconds billed for the wrong fold count"

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
        """Not a stopwatch race despite the timings: each arm's ``fold_seconds``
        is a *prefix sum* of one non-negative per-fold list plus a shared
        overhead, so a stalled fold inflates every K at and above it and can
        never invert the order.  Timing jitter cannot fail this.
        """
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


class TestFoldCountCombineArms:
    """The combine-rule arms (#3115): pooled vs averaged, in two spaces.

    ``xcal`` is already the pooled control - it calls
    ``threshold_from_fold_orderings`` verbatim - so these four arms are the
    challengers, and the contrast factors into two legs that isolate different
    things: ``xcal -> tmean`` is pooling vs averaging with the space held fixed,
    and ``tmean -> qmean`` is score space vs quantile space with the combine held
    fixed.  What these pin is that the arms are *emitted*, that they are wired to
    the same fold prefix as the control, and that the exact K=1 control holds
    end to end - not which of them wins, which is what the run is for.
    """

    def test_score_space_arms_need_no_safe_thresholds(self):
        """They read only the fold orderings, which every calibratable step has."""
        rows = _run(fold_counts=_COUNTS, safe=False)
        for combine in ("tmean", "tmedian"):
            assert any(r["gmm_variant"] == f"folds_k4_{combine}" for r in rows), combine

    def test_quantile_space_arms_need_the_fold_haystacks(self):
        """They carry each cut through its fold's own distribution, so they share
        the anchored arm's condition rather than the control's."""
        off = _run(fold_counts=_COUNTS, safe=False)
        assert not any(r["gmm_variant"].endswith(("_qmean", "_qmedian")) for r in off)
        on = _run(fold_counts=_COUNTS, safe=True)
        assert any(r["gmm_variant"] == "folds_k4_qmean" for r in on), "no quantile-space arm"

    def test_single_fold_score_space_arm_is_the_pooled_arm(self):
        """The exact control, asserted through the harness and not only the rule.

        At K=1 there is one cut to average, so ``tmean`` must reproduce
        ``xcal``'s threshold *and* every metric computed from it.  A mis-sliced
        prefix, or haystacks out of step with their orderings, breaks this.
        """
        rows = _run(fold_counts=_COUNTS, safe=True)
        xcal = _arm_rows(rows, "folds_k1_xcal")
        assert xcal, "no K=1 control arm"
        for combine in ("tmean", "tmedian"):
            arm = _arm_rows(rows, f"folds_k1_{combine}")
            assert arm, combine
            for t, row in arm.items():
                assert row["threshold"] == xcal[t]["threshold"], (combine, t)
                assert row["cost"] == xcal[t]["cost"], (combine, t)
                assert row["regret"] == xcal[t]["regret"], (combine, t)

    def test_mean_and_median_are_one_arm_below_three_folds(self):
        """Production's ``calibrate_count=2`` is exactly where the question hides."""
        rows = _run(fold_counts=_COUNTS, safe=True)
        for k in (1, 2):
            for family in ("t", "q"):
                mean = _arm_rows(rows, f"folds_k{k}_{family}mean")
                median = _arm_rows(rows, f"folds_k{k}_{family}median")
                assert mean and set(mean) == set(median), (k, family)
                for t in mean:
                    assert mean[t]["threshold"] == median[t]["threshold"], (k, family, t)

    def test_some_step_actually_separates_the_rules(self):
        """A run where every arm agrees measures nothing; this fixture must not be one."""
        rows = _run(fold_counts=_COUNTS, safe=True)
        xcal = _arm_rows(rows, "folds_k4_xcal")
        moved = {
            combine: sum(
                1 for t, r in _arm_rows(rows, f"folds_k4_{combine}").items() if r["threshold"] != xcal[t]["threshold"]
            )
            for combine in ("tmean", "tmedian", "qmean", "qmedian")
        }
        assert all(n > 0 for n in moved.values()), moved

    def test_n_folds_used_counts_only_contributed_cuts(self):
        """NaN where no fold ever "contributes a cut" - the pooled arm and the blend.

        Reporting K there would read as agreement with the combining arms and
        hide the asymmetry the study is about: a single-class fold is silently
        *in* a pooled quantile and explicitly *out* of a mean.
        """
        rows = [r for r in _run(fold_counts=_COUNTS, safe=True) if r["gmm_variant"].startswith("folds_k")]
        assert rows
        for r in rows:
            k = int(r["fold_count"])
            arm = r["gmm_variant"].split(f"folds_k{k}_", 1)[1]
            used = r["n_folds_used"]
            if arm in ("xcal", "blend"):
                assert np.isnan(used), r["gmm_variant"]
            else:
                assert 1 <= used <= k, (r["gmm_variant"], used)


class TestFoldCountAnchoredQmedianArm:
    """#3115's contamination question, put on the **shipped** rule.

    ``fold_anchored_gmm_threshold`` already combines its per-fold cuts in
    quantile space; production picks the mean (``FOLD_ANCHOR_COMBINE``).  This
    arm re-cuts the *same* per-fold fits under the median, so the difference
    between the two rows is the combine and nothing else.
    """

    def test_only_under_safe_thresholds(self):
        off = _run(fold_counts=_COUNTS, safe=False)
        assert not any(r["gmm_variant"].endswith("_anchored_qmedian") for r in off)
        on = _run(fold_counts=_COUNTS, safe=True)
        assert any(r["gmm_variant"] == "folds_k4_anchored_qmedian" for r in on)

    def test_shares_the_anchored_arms_fits(self):
        """Same fits, same fold count, same anchored/unanchored split - only the
        combine differs, which is what makes the pair a clean contrast."""
        rows = _run(fold_counts=_COUNTS, safe=True)
        mean = _arm_rows(rows, "folds_k4_anchored")
        median = _arm_rows(rows, "folds_k4_anchored_qmedian")
        assert mean and set(mean) == set(median)
        for t, row in mean.items():
            assert row["threshold_provenance"] == median[t]["threshold_provenance"], t

    def test_collapses_onto_production_below_three_folds(self):
        """Mean and median of at most two quantiles are the same number."""
        rows = _run(fold_counts=_COUNTS, safe=True)
        for k in (1, 2):
            mean = _arm_rows(rows, f"folds_k{k}_anchored")
            median = _arm_rows(rows, f"folds_k{k}_anchored_qmedian")
            assert mean and set(mean) == set(median), k
            for t, row in mean.items():
                assert row["threshold"] == median[t]["threshold"], (k, t)


class TestFoldCountAnchoredArm:
    """The arm that measures **production's** threshold rule across K (#3116).

    The other two arms cannot: ``xcal`` is the raw conformal cut, and ``blend``
    is the retired ``cap50`` mix-in whose GMM half is a single unanchored fit on
    the sim haystack - K-independent by construction, so only its x-cal half
    ever moved.  The shipped path fits one *anchored* mixture per fold, so K
    multiplies the number of mixtures combined.
    """

    def test_anchored_arm_only_under_safe_thresholds(self):
        off = _run(fold_counts=_COUNTS, safe=False)
        assert not any(r["gmm_variant"].endswith("_anchored") for r in off)
        on = _run(fold_counts=_COUNTS, safe=True)
        assert any(r["gmm_variant"].endswith("_anchored") for r in on), "no anchored fold-count arm"

    def test_arm_at_live_count_reproduces_the_shipped_threshold(self):
        """The control that licenses the arm: at K == calibrate_count it *is* production.

        The base row's threshold on a safe-threshold run is
        ``fold_anchored_gmm_threshold`` over the live folds.  The K=2 anchored
        arm re-derives it from the Kmax run's own fold prefix, so any drift
        between the arm and the shipped rule - a mis-sliced prefix, haystacks
        out of step with their orderings, the wrong inclusion - shows up here as
        an inequality rather than as a plausible number in a report.
        """
        rows = _run(fold_counts=_COUNTS, calibrate_count=2, safe=True)
        base = _base_rows(rows)
        live = _arm_rows(rows, "folds_k2_anchored")
        assert live, "no live-count anchored arm emitted"
        compared = 0
        for t, arm in live.items():
            # Only on steps the shipped rule actually anchored; a step that fell
            # back to the blend has no fold-anchored threshold to reproduce.
            if base[t]["threshold_provenance"].startswith("fold_anchored"):
                assert arm["threshold"] == base[t]["threshold"], t
                assert arm["threshold_provenance"] == base[t]["threshold_provenance"], t
                compared += 1
        assert compared, "no step reached a fold-anchored cut"

    def test_anchored_arm_fits_k_mixtures_at_k(self):
        """K must actually drive the number of anchored mixtures - the point of the arm.

        Asserted through **provenance** rather than through the threshold.
        ``fold_anchored_gmm_threshold`` reports ``fold_anchored[a/k]`` naming
        how many folds it anchored, so a mis-sliced prefix - the live count
        reused at every K, the haystacks out of step with their orderings -
        shows up here as the wrong ``k`` and cannot be missed.

        The realized *threshold* is deliberately not the assertion: it is a
        quantile carried onto the final model's haystack, and this fixture's
        haystack is small enough that neighbouring quantiles routinely land on
        the same order statistic.  Two fold counts producing the same float
        here is therefore an artefact of a 40-media-per-category fixture, not
        evidence that the fit was reused - which is exactly why the count, not
        the value, is what this pins.
        """
        rows = _run(fold_counts=_COUNTS, calibrate_count=2, safe=True)
        seen: dict[int, set[str]] = {}
        for k in _COUNTS:
            for r in _arm_rows(rows, f"folds_k{k}_anchored").values():
                seen.setdefault(k, set()).add(r["threshold_provenance"])
        assert set(seen) == set(_COUNTS), f"missing anchored arms: {set(_COUNTS) - set(seen)}"
        for k, provs in seen.items():
            anchored = {p for p in provs if p.startswith("fold_anchored[")}
            assert anchored, f"K={k} never reached an anchored fit: {provs}"
            # `[a/k]` - `a` folds anchored of `k` used; `k` is the knob.
            assert {p.split("/")[1].rstrip("]") for p in anchored} == {str(k)}, (k, anchored)

    def test_extra_haystacks_do_not_perturb_the_live_threshold(self):
        """Scoring the Kmax fold models must leave the shipped cut byte-identical.

        The extra haystacks ride in ``fold_count_data`` precisely so the live
        fit keeps using only ``calibrate_count`` of them; this pins that the
        observer effect is zero.
        """
        screened = _base_rows(_run(fold_counts=_COUNTS, calibrate_count=2, safe=True))
        plain = _base_rows(_run(fold_counts=None, calibrate_count=2, safe=True))
        compared = 0
        for t, row in screened.items():
            if t in plain:
                assert row["threshold"] == plain[t]["threshold"], t
                assert row["threshold_provenance"] == plain[t]["threshold_provenance"], t
                compared += 1
        assert compared, "no comparable steps"


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
        """Two *independent* sweeps must agree, so this bypasses the memoized
        ``_run`` — a cache hit would compare one list against itself."""
        a = _run_uncached(fold_counts=_COUNTS)
        b = _run_uncached(fold_counts=_COUNTS)
        assert len(a) == len(b)
        for ra, rb in zip(a, b, strict=True):
            assert ra["gmm_variant"] == rb["gmm_variant"]
            assert ra["threshold"] == rb["threshold"]
            assert ra["regret"] == rb["regret"]
