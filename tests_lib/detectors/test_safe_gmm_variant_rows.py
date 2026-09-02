"""Tests for the safe-threshold cut variant rows in the eval harness (#2799, #2836).

With ``safe_thresholds=True`` and ``emit_calibration_metrics=True`` the harness
emits one extra metric row per cut variant at every trainable step - the #2799
and #2836 measurement arms - plus a per-(step, geometry) decomposition frame
into ``cut_diag_sink``.  The load-bearing invariant is that the arm at the
*production* estimator settings reproduces the base row's threshold
bit-for-bit, so the study measures the shipped code and every other variant
differs from it along exactly one named axis.  Since the population-anchored
adoption that arm is the fold-anchored fit at the shipped constants
(:data:`~vtscore.training.thresholds.FOLD_ANCHOR_WEIGHT` etc., today κ=0.3 with
the ``mid_tilt`` cut and the quantile mean); the ``pooled_*`` family now
measures the retired schedule blend.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.cut_rules import CUT_KIND_MIDPOINT
from vtscore.eval.voting_columns import CALIBRATION_COLUMNS, CUT_DIAGNOSTIC_COLUMNS
from vtscore.eval.arms_safe_gmm import _ORACLE_VARIANTS, _SAFE_GMM_VARIANTS, _safe_gmm_variant_rows
from vtscore.eval.voting_iterations import simulate_voting_iterations
from vtscore.training.thresholds import (
    CUT_KIND_INTERIOR,
    FOLD_ANCHOR_COMBINE,
    FOLD_ANCHOR_CUT_RULE,
    FOLD_ANCHOR_WEIGHT,
)

# Reuse the synthetic planted-patch dataset builders from the Max-Patch tests.
from .sweep_cache import memoize_sweep
from .test_max_patch_style import _planted_dataset

# Nine of the tests below sweep ``max_patch`` at identical settings — five
# reading only the metric rows, four also reading the decomposition frame — so
# each of those two shapes is simulated once per worker and this module is
# pinned to one worker for the cache to hit.  Shared rows are read-only.
# See sweep_cache.py.
pytestmark = pytest.mark.xdist_group("safe-gmm-variant-rows")

_VARIANT_NAMES = {name for name, _fit, _rule in _SAFE_GMM_VARIANTS}
#: Variants that must appear at every trainable step.  The label-reading
#: diagnostics may legitimately have no cut on a given fit (no root, or a step
#: with one class in the sim set) and are emitted only when they do.
_ALWAYS_EMITTED = _VARIANT_NAMES - set(_ORACLE_VARIANTS)


def _run_safe_uncached(style, seed=0, max_steps=16, diag_sink=None, **kw):
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
        cut_diag_sink=diag_sink,
        **kw,
    )


_run_safe_no_sink = memoize_sweep(_run_safe_uncached)


@memoize_sweep
def _run_safe_with_diag(style, **kw):
    """A sweep that also collects the decomposition frame, as a cacheable pair.

    The frame is delivered through a caller-owned list, which cannot be part of
    a cache key — every caller passes a fresh empty one. Running the sweep with
    a sink this wrapper owns, and returning ``(rows, diagnostics)``, makes the
    whole result cacheable; :func:`_run_safe` replays the diagnostics into
    whichever list the test handed in.
    """
    sink: list[dict] = []
    rows = _run_safe_uncached(style, diag_sink=sink, **kw)
    return rows, sink


def _run_safe(style, seed=0, max_steps=16, diag_sink=None, **kw):
    """Memoized sweep, with or without the decomposition frame.

    Passing a sink is not a different simulation, only a different *view* of
    one, so both shapes are cached; see :mod:`.sweep_cache` for why the results
    must be treated as read-only.
    """
    if diag_sink is None:
        return _run_safe_no_sink(style, seed=seed, max_steps=max_steps, **kw)
    rows, diagnostics = _run_safe_with_diag(style, seed=seed, max_steps=max_steps, **kw)
    diag_sink.extend(diagnostics)
    return rows


class TestSafeGmmVariantRows:
    def test_every_step_emits_base_plus_all_variants(self):
        rows = _run_safe("max_patch")
        assert rows, "no rows produced"
        by_step: dict[int, set[str]] = {}
        for r in rows:
            by_step.setdefault(r["t"], set()).add(r["gmm_variant"])
        for variants_at_t in by_step.values():
            assert _ALWAYS_EMITTED | {""} <= variants_at_t
            assert variants_at_t <= {"", *_VARIANT_NAMES}

    def test_columns_and_tags(self):
        rows = _run_safe("max_patch")
        for r in rows:
            assert set(CALIBRATION_COLUMNS).issubset(r.keys())
        base = [r for r in rows if r["gmm_variant"] == ""]
        variant = [r for r in rows if r["gmm_variant"] != ""]
        assert base and variant
        # The production operating point is the base row; its provenance names
        # the shipped estimator (the fold-anchored fit, or the blend fallback
        # when no fold produced one), and it records the pre-fusion conformal
        # cut alongside.
        for r in base:
            assert r["threshold_provenance"].startswith("fold_anchored[") or r["threshold_provenance"] == "gmm_blend"
            assert np.isfinite(r["xcal_threshold"])
        for r in variant:
            assert r["pool_variant"] == "max"
            assert 0.0 <= r["blend_weight"] <= 1.0 or np.isnan(r["blend_weight"])

    def test_production_anchored_arm_reproduces_the_shipped_threshold(self):
        """The harness must not deviate from the app at the shipped settings.

        The arm named by the ``FOLD_ANCHOR_*`` constants *is* the production
        estimator, so its threshold has to equal the base row's - the value the
        step actually shipped - at every step.  Reading the arm's settings off
        those constants rather than restating them is what keeps this honest
        when the shipped operating point moves (κ=1/``rate`` → κ=0.3/``mid`` →
        κ=0.3/``mid_tilt``).
        """
        rows = _run_safe(
            "max_patch",
            anchored_thresholds=True,
            anchored_weights=[FOLD_ANCHOR_WEIGHT],
            anchored_rules=[FOLD_ANCHOR_CUT_RULE],
            anchored_fold_combines=[FOLD_ANCHOR_COMBINE],
        )
        arm_name = f"fold_anchored_w{FOLD_ANCHOR_WEIGHT:g}_{FOLD_ANCHOR_CUT_RULE}_{FOLD_ANCHOR_COMBINE}"
        base = {r["t"]: r for r in rows if r["gmm_variant"] == ""}
        prod = {r["t"]: r for r in rows if r["gmm_variant"] == arm_name}
        assert prod, "the production anchored arm emitted no rows"
        # ``mid_tilt`` is a fold-level rule: the label-anchored family must
        # skip it rather than crash on it, so no ``anchored_w*`` arm exists.
        assert not any(r["gmm_variant"].startswith("anchored_w") for r in rows)
        for t, arm in prod.items():
            if base[t]["threshold_provenance"] == "gmm_blend":
                continue  # the step fell back to the blend; the arm is not it
            assert arm["threshold"] == base[t]["threshold"], f"step {t}: harness diverged from the app"
            assert arm["threshold_provenance"] == base[t]["threshold_provenance"]

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
            assert variants["image_priorfree"]["gmm_cut"] == variants["pooled_priorfree"]["gmm_cut"]

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


class TestCutDiagnosticFrame:
    """The #2836 decomposition side frame."""

    def test_one_row_per_step_per_geometry_with_the_declared_columns(self):
        diag: list[dict] = []
        rows = _run_safe("max_patch", diag_sink=diag)
        assert diag, "no cut-diagnostic rows produced"
        steps = {r["t"] for r in rows if r["gmm_variant"] == ""}
        for t in steps:
            assert {d["geometry"] for d in diag if d["t"] == t} == {"pooled", "image"}
        for d in diag:
            assert set(CUT_DIAGNOSTIC_COLUMNS).issubset(d.keys())

    def test_cuts_agree_with_the_variant_rows(self):
        """The frame is a second view of the same fit, not a second fit."""
        diag: list[dict] = []
        rows = _run_safe("max_patch", diag_sink=diag)
        pooled = {d["t"]: d for d in diag if d["geometry"] == "pooled"}
        for r in rows:
            if r["gmm_variant"] not in ("pooled_mid", "pooled_cross", "pooled_priorfree"):
                continue
            rule = r["gmm_variant"].removeprefix("pooled_")
            if r["cut_fallback"]:
                continue  # the row reports the midpoint it fell back to
            assert r["gmm_cut"] == pooled[r["t"]][f"tau_{rule}"]

    def test_every_swept_row_reports_one_of_this_familys_two_outcomes(self):
        """Only two kinds are reachable here: the rule's own cut, or the midpoint.

        This family never *continues* past the inter-mean interval and never
        reports production's degenerate branch, so a row carrying either of
        those kinds would mean the two families' vocabularies had been crossed.
        """
        rows = _run_safe("max_patch")
        for r in rows:
            if r["gmm_variant"] in ("xcal_only", *_ORACLE_VARIANTS) or not r["gmm_variant"].startswith(
                ("pooled_", "image_")
            ):
                continue
            kind = r["cut_fallback_kind"]
            assert kind in (CUT_KIND_INTERIOR, CUT_KIND_MIDPOINT), (r["gmm_variant"], kind)
            assert r["cut_fallback"] == int(bool(kind))

    def test_a_fallen_back_row_names_the_midpoint_it_substituted(self):
        """The divergence from production is kept, but it is no longer invisible.

        This family substitutes the fit's own midpoint - a neutral,
        rule-independent stand-in, which is what keeps ``rate`` commensurable
        with the ``cross``/``priorfree`` siblings it is differenced against.
        Production's ``rate`` continues past the component mean instead, so on
        exactly these steps the arm is *not* a stand-in for the app.  The row
        has to say which of the two it is, or an analysis that reads a
        ``*_rate`` arm as "what the app would have done" is silently wrong on
        them (issue #2900).

        The substitution is provoked here rather than waited for.  It used to be
        read off whatever the ``max_patch`` sweep happened to produce, which
        made the assertion hostage to the fixture: the planted dataset's fits
        stopped declining the moment the shipped head changed, and the branch
        went silently unexercised while the test still passed its other rows.
        A near-tied sim distribution declines the EVT guards by construction on
        any head, so the path is exercised on purpose.
        """
        scores = [0.5] * 40 + [0.5001] * 40
        labels = np.array([1.0] * 40 + [0.0] * 40)
        rows, diag = _safe_gmm_variant_rows(
            details={"xcal_threshold": 0.5, "n_votes": 20, "n_good": 10, "n_bad": 10},
            base_scores=np.array(scores),
            base_labels=labels,
            sim_scores_by_geometry={"pooled": scores, "image": scores},
            sim_labels_by_geometry={"pooled": labels, "image": labels},
            inclusion=0,
            n_pool_rows=float(len(scores)),
        )
        tau_mid = {d["geometry"]: d["tau_mid"] for d in diag}["pooled"]

        seen_fallback = 0
        for r in rows:
            if r["gmm_variant"] in ("xcal_only", *_ORACLE_VARIANTS) or not r["gmm_variant"].startswith("pooled_"):
                continue
            kind = r["cut_fallback_kind"]
            assert kind in (CUT_KIND_INTERIOR, CUT_KIND_MIDPOINT), (r["gmm_variant"], kind)
            assert r["cut_fallback"] == int(bool(kind))
            if kind == CUT_KIND_MIDPOINT:
                seen_fallback += 1
                # The substituted value really is that fit's midpoint, not the
                # rule's own extrapolation.
                assert r["gmm_cut"] == tau_mid
        assert seen_fallback, "the crafted tie should have declined at least one EVT guard"

    def test_prevalence_is_a_fraction(self):
        diag: list[dict] = []
        _run_safe("max_patch", diag_sink=diag)
        for d in diag:
            assert 0.0 <= d["sim_prevalence"] <= 1.0
            assert d["sim_n"] > 0

    def test_no_diagnostic_rows_without_a_sink(self):
        rows = _run_safe("max_patch")
        assert rows  # the run still works with cut_diag_sink=None
