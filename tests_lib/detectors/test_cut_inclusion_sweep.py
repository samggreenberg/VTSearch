"""Tests for the cut-rule x inclusion side frame (issue #2865).

The shipped ``mid`` cut was chosen by two calibration runs that scored every
arm at inclusion 0, and a bare midpoint of two component means ignores the cost
weights inclusion arrives as - so it made the Inclusion knob a no-op for every
detector with usable calibration folds.  ``mid_tilt`` restored the tilt while
reproducing the measured arm bit-for-bit at 0; this frame is the sweep that
prices that tilt against its alternatives.

What these tests pin is the frame's *decision content*, not its numbers: that
each candidate rule is present, that every row is scored under its own ``k``
(otherwise regret is not comparable along the knob), that the two rules which
must coincide with the measured arm at inclusion 0 actually do, and - the
observable the whole issue turns on - that ``admitted_frac`` separates a rule
that moves the admitted set from one that cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.calibration_metrics import inclusion_weights, operating_cost
from vtscore.eval.voting_columns import CUT_INCLUSION_COLUMNS
from vtscore.eval.voting_iterations import simulate_voting_iterations

# Reuse the synthetic planted-patch dataset builders from the Max-Patch tests.
from .sweep_cache import memoize_sweep
from .test_max_patch_style import _planted_dataset

# Six of the tests below sweep the identical default frame and assert on
# different columns of it, so the sweep is memoized per worker and this module
# is pinned to one worker for the cache to hit.  Rows handed back are shared:
# read them, never mutate them.  See sweep_cache.py.
pytestmark = pytest.mark.xdist_group("cut-inclusion-sweep")

#: The knob positions the frame sweeps.  Deliberately near the ends of the
#: ``[-10, 10]`` range rather than clustered around zero: ``rate`` - and
#: therefore the ``mid_tilt`` shift composed from it - is *invariant* to the
#: cost weights while its root stays inside the inter-mean interval, and only
#: starts moving once the root is pushed out (see ``gmm_cut_from_fit``).  How
#: far out that takes depends on how well separated the fitted components are,
#: which depends on the head: a narrow grid that happened to show movement
#: under the logistic head showed none under the shipped linear SVM, whose
#: fits keep the root interior further along the knob.  A frame whose subject
#: is "does this rule read inclusion at all" has to sweep where the answer can
#: be yes.
_KS = [-8, -4, 0, 4, 8]
_RULES = ["mid", "mid_tilt", "rate", "cross_tilt", "q_tilt"]


@memoize_sweep
def _run(seed=0, max_steps=12, ks=None, rules=None, qtilt_steps=None):
    medias, _ = _planted_dataset(n_per_cat=40, seed=seed)
    sink: list[dict] = []
    simulate_voting_iterations(
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
        anchored_weights=[0.3],
        anchored_rules=list(rules if rules is not None else _RULES),
        anchored_fold_combines=["qmean"],
        cut_inclusion_ks=list(ks if ks is not None else _KS),
        cut_inclusion_sink=sink,
        cut_inclusion_qtilt_steps=qtilt_steps,
    )
    return sink


class TestCutInclusionFrame:
    def test_every_candidate_rule_is_swept_over_the_whole_knob(self):
        sink = _run()
        assert sink, "no cut-inclusion rows produced"
        assert {r["cut_rule"] for r in sink} == set(_RULES)
        # Each arm is a complete grid over k - a rule that silently dropped a
        # k would make its regret curve incomparable to the others'.
        by_arm: dict[tuple, set[int]] = {}
        for r in sink:
            by_arm.setdefault((r["t"], r["arm"]), set()).add(r["inclusion_k"])
        assert all(ks == set(_KS) for ks in by_arm.values())

    def test_columns_and_ranges(self):
        sink = _run()
        for r in sink:
            assert set(CUT_INCLUSION_COLUMNS).issubset(r.keys())
            assert np.isfinite(r["cut_threshold"])
            assert 0.0 <= r["admitted_frac"] <= 1.0
            assert r["n_admitted"] == pytest.approx(r["admitted_frac"] * r["n_test"], abs=1.0)
            # The oracle reads the labels of the set it optimises, so it can
            # never cost more than the rule's own cut at the same k.
            assert r["k_oracle_cost"] <= r["cut_cost"] + 1e-9
            # Each column is rounded independently for the CSV, so the identity
            # holds only to a few ulps of that rounding, not exactly.
            assert r["cut_regret"] == pytest.approx(r["cut_cost"] - r["k_oracle_cost"], abs=5e-6)

    def test_each_row_is_scored_at_its_own_k_not_the_run_inclusion(self):
        """The point of the frame: cost/regret must price inclusion at the row's
        own ``k``.  Scoring every row at the run's reporting inclusion (0 here)
        would make an arm's regret curve flat in ``k`` by construction and the
        comparison meaningless."""
        sink = _run()
        for r in sink:
            wf, wn = inclusion_weights(r["inclusion_k"])
            # The rate columns are rounded before the weights multiply them, so
            # the tolerance has to carry the weights too.
            assert r["cut_cost"] == pytest.approx(wf * r["cut_fpr"] + wn * r["cut_fnr"], abs=(wf + wn) * 1e-6)

    def test_mid_is_the_inclusion_blind_null_and_the_tilts_are_not(self):
        """This *is* the bug #2865 reports, pinned as a measurement.

        ``mid`` returns one threshold for the entire knob; every candidate that
        replaces it must move both the threshold and - the part that actually
        matters to a user - the admitted set.
        """
        sink = _run()
        moved_admitted: dict[str, bool] = {}
        for rule in _RULES:
            per_step: dict[int, set[float]] = {}
            per_step_adm: dict[int, set[float]] = {}
            for r in sink:
                if r["cut_rule"] != rule:
                    continue
                per_step.setdefault(r["t"], set()).add(r["cut_threshold"])
                per_step_adm.setdefault(r["t"], set()).add(r["admitted_frac"])
            assert per_step, f"no rows for rule {rule}"
            if rule == "mid":
                assert all(len(v) == 1 for v in per_step.values()), "mid must be constant in inclusion"
                assert all(len(v) == 1 for v in per_step_adm.values())
            else:
                assert any(len(v) > 1 for v in per_step.values()), f"{rule} never moved the threshold"
            moved_admitted[rule] = any(len(v) > 1 for v in per_step_adm.values())
        # A rule that moves the threshold without moving the admitted set has
        # not fixed anything; at least the by-construction rule must clear it.
        assert moved_admitted["q_tilt"], "q_tilt moves the admitted fraction by construction"

    def test_the_tilts_reproduce_the_measured_midpoint_arm_at_inclusion_zero(self):
        """``mid_tilt`` and ``q_tilt`` are both defined as ``mid`` plus a shift
        that vanishes at inclusion 0.  That identity is what lets #2865 restore
        the knob without re-opening the anchor-mass runs' κ=0.3 recommendation,
        so it is pinned exactly rather than approximately."""
        sink = _run()
        at_zero: dict[tuple[int, str], float] = {
            (r["t"], r["cut_rule"]): r["cut_threshold"] for r in sink if r["inclusion_k"] == 0
        }
        steps = {t for (t, _rule) in at_zero}
        assert steps
        for t in steps:
            assert at_zero[(t, "mid_tilt")] == at_zero[(t, "mid")]
            assert at_zero[(t, "q_tilt")] == at_zero[(t, "mid")]

    def test_thresholds_are_monotone_in_inclusion(self):
        """The nesting contract: everything admitted at ``k`` stays admitted at
        ``k + 1``, so the threshold must be non-increasing in ``k`` for every
        candidate rule, not only the shipped one."""
        sink = _run()
        by_arm: dict[tuple, list[tuple[int, float]]] = {}
        for r in sink:
            by_arm.setdefault((r["t"], r["arm"]), []).append((r["inclusion_k"], r["cut_threshold"]))
        for key, points in by_arm.items():
            thr = [t for _k, t in sorted(points)]
            assert all(b <= a + 1e-9 for a, b in zip(thr, thr[1:], strict=False)), f"{key} not monotone: {thr}"

    def test_qtilt_step_expands_into_separate_arms(self):
        """``q_tilt``'s step size is a free parameter with no principled value,
        so the sweep has to price it; every other rule must ignore it rather
        than silently duplicating rows."""
        sink = _run(qtilt_steps=[0.01, 0.05])
        q_arms = {r["arm"] for r in sink if r["cut_rule"] == "q_tilt"}
        assert len(q_arms) == 2, q_arms
        assert {r["qtilt_step"] for r in sink if r["cut_rule"] == "q_tilt"} == {0.01, 0.05}
        # A bigger step must move the admitted set further across the knob.
        span: dict[float, float] = {}
        for step in (0.01, 0.05):
            rows = [r for r in sink if r["cut_rule"] == "q_tilt" and r["qtilt_step"] == step]
            per_step = {}
            for r in rows:
                per_step.setdefault(r["t"], []).append(r["admitted_frac"])
            span[step] = max(max(v) - min(v) for v in per_step.values())
        assert span[0.05] > span[0.01]
        for r in sink:
            if r["cut_rule"] != "q_tilt":
                assert np.isnan(r["qtilt_step"])

    def test_off_by_default(self):
        """Every other calibration study must be byte-unchanged by this frame."""
        sink = _run(ks=[])
        assert sink == []


class TestRateIsPriorFree:
    """The premise #2865's candidate 2 rests on, checked rather than assumed.

    The issue proposes "drop the mixture-weight factor: ``lam = fnr/fpr``
    instead of ``(fnr/fpr)*(w_lo/w_hi)``" on the grounds that ``mid`` beat
    ``rate`` because ``mid`` ignores the acquisition-biased mixture weights and
    ``rate`` reads them.  It does not: the prior-odds factor in ``rate``'s
    ``lam`` cancels the ``w_lo/w_hi`` inside ``_rate_cut``'s ``offset``
    exactly.  So the proposed candidate is what ``rate`` already computes, and
    the rule that genuinely retains the priors is ``cross_tilt``.
    """

    @staticmethod
    def _fit(w_lo):
        from vtscore.training.thresholds import GmmFit1D

        return GmmFit1D(w_lo=w_lo, mu_lo=0.2, var_lo=0.05, w_hi=1.0 - w_lo, mu_hi=0.7, var_hi=0.01)

    def test_rate_is_invariant_to_the_mixture_weights_on_its_interior_root(self):
        from vtscore.training.thresholds import CUT_KIND_INTERIOR, gmm_cut_from_fit

        for k in (-4, -1, 0, 1, 4):
            wf, wn = inclusion_weights(k)
            cuts = []
            for w_lo in (0.5, 0.9, 0.99):
                cut, kind = gmm_cut_from_fit(self._fit(w_lo), "rate", wf, wn)
                assert kind == CUT_KIND_INTERIOR, "fixture must stay on the interior branch"
                cuts.append(cut)
            assert cuts[0] == pytest.approx(cuts[1]) == pytest.approx(cuts[2])

    def test_cross_tilt_does_read_them(self):
        from vtscore.training.thresholds import gmm_cut_from_fit

        wf, wn = inclusion_weights(0)
        cuts = [gmm_cut_from_fit(self._fit(w), "cross_tilt", wf, wn)[0] for w in (0.5, 0.9, 0.99)]
        assert len(set(np.round(cuts, 9))) == 3, cuts

    def test_cross_tilt_is_monotone_in_inclusion(self):
        from vtscore.training.thresholds import gmm_cut_from_fit

        cuts = []
        for k in range(-10, 11):
            wf, wn = inclusion_weights(k)
            cuts.append(gmm_cut_from_fit(self._fit(0.9), "cross_tilt", wf, wn)[0])
        assert all(b <= a + 1e-12 for a, b in zip(cuts, cuts[1:], strict=False)), cuts

    def test_unknown_rule_still_rejected(self):
        from vtscore.training.thresholds import gmm_cut_from_fit

        with pytest.raises(ValueError, match="unknown cut rule"):
            gmm_cut_from_fit(self._fit(0.5), "mid_tilt")


def test_operating_cost_agrees_with_the_frame():
    """Guards the frame against silently pricing a different loss than the one
    ``inclusion_cost_weights`` defines - the single definition the shipped cut
    reads too."""
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    wf, wn = inclusion_weights(2)
    cost, fpr, fnr = operating_cost(scores, labels, 0.5, wf, wn)
    assert cost == pytest.approx(wf * fpr + wn * fnr)


class TestHingeRulesInTheFrame:
    """Issue #3557: the sign-dependent rules ride the same re-cut as #2865's arms."""

    _HINGE_RULES = ["mid_tilt", "cross_tilt", "hinge", "hinge_raw", "hinge_cont"]
    _HINGE_KS = [-4, -1, -0.25, 0, 1, 4]

    def test_hinges_match_the_incumbent_at_and_above_zero_and_carry_the_seam(self):
        sink = _run(ks=self._HINGE_KS, rules=self._HINGE_RULES)
        assert {r["cut_rule"] for r in sink} == set(self._HINGE_RULES)
        by = {(r["t"], r["cut_rule"], r["inclusion_k"]): r for r in sink}
        steps = {t for (t, _rule, _k) in by}
        assert steps
        for t in steps:
            for k in (0, 1, 4):
                for rule in ("hinge", "hinge_raw", "hinge_cont"):
                    assert by[(t, rule, k)]["cut_threshold"] == by[(t, "mid_tilt", k)]["cut_threshold"]
            # The seam columns are per-step constants of the shared fit.
            seams = {(r["seam_q_mid"], r["seam_q_cross0"]) for r in sink if r["t"] == t}
            assert len(seams) == 1
            # And they are the quantities the rules read: mid_tilt sits at q_mid
            # at inclusion 0, which is exactly what the frame's quantile says.
            assert by[(t, "mid_tilt", 0)]["fold_quantile"] == pytest.approx(
                by[(t, "mid_tilt", 0)]["seam_q_mid"], abs=2e-6
            )
            # The guarded hinge is never more inclusive than the incumbent below 0.
            for k in (-4, -1, -0.25):
                assert by[(t, "hinge", k)]["cut_threshold"] >= by[(t, "mid_tilt", k)]["cut_threshold"]

    def test_fractional_stops_are_scored_at_their_own_weights(self):
        sink = _run(ks=self._HINGE_KS, rules=self._HINGE_RULES)
        rows = [r for r in sink if r["inclusion_k"] == -0.25]
        assert rows
        wf, wn = inclusion_weights(-0.25)
        for r in rows:
            assert r["cut_cost"] == pytest.approx(wf * r["cut_fpr"] + wn * r["cut_fnr"], abs=(wf + wn) * 1e-6)


def test_live_cut_rule_moves_the_live_threshold_onto_that_rule():
    """The #3557 run-level knob: with ``live_cut_rule`` set and reporting at
    inclusion -8 (well below the hinge's seam, where
    the planted fixture's rules actually separate), every fold-anchored step's live
    threshold is the frame's re-cut of that same rule at k=-8 - so a run-level
    arm and the paired re-cut are the same object - and the default arm is the
    incumbent's.  ``mid`` rides along because the planted fixture's fits are
    near 50/50, where every hinge coincides with ``mid_tilt``: the inert rule is
    the one that proves the knob reached the live cut at all."""
    medias, _ = _planted_dataset(n_per_cat=40, seed=0)

    def run(live):
        sink: list[dict] = []
        rows = simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=0,
            dataset_name="planted",
            inclusion=-8,
            region_voting=True,
            safe_thresholds=True,
            max_steps=10,
            style="max_patch",
            emit_calibration_metrics=True,
            anchored_thresholds=True,
            anchored_weights=[0.3],
            anchored_rules=["mid_tilt", "mid", "hinge"],
            anchored_fold_combines=["qmean"],
            cut_inclusion_ks=[-8],
            cut_inclusion_sink=sink,
            live_cut_rule=live,
        )
        return rows, sink

    for live, rule in ((None, "mid_tilt"), ("mid", "mid"), ("hinge", "hinge")):
        rows, sink = run(live)
        recut = {r["t"]: r["cut_threshold"] for r in sink if r["cut_rule"] == rule}
        live_rows = [
            r
            for r in rows
            if not r.get("gmm_variant") and str(r.get("threshold_provenance", "")).startswith("fold_anchored")
        ]
        assert live_rows, "no fold-anchored steps to compare"
        compared = [r for r in live_rows if r["t"] in recut]
        assert compared
        for r in compared:
            assert r["threshold"] == pytest.approx(recut[r["t"]], abs=2e-6), (live, r["t"])
        if live == "mid":
            incumbent = {r["t"]: r["cut_threshold"] for r in sink if r["cut_rule"] == "mid_tilt"}
            assert any(abs(r["threshold"] - incumbent[r["t"]]) > 1e-4 for r in compared), "knob never reached the cut"
