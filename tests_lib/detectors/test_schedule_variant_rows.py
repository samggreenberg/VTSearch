"""The mix-in schedule rows on today's threshold stack (#2841, #3551).

The shipped threshold is the fold-anchored fused cut and the schedule blend is
only its fallback.  A schedule row therefore means one of two things, and the
load-bearing invariants are:

* on a **fallback** step the row for the schedule production resolves must
  reproduce the base row's threshold bit-for-bit - the fidelity that licenses
  every other schedule row on those steps;
* every row says which kind of step it sits on (``shipped_provenance``), so an
  analysis can never pool fallback rows with replacement counterfactuals.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.arms_schedule import _schedule_variant_rows
from vtscore.eval.voting_iterations import simulate_voting_iterations
from vtscore.training.blend_schedules import BlendContext, production_schedule_for
from vtscore.training.thresholds import NO_GOOD_THRESHOLD, calculate_safe_threshold
from vtscore.utils.scores import NON_FINITE_SCORE_SENTINEL

from .test_max_patch_style import _planted_dataset

pytestmark = pytest.mark.xdist_group("schedule-variant-rows")

_SHIPPED = production_schedule_for(region_voting=True)
_VARIANTS = [_SHIPPED, "rare:lo=1:hi=8", "corridor:w=0.2", "corridor_ramp"]


def _details(provenance: str, fallback: float | None, n_good: int, n_bad: int) -> dict:
    return {
        "xcal_threshold": NO_GOOD_THRESHOLD if fallback is not None else 0.62,
        "n_votes": n_good + n_bad,
        "n_good": n_good,
        "n_bad": n_bad,
        "provenance": provenance,
        "fold_fallback": fallback,
        "fold_orderings": [],
    }


def _scores(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.2, 0.05, 200), rng.normal(0.8, 0.05, 40)]).clip(0, 1).tolist()


class TestFallbackFidelity:
    def test_shipped_schedule_row_is_what_the_app_ships_on_a_fallback_step(self):
        """One positive among 31 votes - the folds cannot calibrate with a single
        vote of a class, so the app blends ``NO_GOOD_THRESHOLD`` with the GMM
        under the shipped schedule, and so must the row."""
        pool = _scores()
        test_scores, test_labels = np.array([0.1, 0.9]), np.array([0.0, 1.0])
        details = _details("gmm_blend", 0.5, n_good=1, n_bad=30)
        rows = _schedule_variant_rows(details, test_scores, test_labels, pool, 0, 1.0, _VARIANTS)
        by = {r["schedule"]: r for r in rows}
        ctx = BlendContext(31, 1, 30)
        app = calculate_safe_threshold(NO_GOOD_THRESHOLD, pool, ctx, schedule=_SHIPPED)
        assert by[_SHIPPED]["threshold"] == pytest.approx(app, abs=1e-6)
        # ... and a rarer-class ramp starting at one vote reads a lone positive
        # as "no calibration": pure GMM, whatever the total.
        gmm_only = calculate_safe_threshold(NO_GOOD_THRESHOLD, pool, ctx, schedule="pure_gmm")
        assert by["rare:lo=1:hi=8"]["threshold"] == pytest.approx(gmm_only, abs=1e-6)

    def test_unscorable_media_are_dropped_before_the_fit_as_the_app_drops_them(self):
        pool = _scores() + [NON_FINITE_SCORE_SENTINEL] * 25
        details = _details("gmm_blend", 0.5, n_good=0, n_bad=30)
        rows = _schedule_variant_rows(details, np.array([0.1, 0.9]), np.array([0.0, 1.0]), pool, 0, 1.0, [_SHIPPED])
        app = calculate_safe_threshold(NO_GOOD_THRESHOLD, pool, BlendContext(30, 0, 30), schedule=_SHIPPED)
        assert rows[0]["threshold"] == pytest.approx(app, abs=1e-6)

    def test_every_row_names_the_kind_of_step_it_sits_on(self):
        for prov, fb in (("gmm_blend", 0.5), ("fold_anchored[2/2]", None)):
            rows = _schedule_variant_rows(
                _details(prov, fb, 3, 20), np.array([0.1, 0.9]), np.array([0.0, 1.0]), _scores(), 0, 1.0, _VARIANTS
            )
            assert {r["shipped_provenance"] for r in rows} == {prov}
            assert {r["fold_fallback"] for r in rows} == {"" if fb is None else str(fb)}


@pytest.fixture(scope="module")
def harness_rows():
    medias, _ = _planted_dataset(n_per_cat=40, seed=0)
    return simulate_voting_iterations(
        medias,
        target_category="cat0",
        seed=0,
        dataset_name="planted",
        inclusion=0,
        region_voting=True,
        safe_thresholds=True,
        max_steps=16,
        style="max_patch",
        emit_calibration_metrics=True,
        schedule_variants=_VARIANTS,
    )


class TestHarness:
    def test_shipped_schedule_reproduces_every_fallback_step(self, harness_rows):
        base = {r["t"]: r for r in harness_rows if r.get("schedule") == "" and r.get("gmm_variant") == ""}
        sched = [r for r in harness_rows if r.get("schedule") == _SHIPPED]
        assert sched, "no schedule rows were emitted"
        fallback = [r for r in sched if r["shipped_provenance"] == "gmm_blend"]
        # The planted run's cold start falls back (one class under two votes),
        # so the fidelity check is not vacuous.
        assert fallback, "the planted run produced no fallback step to check"
        for r in fallback:
            assert r["threshold"] == base[r["t"]]["threshold"]
        for r in sched:
            assert r["shipped_provenance"] == base[r["t"]]["threshold_provenance"]

    def test_every_requested_schedule_emits_at_every_step(self, harness_rows):
        per_t: dict[int, set[str]] = {}
        for r in harness_rows:
            if r.get("schedule"):
                per_t.setdefault(r["t"], set()).add(r["schedule"])
        assert per_t
        assert all(names == set(_VARIANTS) for names in per_t.values())
