"""The retired live threshold rules (issue #4184).

#4184 draws the deck's calibration ladder as one progression of cost curves, so
each rung has to run closed-loop: its cut is what reporting reads and what
autopilot aims at.  These tests pin what makes a rung *that* rung - that each
rule computes the cut its slide draws, from the inputs its slide names, and
that the knob reaches the live cut (and the acquisition cut) without moving
anything on the default arm.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.calibration_metrics import oracle_cut
from vtscore.eval.live_threshold_rules import (
    LIVE_THRESHOLD_RULES,
    anchored_rawmean_threshold,
    check_live_threshold,
    gmm_midpoint_threshold,
    live_threshold,
    xcal_mincost_threshold,
)
from vtscore.eval.voting_iterations import simulate_voting_iterations
from vtscore.training.blend_schedules import BlendContext
from vtscore.training.thresholds import (
    NO_GOOD_THRESHOLD,
    calculate_safe_threshold,
    fit_fold_anchored_cut,
    fit_gmm_threshold,
    gmm_cut_from_fit,
)

from .sweep_cache import memoize_sweep
from .test_max_patch_style import _planted_dataset

# The runs below repeat per rule across tests, so each is simulated once per
# worker; the module is pinned to one worker for the cache to hit.  Rows are
# shared and must stay read-only.  See sweep_cache.py.
pytestmark = pytest.mark.xdist_group("live-threshold-rules")


# --- rung 1: cross-calibration, as the slide draws it --------------------------


def test_xcal_mincost_averages_each_folds_cost_minimising_cut():
    """Fold 1 separates cleanly, so its cheapest cut is its lowest positive (0.7).
    Fold 2 ties at cost 0.5 between 0.8 and 0.5; ties break toward the higher cut,
    as the February rule did, so it cuts at 0.8.  The rung hands the mean on."""
    fold1 = ([0.9, 0.7, 0.4, 0.2], [1.0, 1.0, 0.0, 0.0])
    fold2 = ([0.8, 0.6, 0.5, 0.1], [1.0, 0.0, 1.0, 0.0])
    assert xcal_mincost_threshold([fold1, fold2], 0) == pytest.approx(0.75)


def test_xcal_mincost_is_the_oracle_cut_per_fold():
    """One definition of 'the cost-minimising cut' in the harness, not two."""
    rng = np.random.default_rng(7)
    folds = []
    for _ in range(3):
        labels = np.array([1.0] * 6 + [0.0] * 9)
        scores = np.where(labels == 1.0, rng.normal(0.7, 0.15, 15), rng.normal(0.35, 0.15, 15))
        folds.append((scores.tolist(), labels.tolist()))
    want = np.mean([oracle_cut(np.asarray(s), np.asarray(y), 1.0, 1.0)[0] for s, y in folds])
    assert xcal_mincost_threshold(folds, 0) == pytest.approx(want)


def test_xcal_mincost_skips_a_fold_that_cannot_place_a_line():
    """A held-out half with one class says nothing about where the line goes."""
    usable = ([0.9, 0.7, 0.4, 0.2], [1.0, 1.0, 0.0, 0.0])
    one_class = ([0.9, 0.1], [1.0, 1.0])
    assert xcal_mincost_threshold([usable, one_class], 0) == pytest.approx(0.7)
    assert xcal_mincost_threshold([one_class], 0) is None
    assert xcal_mincost_threshold([], 0) is None


def test_xcal_mincost_drops_unscored_items_before_cutting():
    """An item the fold model could not score sits at -1.0; as a negative it
    would be the 'easiest' one and cost nothing, but as a positive it would drag
    the cheapest cut below every real score."""
    clean = ([0.9, 0.7, 0.4, 0.2], [1.0, 1.0, 0.0, 0.0])
    dirty = ([0.9, 0.7, 0.4, 0.2, -1.0], [1.0, 1.0, 0.0, 0.0, 1.0])
    assert xcal_mincost_threshold([dirty], 0) == pytest.approx(xcal_mincost_threshold([clean], 0))


# --- rung 2: the GMM midpoint --------------------------------------------------


def test_gmm_midpoint_is_the_label_free_mixture_cut():
    rng = np.random.default_rng(3)
    scores = np.concatenate([rng.normal(0.2, 0.05, 400), rng.normal(0.8, 0.05, 60)]).tolist()
    cut = gmm_midpoint_threshold(scores)
    want, fit = fit_gmm_threshold(scores)
    assert fit is not None
    assert cut == pytest.approx(want)
    assert 0.35 < cut < 0.65, "the midpoint of the two means sits in the valley"


def test_gmm_midpoint_ignores_unscored_items():
    rng = np.random.default_rng(3)
    scores = np.concatenate([rng.normal(0.2, 0.05, 400), rng.normal(0.8, 0.05, 60)]).tolist()
    assert gmm_midpoint_threshold([*scores, -1.0, -1.0, -1.0]) == pytest.approx(gmm_midpoint_threshold(scores))


# --- rung 4's strawman: anchored cuts averaged as raw scores --------------------


def _two_fold_cut():
    """Two folds whose models score on different scales - the slide's premise."""
    rng = np.random.default_rng(11)
    hay1 = np.concatenate([rng.normal(0.2, 0.05, 300), rng.normal(0.6, 0.05, 40)])
    hay2 = np.concatenate([rng.normal(0.4, 0.05, 300), rng.normal(0.9, 0.03, 40)])
    ord1 = ([0.62, 0.58, 0.61, 0.19, 0.22, 0.18], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    ord2 = ([0.91, 0.88, 0.9, 0.41, 0.38, 0.42], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    final = np.concatenate([rng.normal(0.1, 0.03, 300), rng.normal(0.5, 0.03, 40)])
    cut = fit_fold_anchored_cut([hay1, hay2], [ord1, ord2], final.tolist())
    assert cut is not None and len(cut.fits) == 2
    return cut


def test_anchored_rawmean_averages_the_per_fold_midpoints():
    cut = _two_fold_cut()
    want = np.mean([gmm_cut_from_fit(fit, "mid", 1.0, 1.0)[0] for fit in cut.fits])
    assert anchored_rawmean_threshold(cut) == pytest.approx(want)


def test_anchored_rawmean_is_not_the_quantile_transfer():
    """The strawman's whole point: the fold cuts (~0.4 and ~0.65) average to a
    number that means nothing on the final model, whose modes sit at 0.1 and
    0.5 - it lands in the top mode.  The shipped transfer lands in the valley."""
    cut = _two_fold_cut()
    raw = anchored_rawmean_threshold(cut)
    shipped = cut.threshold_at(0)
    assert raw > 0.45, raw
    assert 0.15 < shipped < 0.45, shipped


# --- dispatch and fallbacks ------------------------------------------------------


_FOLDS = [
    ([0.9, 0.7, 0.4, 0.2], [1.0, 1.0, 0.0, 0.0]),
    ([0.8, 0.6, 0.5, 0.1], [1.0, 0.0, 1.0, 0.0]),
]
_CTX = BlendContext(n_labels=20, n_good=8, n_bad=12)


def _haystack() -> list[float]:
    rng = np.random.default_rng(5)
    return np.concatenate([rng.normal(0.2, 0.05, 300), rng.normal(0.8, 0.05, 40)]).tolist()


def _dispatch(rule: str, *, details: dict | None = None, fold_threshold: float = 0.42) -> tuple[float, str]:
    return live_threshold(
        rule,
        fold_threshold=fold_threshold,
        details=details if details is not None else {"fold_orderings": _FOLDS, "fold_fallback": None},
        haystack_scores=_haystack(),
        ctx=_CTX,
        schedule="corridor20",
        fitted_cut=None,
        shipped_threshold=0.33,
        shipped_provenance="gmm_blend",
        inclusion=0,
    )


def test_blend_is_rungs_one_and_two_combined_under_the_schedule():
    cut, prov = _dispatch("blend")
    assert prov == "blend"
    want = calculate_safe_threshold(0.75, _haystack(), _CTX, schedule="corridor20")
    assert cut == pytest.approx(want)


def test_xcal_mincost_falls_back_to_the_fold_sentinel():
    cut, prov = _dispatch("xcal_mincost", details={"fold_orderings": [], "fold_fallback": 0.5}, fold_threshold=0.5)
    assert (cut, prov) == (0.5, "xcal_mincost:fallback")


def test_blend_substitutes_no_good_for_a_cut_never_computed():
    """Production's substitution (``_blend_xcal_input``), applied to this rung's x-cal side."""
    cut, _ = _dispatch("blend", details={"fold_orderings": [], "fold_fallback": 0.5})
    want = calculate_safe_threshold(NO_GOOD_THRESHOLD, _haystack(), _CTX, schedule="corridor20")
    assert cut == pytest.approx(want)


def test_anchored_rawmean_falls_back_to_the_shipped_fallback():
    cut, prov = _dispatch("anchored_rawmean")
    assert cut == 0.33
    assert prov == "anchored_rawmean:fallback:gmm_blend"


def test_gmm_mid_reads_no_vote():
    a, _ = _dispatch("gmm_mid")
    b, _ = _dispatch("gmm_mid", details={"fold_orderings": [], "fold_fallback": 0.5}, fold_threshold=0.5)
    assert a == b


def test_check_refuses_what_the_run_cannot_honour():
    check_live_threshold(None, safe_thresholds=False, live_cut_rule="hinge")
    for rule in LIVE_THRESHOLD_RULES:
        check_live_threshold(rule, safe_thresholds=True, live_cut_rule=None)
    with pytest.raises(ValueError, match="unknown live_threshold"):
        check_live_threshold("conformal", safe_thresholds=True, live_cut_rule=None)
    with pytest.raises(ValueError, match="needs safe_thresholds"):
        check_live_threshold("gmm_mid", safe_thresholds=False, live_cut_rule=None)
    with pytest.raises(ValueError, match="cannot both be set"):
        check_live_threshold("gmm_mid", safe_thresholds=True, live_cut_rule="mid")


# --- the knob reaches the live cut ------------------------------------------------


def _base(rows):
    return [
        r
        for r in rows
        if not str(r.get("gmm_variant") or "").strip()
        and str(r.get("pool_variant") or "") in ("", "max")
        and not str(r.get("schedule") or "").strip()
    ]


@memoize_sweep
def _run(live=None):
    medias, _ = _planted_dataset(n_per_cat=40, seed=0)
    return _base(
        simulate_voting_iterations(
            medias,
            target_category="cat0",
            seed=3,
            dataset_name="synthetic",
            max_steps=30,
            head="linear",
            style="whole_image",
            emit_calibration_metrics=True,
            live_threshold=live,
        )
    )


@pytest.mark.parametrize("rule", LIVE_THRESHOLD_RULES)
def test_every_trained_step_is_cut_by_the_rule(rule):
    rows = [r for r in _run(rule) if np.isfinite(r["threshold"])]
    assert rows
    provs = {str(r["threshold_provenance"]) for r in rows}
    assert all(p.startswith(rule) for p in provs), provs
    assert any(p == rule for p in provs), f"{rule} never cut a step on its own: {provs}"


@pytest.mark.parametrize("rule", LIVE_THRESHOLD_RULES)
def test_acquisition_aims_at_the_rungs_own_cut(rule):
    """The retired rules predate the acquisition offset, and the fit that offset
    would re-cut is not this arm's - so the selector reads the reporting cut."""
    rows = [r for r in _run(rule) if np.isfinite(r["threshold"])]
    assert rows
    for r in rows:
        assert r["acq_threshold"] == pytest.approx(r["threshold"], abs=1e-6)


def test_the_knob_moves_the_cut_and_the_default_arm_is_shipped():
    shipped = [r for r in _run() if np.isfinite(r["threshold"])]
    assert any(str(r["threshold_provenance"]).startswith("fold_anchored") for r in shipped)
    for rule in LIVE_THRESHOLD_RULES:
        ruled = {r["t"]: r["threshold"] for r in _run(rule) if np.isfinite(r["threshold"])}
        assert any(abs(ruled[r["t"]] - r["threshold"]) > 1e-4 for r in shipped if r["t"] in ruled), rule
