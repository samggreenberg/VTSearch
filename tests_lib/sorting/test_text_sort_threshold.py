"""The typed-query sort's line: the guarded rule of issue #3826, and its flag.

A cosine sort of a typed query is usually one broad mode with the matches as a
thin right shoulder, so a two-Gaussian midpoint splits the mode and lands in
its densest region.  :func:`guarded_text_sort_threshold` keeps the mixture only
where its components are separated (converged, then the midpoint) and cuts the
bulk's upper tail everywhere else.  :func:`text_sort_threshold` is the one entry
point the app's text route and the eval harness's opening both call, and with
the default rule it must *be* :func:`calculate_gmm_threshold`.

These are identities and mechanisms, not golden numbers: the measured effect
lives in ``docs/experiments/2026-09-22-text-cut-3826/REPORT.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.training.thresholds import gmm as G
from vtscore.training.thresholds import (
    TEXT_SORT_SEPARATION_D,
    TEXT_SORT_TAIL_K,
    ashman_d,
    bulk_location_scale,
    calculate_gmm_threshold,
    converge_score_gmm,
    fit_score_gmm,
    guarded_text_sort_threshold,
    text_sort_threshold,
)


def _shoulder(n_bulk: int = 5000, n_match: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One broad mode of cosines plus a thin right shoulder: a typical typed-query sort."""
    rng = np.random.default_rng(seed)
    bulk = rng.normal(0.0, 0.02, n_bulk)
    match = rng.normal(0.075, 0.012, n_match)
    scores = np.concatenate([bulk, match])
    labels = np.concatenate([np.zeros(n_bulk, bool), np.ones(n_match, bool)])
    return scores, labels


def _separated(n_lo: int = 800, n_hi: int = 40, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated modes: the single-object-image shape where the mixture is real."""
    rng = np.random.default_rng(seed)
    lo = rng.normal(0.10, 0.015, n_lo)
    hi = rng.normal(0.25, 0.015, n_hi)
    return np.concatenate([lo, hi]), np.concatenate([np.zeros(n_lo, bool), np.ones(n_hi, bool)])


class TestDefaultRuleIsTheShippedLine:
    """Off by default: the flag's absence changes nothing, bit for bit."""

    def test_default_rule_is_the_gmm_midpoint(self):
        assert G.TEXT_SORT_CUT_RULE == "gmm_midpoint"

    @pytest.mark.parametrize("seed", range(5))
    def test_default_equals_calculate_gmm_threshold_exactly(self, seed):
        scores, _ = _shoulder(seed=seed)
        values = scores.tolist()
        assert text_sort_threshold(values) == calculate_gmm_threshold(values)

    def test_explicit_rule_overrides_the_module_default(self, monkeypatch):
        scores, _ = _shoulder()
        values = scores.tolist()
        monkeypatch.setattr(G, "TEXT_SORT_CUT_RULE", "guarded_tail")
        assert text_sort_threshold(values) == guarded_text_sort_threshold(values)[0]
        assert text_sort_threshold(values, rule="gmm_midpoint") == calculate_gmm_threshold(values)

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="unknown text-sort cut rule"):
            text_sort_threshold([0.1, 0.2, 0.3], rule="midpoint_of_vibes")


class TestTheFlagIsReadFromTheEnvironment:
    """``VTSEARCH_TEXT_SORT_CUT`` selects the rule; a typo degrades to the default."""

    def _reload_rule(self, monkeypatch, value):
        import importlib

        monkeypatch.setenv("VTSEARCH_TEXT_SORT_CUT", value)
        mod = importlib.reload(G)
        try:
            return mod.TEXT_SORT_CUT_RULE
        finally:
            monkeypatch.delenv("VTSEARCH_TEXT_SORT_CUT")
            importlib.reload(G)

    def test_guarded_tail_is_selectable(self, monkeypatch):
        assert self._reload_rule(monkeypatch, " Guarded_Tail ") == "guarded_tail"

    def test_a_typo_falls_back_to_the_default(self, monkeypatch):
        assert self._reload_rule(monkeypatch, "guarded-tail") == "gmm_midpoint"


class TestTheTailBranch:
    """An unseparated sort is cut at median + k bulk sigmas, with no optimiser in it."""

    def test_a_shoulder_takes_the_tail_branch(self):
        scores, _ = _shoulder()
        fit = fit_score_gmm(G.gmm_fit_array(scores.tolist()))
        assert ashman_d(fit) < TEXT_SORT_SEPARATION_D
        _cut, branch = guarded_text_sort_threshold(scores.tolist())
        assert branch == "tail"

    def test_the_tail_cut_is_the_closed_form(self):
        scores, _ = _shoulder()
        mu, sigma = bulk_location_scale(scores)
        cut, _ = guarded_text_sort_threshold(scores.tolist())
        assert cut == pytest.approx(mu + TEXT_SORT_TAIL_K * sigma, rel=0, abs=1e-15)

    def test_it_admits_about_the_shoulder_where_the_midpoint_admits_the_mode(self):
        """The finding of #3826 in miniature: the midpoint paints a large share of the
        haystack green; the tail line paints roughly the matches."""
        scores, labels = _shoulder()
        values = scores.tolist()
        tail_cut, _ = guarded_text_sort_threshold(values)
        mid_cut = calculate_gmm_threshold(values)
        tail_adm = scores >= tail_cut
        mid_adm = scores >= mid_cut
        assert mid_adm.sum() > 10 * labels.sum()
        assert 0.5 * labels.sum() <= tail_adm.sum() <= 2 * labels.sum()
        assert (tail_adm & labels).sum() / tail_adm.sum() > 0.8

    def test_the_bulk_scale_ignores_a_heavy_right_shoulder(self):
        """Read from the lower half, sigma barely moves when matches pile up on the right;
        a standard deviation would inflate and drag the cut up into the matches."""
        rng = np.random.default_rng(3)
        bulk = rng.normal(0.0, 0.02, 4000)
        heavy = np.concatenate([bulk, rng.normal(0.08, 0.01, 800)])
        _, s_bulk = bulk_location_scale(bulk)
        _, s_heavy = bulk_location_scale(heavy)
        assert s_heavy == pytest.approx(s_bulk, rel=0.2)
        assert float(np.std(heavy)) > 1.4 * s_bulk

    def test_nonfinite_scores_are_ignored_by_the_bulk(self):
        scores, _ = _shoulder()
        with_nan = np.concatenate([scores, [np.nan, np.inf]])
        assert bulk_location_scale(with_nan) == bulk_location_scale(scores)


class TestTheMixtureBranch:
    """A separated sort keeps the mixture, fitted to convergence, cut at the midpoint."""

    def test_separated_modes_take_the_gmm_branch_between_them(self):
        scores, labels = _separated()
        cut, branch = guarded_text_sort_threshold(scores.tolist())
        assert branch == "gmm"
        assert 0.15 < cut < 0.20
        assert ((scores >= cut) == labels).mean() > 0.99

    def test_the_branch_uses_the_converged_fit(self):
        scores, _ = _separated()
        arr = G.gmm_fit_array(scores.tolist())
        shipped = fit_score_gmm(arr)
        cut, _ = guarded_text_sort_threshold(scores.tolist())
        assert cut == converge_score_gmm(arr, shipped).midpoint()

    def test_convergence_removes_the_start_from_the_answer(self):
        """The reason the separated branch converges: from two different starts the
        shipped 1e-3 stop can differ, the converged midpoint does not."""
        scores, _ = _separated()
        arr = np.asarray(scores, dtype=np.float64)
        xs = np.sort(arr)
        k = int(0.7 * xs.size)
        other = G.GmmFit1D(
            k / xs.size, float(xs[:k].mean()), float(xs[:k].var()),
            1 - k / xs.size, float(xs[k:].mean()), float(xs[k:].var()),
        )  # fmt: skip
        a = converge_score_gmm(arr, G._two_means_init(xs)).midpoint()
        b = converge_score_gmm(arr, other).midpoint()
        assert a == pytest.approx(b, abs=1e-6)


class TestKnownFailureMajorityClass:
    """Documented, not guessed at: a majority-class query the embedder does not
    separate is cut at the top of its own matches (#3826, "a person" in COCO)."""

    def _majority(self, separated: bool) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        gap = 0.15 if separated else 0.02
        neg = rng.normal(0.0, 0.02, 2000)
        pos = rng.normal(gap, 0.02, 3000)  # 60% prevalence
        return np.concatenate([neg, pos]), np.concatenate([np.zeros(2000, bool), np.ones(3000, bool)])

    def test_unseparated_majority_admits_only_a_sliver_of_the_matches(self):
        scores, labels = self._majority(separated=False)
        cut, branch = guarded_text_sort_threshold(scores.tolist())
        assert branch == "tail"
        recall = ((scores >= cut) & labels).sum() / labels.sum()
        assert recall < 0.1  # the limitation, pinned: change this test if a fix lands

    def test_separated_majority_is_handled_by_the_mixture(self):
        scores, labels = self._majority(separated=True)
        cut, branch = guarded_text_sort_threshold(scores.tolist())
        assert branch == "gmm"
        recall = ((scores >= cut) & labels).sum() / labels.sum()
        assert recall > 0.95


class TestFallbacks:
    """The same edges :func:`calculate_gmm_threshold` guards, with the same answers."""

    @pytest.mark.parametrize("scores", [[], [0.3]])
    def test_fewer_than_two_scores(self, scores):
        assert guarded_text_sort_threshold(scores) == (0.5, "fallback")

    def test_constant_sample_cuts_at_the_value(self):
        cut, _ = guarded_text_sort_threshold([0.3] * 10)
        assert cut == pytest.approx(0.3)

    def test_ashman_d_of_identical_components_is_zero(self):
        fit = G.GmmFit1D(0.5, 0.3, 0.0, 0.5, 0.3, 0.0)
        assert ashman_d(fit) == 0.0
        assert math.isinf(ashman_d(G.GmmFit1D(0.5, 0.1, 0.0, 0.5, 0.3, 0.0)))


class TestCallersFollowTheRule:
    """The app's text route and the harness's opening read the same line."""

    def test_harness_bad_phase_line_on_a_text_sort(self, monkeypatch):
        from vtscore.eval.al_strategies import _sort_threshold

        scores, _ = _shoulder()
        ranking = dict(enumerate(scores.tolist()))
        monkeypatch.setattr(G, "TEXT_SORT_CUT_RULE", "guarded_tail")
        assert _sort_threshold(ranking, typed_query=True) == guarded_text_sort_threshold(list(ranking.values()))[0]
        # A known-good centroid sort is not a typed query and keeps the midpoint.
        assert _sort_threshold(ranking) == calculate_gmm_threshold(list(ranking.values()))

    def test_startup_schedule_mid_is_the_text_line(self, monkeypatch):
        from vtscore.eval.startup_schedule import parse_startup_schedule, round_cut

        scores, _ = _shoulder()
        (rnd,) = parse_startup_schedule("n1@mid")
        monkeypatch.setattr(G, "TEXT_SORT_CUT_RULE", "guarded_tail")
        assert round_cut(scores.tolist(), rnd) == guarded_text_sort_threshold(scores.tolist())[0]

    def test_cosine_sort_active_uses_the_text_rule_only_for_text(self, monkeypatch):
        import vtscore.training.thresholds as T
        from vtscore.training.query_sort import cosine_sort_active

        from tests_lib.sorting.test_query_sort import _fill_active_medias

        query = _fill_active_medias()
        monkeypatch.setattr(T, "text_sort_threshold", lambda s: 0.123)
        monkeypatch.setattr(T, "calculate_gmm_threshold", lambda s: 0.456)
        assert cosine_sort_active(query, role="text")[1] == 0.123
        assert cosine_sort_active(query, role="score")[1] == 0.456
