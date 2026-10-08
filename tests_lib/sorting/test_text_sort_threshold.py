"""The typed-query sort's two lines: the guarded display rule of #3826, the midpoint it samples at (#4136).

A cosine sort of a typed query is usually one broad mode with the matches as a
thin right shoulder, so a two-Gaussian midpoint splits the mode and lands in
its densest region.  :func:`guarded_text_sort_threshold` keeps the mixture only
where its components are separated (converged, then the midpoint) and cuts the
bulk's upper tail everywhere else.  That is the **display** line, and since
#4136 the default one.  The sort's **acquisition** cut - where Autopilot's Bad
phase samples - stays the midpoint under every rule, because the #3826 A/B
found the guarded line puts that phase inside the matches.
:func:`text_sort_cuts` draws both from one fit; the app's text route sends
them as ``threshold`` / ``acq_threshold``, and the eval harness's opening reads
the acquisition half through :func:`text_sort_acquisition_threshold`.

These are identities and mechanisms, not golden numbers: the measured effects
live in ``docs/experiments/2026-09-22-text-cut-3826/REPORT.md`` (the display
line), ``docs/experiments/2026-09-23-text-cut-ab-3826/REPORT.md`` (the opening)
and ``docs/experiments/2026-10-06-cost-fbeta-rescore-4582/REPORT.md`` (both,
at the three F-beta presets).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.training.thresholds import gmm as G
from vtscore.training.thresholds import (
    TEXT_SORT_CUT_DEFAULT,
    TEXT_SORT_CUT_RULES,
    TEXT_SORT_SEPARATION_D,
    TEXT_SORT_TAIL_K,
    TextSortCuts,
    ashman_d,
    bulk_location_scale,
    calculate_gmm_threshold,
    converge_score_gmm,
    fit_score_gmm,
    guarded_text_sort_threshold,
    text_sort_acquisition_threshold,
    text_sort_cuts,
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


class TestTheDefaultRule:
    """The guarded line is what a text sort paints by default (#4136); the midpoint is where it samples."""

    def test_the_default_rule_is_the_guarded_line(self):
        assert TEXT_SORT_CUT_DEFAULT == "guarded_tail"
        assert G.TEXT_SORT_CUT_RULE == "guarded_tail"

    @pytest.mark.parametrize("seed", range(5))
    def test_the_default_display_line_is_the_guarded_one(self, seed):
        scores, _ = _shoulder(seed=seed)
        values = scores.tolist()
        assert text_sort_threshold(values) == guarded_text_sort_threshold(values)[0]

    @pytest.mark.parametrize("seed", range(5))
    def test_the_acquisition_cut_is_calculate_gmm_threshold_exactly(self, seed):
        """Bit for bit, under the default rule: the opening's picks are the midpoint arm's by construction."""
        scores, _ = _shoulder(seed=seed)
        values = scores.tolist()
        assert text_sort_cuts(values).acq_threshold == calculate_gmm_threshold(values)
        assert text_sort_acquisition_threshold(values) == calculate_gmm_threshold(values)

    def test_the_opt_out_rule_draws_one_line(self):
        """``gmm_midpoint`` is the pre-#3826 sort: display and acquisition are one number."""
        scores, _ = _shoulder()
        values = scores.tolist()
        cuts = text_sort_cuts(values, rule="gmm_midpoint")
        assert cuts == TextSortCuts(calculate_gmm_threshold(values), calculate_gmm_threshold(values), "midpoint")
        assert text_sort_threshold(values, rule="gmm_midpoint") == calculate_gmm_threshold(values)

    def test_an_explicit_rule_overrides_the_module_default(self, monkeypatch):
        scores, _ = _shoulder()
        values = scores.tolist()
        monkeypatch.setattr(G, "TEXT_SORT_CUT_RULE", "gmm_midpoint")
        assert text_sort_threshold(values) == calculate_gmm_threshold(values)
        assert text_sort_threshold(values, rule="guarded_tail") == guarded_text_sort_threshold(values)[0]

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="unknown text-sort cut rule"):
            text_sort_threshold([0.1, 0.2, 0.3], rule="midpoint_of_vibes")
        with pytest.raises(ValueError, match="unknown text-sort cut rule"):
            text_sort_cuts([0.1, 0.2, 0.3], rule="midpoint_of_vibes")


class TestTheFlagIsReadFromTheEnvironment:
    """``VTSEARCH_TEXT_SORT_CUT`` selects the rule; a typo or no value degrades to the default.

    Tested through the resolver rather than by reloading the module: a reload
    re-creates every module-level object under the other tests in the worker,
    and the package re-export identity test is the first to notice.
    """

    @pytest.mark.parametrize(("value", "rule"), [(" Guarded_Tail ", "guarded_tail"), ("gmm_midpoint", "gmm_midpoint")])
    def test_known_values(self, value, rule):
        assert G.resolve_text_sort_cut_rule(value) == rule

    @pytest.mark.parametrize("value", [None, "", "guarded-tail", "tail"])
    def test_unknown_or_unset_falls_back_to_the_default(self, value):
        assert G.resolve_text_sort_cut_rule(value) == TEXT_SORT_CUT_DEFAULT == "guarded_tail"


class TestTheTailBranch:
    """An unseparated sort is cut at median + k bulk sigmas, with no optimiser in it."""

    def test_a_shoulder_takes_the_tail_branch(self):
        scores, _ = _shoulder()
        fit = fit_score_gmm(G.gmm_fit_array(scores.tolist()))
        assert fit is not None
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
        assert shipped is not None
        converged = converge_score_gmm(arr, shipped)
        assert converged is not None
        cut, _ = guarded_text_sort_threshold(scores.tolist())
        assert cut == converged.midpoint()

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
        start = G._two_means_init(xs)
        assert start is not None
        a, b = converge_score_gmm(arr, start), converge_score_gmm(arr, other)
        assert a is not None and b is not None
        assert a.midpoint() == pytest.approx(b.midpoint(), abs=1e-6)


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


class TestTheTwoCuts:
    """:func:`text_sort_cuts` draws both lines from one fit, and the acquisition half never moves."""

    @pytest.mark.parametrize("rule", TEXT_SORT_CUT_RULES)
    @pytest.mark.parametrize("make", [_shoulder, _separated])
    def test_the_acquisition_cut_is_the_midpoint_under_every_rule(self, make, rule):
        scores, _ = make()
        values = scores.tolist()
        cuts = text_sort_cuts(values, rule=rule)
        assert cuts.acq_threshold == calculate_gmm_threshold(values)
        assert cuts.acq_threshold == text_sort_acquisition_threshold(values)

    @pytest.mark.parametrize("make", [_shoulder, _separated])
    def test_the_display_line_and_its_branch_are_the_guarded_rules(self, make):
        scores, _ = make()
        values = scores.tolist()
        cuts = text_sort_cuts(values, rule="guarded_tail")
        assert (cuts.threshold, cuts.branch) == guarded_text_sort_threshold(values)

    def test_on_a_shoulder_the_two_lines_differ(self):
        """The point of #4136: one number cannot do both jobs on a one-mode sort."""
        scores, labels = _shoulder()
        cuts = text_sort_cuts(scores.tolist(), rule="guarded_tail")
        assert cuts.branch == "tail"
        assert cuts.threshold > cuts.acq_threshold
        # The display line paints about the matches; the acquisition cut sits in the mode.
        assert (scores >= cuts.threshold).sum() <= 2 * labels.sum()
        assert (scores >= cuts.acq_threshold).sum() > 10 * labels.sum()

    @pytest.mark.parametrize("rule", TEXT_SORT_CUT_RULES)
    @pytest.mark.parametrize("scores", [[], [0.3]])
    def test_fewer_than_two_scores(self, scores, rule):
        cuts = text_sort_cuts(scores, rule=rule)
        assert (cuts.threshold, cuts.acq_threshold) == (0.5, 0.5)
        assert cuts.branch == ("fallback" if rule == "guarded_tail" else "midpoint")

    def test_a_constant_sample_cuts_both_lines_at_the_value(self):
        cuts = text_sort_cuts([0.3] * 10, rule="guarded_tail")
        assert cuts.threshold == pytest.approx(0.3)
        assert cuts.acq_threshold == pytest.approx(0.3)
        assert cuts.branch == "gmm"

    def test_the_cuts_are_immutable(self):
        cuts = text_sort_cuts([0.1, 0.2, 0.3, 0.9])
        with pytest.raises(Exception, match="frozen|assign"):
            cuts.threshold = 0.0  # type: ignore[misc]


class TestTheOpeningReadsTheAcquisitionCut:
    """The harness's Bad phase and ``@mid`` sample where the app's Hard select does: the midpoint."""

    @pytest.mark.parametrize("rule", TEXT_SORT_CUT_RULES)
    def test_harness_bad_phase_line_on_a_text_sort(self, monkeypatch, rule):
        from vtscore.eval.al_strategies import _sort_threshold

        scores, _ = _shoulder()
        ranking = dict(enumerate(scores.tolist()))
        monkeypatch.setattr(G, "TEXT_SORT_CUT_RULE", rule)
        assert _sort_threshold(ranking, typed_query=True) == calculate_gmm_threshold(list(ranking.values()))
        # A known-good centroid sort is not a typed query and keeps the midpoint.
        assert _sort_threshold(ranking) == calculate_gmm_threshold(list(ranking.values()))

    def test_the_bad_phase_does_not_read_the_display_line(self, monkeypatch):
        from vtscore.eval.al_strategies import _sort_threshold

        scores, _ = _shoulder()
        ranking = dict(enumerate(scores.tolist()))
        monkeypatch.setattr(G, "TEXT_SORT_CUT_RULE", "guarded_tail")
        assert _sort_threshold(ranking, typed_query=True) != text_sort_threshold(list(ranking.values()))

    @pytest.mark.parametrize("rule", TEXT_SORT_CUT_RULES)
    def test_startup_schedule_mid_is_the_acquisition_cut(self, monkeypatch, rule):
        from vtscore.eval.startup_schedule import parse_startup_schedule, round_cut

        scores, _ = _shoulder()
        (rnd,) = parse_startup_schedule("n1@mid")
        monkeypatch.setattr(G, "TEXT_SORT_CUT_RULE", rule)
        assert round_cut(scores.tolist(), rnd) == text_sort_acquisition_threshold(scores.tolist())

    def test_cosine_sort_active_uses_the_text_rule_only_for_text(self, monkeypatch):
        import vtscore.training.thresholds as T
        from vtscore.training.query_sort import cosine_sort_active, text_sort_active

        from tests_lib.sorting.test_query_sort import _fill_active_medias

        query = _fill_active_medias()
        monkeypatch.setattr(T, "text_sort_cuts", lambda s, beta=None: TextSortCuts(0.123, 0.0456, "tail"))
        monkeypatch.setattr(T, "calculate_gmm_threshold", lambda s: 0.456)
        assert cosine_sort_active(query, role="text")[1] == 0.123
        assert cosine_sort_active(query, role="score")[1] == 0.456
        _results, cuts = text_sort_active(query)
        assert cuts == TextSortCuts(0.123, 0.0456, "tail")

    def test_text_sort_active_rounds_both_lines_to_four_decimals(self, monkeypatch):
        import vtscore.training.thresholds as T
        from vtscore.training.query_sort import text_sort_active

        from tests_lib.sorting.test_query_sort import _fill_active_medias

        query = _fill_active_medias()
        monkeypatch.setattr(T, "text_sort_cuts", lambda s, beta=None: TextSortCuts(0.123456, 0.0456789, "gmm"))
        _results, cuts = text_sort_active(query)
        assert cuts == TextSortCuts(0.1235, 0.0457, "gmm")


class TestTheOpeningIsBlindToTheDisplayRule:
    """The assertion #4136 asked of the A/B harness, run on the harness itself.

    Two openings that differ only in ``VTSEARCH_TEXT_SORT_CUT`` vote the same
    items in the same order, because every pre-detector pick reads the
    acquisition cut and that cut is the midpoint under both rules.  A GRID
    re-run of ``launch_ab_3826.sh`` could only confirm this with sampling
    noise on top; the identity is exact, so it is pinned here instead.
    """

    STEPS = 40
    #: A typed-query sort shaped like the real ones: a few matches as a thin
    #: shoulder on one broad mode.  The guarded rule takes its ``tail`` branch
    #: here, so its line sits far above the midpoint (27 items admitted against
    #: 332).  On a sort where the guarded rule keeps the mixture (its ``gmm``
    #: branch) the two cuts land at the same rank and an opening that wrongly
    #: sampled at the display line would vote the same items - the test could
    #: not fail.
    N_POS, N_NEG = 25, 800

    def _dataset(self):
        from tests_lib.detectors.test_startup_schedule import _seeded_dataset

        return _seeded_dataset(n_pos=self.N_POS, n_neg=self.N_NEG)

    def _picks(self, schedule):
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        medias, seed_scores = self._dataset()
        log: list[dict] = []
        simulate_voting_iterations(
            medias,
            target_category="target",
            seed=3,
            dataset_name="stub",
            max_steps=self.STEPS,
            seed_scores=seed_scores,
            atlas_min_node_size=8,
            startup_schedule=schedule,
            spot_check="off",
            pick_sink=log,
        )
        return [(p["phase"], p["picked_id"], p["picked_label"]) for p in log]

    def test_the_rules_sit_at_different_ranks_on_this_sort(self):
        """Teeth: the two lines must admit different items, or the identity below is vacuous."""
        _medias, seed_scores = self._dataset()
        values = np.asarray(list(seed_scores.values()))
        cuts = text_sort_cuts(list(values), rule="guarded_tail")
        assert cuts.branch == "tail"
        above_display = int((values >= cuts.threshold).sum())
        above_acq = int((values >= cuts.acq_threshold).sum())
        assert above_acq > 5 * above_display, (above_display, above_acq)

    @pytest.mark.parametrize("schedule", [None, "PRODUCTION_STARTUP"])
    def test_the_same_items_are_voted_under_both_rules(self, monkeypatch, schedule):
        from vtscore.eval.startup_schedule import PRODUCTION_STARTUP

        spec = PRODUCTION_STARTUP if schedule else None
        picks: dict[str, list[tuple[str, int, int]]] = {}
        for rule in TEXT_SORT_CUT_RULES:
            monkeypatch.setattr(G, "TEXT_SORT_CUT_RULE", rule)
            picks[rule] = self._picks(spec)
        assert picks["guarded_tail"] == picks["gmm_midpoint"]
        phases = [ph for ph, _, _ in picks["gmm_midpoint"]]
        assert "hard" in phases, "the run never left the opening, so the comparison covers no learned pick"
        assert ("bad" in phases) or ("s1" in phases), "no Bad-phase pick was made"


class TestTheCountLine:
    """#4603: at a balance of beta 1 or below the display line keeps round(c(beta) * n_hat) of the sort.

    ``n_hat`` is the excess over a Gaussian bulk above median + 4 sigma (sigma from the MAD); c is 3/8 at
    beta 1/4 and 1 at beta 1.  Above beta 1 the guarded line stays, and the acquisition cut never moves.
    """

    @staticmethod
    def _sort(n_bulk: int = 20000, n_match: int = 100, seed: int = 0) -> list[float]:
        rng = np.random.default_rng(seed)
        bulk = rng.normal(0.0, 0.02, n_bulk)
        matches = rng.normal(0.14, 0.01, n_match)  # far above median + 4 sigma
        return [float(v) for v in np.concatenate([bulk, matches])]

    @staticmethod
    def _kept(values: list[float], threshold: float) -> int:
        return int((np.asarray(values) >= threshold).sum())

    def test_beta_1_keeps_about_the_matches(self):
        values = self._sort()
        cuts = text_sort_cuts(values, beta=1.0)
        assert cuts.branch == "count"
        assert 95 <= self._kept(values, cuts.threshold) <= 105

    def test_beta_quarter_keeps_three_eighths_of_them(self):
        values = self._sort()
        kept_quarter = self._kept(values, text_sort_cuts(values, beta=0.25).threshold)
        kept_one = self._kept(values, text_sort_cuts(values, beta=1.0).threshold)
        assert kept_quarter == pytest.approx(0.375 * kept_one, abs=2)
        assert G.TEXT_SORT_COUNT_EXPONENT == pytest.approx(math.log(3 / 8) / math.log(1 / 4))

    @pytest.mark.parametrize("beta", [None, 4.0, 1.5])
    def test_above_beta_1_or_without_a_balance_the_guarded_line_stays(self, beta):
        values = self._sort()
        assert text_sort_cuts(values, beta=beta) == text_sort_cuts(values)
        assert text_sort_cuts(values, beta=beta).branch != "count"

    def test_the_acquisition_cut_never_moves_with_the_balance(self):
        values = self._sort()
        acq = {text_sort_cuts(values, beta=b).acq_threshold for b in (None, 0.25, 1.0, 4.0)}
        assert acq == {calculate_gmm_threshold(values)}

    @pytest.mark.parametrize("values", [[0.3] * 500, [0.1 * i for i in range(30)]])
    def test_no_spread_or_too_few_scores_keep_the_guarded_line(self, values):
        assert text_sort_cuts(values, beta=0.25) == text_sort_cuts(values)

    def test_text_sort_threshold_passes_the_balance(self):
        values = self._sort()
        assert text_sort_threshold(values, beta=1.0) == text_sort_cuts(values, beta=1.0).threshold

    def test_text_sort_active_draws_at_the_active_balance(self, monkeypatch):
        import vtscore.state as S
        import vtscore.training.thresholds as T
        from vtscore.training.query_sort import text_sort_active

        from tests_lib.sorting.test_query_sort import _fill_active_medias

        query = _fill_active_medias()
        seen: list[float | None] = []

        def fake_cuts(scores, beta=None):
            seen.append(beta)
            return TextSortCuts(0.5, 0.25, "count")

        monkeypatch.setattr(T, "text_sort_cuts", fake_cuts)
        monkeypatch.setattr(S, "get_beta", lambda: 0.25)
        text_sort_active(query)
        text_sort_active(query, beta=4.0)
        assert seen == [0.25, 4.0]
