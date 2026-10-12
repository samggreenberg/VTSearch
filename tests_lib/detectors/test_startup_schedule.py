"""Parameterised Autopilot openings (issue #3267).

Two things need pinning, and they pull against each other:

* the **grammar and the state machine**, so an arm's spec means exactly what its
  launch script says it means - a silently misparsed round would make a study
  measure an opening nobody wrote; and
* the **default**, so that :data:`~vtscore.eval.startup_schedule.
  PRODUCTION_STARTUP` is not merely *similar* to the app's own opening but
  reproduces it click for click.  That is what licenses reading a study's
  control arm as "what users get today" (see "The Eval Default Arm IS the App"
  in ``docs/EVAL.md``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.eval.autopilot_flow import (
    BAD_TARGET,
    GOOD_TARGET,
    MORE_TARGET,
    AutopilotFlow,
    app_has_detector,
)
from vtscore.eval.startup_schedule import (
    PRODUCTION_STARTUP,
    StartupRound,
    StartupState,
    is_startup_phase,
    parse_startup_schedule,
    round_cut,
)
from vtscore.eval.voting_columns import PICK_COLUMNS
from vtscore.eval.voting_iterations import simulate_voting_iterations

DIM = 24


def _seeded_dataset(n_pos=70, n_neg=200, seed=0):
    """A single-vector dataset with a text-sortable query.

    Positives sit near a planted centre and the query points at it, so the seed
    sort is genuinely informative - which is the situation every schedule round
    is defined against.  Returns ``(medias, seed_scores)``.
    """
    rng = np.random.default_rng(seed)
    centre = rng.standard_normal(DIM).astype(np.float32)
    medias: dict[int, dict] = {}
    for i in range(n_pos + n_neg):
        positive = i < n_pos
        vec = centre * (1.2 if positive else 0.0) + rng.standard_normal(DIM).astype(np.float32) * 0.9
        medias[i] = {
            "id": i,
            "category": "target" if positive else "other",
            "embedding": vec,
            "embeddings": {"stub": vec},
        }
    query = centre + rng.standard_normal(DIM).astype(np.float32) * 0.3

    def cos(v):
        return float(v @ query / (np.linalg.norm(v) * np.linalg.norm(query) + 1e-9))

    return medias, {i: cos(m["embedding"]) for i, m in medias.items()}


def _run(schedule, *, max_steps=24, seed=3, n_pos=70):
    medias, seed_scores = _seeded_dataset(n_pos=n_pos)
    picks: list[dict] = []
    rows = simulate_voting_iterations(
        medias,
        target_category="target",
        seed=seed,
        dataset_name="stub",
        max_steps=max_steps,
        seed_scores=seed_scores,
        atlas_min_node_size=8,
        startup_schedule=schedule,
        # The opening is the subject here; the floor's spot check after the
        # voting steps (#4272) would only add picks past max_steps.
        spot_check="off",
        pick_sink=picks,
    )
    return rows, picks


# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------


class TestParsing:
    def test_production_spec_parses_to_the_apps_own_targets(self):
        rounds = parse_startup_schedule(PRODUCTION_STARTUP)
        # Since #4740 the photo opening has no More walk (MORE_TARGET 0).
        assert MORE_TARGET == 0
        assert rounds == (
            StartupRound(stop="good", n=GOOD_TARGET, cut="top"),
            StartupRound(stop="bad", n=BAD_TARGET, cut="mid"),
        )

    @pytest.mark.parametrize(
        "spec",
        ["g3@top", "b4@mid", "n8@k-3", "n8@k0", "n8@k2", "n6@q0.05", "n6@q0.5", "g20+dry1/8@top", "n30+dry2/10@q0.05"],
    )
    def test_round_trips_through_its_spec(self, spec):
        (rnd,) = parse_startup_schedule(spec)
        assert rnd.spec() == spec

    @pytest.mark.parametrize(
        "spec",
        [
            "",
            "g3",
            "g3@",
            "@top",
            "x3@top",
            "g0@top",
            "g3@k",
            "g3@warm",
            "n3@q",
            "g-1@top",
            "n3@k1.5",
            "g20+dry1@top",
            "g20+dry/8@top",
            "g20dry1/8@top",
            "g20@top+dry1/8",
            "g20+dry1/8/2@top",
        ],
    )
    def test_junk_is_rejected_not_guessed_at(self, spec):
        """A misparsed arm measures an opening nobody wrote."""
        with pytest.raises(ValueError):
            parse_startup_schedule(spec)

    def test_whitespace_and_trailing_commas_are_tolerated(self):
        assert parse_startup_schedule(" g3@top , b4@mid ,") == parse_startup_schedule(PRODUCTION_STARTUP)

    def test_dry_stop_parses_to_its_count_and_window(self):
        assert parse_startup_schedule("g20+dry1/8@top,b4@mid") == (
            StartupRound(stop="good", n=20, cut="top", dry_goods=1, dry_window=8),
            StartupRound(stop="bad", n=BAD_TARGET, cut="mid"),
        )

    def test_a_bad_round_refuses_a_dry_stop(self):
        """The stop counts goods, so a round hunting negatives would call itself
        dry on exactly the picks it wants."""
        with pytest.raises(ValueError, match="bad round"):
            parse_startup_schedule("g3@top,b4+dry1/8@mid")

    @pytest.mark.parametrize("spec", ["g20+dry0/8@top", "g20+dry9/8@top", "g20+dry1/0@top"])
    def test_a_dry_stop_that_could_never_fire_or_always_fires_is_rejected(self, spec):
        with pytest.raises(ValueError, match="0 < m <= w"):
            parse_startup_schedule(spec)


# ---------------------------------------------------------------------------
# Where a round cuts
# ---------------------------------------------------------------------------


class TestRoundCut:
    def setup_method(self):
        _, self.scores = _seeded_dataset(seed=1)
        self.values = list(self.scores.values())

    def _cut(self, spec):
        (rnd,) = parse_startup_schedule(f"n1@{spec}")
        return round_cut(self.values, rnd)

    def test_top_is_above_every_score(self):
        """Not a sentinel: with the cut above the sort, rank 0 is the first row
        at-or-below it, which is exactly what the Good phase's `top` select does."""
        assert self._cut("top") == math.inf

    def test_mid_is_the_shipped_cosine_sort_cut(self):
        from vtscore.training.thresholds import calculate_gmm_threshold

        assert self._cut("mid") == pytest.approx(calculate_gmm_threshold(self.values))

    def test_more_negative_inclusion_raises_the_cut(self):
        """The direction the whole study rests on: a negative inclusion prices a
        false alarm higher, raises the cut, and moves the pick *up* the ranking."""
        cuts = [self._cut(f"k{k}") for k in (4, 2, 0, -2, -4, -6)]
        assert cuts == sorted(cuts), cuts

    def test_quantile_names_a_rank_position_directly(self):
        deep, shallow = self._cut("q0.4"), self._cut("q0.05")
        assert shallow > deep
        assert np.mean([s >= shallow for s in self.values]) == pytest.approx(0.05, abs=0.02)

    def test_unfittable_distribution_falls_back_to_the_shipped_cut(self):
        from vtscore.training.thresholds import calculate_gmm_threshold

        (rnd,) = parse_startup_schedule("n1@k-3")
        flat = [0.5]
        assert round_cut(flat, rnd) == pytest.approx(calculate_gmm_threshold(flat))


# ---------------------------------------------------------------------------
# The round state machine
# ---------------------------------------------------------------------------


class TestStartupState:
    def test_good_and_bad_stops_read_global_counts(self):
        """The app's Good phase ends on the third *positive*, however many
        negatives the top of the sort happened to hand back on the way."""
        st = StartupState(parse_startup_schedule("g3@top,b4@mid"))
        st.advance(good_count=2, bad_count=9, remaining_unlabeled=100)
        assert st.index == 0
        st.advance(good_count=3, bad_count=9, remaining_unlabeled=100)
        assert st.done  # the bad target was already met by those nine

    def test_click_stops_count_only_this_rounds_clicks(self):
        st = StartupState(parse_startup_schedule("n2@top,n2@mid"))
        for _ in range(2):
            st.on_click()
            st.advance(good_count=1, bad_count=1, remaining_unlabeled=100)
        assert st.index == 1
        assert st.clicks_in_round == 0

    def test_stops_are_capped_by_what_the_pool_can_supply(self):
        """A target the collection cannot meet must not strand the trajectory."""
        st = StartupState(parse_startup_schedule("g9@top,b9@mid"))
        st.advance(good_count=2, bad_count=1, remaining_unlabeled=0)
        assert st.done

    def test_schedule_will_not_finish_with_one_class_empty(self):
        """Handing a learned sort a one-class labelset leaves the selector
        picking at random, which would make the arm uninterpretable."""
        st = StartupState(parse_startup_schedule("n1@top"))
        st.on_click()
        st.advance(good_count=4, bad_count=0, remaining_unlabeled=50)
        assert not st.done
        st.on_click()
        assert st.extended_clicks == 1
        st.advance(good_count=4, bad_count=1, remaining_unlabeled=49)
        assert st.done

    def test_an_exhausted_pool_ends_the_schedule_even_one_class_short(self):
        st = StartupState(parse_startup_schedule("n1@top"))
        st.on_click()
        st.advance(good_count=4, bad_count=0, remaining_unlabeled=0)
        assert st.done

    def test_phase_names(self):
        st = StartupState(parse_startup_schedule("n1@top,n1@mid"))
        assert st.phase_name() == "s0"
        st.on_click()
        st.advance(1, 1, 10)
        assert st.phase_name() == "s1"
        assert is_startup_phase("s0") and is_startup_phase("s12")
        assert not is_startup_phase("hard") and not is_startup_phase("s") and not is_startup_phase("")


def _vote(st, outcomes, *, good=0, bad=0, remaining=1000):
    """Cast *outcomes* (a string of ``G``/``B``) at *st* the way the harness
    does, one click and one advance per vote; returns the round index after
    each vote."""
    after = []
    for o in outcomes:
        g = o == "G"
        good, bad, remaining = good + g, bad + (not g), remaining - 1
        st.on_click(g)
        st.advance(good_count=good, bad_count=bad, remaining_unlabeled=remaining)
        after.append(st.index)
    return after


class TestDryStop:
    """#4222's adaptive stop: a round ends on its target *or* once its last
    ``w`` picks held fewer than ``m`` goods, whichever comes first.

    The follow-on rounds here ask for more bads than the walk casts, so the
    index lands on them rather than skipping past a target already met.
    """

    def test_the_walk_ends_when_its_window_runs_dry(self):
        st = StartupState(parse_startup_schedule("g20+dry1/4@top,b20@mid"))
        assert _vote(st, "GBBBB") == [0, 0, 0, 0, 1]

    def test_exactly_m_goods_in_the_window_is_not_dry(self):
        """ "Fewer than m", pinned at the boundary: 2 of the last 4 keeps a
        ``+dry2/4`` walk going, 1 of the last 4 ends it."""
        st = StartupState(parse_startup_schedule("g20+dry2/4@top,b4@mid"))
        assert _vote(st, "GBGBB") == [0, 0, 0, 0, 1]

    def test_the_window_slides_rather_than_counting_the_whole_round(self):
        """A round-long count would never run dry after its first good; the
        stop is about the *recent* rate."""
        st = StartupState(parse_startup_schedule("g20+dry1/3@top,b20@mid"))
        assert _vote(st, "BBGBBB") == [0, 0, 0, 0, 0, 1]

    def test_a_round_cannot_run_dry_before_its_window_fills(self):
        st = StartupState(parse_startup_schedule("g20+dry1/8@top,b4@mid"))
        assert _vote(st, "B" * 7) == [0] * 7
        assert not st.ran_dry()
        assert _vote(st, "B", bad=7, remaining=993) == [1]

    def test_the_window_counts_only_this_rounds_picks(self):
        """Round 0's bads are not round 1's evidence: carried over, they would
        call round 1 dry on its first click."""
        st = StartupState(parse_startup_schedule("n3@top,g20+dry1/3@top,b20@mid"))
        assert _vote(st, "GBB") == [0, 0, 1]
        assert _vote(st, "BBB", good=1, bad=2, remaining=997) == [1, 1, 2]

    def test_the_target_still_ends_the_round(self):
        st = StartupState(parse_startup_schedule("g3+dry1/8@top,b4@mid"))
        assert _vote(st, "GGG") == [0, 0, 1]

    def test_a_click_round_can_run_dry_too(self):
        st = StartupState(parse_startup_schedule("n30+dry1/4@q0.05,b20@mid"))
        assert _vote(st, "GBBBB") == [0, 0, 0, 0, 1]

    def test_the_quorum_hold_still_applies(self):
        """A walk that runs dry before its first good is still one class short;
        the schedule holds on its last round until a good arrives, as it would
        for any other stop."""
        st = StartupState(parse_startup_schedule("g20+dry1/4@top"))
        _vote(st, "BBBB")
        assert st.ran_dry() and st.held_for_quorum and not st.done
        _vote(st, "B", bad=4, remaining=996)
        assert st.extended_clicks == 1 and not st.done
        _vote(st, "G", bad=5, remaining=995)
        assert st.done

    def test_a_dry_round_refuses_a_click_without_its_outcome(self):
        """Counting nothing, the stop would never fire and the arm would run as
        the plain round its spec says it is not."""
        st = StartupState(parse_startup_schedule("g20+dry1/4@top"))
        with pytest.raises(ValueError, match="outcome"):
            st.on_click()

    def test_a_plain_round_still_needs_no_outcome(self):
        st = StartupState(parse_startup_schedule("n1@top"))
        st.on_click()
        st.advance(good_count=1, bad_count=1, remaining_unlabeled=10)
        assert st.done


class TestFlowIntegration:
    def test_no_detector_is_on_screen_during_a_round(self):
        """A round is on the seed sort by construction, whatever the vote count."""
        assert not app_has_detector("s0")
        assert not app_has_detector("s3")

    def test_the_apps_machine_resumes_once_the_schedule_is_spent(self):
        flow = AutopilotFlow(startup=StartupState(parse_startup_schedule("n1@top")))
        assert flow.phase == "s0"
        flow.update(good_count=2, bad_count=2, remaining_unlabeled=50, span=None)
        assert flow.phase == "hard"

    def test_the_flow_hands_the_vote_to_a_dry_round(self):
        flow = AutopilotFlow(startup=StartupState(parse_startup_schedule("g20+dry1/2@top")))
        assert flow.update(1, 0, remaining_unlabeled=50, span=None, last_vote_good=True) == "s0"
        assert flow.update(1, 1, remaining_unlabeled=49, span=None, last_vote_good=False) == "s0"
        assert flow.update(1, 2, remaining_unlabeled=48, span=None, last_vote_good=False) == "hard"

    def test_a_dry_round_rejects_an_update_without_the_vote(self):
        flow = AutopilotFlow(startup=StartupState(parse_startup_schedule("g20+dry1/2@top")))
        with pytest.raises(ValueError, match="outcome"):
            flow.update(1, 0, remaining_unlabeled=50, span=None)

    def test_without_a_schedule_the_flow_is_untouched(self):
        flow = AutopilotFlow()
        assert flow.phase == "good"
        assert flow.update(good_count=GOOD_TARGET, bad_count=0, remaining_unlabeled=50, span=None) == "bad"


# ---------------------------------------------------------------------------
# The default arm IS the app
# ---------------------------------------------------------------------------


class TestProductionScheduleIsTheDefault:
    #: Long enough to hand over and run Hard for a while, so the comparison
    #: covers the whole opening and the first learned picks.
    STEPS = 60

    def test_it_reproduces_the_default_opening_click_for_click(self):
        base_rows, base_picks = _run(None, max_steps=self.STEPS)
        prod_rows, prod_picks = _run(PRODUCTION_STARTUP, max_steps=self.STEPS)
        assert [p["picked_id"] for p in prod_picks] == [p["picked_id"] for p in base_picks]
        assert [r["cost"] for r in prod_rows] == [r["cost"] for r in base_rows]
        assert any(p["phase"] == "hard" for p in base_picks), "the run never left the opening"

    def test_it_reproduces_a_sparse_pool_click_for_click(self):
        """A sparse pool, where the Bad round meets its target before the Good one would."""
        base_rows, base_picks = _run(None, max_steps=self.STEPS, n_pos=8)
        prod_rows, prod_picks = _run(PRODUCTION_STARTUP, max_steps=self.STEPS, n_pos=8)
        assert "hard" in [p["phase"] for p in base_picks], "the run never left the opening"
        assert [p["picked_id"] for p in prod_picks] == [p["picked_id"] for p in base_picks]
        assert [r["cost"] for r in prod_rows] == [r["cost"] for r in base_rows]

    def test_only_the_phase_labels_differ(self):
        _, base_picks = _run(None, max_steps=self.STEPS)
        _, prod_picks = _run(PRODUCTION_STARTUP, max_steps=self.STEPS)
        # No More walk on photos since #4740: Good, Bad, then Hard.
        assert not any(b["phase"] == "more" for b in base_picks)
        assert [p["phase"] for p in base_picks][:GOOD_TARGET] == ["good"] * GOOD_TARGET
        assert [p["phase"] for p in prod_picks][:GOOD_TARGET] == ["s0"] * GOOD_TARGET
        bad = [(b["phase"], p["phase"]) for b, p in zip(base_picks, prod_picks) if b["phase"] == "bad"]
        assert bad and all(p == "s1" for _, p in bad)
        # And both hand over to the same learned phase at the same click.
        tail = [(b["phase"], p["phase"]) for b, p in zip(base_picks, prod_picks) if b["phase"] == "hard"]
        assert tail and all(p == "hard" for _, p in tail)

    def test_a_default_run_leaves_the_column_blank(self):
        rows, picks = _run(None)
        assert {r["startup_schedule"] for r in rows} == {""}
        assert {p["startup_round"] for p in picks} == {-1}


# ---------------------------------------------------------------------------
# What the study reads
# ---------------------------------------------------------------------------


class TestPickLog:
    def test_records_every_click_including_the_untrainable_opening(self):
        """The main frame starts at the first Good, the first click Test gives a
        detector at (#4643), so the clicks before it - inside the opening, the
        whole subject of #3267 - are exactly what it does not record."""
        rows, picks = _run("n6@q0.02,n8@q0.25")
        assert len(picks) == 24
        assert [p["t"] for p in picks] == list(range(1, 25))
        first_good = min(p["t"] for p in picks if p["picked_label"] == 1)
        assert [r["t"] for r in rows] == list(range(first_good, 25))

    def test_every_declared_column_is_present(self):
        _, picks = _run("n6@q0.02,n8@q0.25")
        for row in picks:
            assert set(row) == set(PICK_COLUMNS)

    def test_carries_where_on_the_seed_sort_each_click_came_from(self):
        _, picks = _run("n6@q0.02,n8@q0.25")
        opening = [p for p in picks if p["startup_round"] == 0]
        assert opening
        assert all(0.0 <= p["picked_seed_percentile"] <= 1.0 for p in opening)
        # Round 0 cuts at the 2nd percentile, so its picks come off the top.
        assert max(p["picked_seed_percentile"] for p in opening) < 0.2
        assert all(p["startup_cut_percentile"] == pytest.approx(0.02, abs=0.01) for p in opening)

    def test_the_arm_is_named_on_every_row(self):
        rows, picks = _run("n6@q0.02,n8@q0.25")
        assert {r["startup_schedule"] for r in rows} == {"n6@q0.02,n8@q0.25"}
        assert {p["startup_schedule"] for p in picks} == {"n6@q0.02,n8@q0.25"}


class TestTheLeverMoves:
    def test_a_shallower_opening_mines_more_positives(self):
        """The mechanism, stated as the study will read it: sampling nearer the
        top of the seed sort returns more Goods per click."""
        _, shallow = _run("n10@q0.03,n6@q0.4")
        _, deep = _run("n10@q0.35,n6@q0.4")
        assert sum(p["picked_label"] for p in shallow[:10]) > sum(p["picked_label"] for p in deep[:10])


class TestDryStopInTheHarness:
    def test_a_dry_stop_that_never_fires_changes_nothing(self):
        """The three goods arrive long before a 50-pick window fills, so this arm
        *is* the plain two-round opening, click for click."""
        _, prod = _run("g3@top,b4@mid")
        _, dry = _run("g3+dry1/50@top,b4@mid")
        assert [p["picked_id"] for p in dry] == [p["picked_id"] for p in prod]

    def test_the_walk_hands_over_at_its_first_dry_window(self):
        """Against the same walk with no dry stop: identical picks down the top
        of the sort, then a handover at the first 3 bads in a row, well short of
        the 60 goods the plain round would still be walking for."""
        _, plain = _run("g60@top,b4@mid", max_steps=42)
        _, dry = _run("g60+dry1/3@top,b4@mid", max_steps=42)
        walk = [p for p in dry if p["startup_round"] == 0]
        labels = "".join(str(p["picked_label"]) for p in walk)
        assert labels.endswith("000") and "000" not in labels[:-1]
        assert [p["picked_id"] for p in walk] == [p["picked_id"] for p in plain[: len(walk)]]
        assert all(p["startup_round"] == 0 for p in plain)
        handover = dry[len(walk)]
        assert handover["phase"] != "s0"
        assert walk[-1]["n_good"] < 60


class TestGuards:
    def test_a_schedule_needs_a_seed_sort(self):
        medias, _ = _seeded_dataset()
        with pytest.raises(ValueError, match="seed_scores"):
            simulate_voting_iterations(medias, target_category="target", seed=1, max_steps=5, startup_schedule="n4@top")

    def test_a_schedule_needs_the_apps_phase_machine(self):
        medias, seed_scores = _seeded_dataset()
        with pytest.raises(ValueError, match="autopilot_fidelity"):
            simulate_voting_iterations(
                medias,
                target_category="target",
                seed=1,
                max_steps=5,
                seed_scores=seed_scores,
                autopilot_fidelity=False,
                startup_schedule="n4@top",
            )
