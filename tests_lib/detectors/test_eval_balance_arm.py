"""The eval harness's default arm draws the line a live detector draws (#4272, #4413).

The app's reporting line under a balance keeps a set - the mixture's F-beta
argmax under the balance's cap, or what the check's shape makes of the set
its spot check ended on - and the default user has a balance, so the harness's
default arm has to do the same, or every study measures a detector nobody
ships.  The rule itself is shared (``balance_line`` / ``balance_state`` /
``SpotCheck``); what these tests pin is the harness's side of the plumbing:
that the run's spot check cuts its bands off the ranking, answers its picks
from ground truth, casts them as votes and retrains on them, and that the arms
the knob names behave as named.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.eval.autopilot_flow import pick_provenance
from vtscore.eval.voting_iterations import simulate_voting_iterations
from vtscore.training.thresholds import (
    BALANCE_CHECKED,
    BALANCE_STATES,
    BALANCE_UNCHECKED,
    DEFAULT_BETA,
    NO_BALANCE,
    WEAK_CHECK_COOLDOWN,
    WEAK_CHECK_MIN_VOTES,
    WEAK_SEPARATION_D,
    balance_schedule,
)


def _separable(n_per_cat: int = 20, dim: int = 16) -> dict[int, dict]:
    rng = np.random.RandomState(0)
    medias: dict[int, dict] = {}
    for i in range(2 * n_per_cat):
        cat = "alpha" if i < n_per_cat else "beta"
        centre = 1.0 if cat == "alpha" else -1.0
        emb = rng.normal(centre, 0.6, dim).astype(np.float32)
        medias[i + 1] = {"id": i + 1, "embeddings": {"emb": emb}, "category": cat}
    return medias


class TestTheRetiredProvenanceRecord:
    """``pick_provenance`` fed the #4245 evidence filter, which retired with the estimate (#4362)."""

    @pytest.mark.parametrize(
        ("phase", "sort_kind", "select_mode"),
        [("good", "text", "top"), ("bad", "text", "hard"), ("hard", "learned", "hard"), ("new", "learned", "new")],
    )
    def test_it_still_answers_from_its_table_with_a_warning(self, phase, sort_kind, select_mode):
        with pytest.deprecated_call(match="pick_provenance"):
            recorded = pick_provenance(phase)
        assert recorded == {"flow": "autopilot", "phase": phase, "select_mode": select_mode, "sort_kind": sort_kind}

    @pytest.mark.parametrize("phase", [None, "idle", "done", "exhausted"])
    def test_no_labelling_phase_records_nothing(self, phase):
        with pytest.deprecated_call():
            assert pick_provenance(phase) is None


class TestTheArms:
    """The default arm (the app's balance) and the Inclusion arm."""

    def _run(self, **kwargs):
        picks: list[dict] = []
        rows = simulate_voting_iterations(
            _separable(), "alpha", seed=0, max_steps=15, calibrate_count=2, pick_sink=picks, **kwargs
        )
        return rows, picks

    def test_the_default_arm_is_the_apps_balance(self):
        """No knob pinned draws the line where the app does: the balance at DEFAULT_BETA (#4413)."""
        rows, _picks = self._run()
        assert rows
        assert all(r["beta"] == DEFAULT_BETA and "min_precision" not in r for r in rows)
        assert {r["floor_status"] for r in rows} <= set(BALANCE_STATES)

    def test_before_the_check_the_line_keeps_the_unchecked_cap(self, schedule_only):
        """The count line before a check: the top K unvoted, K the balance's cap, capped by the corpus.

        Since #4452 the default arm draws the labels' line; the count line is the forced-shape arm's.
        """
        rows, _picks = self._run(walk_shape="advisory")
        steps = [r for r in rows if r["phase"] != "check"]
        assert steps and all(r["floor_status"] == BALANCE_UNCHECKED for r in steps)
        k = balance_schedule(DEFAULT_BETA).candidate
        # `n_remainder` is the unvoted sim set after this step's vote: the candidate.
        assert all(r["floor_count"] == min(k, r["n_remainder"]) for r in steps)
        assert all(math.isnan(r["range_lo"]) and r["check_labelled"] == -1 and r["check_stale"] == -1 for r in steps)

    def test_before_the_check_the_mixture_can_only_lower_the_count(self):
        """The app's unchecked line: the smaller of the cap and the mixture's F-beta argmax (#4389, #4413).

        The harness hands the votes so far to the same ``fbeta_count`` the app
        anchored on before #4452; the count line is now the forced-shape arm's.
        """
        rows, _picks = self._run(walk_shape="advisory")
        steps = [r for r in rows if r["phase"] != "check"]
        k = balance_schedule(DEFAULT_BETA).candidate
        assert steps and all(1 <= r["floor_count"] <= min(k, r["n_remainder"]) for r in steps)

    def test_the_check_runs_once_the_steps_are_spent_and_its_votes_enter_training(self):
        rows, picks = self._run()
        steps = [r for r in rows if r["phase"] != "check"]
        check_rows = [r for r in rows if r["phase"] == "check"]
        assert check_rows, "the default arm checks the line"
        assert max(r["t"] for r in steps) == 15
        # One row per round, each after that round's picks were cast and the model retrained.
        assert all(r["t"] > 15 for r in check_rows)
        assert all(r["n_good"] + r["n_bad"] == r["t"] for r in check_rows)
        # The check's picks are logged as votes with the app's provenance flow.
        check_picks = [p for p in picks if p["phase"] == "check"]
        assert len(check_picks) == check_rows[-1]["t"] - 15
        assert all(p["t"] > 15 for p in check_picks)
        # The last row carries the result: the set the walk audited, and its range.
        last = check_rows[-1]
        assert last["floor_status"] == BALANCE_CHECKED
        assert 0.0 <= last["range_lo"] <= last["range_hi"] <= 1.0
        assert last["check_labelled"] >= 1 and 0 <= last["check_right"] <= last["check_labelled"]
        assert last["check_stale"] in (0, 1)
        assert last["check_audited"] >= 1

    def test_switching_the_check_off_leaves_the_run_unchecked(self):
        rows, picks = self._run(spot_check="off")
        assert all(r["phase"] != "check" for r in rows)
        assert all(r["floor_status"] == BALANCE_UNCHECKED for r in rows)
        assert all(p["phase"] != "check" for p in picks)

    def test_the_voting_steps_are_byte_identical_with_or_without_the_check(self):
        """The check is seeded after every trajectory draw: switching it on changes nothing before it."""
        with_check, _ = self._run()
        without, _ = self._run(spot_check="off")
        steps = [r for r in with_check if r["phase"] != "check"]
        assert [r["t"] for r in steps] == [r["t"] for r in without]
        # The plain frame carries no raw threshold; these all read it.
        for col in ("report_pool_percentile", "cost", "n_flagged", "acq_threshold", "floor_count"):
            assert [r[col] for r in steps] == [r[col] for r in without], col

    def test_an_unknown_check_knob_is_refused(self):
        with pytest.raises(ValueError, match="spot_check"):
            self._run(spot_check="sometimes")

    def test_the_inclusion_arm_reports_no_balance_and_never_checks(self):
        rows, _picks = self._run(beta=NO_BALANCE)
        assert rows
        assert all(math.isnan(r["beta"]) for r in rows)
        assert all(r["floor_status"] == "" and r["floor_count"] == -1 for r in rows)
        assert all(r["phase"] != "check" for r in rows)

    def test_the_precision_floors_arm_is_gone(self):
        """#4421: the floor's knob went with the floor, loudly."""
        with pytest.raises(TypeError, match="min_precision"):
            self._run(min_precision=0.5)


class TestTheObjectiveColumns:
    """#4584: every row carries the objective at its own beta, so no analyzer re-derives it."""

    def _rows(self, **kwargs):
        return simulate_voting_iterations(_separable(), "alpha", seed=0, max_steps=12, calibrate_count=2, **kwargs)

    @pytest.mark.parametrize(("beta", "preset"), [(None, "fbeta_b1"), (0.25, "fbeta_b025"), (4.0, "fbeta_b4")])
    def test_fbeta_is_the_rows_own_preset_column(self, beta, preset):
        rows = self._rows(**({} if beta is None else {"beta": beta}))
        assert rows
        scored = [r for r in rows if math.isfinite(r["recall"])]
        assert scored
        for r in scored:
            assert r["beta"] == (DEFAULT_BETA if beta is None else beta)
            assert math.isfinite(r["fbeta"]) and r["fbeta"] == r[preset]

    def test_it_is_what_the_rows_rates_say(self):
        """The count form on the row and the rate form an analyzer back-fills with agree."""
        from vtscore.eval.calibration_metrics import fbeta_from_rates  # noqa: PLC0415

        rows = [r for r in self._rows(beta=0.5) if math.isfinite(r["recall"])]
        got = np.array([r["fbeta"] for r in rows])
        want = fbeta_from_rates([r["precision"] for r in rows], [r["recall"] for r in rows], 0.5)
        np.testing.assert_allclose(got, want, atol=2e-5)

    def test_the_inclusion_arm_has_no_objective_but_keeps_the_presets(self):
        rows = [r for r in self._rows(beta=NO_BALANCE) if math.isfinite(r["recall"])]
        assert rows
        assert all(math.isnan(r["fbeta"]) for r in rows)
        assert all(math.isfinite(r["fbeta_b1"]) for r in rows)


class TestTheBalanceArm:
    """#4413: the line drawn at F-beta's beta, and the end-of-run check as the F-beta walk."""

    def _run(self, **kwargs):
        picks: list[dict] = []
        rows = simulate_voting_iterations(
            _separable(), "alpha", seed=0, max_steps=15, calibrate_count=2, pick_sink=picks, **kwargs
        )
        return rows, picks

    def test_the_walks_arms_reach_the_end_of_run_check(self):
        """#4427: picks / tol / fine / guard shape the end-of-run balance walk; off, it is the app's."""
        rows, _ = self._run(beta=1.0, walk_picks=3, walk_tol=0.02, walk_fine=True, walk_guard=0.5)
        check_rows = [r for r in rows if r["phase"] == "check"]
        assert check_rows, "the balance arm checks the line"
        assert check_rows[0]["check_labelled"] <= 3, "three picks a band"
        with pytest.raises(ValueError, match="tol must be"):
            self._run(beta=1.0, walk_tol=-0.1)
        with pytest.raises(ValueError, match="guard must be"):
            self._run(beta=1.0, walk_guard=-1.0)

    def test_the_default_check_is_advisory_at_beta_one_and_the_full_walk_can_be_forced(self):
        """#4427: the arm's check follows the app's shape; at beta 1 the walk audits but the line keeps the
        unchecked rule's count; ``walk_shape="walk"`` is the full walk whose end moves the line."""
        rows, picks = self._run(beta=1.0)
        check_rows = [r for r in rows if r["phase"] == "check"]
        assert check_rows and [p for p in picks if p["phase"] == "check"], "the check ran and voted"
        last = check_rows[-1]
        assert last["floor_status"] == BALANCE_CHECKED and last["check_audited"] >= 1
        assert last["check_audited"] == max(r["check_audited"] for r in check_rows), "the walk's end is on the row"
        full, _ = self._run(beta=1.0, walk_shape="walk")
        last_full = [r for r in full if r["phase"] == "check"][-1]
        assert last_full["floor_count"] == last_full["check_audited"], "the full walk's end is the line"
        with pytest.raises(ValueError, match="walk_shape must be"):
            self._run(beta=1.0, walk_shape="sideways")

    def test_a_trim_walk_ends_at_or_above_its_start(self):
        """#4427's trim shape: the check runs and can only cut the line, never deepen it."""
        rows, picks = self._run(beta=1.0, walk_shape="trim")
        check_rows = [r for r in rows if r["phase"] == "check"]
        assert check_rows and [p for p in picks if p["phase"] == "check"], "the check ran and voted"
        assert check_rows[-1]["floor_count"] <= check_rows[0]["floor_count"], "it never deepened"
        assert check_rows[-1]["floor_count"] == check_rows[-1]["check_audited"], "and the line is its end"

    def test_the_line_is_the_balances_under_its_cap(self):
        rows, _ = self._run(beta=1.0)
        steps = [r for r in rows if r["phase"] not in ("check", "")]
        assert steps and all(r["beta"] == 1.0 for r in steps)
        assert all(r["floor_status"] == BALANCE_UNCHECKED for r in steps)
        # No cap since #4452: the labels' line keeps what clears it, possibly none.
        assert all(0 <= r["floor_count"] <= r["n_remainder"] for r in steps)

    def test_the_check_is_the_f_beta_walk_and_ends_checked(self):
        rows, picks = self._run(beta=1.0)
        check_rows = [r for r in rows if r["phase"] == "check"]
        assert check_rows, "the balance arm checks the line"
        last = check_rows[-1]
        assert last["floor_status"] == BALANCE_CHECKED and last["beta"] == 1.0
        # Advisory at beta 1 (#4427): the audited set is the walk's end; the count is the unchecked rule's,
        # which on this tiny corpus can be every unvoted item left - possibly none.
        assert last["check_audited"] >= 1 and last["floor_count"] >= 0
        assert 0.0 <= last["range_lo"] <= last["range_hi"] <= 1.0
        assert last["check_labelled"] >= 5 and all(p["phase"] != "check" or p["t"] > 15 for p in picks)

    def test_a_recall_leaning_balance_keeps_at_least_as_much(self):
        lo, _ = self._run(beta=0.5)
        hi, _ = self._run(beta=2.0)
        at = lambda rows: [r["floor_count"] for r in rows if r["phase"] not in ("check", "")]  # noqa: E731
        assert sum(at(hi)) >= sum(at(lo))

    def test_a_balance_outside_the_range_kills_the_cell_before_it_runs(self):
        with pytest.raises(ValueError, match="beta must be in"):
            self._run(beta=9.0)


class TestTheWeakSeparationPrompt:
    """#4496: ``spot_check="weak"`` - the app prompts a check when the labels separate weakly."""

    def _run(self, n_per_cat: int = 20, max_steps: int = 15, **kwargs):
        picks: list[dict] = []
        rows = simulate_voting_iterations(
            _separable(n_per_cat), "alpha", seed=0, max_steps=max_steps, calibrate_count=2, pick_sink=picks,
            beta=1.0, **kwargs,
        )  # fmt: skip
        return rows, picks

    @staticmethod
    def _prompts(picks: list[dict]) -> list[list[int]]:
        """The prompted checks, as runs of prompted picks in the log's order (a round's picks share one ``t``)."""
        runs: list[list[int]] = []
        previous = None
        for p in picks:
            if p["phase"] == "prompt":
                if previous != "prompt":
                    runs.append([])
                runs[-1].append(p["t"])
            previous = p["phase"]
        return runs

    def test_a_weak_session_checks_mid_run_and_the_picks_are_clicks(self):
        rows, picks = self._run(
            n_per_cat=40, max_steps=25, spot_check="weak", weak_separation=math.inf, weak_min_t=6, weak_phase="any"
        )
        prompts = self._prompts(picks)
        assert len(prompts) == 1, "prompted once"
        assert prompts[0][0] > 6, "not before the first eligible click"
        prompt_rows = [r for r in rows if r["phase"] == "prompt"]
        assert prompt_rows and all(r["n_good"] + r["n_bad"] == r["t"] for r in prompt_rows)
        # The prompted picks spend the voting budget: ordinary clicks resume after them and stop at the same count.
        ordinary = [r for r in rows if r["phase"] not in ("check", "prompt")]
        assert any(r["t"] > prompts[0][-1] for r in ordinary), "voting resumed"
        assert max(r["t"] for r in ordinary) == 25
        # And the end-of-run check still runs after them.
        end = [r for r in rows if r["phase"] == "check"]
        assert end and all(r["t"] > 25 for r in end)

    def test_a_session_that_never_separates_weakly_is_the_end_check_run(self):
        """No prompt draws nothing from the RNG: the run is byte-identical to ``spot_check="end"``."""
        shipped, shipped_picks = self._run(spot_check="end")
        weak, weak_picks = self._run(spot_check="weak", weak_separation=-math.inf)
        assert [p["picked_id"] for p in weak_picks] == [p["picked_id"] for p in shipped_picks]
        assert [(r["t"], r["phase"], r["floor_count"]) for r in weak] == [
            (r["t"], r["phase"], r["floor_count"]) for r in shipped
        ]

    def test_a_learned_only_prompt_waits_for_the_flow_to_leave_its_opening(self):
        """#4496: the app trains no detector on the text-sort opening, so ``weak_phase="learned"`` waits."""
        _rows, any_picks = self._run(
            n_per_cat=40, max_steps=40, spot_check="weak", weak_separation=math.inf, weak_phase="any"
        )
        _rows, picks = self._run(
            n_per_cat=40, max_steps=40, spot_check="weak", weak_separation=math.inf, weak_phase="learned"
        )
        before = [
            picks[i - 1]["phase"]
            for i, p in enumerate(picks)
            if p["phase"] == "prompt" and i and picks[i - 1]["phase"] != "prompt"
        ]
        assert all(ph not in ("good", "bad", "more") for ph in before), before
        first_any = next((p["t"] for p in any_picks if p["phase"] == "prompt"), None)
        first_learned = next((p["t"] for p in picks if p["phase"] == "prompt"), None)
        assert first_any is not None
        assert first_learned is None or first_learned >= first_any, "never earlier than anywhere-prompting"
        with pytest.raises(ValueError, match="weak_phase"):
            self._run(spot_check="weak", weak_phase="sometimes")

    def test_a_prompted_check_spends_no_more_than_the_budget(self):
        """Its opening bands must fit the votes left, and a deeper band that would overrun ends it: clicks end at max_steps."""
        rows, _picks = self._run(
            n_per_cat=40, max_steps=25, spot_check="weak", weak_separation=math.inf, weak_min_t=6, weak_phase="any"
        )
        clicks = [r for r in rows if r["phase"] != "check"]
        assert any(r["phase"] == "prompt" for r in clicks)
        assert max(r["t"] for r in clicks) == 25, "the prompted check stayed inside the budget, and voting resumed"
        _rows, late = self._run(
            n_per_cat=40, max_steps=25, spot_check="weak", weak_separation=math.inf, weak_min_t=20, weak_phase="any"
        )
        assert not [p for p in late if p["phase"] == "prompt"], "no room left at click 20 for a check's opening bands"

    def test_the_default_arm_checks_where_the_apps_rule_says(self):
        """Since the owner's ruling (2026-10-05) the default is the app's: ``weak`` at the app's constants."""
        import inspect

        params = inspect.signature(simulate_voting_iterations).parameters
        assert params["spot_check"].default == "weak"
        assert params["weak_separation"].default == WEAK_SEPARATION_D
        assert params["weak_min_t"].default == WEAK_CHECK_MIN_VOTES
        assert params["weak_repeat"].default == WEAK_CHECK_COOLDOWN
        assert params["weak_phase"].default == "learned", "the app trains no detector on Autopilot's opening"

    def test_the_prompt_returns_after_its_cooldown_while_separation_stays_weak(self):
        once, once_picks = self._run(
            n_per_cat=100, max_steps=90, spot_check="weak", weak_separation=math.inf, weak_repeat=0, weak_phase="any"
        )
        again, again_picks = self._run(
            n_per_cat=100, max_steps=90, spot_check="weak", weak_separation=math.inf, weak_repeat=5, weak_phase="any"
        )
        assert len(self._prompts(once_picks)) == 1
        prompts = self._prompts(again_picks)
        assert len(prompts) >= 2, "the prompt came back"
        assert all(b[0] - a[-1] >= 5 for a, b in zip(prompts, prompts[1:])), "after at least the cooldown"


class TestTheBalanceAwareAcquisitionArm:
    """#4409, #4413: the acquisition cut at a share of the depth of the mixture's F-beta argmax."""

    def _run(self, beta, **kwargs):
        picks: list[dict] = []
        rows = simulate_voting_iterations(
            _separable(), "alpha", seed=0, max_steps=15, calibrate_count=2, pick_sink=picks,
            beta=beta, spot_check="off", **kwargs,
        )  # fmt: skip
        return rows, picks

    def test_it_is_a_live_arm_not_the_shipped_cut(self):
        shipped, picks_shipped = self._run(1.0)
        aware, picks_aware = self._run(1.0, acq_inclusion_offset=0, acq_p_crossing=0.5)
        # The cut differs; on a 40-item fixture the opening takes every pick, so
        # whether the picks follow is for the bench run to show.
        assert [r["acq_threshold"] for r in shipped] != [r["acq_threshold"] for r in aware]
        assert len(picks_shipped) == len(picks_aware)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"acq_p_crossing": 1.0}, "acq_inclusion_offset=0"),
            ({"acq_inclusion_offset": 0, "acq_p_crossing": 0.0}, "must be > 0"),
            ({"acq_inclusion_offset": 0, "acq_rank_percentile": 0.9, "acq_p_crossing": 1.0}, "acq_inclusion_offset=0"),
        ],
    )
    def test_a_malformed_arm_dies_before_it_runs(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self._run(1.0, **kwargs)

    def test_it_needs_a_balance(self):
        with pytest.raises(ValueError, match="needs a balance"):
            self._run(NO_BALANCE, acq_inclusion_offset=0, acq_p_crossing=1.0)

    def test_the_default_arm_under_a_balance_is_the_shipped_factor_and_off_is_the_offset_cut(self):
        """#4409 / #4427: a balance arm with no acq knob is the shipped cut, line - 4 again since the revert; a
        number is the argmax arm and "off" the offset whatever ships."""
        from vtscore.eval.voting_iterations import resolve_acquisition_factor
        from vtscore.training.thresholds import ACQUISITION_ARGMAX_FACTOR

        assert resolve_acquisition_factor(None, 1.0) is ACQUISITION_ARGMAX_FACTOR is None
        assert resolve_acquisition_factor(None, None) is None and resolve_acquisition_factor("off", 1.0) is None
        assert resolve_acquisition_factor(0.25, 2.0) == 0.25
        shipped, _ = self._run(1.0)
        offset, _ = self._run(1.0, acq_p_crossing="off")
        assert [r["acq_threshold"] for r in shipped] == [r["acq_threshold"] for r in offset]
        argmax, _ = self._run(1.0, acq_inclusion_offset=0, acq_p_crossing=0.5)
        assert [r["acq_threshold"] for r in argmax] != [r["acq_threshold"] for r in offset]
        with pytest.raises(ValueError, match="must be > 0"):
            self._run(1.0, acq_inclusion_offset=0, acq_p_crossing="x")

    def test_under_a_balance_it_samples_at_the_f_beta_argmax(self):
        """#4413: a precision-leaning balance samples higher than a recall-leaning one."""
        hi, _ = self._run(0.5, acq_inclusion_offset=0, acq_p_crossing=1.0)
        lo, _ = self._run(2.0, acq_inclusion_offset=0, acq_p_crossing=1.0)
        a_hi = [r["acq_threshold"] for r in hi if r["t"] >= 5 and r["phase"] != "check"]
        a_lo = [r["acq_threshold"] for r in lo if r["t"] >= 5 and r["phase"] != "check"]
        assert a_hi and len(a_hi) == len(a_lo)
        assert sum(a_hi) / len(a_hi) >= sum(a_lo) / len(a_lo)
