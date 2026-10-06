"""The Test arm (#4523): Test mode's autopilot on the withheld half, read against the truth.

Pinned here: the arm runs the shipped phase machine to Done on a planted
ranking and reports the truth beside every range; its inputs are Find's own
line on the withheld half with the model's posteriors aligned to the ranking;
the verdict rule reads as documented; the harness emits one row per session
under a balance and none under the floor arm, without touching the
trajectory; and a replay of the saved snapshot reproduces the harness's row,
which is what lets the study price budgets without retraining.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vtscore.eval.line_test_arm import (
    LEAN_GAIN,
    LINE_FALLBACK,
    LINE_LABELS,
    LINE_NONE,
    RETRAIN_BAR,
    VERDICT_LEAN,
    VERDICT_NOTHING,
    VERDICT_RETRAIN,
    VERDICT_SHIP,
    best_cut,
    line_test_inputs,
    line_test_row,
    oracle_verdict,
    read_verdict,
    row_from_snapshot,
    simulate_line_test,
    truth_at,
)
from vtscore.eval.voting_columns import LINE_TEST_COLUMNS
from vtscore.eval.voting_iterations import simulate_voting_iterations
from vtscore.training.thresholds import (
    DEFAULT_BUDGETS,
    DEFAULT_MIN_PRECISION,
    PHASE_DONE,
    PHASE_NOTHING,
    STOP_REASONS,
    LineBudgets,
    band_edges,
)
from vtscore.training.thresholds.labels_line import ClassScoreModel, LabelsLine

IDENT = ("seed", "dataset", "category", "calibration_seed", "style")


def _planted(n: int = 1200, line: int = 64, above_rate: float = 0.7, below: int = 6, seed: int = 0):
    """A ranking of *n* whose line keeps the top *line*: ``(truth, posteriors)``, the model calibrated per band."""
    rng = np.random.default_rng(seed)
    truth = np.zeros(n, dtype=bool)
    post = np.zeros(n)
    for lo, hi in zip(band_edges(line), band_edges(line)[1:]):
        count = int(round(above_rate * (hi - lo)))
        truth[rng.choice(np.arange(lo, hi), size=count, replace=False)] = True
        post[lo:hi] = count / (hi - lo)
    tail = rng.choice(np.arange(line, n), size=below, replace=False)
    truth[tail] = True
    post[line:] = below / (n - line)
    return truth, post


class TestSimulateLineTest:
    def test_runs_to_done_and_reports_the_truth_beside_the_ranges(self):
        truth, post = _planted()
        out = simulate_line_test(truth, 64, 1.0, posteriors=post, seed=1)
        assert out["phase"] == PHASE_DONE
        assert out["matches_stop"] in STOP_REASONS and out["misses_stop"] in STOP_REASONS
        assert out["picks_above"] + out["picks_below"] == out["picks_total"] > 0
        assert out["picks_above"] <= DEFAULT_BUDGETS.matches_picks + DEFAULT_BUDGETS.picks_per_round
        p, r, f = truth_at(truth, 64, 1.0)
        assert out["precision_true"] == pytest.approx(p)
        assert out["recall_true"] == pytest.approx(r)
        assert out["fbeta_true"] == pytest.approx(f)
        assert out["positives_above_true"] == int(truth[:64].sum())
        assert out["positives_below_true"] == int(truth[64:].sum())
        assert out["precision_lo"] <= out["precision_point"] <= out["precision_hi"]
        assert out["precision_held"] == int(out["precision_lo"] - 1e-9 <= p <= out["precision_hi"] + 1e-9)
        assert out["n_edges"] == out["edges_precision_held"] or out["edges_precision_held"] < out["n_edges"]
        assert out["verdict_match"] == int(out["verdict"] == out["oracle_verdict"])

    def test_the_ranges_hold_a_planted_truth_at_about_the_stated_level(self):
        """Over many seeds the precision range at Done contains the truth at least 90% of the time (95% nominal)."""
        truth, post = _planted()
        held = [simulate_line_test(truth, 64, 1.0, posteriors=post, seed=s)["precision_held"] for s in range(40)]
        assert np.mean(held) >= 0.9

    def test_a_tight_target_costs_more_picks_than_a_loose_one(self):
        truth, post = _planted()
        loose = simulate_line_test(truth, 64, 1.0, posteriors=post, budgets=LineBudgets(matches_width=0.5), seed=2)
        tight = simulate_line_test(truth, 64, 1.0, posteriors=post, budgets=LineBudgets(matches_width=0.1), seed=2)
        assert tight["picks_above"] >= loose["picks_above"]

    def test_a_budget_caps_the_picks_and_is_named_as_the_stop(self):
        truth, post = _planted()
        out = simulate_line_test(
            truth, 64, 1.0, posteriors=post, budgets=LineBudgets(matches_width=0.01, matches_picks=25), seed=3
        )
        assert out["matches_stop"] == "budget"
        assert out["picks_above"] == 25  # past the first pass's 20 (four bands above the line, #4560)

    def test_a_line_keeping_fewer_than_a_round_is_nothing_to_test(self):
        truth, post = _planted()
        out = simulate_line_test(truth, 3, 1.0, posteriors=post, seed=0)
        assert out["phase"] == PHASE_NOTHING
        assert out["picks_total"] == 0
        assert math.isnan(out["precision_point"]) and out["precision_held"] == -1
        assert out["verdict"] == out["oracle_verdict"] == VERDICT_NOTHING

    def test_seeded(self):
        truth, post = _planted()
        a = simulate_line_test(truth, 64, 1.0, posteriors=post, seed=7)
        b = simulate_line_test(truth, 64, 1.0, posteriors=post, seed=7)
        c = simulate_line_test(truth, 64, 1.0, posteriors=post, seed=8)
        assert a == b
        assert a != c  # a different seed deals different picks

    def test_with_a_class_model_the_walk_below_the_line_runs_to_its_budget(self):
        """#4542: the default arm walks to the misses budget with a model and stops at a dry band without one."""
        truth, post = _planted()
        out = simulate_line_test(truth, 64, 1.0, posteriors=post, seed=1)
        assert out["misses_stop"] == "budget" and out["picks_below"] == DEFAULT_BUDGETS.misses_picks
        bare = simulate_line_test(truth, 64, 1.0, posteriors=None, seed=1)
        assert bare["misses_stop"] in ("dry_run", "width") and bare["picks_below"] < DEFAULT_BUDGETS.misses_picks

    def test_without_a_model_the_tail_counts_nothing_and_says_so(self):
        truth, _ = _planted()
        out = simulate_line_test(truth, 64, 1.0, posteriors=None, seed=0)
        assert out["tail_positives_model"] == 0.0
        assert out["tail_from_model"] in (0, 1)
        assert out["positives_below_true"] == 6


class TestTheTruthHelpers:
    def test_truth_at(self):
        truth = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
        p, r, f = truth_at(truth, 3, 1.0)
        assert (p, r) == (2 / 3, 2 / 3)
        assert f == pytest.approx(2 / 3)
        assert truth_at(truth, 0, 1.0) == (0.0, 0.0, 0.0)

    def test_best_cut_is_the_f_beta_argmax(self):
        truth = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
        k, f = best_cut(truth, 1.0)
        assert k == 4 and f == pytest.approx(6 / 7)  # all three positives in the top 4: P 3/4, R 1
        assert best_cut(np.zeros(5, dtype=bool), 1.0) == (0, 0.0)


class TestTheVerdictRule:
    class _E:
        def __init__(self, point, lo, hi):
            self.point, self.lo, self.hi = point, lo, hi

    class _Edge:
        def __init__(self, count, fbeta):
            self.count, self.fbeta = count, fbeta

    def _est(self, fbeta, edges) -> Any:
        """A stand-in for `LineEstimates` carrying only what `read_verdict` reads."""
        return SimpleNamespace(fbeta=fbeta, at_edges=edges)

    def test_lean_when_another_edge_reads_better_by_the_gain(self):
        est = self._est(
            self._E(0.6, 0.5, 0.7),
            [self._Edge(64, self._E(0.6, 0.5, 0.7)), self._Edge(128, self._E(0.6 + LEAN_GAIN, 0.5, 0.8))],
        )
        assert read_verdict(est, 64) == VERDICT_LEAN

    def test_retrain_when_the_range_cannot_reach_the_bar(self):
        est = self._est(self._E(0.3, 0.2, RETRAIN_BAR - 0.01), [self._Edge(64, self._E(0.3, 0.2, 0.4))])
        assert read_verdict(est, 64) == VERDICT_RETRAIN

    def test_ship_otherwise(self):
        est = self._est(
            self._E(0.6, 0.5, 0.7), [self._Edge(64, self._E(0.6, 0.5, 0.7)), self._Edge(32, self._E(0.61, 0.5, 0.7))]
        )
        assert read_verdict(est, 64) == VERDICT_SHIP

    def test_the_oracle_reads_the_truth_under_the_same_rule(self):
        edges = [8, 16, 32, 64, 128, 200]
        truth = np.zeros(200, dtype=bool)
        truth[:10] = True  # the top 10 are the positives: a line at 64 should lean to the edge at 16
        assert oracle_verdict(truth, 64, edges, 1.0) == VERDICT_LEAN
        truth = np.zeros(200, dtype=bool)
        truth[:64] = True  # the line is exactly right: ship
        assert oracle_verdict(truth, 64, edges, 1.0) == VERDICT_SHIP
        # 22 of the top 64 right (every third rank) and 20 more positives spread thinly below:
        # F1 at the line is 0.42, and every other edge reads lower, so the truth says retrain.
        truth = np.zeros(200, dtype=bool)
        truth[:64:3] = True
        truth[64:200:7] = True
        at_line = truth_at(truth, 64, 1.0)[2]
        gains = [truth_at(truth, e, 1.0)[2] - at_line for e in edges if e != 64]
        assert at_line < RETRAIN_BAR and max(gains) < LEAN_GAIN, (at_line, gains)
        assert oracle_verdict(truth, 64, edges, 1.0) == VERDICT_RETRAIN


class TestTheInputs:
    def _scores(self, n=400, pos=30, seed=0):
        rng = np.random.default_rng(seed)
        labels = np.zeros(n)
        labels[rng.choice(n, pos, replace=False)] = 1.0
        scores = np.clip(rng.normal(0.3, 0.1, n) + 0.35 * labels, 0.01, 0.99)
        return scores, labels

    def test_the_labels_line_on_the_corpus_draws_the_line_and_aligns_the_posteriors(self):
        scores, labels = self._scores()
        model = ClassScoreModel(mu_pos=1.0, mu_neg=-1.0, sigma=0.5, n_pos=10, n_neg=20)
        line = LabelsLine(model, 0.5).on_corpus(scores)
        got = line_test_inputs(scores, labels, line, None, beta=1.0)
        assert got.line_source == LINE_LABELS
        assert got.line_count == int(np.count_nonzero(scores >= line.threshold(1.0)))
        order = np.argsort(-scores, kind="stable")
        assert np.array_equal(got.truth, labels[order] >= 0.5)
        assert got.posteriors is not None and got.posteriors.shape == scores.shape
        assert line.unvoted_posteriors is not None
        assert np.array_equal(got.posteriors, line.unvoted_posteriors)

    def test_an_unscorable_item_ranks_last_and_takes_no_posterior(self):
        scores, labels = self._scores()
        scores[5] = float("nan")
        model = ClassScoreModel(mu_pos=1.0, mu_neg=-1.0, sigma=0.5, n_pos=10, n_neg=20)
        line = LabelsLine(model, 0.5).on_corpus(scores)
        got = line_test_inputs(scores, labels, line, None, beta=1.0)
        assert got.posteriors is not None and got.posteriors.size == scores.size
        assert got.posteriors[-1] == 0.0  # the NaN ranks last
        assert np.all((got.posteriors >= 0.0) & (got.posteriors <= 1.0))

    def test_the_fallback_cut_and_no_line(self):
        scores, labels = self._scores()
        got = line_test_inputs(scores, labels, None, 0.6, beta=1.0)
        assert got.line_source == LINE_FALLBACK and got.posteriors is None
        assert got.line_count == int(np.count_nonzero(scores >= 0.6))
        none = line_test_inputs(scores, labels, None, None, beta=1.0)
        assert none.line_source == LINE_NONE and none.line_count == 0
        assert simulate_line_test(none.truth, none.line_count, 1.0)["phase"] == PHASE_NOTHING

    def test_the_row_carries_exactly_the_frames_columns(self):
        scores, labels = self._scores()
        row = line_test_row(150, scores, labels, 1.0, fallback_threshold=0.6, seed=0)
        assert set(row) | set(IDENT) == set(LINE_TEST_COLUMNS)
        assert row["t"] == 150 and row["beta"] == 1.0 and row["test_seed"] == 0
        assert row["matches_width"] == DEFAULT_BUDGETS.matches_width
        # No class model: no preset line to re-estimate (#4540).
        assert row["preset_b1_count"] == -1 and row["preset_b1_fbeta_held"] == -1

    def test_the_presets_are_what_lean_the_threshold_shows(self):
        """#4540: each balance preset's count is the labels line's at that beta, read like the app's route."""
        scores, labels = self._scores()
        model = ClassScoreModel(mu_pos=1.0, mu_neg=-1.0, sigma=0.5, n_pos=10, n_neg=20)
        line = LabelsLine(model, 0.5).on_corpus(scores)
        row = line_test_row(150, scores, labels, 1.0, find_on_test=line, seed=0)
        order = np.argsort(-scores, kind="stable")
        truth = labels[order] >= 0.5
        for beta, tag in ((0.25, "b025"), (1.0, "b1"), (4.0, "b4")):
            count = int(np.count_nonzero(scores >= line.threshold(beta)))
            assert row[f"preset_{tag}_count"] == count
            if row["phase"] == PHASE_DONE and count > 0:
                p, r, f = truth_at(truth, count, beta)
                assert row[f"preset_{tag}_precision_true"] == pytest.approx(p)
                assert row[f"preset_{tag}_fbeta_true"] == pytest.approx(f)
                assert row[f"preset_{tag}_precision_held"] in (0, 1)
        # The preset at the session's own beta is the line itself, so its range is the line's.
        if row["phase"] == PHASE_DONE:
            assert row["preset_b1_count"] == row["line_count"]
            assert row["preset_b1_precision_point"] == pytest.approx(row["precision_point"])

    def test_a_snapshot_replays_through_the_same_row(self):
        scores, labels = self._scores()
        model = ClassScoreModel(mu_pos=1.0, mu_neg=-1.0, sigma=0.5, n_pos=10, n_neg=20)
        snap = {
            "t": 150,
            "beta": 1.0,
            "scores": scores,
            "labels": labels,
            "model": model.as_dict(),
            "train_threshold": 0.6,
        }
        direct = line_test_row(
            150, scores, labels, 1.0, find_on_test=LabelsLine(model, 0.5).on_corpus(scores), fallback_threshold=0.6
        )
        assert row_from_snapshot(snap) == direct
        keep = np.ones(scores.size, dtype=bool)
        keep[::3] = False
        thinned = row_from_snapshot(snap, keep=keep)
        assert thinned["n_test"] == int(keep.sum())
        with pytest.raises(ValueError):
            row_from_snapshot({**snap, "beta": float("nan")})


def _separable(n_per_cat: int = 20, dim: int = 16) -> dict[int, dict]:
    rng = np.random.RandomState(0)
    medias: dict[int, dict] = {}
    for i in range(2 * n_per_cat):
        cat = "alpha" if i < n_per_cat else "beta"
        centre = 1.0 if cat == "alpha" else -1.0
        emb = rng.normal(centre, 0.6, dim).astype(np.float32)
        medias[i + 1] = {"id": i + 1, "embeddings": {"emb": emb}, "category": cat}
    return medias


class TestTheHarnessArm:
    def _run(self, **kwargs):
        sink: list[dict] = []
        snaps: list[dict] = []
        rows = simulate_voting_iterations(
            _separable(),
            "alpha",
            seed=0,
            max_steps=15,
            calibrate_count=2,
            style="whole_image",
            emit_calibration_metrics=True,
            line_test_sink=sink,
            test_score_sink=snaps,
            **kwargs,
        )
        return rows, sink, snaps

    def test_one_row_per_session_under_a_balance(self):
        rows, sink, _ = self._run(beta=1.0)
        assert rows and len(sink) == 1
        row = sink[0]
        assert set(row) == set(LINE_TEST_COLUMNS)
        assert row["beta"] == 1.0 and row["phase"] in (PHASE_DONE, PHASE_NOTHING)
        assert row["dataset"] == rows[0]["dataset"] and row["category"] == "alpha" and row["seed"] == 0
        assert row["t"] == max(r["t"] for r in rows if r["phase"] != "check")

    def test_no_row_under_the_floor_arm(self):
        """The Test is the balance line's: a floor draws no F-beta line to test."""
        _rows, sink, _ = self._run(min_precision=DEFAULT_MIN_PRECISION)
        assert sink == []

    def test_the_arm_does_not_touch_the_trajectory(self):
        with_arm, _, _ = self._run(beta=1.0)
        without = simulate_voting_iterations(
            _separable(),
            "alpha",
            seed=0,
            max_steps=15,
            calibrate_count=2,
            style="whole_image",
            emit_calibration_metrics=True,
            beta=1.0,
        )
        from vtscore.eval.voting_columns import TIMING_COLUMNS

        def _same(a, b) -> bool:
            """Value equality that reads NaN as reproducing itself."""
            if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
                return True
            return bool(a == b)

        assert len(with_arm) == len(without)
        for r, s in zip(with_arm, without):
            assert set(r) == set(s)
            for k in r:
                if k not in TIMING_COLUMNS:
                    assert _same(r[k], s[k]), (k, r[k], s[k])

    def test_the_saved_snapshot_replays_to_the_harness_row(self):
        """What the study rests on: `row_from_snapshot` on the dumped `last` snapshot is the harness's row."""
        _rows, sink, snaps = self._run(beta=1.0)
        ordinary = [s for s in snaps if s["phase"] != "check"]
        assert ordinary
        replay = row_from_snapshot(ordinary[-1], seed=0)
        harness = {k: v for k, v in sink[0].items() if k not in IDENT}
        assert replay == harness
