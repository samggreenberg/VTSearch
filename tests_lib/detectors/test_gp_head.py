"""The Gaussian-process arms of issue #3954.

Three additions, each pinned here on small synthetic embeddings so nothing
needs a model download:

* the ``gp_*`` sweep trainers (:mod:`vtscore.eval.sweep_trainers`) - fit,
  score, report a bounded per-item spread, parse their ``@`` parameters;
* the ``gp_*`` step-trainer path (:mod:`vtscore.eval.step_trainers`), which
  gives the voting simulation a :class:`StepModel` with ``predict_std``;
* the two uncertainty-driven strategies (:mod:`vtscore.eval.al_strategies`),
  which keep the Autopilot phases and swap only the Hard pick - and refuse to
  run against a trainer that reports no spread.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from vtscore.eval.al_strategies import (
    AUTOPILOT_STRATEGIES,
    STRATEGIES,
    ALContext,
    _uncertainty_pick,
    is_autopilot_strategy,
    select_next,
)
from vtscore.eval.step_trainers import _train_and_calibrate
from vtscore.eval.sweep_trainers import (
    SWEEP_TRAINERS,
    _parse_gp_spec,
    _sigmoid_posterior_std,
    resolve_trainer,
)
from vtscore.eval.voting_iterations import simulate_voting_iterations


def _blobs(dim: int = 16, n_per: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated clusters on the unit sphere, like L2-normalised embeddings."""
    rng = np.random.RandomState(seed)
    centres = np.eye(2, dim, dtype=np.float32) * 2.0
    X, y = [], []
    for c in range(2):
        for _ in range(n_per):
            e = (centres[c] + rng.normal(0, 0.25, dim)).astype(np.float32)
            X.append(e / np.linalg.norm(e))
            y.append(1 if c == 0 else 0)
    return np.stack(X), np.asarray(y, dtype=np.int32)


def _clips(dim: int = 16, n_per: int = 40, seed: int = 0) -> dict[int, dict]:
    X, y = _blobs(dim, n_per, seed)
    return {
        i + 1: {"id": i + 1, "embeddings": {"emb": X[i]}, "category": "cat0" if y[i] == 1 else "cat1"}
        for i in range(len(y))
    }


# ---------------------------------------------------------------------------
# Sweep trainers
# ---------------------------------------------------------------------------


class TestGPSweepTrainers:
    @pytest.mark.parametrize("name", ["gp_rbf", "gp_dot"])
    def test_registered_and_separates_the_blobs(self, name):
        assert name in SWEEP_TRAINERS
        X, y = _blobs()
        idx = np.r_[0:4, 40:44]  # 4 positives, 4 negatives - a quorum-sized vote set
        predict = resolve_trainer(name)(X[idx], y[idx], 0)
        scores, std = predict(X)
        assert scores.shape == (80,) and std.shape == (80,)
        assert np.all((scores >= 0.0) & (scores <= 1.0))
        assert scores[:40].mean() > scores[40:].mean()
        # A std of a [0, 1] variable is bounded by 0.5, and every item has one.
        assert np.all((std >= 0.0) & (std <= 0.5))

    def test_spread_is_smaller_near_the_training_votes(self):
        X, y = _blobs()
        idx = np.r_[0:4, 40:44]
        _, std = resolve_trainer("gp_rbf")(X[idx], y[idx], 0)(X)
        # The voted items are the GP's own conditioning points.
        assert std[idx].mean() <= std.mean()

    def test_fixed_hyperparameters_skip_ml_ii(self):
        X, y = _blobs()
        idx = np.r_[0:4, 40:44]
        fixed = resolve_trainer("gp_rbf@ls=0.5,amp=2,fixed")(X[idx], y[idx], 0)
        scores, _ = fixed(X)
        assert scores[:40].mean() > scores[40:].mean()

    def test_single_class_labels_are_refused(self):
        X, y = _blobs()
        with pytest.raises(ValueError):
            resolve_trainer("gp_rbf")(X[:4], y[:4], 0)

    def test_deterministic_for_a_seed(self):
        X, y = _blobs()
        idx = np.r_[0:4, 40:44]
        a, _ = resolve_trainer("gp_dot")(X[idx], y[idx], 3)(X)
        b, _ = resolve_trainer("gp_dot")(X[idx], y[idx], 3)(X)
        np.testing.assert_allclose(a, b)


class TestGPSpec:
    def test_parses_every_token(self):
        assert _parse_gp_spec("gp_rbf") == ("rbf", {})
        assert _parse_gp_spec("gp_dot@ls=0.5,amp=2,fixed") == (
            "dot",
            {"length_scale": 0.5, "amplitude": 2.0, "optimize": False},
        )

    def test_rejects_unknown_kernel_and_parameter(self):
        with pytest.raises(KeyError):
            _parse_gp_spec("gp_matern")
        with pytest.raises(ValueError):
            _parse_gp_spec("gp_rbf@C=3")
        with pytest.raises(KeyError):
            resolve_trainer("gp_matern")

    def test_svm_specs_still_resolve(self):
        assert callable(resolve_trainer("svm_rbf@C=3,gamma=scale"))


class TestSigmoidPosteriorStd:
    def test_matches_monte_carlo(self):
        rng = np.random.default_rng(42)
        mean = np.array([0.0, 2.0, -3.0])
        std = np.array([1.0, 3.0, 0.2])
        mc = np.array([expit(m + s * rng.standard_normal(200_000)).std() for m, s in zip(mean, std)])
        np.testing.assert_allclose(_sigmoid_posterior_std(mean, std), mc, atol=2e-3)

    def test_zero_latent_spread_is_zero(self):
        assert _sigmoid_posterior_std(np.array([1.0]), np.array([0.0]))[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Step trainer
# ---------------------------------------------------------------------------


class TestGPStepTrainer:
    def test_dispatches_and_exposes_predict_std(self):
        clips = _clips()
        good = {1: None, 2: None, 3: None}
        bad = {41: None, 42: None, 43: None, 44: None}
        step, threshold, n_labels, timings, details = _train_and_calibrate(
            "gp_rbf",
            good,
            bad,
            clips,
            "cat0",
            region_voting=False,
            input_dim=16,
            inclusion=0,
            calibrate_count=2,
            calibration_fraction=0.5,
        )
        assert n_labels == 7
        assert step.backend == "sklearn-gp" and step.torch_model is None
        assert step.predict_std is not None
        X = np.stack([clips[i]["embeddings"]["emb"] for i in sorted(clips)])
        p = step.predict(X)
        s = step.predict_std(X)
        assert p.shape == s.shape == (80,)
        assert 0.0 <= threshold <= 1.0
        assert set(timings) == {"train_seconds", "xcal_seconds"}
        assert details == {}

    def test_app_and_svm_paths_carry_no_spread(self):
        clips = _clips()
        good = {1: None, 2: None, 3: None}
        bad = {41: None, 42: None, 43: None, 44: None}
        for trainer in ("app", "svm_linear"):
            step, *_ = _train_and_calibrate(
                trainer,
                good,
                bad,
                clips,
                "cat0",
                region_voting=False,
                input_dim=16,
                inclusion=0,
                calibrate_count=2,
                calibration_fraction=0.5,
            )
            assert step.predict_std is None


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _ctx(pool_ids, *, scores, uncertainty, threshold=0.5, phase: str | None = "hard", model=object()):
    return ALContext(
        pool_ids=list(pool_ids),
        embeddings={i: np.zeros(4, np.float32) for i in pool_ids},
        labeled={},
        scores=scores,
        model=model,
        threshold=threshold,
        atlas=None,
        rng=np.random.RandomState(0),
        phase=phase,
        uncertainty=uncertainty,
    )


class TestUncertaintyStrategies:
    def test_registry_and_family(self):
        assert {"autopilot_uncertainty", "autopilot_maxvar"} <= set(STRATEGIES)
        assert AUTOPILOT_STRATEGIES == frozenset(STRATEGIES)
        assert is_autopilot_strategy("autopilot_uncertainty")
        assert not is_autopilot_strategy("random")

    def test_straddle_prefers_the_item_the_posterior_could_flip(self):
        # id 1 sits exactly on the cut but is certain (straddle 0.002); id 2 is
        # a little off the cut and wide open (0.34); id 3 is far and certain.
        scores = {1: 0.50, 2: 0.55, 3: 0.95}
        spread = {1: 0.001, 2: 0.20, 3: 0.01}
        ctx = _ctx([1, 2, 3], scores=scores, uncertainty=spread)
        assert _uncertainty_pick(ctx, straddle=True) == 2
        assert select_next("autopilot_uncertainty", ctx) == 2

    def test_maxvar_ignores_the_cut(self):
        scores = {1: 0.50, 2: 0.55, 3: 0.95}
        spread = {1: 0.001, 2: 0.20, 3: 0.30}
        ctx = _ctx([1, 2, 3], scores=scores, uncertainty=spread)
        assert _uncertainty_pick(ctx, straddle=False) == 3
        assert select_next("autopilot_maxvar", ctx) == 3

    def test_only_unlabeled_pool_items_are_candidates(self):
        scores = {1: 0.50, 2: 0.55, 3: 0.95, 9: 0.5}
        spread = {1: 0.001, 2: 0.20, 3: 0.01, 9: 0.49}
        ctx = _ctx([1, 2, 3], scores=scores, uncertainty=spread)  # 9 is voted
        assert _uncertainty_pick(ctx, straddle=True) == 2
        assert _uncertainty_pick(ctx, straddle=False) == 2

    def test_refuses_a_model_without_a_spread(self):
        ctx = _ctx([1, 2], scores={1: 0.4, 2: 0.6}, uncertainty=None)
        with pytest.raises(ValueError, match="per-item uncertainty"):
            select_next("autopilot_uncertainty", ctx)

    def test_refuses_the_legacy_flow(self):
        ctx = _ctx([1, 2], scores={1: 0.4, 2: 0.6}, uncertainty={1: 0.1, 2: 0.1}, phase=None)
        with pytest.raises(ValueError, match="faithful autopilot flow"):
            select_next("autopilot_maxvar", ctx)

    def test_pre_detector_phases_are_the_apps(self):
        # Before a model exists the Good phase is delegated untouched: with a
        # seed sort it is the top of that sort, spread or no spread.
        ctx = ALContext(
            pool_ids=[1, 2, 3],
            embeddings={i: np.zeros(4, np.float32) for i in (1, 2, 3)},
            labeled={},
            scores={},
            model=None,
            threshold=0.5,
            atlas=None,
            rng=np.random.RandomState(0),
            seed_scores={1: 0.1, 2: 0.9, 3: 0.5},
            phase="good",
            uncertainty=None,
        )
        assert select_next("autopilot_uncertainty", ctx) == 2
        assert select_next("autopilot", ctx) == 2


class TestGPVotingSimulation:
    @pytest.mark.parametrize("strategy", ["autopilot", "autopilot_uncertainty", "autopilot_maxvar"])
    def test_gp_trainer_runs_the_loop(self, strategy):
        rows = simulate_voting_iterations(
            _clips(),
            "cat0",
            seed=0,
            trainer="gp_rbf",
            strategy=strategy,
            max_steps=16,
            safe_thresholds=False,
        )
        assert rows, "the loop should train once a Good and a Bad vote coexist"
        assert {r["trainer"] for r in rows} == {"gp_rbf"}
        assert {r["strategy"] for r in rows} == {strategy}
        assert all(r["head"] == "" for r in rows)
        # The phase machine ran: the family gates the atlas and flow, not the one name.
        assert all(r["phase"] for r in rows)
        assert rows[-1]["cost"] <= rows[0]["cost"]

    def test_app_trainer_under_an_uncertainty_strategy_is_loud(self):
        with pytest.raises(ValueError, match="per-item uncertainty"):
            simulate_voting_iterations(
                _clips(),
                "cat0",
                seed=0,
                trainer="app",
                strategy="autopilot_uncertainty",
                max_steps=12,
                safe_thresholds=False,
            )
