"""``--head-strategy`` must reach the region-voting path, or fail loudly.

Region voting trains through :func:`train_rv_head`, which expresses the model as a torch
``hidden_dim`` (the knob ``cross_calibration_threshold_cached`` and ``train_model`` share)
rather than as a pluggable ``trainer_fn``. Before this was wired, ``--head-strategy`` was a
silent no-op on every ``hac`` + DINO cell and the path always trained an auto-sized MLP,
diverging from the shipped detector, which has used the linear/logistic head since #2790.

These tests pin the mapping (:func:`rv_hidden_dim`), the production-parity case, and the
refusal for the sklearn-only arms so the no-op can't come back unnoticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from vtscore.eval.region_curve import RV_HEAD_STRATEGIES, rv_hidden_dim, train_rv_head
from vtscore.training.mlp import LINEAR_HEAD, _auto_hidden_dim


class TestHiddenDimMapping:
    def test_linear_resolves_to_the_production_head(self):
        """#2790: the app ships Linear(input_dim, 1) + BCE, i.e. hidden_dim == 0."""
        assert rv_hidden_dim("linear", n_votes=40, n_good=10) == LINEAR_HEAD
        assert LINEAR_HEAD == 0

    def test_mlp_keeps_the_historical_auto_width(self):
        for n_votes in (6, 40, 400):
            assert rv_hidden_dim("mlp", n_votes=n_votes, n_good=10) == _auto_hidden_dim(n_votes)

    @pytest.mark.parametrize(("n_good", "expected"), [(1, 4), (2, 8), (10, 40), (16, 64), (100, 64)])
    def test_reg_mlp_anneals_capacity_on_n_good(self, n_good, expected):
        # max(4, min(64, 4 * n_good)) mirrors RegMLPHead's capacity rule.
        assert rv_hidden_dim("reg-mlp", n_votes=999, n_good=n_good) == expected

    def test_anneal_linear_switches_class_at_the_anneal_point(self):
        assert rv_hidden_dim("anneal-linear", n_votes=40, n_good=7) == LINEAR_HEAD
        assert rv_hidden_dim("anneal-linear", n_votes=40, n_good=8) == _auto_hidden_dim(40)

    def test_anneal_reg_switches_class_at_the_anneal_point(self):
        assert rv_hidden_dim("anneal-reg", n_votes=40, n_good=7) == max(4, min(64, 28))
        assert rv_hidden_dim("anneal-reg", n_votes=40, n_good=8) == _auto_hidden_dim(40)

    def test_anneal_point_is_overridable(self):
        assert rv_hidden_dim("anneal-linear", n_votes=40, n_good=8, anneal_k=16) == LINEAR_HEAD

    @pytest.mark.parametrize("strategy", ["svm", "anneal-svm"])
    def test_sklearn_arms_are_refused_not_silently_downgraded(self, strategy):
        with pytest.raises(ValueError, match="unavailable under region voting"):
            rv_hidden_dim(strategy, n_votes=40, n_good=10)

    def test_unknown_strategy_is_refused(self):
        with pytest.raises(ValueError, match="unavailable under region voting"):
            rv_hidden_dim("does-not-exist", n_votes=40, n_good=10)

    def test_supported_set_is_exactly_the_torch_expressible_arms(self):
        assert RV_HEAD_STRATEGIES == {"mlp", "linear", "reg-mlp", "anneal-linear", "anneal-reg"}


def _budget(dim: int = 8, n_pos: int = 6, n_bags: int = 6, rows_per_bag: int = 3):
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((n_pos, dim)).astype(np.float32) + 2.0
    bags = [rng.standard_normal((rows_per_bag, dim)).astype(np.float32) for _ in range(n_bags)]
    return pos, bags, dim


class TestTrainRvHead:
    def test_defaults_to_mlp_so_existing_callers_are_unchanged(self):
        pos, bags, dim = _budget()
        out = train_rv_head(pos, bags, dim, 0, inclusion=0, safe_thresholds=True, calibrate_count=2, cal_fraction=0.5)
        assert out is not None

    def test_linear_head_trains_and_scores(self):
        pos, bags, dim = _budget()
        out = train_rv_head(
            pos,
            bags,
            dim,
            0,
            inclusion=0,
            safe_thresholds=True,
            calibrate_count=2,
            cal_fraction=0.5,
            head_strategy="linear",
            n_good=pos.shape[0],
        )
        assert out is not None
        predict, thr, n_votes = out
        scores = np.asarray(predict(np.vstack([pos, bags[0]])))
        assert scores.shape[0] == pos.shape[0] + bags[0].shape[0]
        assert np.isfinite(thr)
        assert n_votes > 0

    def test_linear_and_mlp_produce_different_models(self):
        """Guards the regression this fixes: the strategy must actually change the head."""
        pos, bags, dim = _budget()
        kw = dict(inclusion=0, safe_thresholds=False, calibrate_count=2, cal_fraction=0.5, n_good=pos.shape[0])
        lin = train_rv_head(pos, bags, dim, 0, head_strategy="linear", **kw)
        mlp = train_rv_head(pos, bags, dim, 0, head_strategy="mlp", **kw)
        assert lin is not None and mlp is not None
        probe = np.vstack([pos, bags[0]])
        assert not np.allclose(np.asarray(lin[0](probe)), np.asarray(mlp[0](probe)))

    def test_sklearn_arm_raises_rather_than_training_an_mlp(self):
        pos, bags, dim = _budget()
        with pytest.raises(ValueError, match="unavailable under region voting"):
            train_rv_head(
                pos,
                bags,
                dim,
                0,
                inclusion=0,
                safe_thresholds=True,
                calibrate_count=2,
                cal_fraction=0.5,
                head_strategy="svm",
                n_good=pos.shape[0],
            )


class TestSweepGuard:
    """Both bag routes must be refused up front, not just --region-voting.

    ``_train_pool_head`` takes the train_rv_head branch on ``region_voting OR
    neg_regions``, and ``--neg-regions`` applies to every proposal and embedder, so a
    guard that only inspects ``--region-voting`` lets ``--neg-regions --head-strategy
    svm`` through to silently train an MLP.
    """

    @staticmethod
    def _run(*argv):
        import subprocess

        # S603: fixed argv, no shell, every element a literal from this file. Spawning the
        # real CLI is the point -- the guard lives in argparse validation, so an in-process
        # call would not exercise it.
        return subprocess.run(  # noqa: S603
            [sys.executable, "scripts/sod/sweep.py", *argv, "--out-dir", "/tmp/vts-guard-check"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(Path(__file__).resolve().parents[2]),
        )

    def test_neg_regions_with_an_sklearn_head_is_refused(self):
        r = self._run(
            "--datasets", "vg_s", "--classes", "car", "--embedders", "siglip",
            "--proposals", "whole", "--neg-regions", "--no-region-voting", "--head-strategy", "svm",
        )  # fmt: skip
        assert r.returncode != 0
        assert "--neg-regions" in r.stderr
        assert "train_rv_head" in r.stderr

    def test_region_voting_with_an_sklearn_head_is_refused(self):
        r = self._run(
            "--datasets", "vg_s", "--classes", "car", "--embedders", "dinov3",
            "--proposals", "hac", "--region-voting", "--head-strategy", "svm",
        )  # fmt: skip
        assert r.returncode != 0
        assert "--region-voting" in r.stderr

    def test_box_pool_path_allows_the_sklearn_heads(self):
        """No bag route means make_head, so every arm including svm is legal."""
        r = self._run(
            "--datasets", "definitely-not-a-dataset", "--classes", "car", "--embedders", "dinov3",
            "--proposals", "hac", "--no-region-voting", "--no-neg-regions", "--head-strategy", "svm",
        )  # fmt: skip
        # Must get PAST argument validation (then fail later on the bogus dataset).
        assert "head-strategy" not in r.stderr
