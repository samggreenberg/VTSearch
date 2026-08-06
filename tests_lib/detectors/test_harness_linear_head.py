"""The harness can run the **production** head (#2799 needs it; torch tests).

The calibration/voting harness historically trained a small auto-sized MLP,
while the live detector has trained a linear (logistic) head since #2790/#2809.
Measuring a shipped default like ``safe_thresholds`` on the wrong head measures
the wrong product, so ``simulate_voting_iterations(head=...)`` selects it.

These tests pin the two things that make a ``head="linear"`` run faithful:

* the width reaches **both** the final model and the calibration folds (the
  folds must share the final model's architecture — production threads one
  width through ``_train_and_score_xy`` for exactly this reason);
* the final per-step model really is a single ``Linear(d, 1)``, i.e. logistic
  regression, on the whole-image path *and* the region/style path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

import vtscore.eval.voting_iterations as vi
from vtscore.eval.patch_styles import resolve_style
from vtscore.training.mlp import LINEAR_HEAD, _auto_hidden_dim

from .test_max_patch_style import DIM, _planted_dataset


def _votes(medias, n=6):
    goods = {m["id"]: None for m in list(medias.values()) if m["category"] == "cat0"}
    bads = {m["id"]: None for m in list(medias.values()) if m["category"] == "cat1"}
    return dict(list(goods.items())[:n]), dict(list(bads.items())[:n])


def test_resolve_hidden_dim_maps_heads_to_widths():
    assert vi._resolve_hidden_dim("linear", 40) == LINEAR_HEAD == 0
    assert vi._resolve_hidden_dim("mlp", 40) == _auto_hidden_dim(40)
    with pytest.raises(ValueError, match="unknown head"):
        vi._resolve_hidden_dim("logreg", 40)


def test_unknown_head_is_rejected_early():
    medias, _ = _planted_dataset(n_per_cat=6, seed=0)
    with pytest.raises(ValueError, match="unknown head"):
        vi.simulate_voting_iterations(medias, target_category="cat0", seed=0, head="logreg", max_steps=1)


def test_head_does_not_apply_to_the_svm_trainer():
    medias, _ = _planted_dataset(n_per_cat=6, seed=0)
    with pytest.raises(ValueError, match="only applies to the torch trainer"):
        vi.simulate_voting_iterations(
            medias, target_category="cat0", seed=0, trainer="svm_linear", head="linear", max_steps=1
        )


@pytest.mark.parametrize("style", [None, "max_patch"])
def test_linear_head_reaches_the_final_model_and_the_calibration_folds(style, monkeypatch):
    """``head="linear"`` must set ``hidden_dim=0`` on the fit *and* the folds."""
    medias, _ = _planted_dataset(n_per_cat=10, seed=0)
    good_votes, bad_votes = _votes(medias)

    seen: dict[str, list] = {"train": [], "calib": []}
    real_train = vi.train_model

    def spy_train(X, y, input_dim, **kw):
        seen["train"].append(kw.get("hidden_dim"))
        return real_train(X, y, input_dim, **kw)

    monkeypatch.setattr(vi, "train_model", spy_train)
    for name in ("calibration_folds", "compute_grouped_fold_node_scores"):
        real = getattr(vi, name)

        def spy_calib(*args, _real=real, **kw):
            seen["calib"].append(kw.get("hidden_dim"))
            return _real(*args, **kw)

        monkeypatch.setattr(vi, name, spy_calib)

    style_obj = None if style is None else resolve_style(style)

    step, threshold, _n, _timings, _details = vi._train_and_calibrate(
        "mlp",
        good_votes,
        bad_votes,
        medias,
        "cat0",
        region_voting=True,
        input_dim=DIM,
        inclusion=0,
        calibrate_count=2,
        calibration_fraction=0.5,
        head="linear",
        style_obj=style_obj,
    )

    assert seen["train"], "the final model was never trained"
    assert set(seen["train"]) == {LINEAR_HEAD}
    assert seen["calib"], "the calibration folds were never fitted"
    assert set(seen["calib"]) == {LINEAR_HEAD}
    # A single Linear(d, 1) with no hidden layer == logistic regression.
    assert step.torch_model is not None
    layers = [m for m in step.torch_model if isinstance(m, nn.Linear)]
    assert len(layers) == 1
    assert layers[0].out_features == 1
    assert not any(isinstance(m, nn.ReLU) for m in step.torch_model)
    assert np.isfinite(threshold)


def test_mlp_head_is_still_the_default_and_keeps_a_hidden_layer():
    """Default runs stay byte-comparable to the published #2781 numbers."""
    medias, _ = _planted_dataset(n_per_cat=10, seed=0)
    good_votes, bad_votes = _votes(medias)

    step, _threshold, n_labels, _t, _d = vi._train_and_calibrate(
        "mlp",
        good_votes,
        bad_votes,
        medias,
        "cat0",
        region_voting=True,
        input_dim=DIM,
        inclusion=0,
        calibrate_count=2,
        calibration_fraction=0.5,
    )
    assert step.torch_model is not None
    layers = [m for m in step.torch_model if isinstance(m, nn.Linear)]
    assert len(layers) == 2
    assert layers[0].out_features == _auto_hidden_dim(n_labels)


def test_rows_record_the_head_and_linear_runs_end_to_end():
    medias, _ = _planted_dataset(n_per_cat=30, seed=0)
    rows = vi.simulate_voting_iterations(
        medias,
        target_category="cat0",
        seed=0,
        dataset_name="planted",
        inclusion=0,
        region_voting=True,
        safe_thresholds=True,
        max_steps=12,
        style="max_patch",
        head="linear",
        emit_calibration_metrics=True,
    )
    assert rows, "no rows produced"
    assert {r["head"] for r in rows} == {"linear"}
    assert "head" in vi._IDENT_COLUMNS
    # The #2799 variant rows still ride along under safe_thresholds.
    assert {r["gmm_variant"] for r in rows} >= {"", "xcal_only", "pooled_cross"}
    for r in rows:
        assert np.isfinite(r["threshold"])
