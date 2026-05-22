"""Tests for ``vtscore.utils.scores`` — NaN/Inf cannot leak into JSON scores.

Logical-bug audit M13: ``learned_scores`` in ``/api/votes`` (and the cousin
endpoints ``/api/learned-sort``, ``/api/find-label``, ``/api/find``,
``/api/label-file-sort``) used to emit the literal token ``NaN`` when the
MLP destabilised.  Browser ``JSON.parse`` rejects it, breaking the UI.

These tests pin the invariant at the source (``sigmoid_to_finite_scores``)
and at the consumers (``train_and_score``, ``labelset_train_and_score``).
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
import torch
import torch.nn as nn

from vtscore.utils.scores import (
    NON_FINITE_SCORE_SENTINEL,
    finite_or,
    sigmoid_to_finite_scores,
)


class TestFiniteOr:
    """Single-value sentinel substitution used by the ``/api/votes`` guard."""

    def test_finite_passthrough(self):
        assert finite_or(0.5) == 0.5
        assert finite_or(-1.0) == -1.0
        assert finite_or(0.0) == 0.0

    def test_nan_replaced(self):
        assert finite_or(float("nan")) == NON_FINITE_SCORE_SENTINEL

    def test_pos_inf_replaced(self):
        assert finite_or(float("inf")) == NON_FINITE_SCORE_SENTINEL

    def test_neg_inf_replaced(self):
        assert finite_or(float("-inf")) == NON_FINITE_SCORE_SENTINEL

    def test_custom_default(self):
        assert finite_or(float("nan"), default=0.5) == 0.5


class TestSigmoidToFiniteScores:
    """Tensor-level helper used everywhere a sigmoid output reaches JSON."""

    def test_finite_logits_round_trip(self):
        logits = torch.tensor([[-2.0], [0.0], [2.0]])
        scores = sigmoid_to_finite_scores(logits)
        assert scores == pytest.approx([0.1192, 0.5, 0.8808], abs=1e-3)
        assert all(math.isfinite(s) for s in scores)

    def test_nan_logits_replaced_with_sentinel(self):
        logits = torch.tensor([[float("nan")], [1.0], [float("nan")]])
        scores = sigmoid_to_finite_scores(logits)
        assert scores[0] == NON_FINITE_SCORE_SENTINEL
        assert scores[2] == NON_FINITE_SCORE_SENTINEL
        assert scores[1] == pytest.approx(0.7311, abs=1e-3)
        assert all(math.isfinite(s) for s in scores)

    def test_pos_inf_logits_produce_finite_scores(self):
        # ``sigmoid(+inf) == 1.0`` already, no sentinel needed — but assert
        # the result is finite so a numerical edge can't sneak through.
        scores = sigmoid_to_finite_scores(torch.tensor([[float("inf")]]))
        assert math.isfinite(scores[0])

    def test_custom_default(self):
        scores = sigmoid_to_finite_scores(torch.tensor([[float("nan")]]), default=0.5)
        assert scores[0] == 0.5

    def test_unsqueezed_input_handled(self):
        # ``squeeze(-1)`` handles both ``(N, 1)`` and ``(N,)`` inputs.
        scores = sigmoid_to_finite_scores(torch.tensor([0.0, float("nan"), 2.0]))
        assert scores[0] == pytest.approx(0.5, abs=1e-3)
        assert scores[1] == NON_FINITE_SCORE_SENTINEL
        assert math.isfinite(scores[2])


class _NaNProducingModel(nn.Module):
    """A toy model whose forward pass deterministically returns NaN logits.

    Used to simulate "MLP destabilised during training" without actually
    having to provoke training divergence (which is non-deterministic).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):  # noqa: D401 - torch nn.Module override
        out = self.linear(x)
        # Poison the output uniformly so the downstream code paths see NaN
        # without depending on training pathology.
        return out * float("nan")


class TestScoreAllMediaNaNSafety:
    """``_score_all_media`` (non-region path) must not leak NaN to results."""

    def test_non_region_path_replaces_nan_with_sentinel(self):
        from vtscore.detectors.training import _score_all_media

        rng = np.random.default_rng(0)
        clips: dict[int, dict] = {}
        for cid in (1, 2, 3):
            clips[cid] = {
                "embedding": rng.standard_normal(8).astype(np.float32),
                "embedder": "test",
            }
        model = _NaNProducingModel(input_dim=8).eval()

        # ``_score_all_media`` is typed as ``nn.Sequential``, but its body only
        # calls the model and reads ``.parameters()`` — both ``nn.Module`` APIs.
        # ``cast`` quiets pyright without changing runtime behaviour.
        all_ids, scores, _best = _score_all_media(cast(nn.Sequential, model), clips)

        assert all_ids == [1, 2, 3]
        assert all(math.isfinite(s) for s in scores), scores
        # ``_score_all_media`` seeds with -1.0; sentinel-substituted scores
        # never beat that floor, so every entry stays at -1.0.
        assert scores == [-1.0, -1.0, -1.0]


class TestLabelsetTrainAndScoreNaNSafety:
    """``labelset_train_and_score`` was the primary M13 leak site —
    sigmoid outputs flowed straight to results with no finite filter.

    We pre-populate ``det_ctx.label_embeddings`` and patch out
    ``populate_label_embeddings`` / ``record_detector_embedder`` so the
    test focuses on the scoring tail of the function instead of dragging
    in the resolver, registry, and disk I/O.
    """

    def test_nan_model_produces_finite_scores(self, monkeypatch):
        from vtscore.datasets.labelset import LabeledElement, LabelSet
        from vtscore.detectors import labelset_training, registry as det_registry
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.state.core import DetectorContext
        from vtscore.training import mlp as mlp_mod

        rng = np.random.default_rng(0)
        dim = 8

        # Replace the real ``train_model`` with one that returns a model
        # that always produces NaN logits.  This is what M13 simulates:
        # the trainer "successfully" produced an MLP, but its weights have
        # destabilised so every forward pass is NaN.
        def fake_train_model(X, y, input_dim, inclusion, hidden_dim=None):  # noqa: ARG001
            return _NaNProducingModel(input_dim).eval()

        # ``labelset_train_and_score`` does a local ``from vtscore.training.mlp
        # import train_model`` — patch the source module so the local import
        # resolves to our fake.
        monkeypatch.setattr(mlp_mod, "train_model", fake_train_model)

        # Bypass embedding resolution: we've pre-populated the cache below.
        monkeypatch.setattr(labelset_training, "populate_label_embeddings", lambda *a, **kw: 0)
        # Avoid registry disk writes — irrelevant for this test.
        monkeypatch.setattr(det_registry, "record_detector_embedder", lambda *a, **kw: None)

        det_ctx = DetectorContext(detector_id="test-det", media_type="audio")
        good_elem = LabeledElement(
            origin={"importer": "test", "params": {}},
            origin_name="g",
            label="good",
            md5="g" * 32,
        )
        bad_elem = LabeledElement(
            origin={"importer": "test", "params": {}},
            origin_name="b",
            label="bad",
            md5="b" * 32,
        )
        labelset = LabelSet(elements=[good_elem, bad_elem])
        det_ctx.label_embeddings[stable_element_id(good_elem)] = rng.standard_normal(dim).astype(np.float32)
        det_ctx.label_embeddings[stable_element_id(bad_elem)] = rng.standard_normal(dim).astype(np.float32)

        # IDs 100+ to avoid clashing with the conftest-loaded test medias (1-20).
        clips_dict = {
            cid: {
                "embedding": rng.standard_normal(dim).astype(np.float32),
                "embedder": "test",
                "media_type": "audio",
                "md5": f"m{cid:031d}",
            }
            for cid in (100, 101, 102)
        }

        results, threshold, model = labelset_training.labelset_train_and_score(
            det_ctx,
            labelset,
            media_type="audio",
            clips_dict=clips_dict,
        )

        assert model is not None
        assert math.isfinite(threshold)
        assert len(results) == 3
        for entry in results:
            assert math.isfinite(entry["score"]), entry
            assert entry["score"] == NON_FINITE_SCORE_SENTINEL
