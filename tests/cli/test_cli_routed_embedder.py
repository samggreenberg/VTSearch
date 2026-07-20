"""CLI direct-scoring builds the matrix in the detector's embedder space.

Regression for the bug where ``_score_direct_all`` (the legacy, no-declared-
``media_type`` path of ``_score_medias_with_detectors``) scored every media
against its **primary** embedding via ``get_embedding_matrix_for_snap(medias)``
with no embedder name.  On a single-embedder dataset primary == the detector's
trained space, so there was no bug; on a *trio* dataset whose primary differs
from that space, the per-chunk scores were silently wrong.

The group key already carries the detectors' shared ``embedder_name`` (they all
trained in it), so the fix threads it into the matrix build.  These arrange the
two embedder spaces to give opposite verdicts, catching a wrong-space matrix as
a flipped Good/Bad.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import app as app_module  # noqa: F401  (activates the default dataset context)
from vtscore.cli import _score_direct_all

DIM = 4


def _basis(i: int) -> np.ndarray:
    return np.eye(DIM, dtype=np.float32)[i]


def _trio_medias() -> dict[int, dict]:
    """siglip (primary) + dinov3_patch, with the "on" vector swapped per space."""
    return {
        1: {
            "id": 1,
            "media_type": "image",
            "embedder": "siglip",
            "embeddings": {"siglip": _basis(0), "dinov3_patch": _basis(1)},
            "filename": "m1.png",
            "md5": "m1",
        },
        2: {
            "id": 2,
            "media_type": "image",
            "embedder": "siglip",
            "embeddings": {"siglip": _basis(1), "dinov3_patch": _basis(0)},
            "filename": "m2.png",
            "md5": "m2",
        },
    }


def _fires_on_dim1_mlp() -> nn.Module:
    linear = nn.Linear(DIM, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[0.0, 10.0, 0.0, 0.0]]))
        linear.bias.copy_(torch.tensor([-5.0]))
    return nn.Sequential(linear).eval()


def _hit_ids(result: dict) -> set[int]:
    return {h["id"] for h in result["hits"]}


class TestScoreDirectRoutesEmbedder:
    def test_scores_in_named_embedder_not_primary(self):
        detector_mlps = {"d": {"mlp": _fires_on_dim1_mlp(), "threshold": 0.5}}
        # dinov3_patch space: media 1 is e1 (Good), media 2 is e0 (Bad).
        out = _score_direct_all(["d"], detector_mlps, _trio_medias(), "dinov3_patch")
        assert _hit_ids(out["d"]) == {1}

    def test_default_embedder_is_primary(self):
        detector_mlps = {"d": {"mlp": _fires_on_dim1_mlp(), "threshold": 0.5}}
        # No embedder name → primary (siglip) space, where the verdicts flip:
        # media 2 is e1 (Good), media 1 is e0 (Bad).  Guards that the fix leaves
        # the single-embedder default untouched.
        out = _score_direct_all(["d"], detector_mlps, _trio_medias(), "")
        assert _hit_ids(out["d"]) == {2}
