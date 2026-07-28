"""Cross-dataset Find scores each detector in its *routed* embedder space.

Regression for the bug where ``_score_dataset`` built the embedding matrix from
each media's **primary** vector (``get_embedding_matrix_for_snap`` with no
embedder name) rather than the dataset's role-bound score embedder.  On a
single-embedder dataset primary == score embedder, so there was no observable
bug; on a *trio* dataset whose primary differs from the detector's keying
embedder, Find silently mis-scored every media.

These build a two-embedder image dataset whose recorded **primary** is
``siglip`` while its **patch** slot (the score-precedence winner) is
``dinov3_patch``, then arrange the two spaces to give *opposite* verdicts, so a
matrix built from the wrong space is caught by a flipped Good/Bad.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import app as app_module  # noqa: F401  (ensures routes are registered)
from vtsearch.routes.detectors import find as find_mod

DIM = 4


def _basis(i: int) -> np.ndarray:
    return np.eye(DIM, dtype=np.float32)[i]


def _trio_medias() -> dict[int, dict]:
    """Two image medias bound to siglip (primary) + dinov3_patch (patch slot).

    In the **dinov3_patch** space media 1 is the "on" vector (e1) and media 2 is
    "off" (e0); in the **siglip** space the assignment is swapped.  A detector
    scored in dinov3_patch therefore calls media 1 Good / media 2 Bad, and a
    detector mistakenly scored in the primary (siglip) space would flip that.
    """
    return {
        1: {
            "id": 1,
            "media_type": "image",
            "embedder": "siglip",  # primary != score embedder
            "embeddings": {"siglip": _basis(0), "dinov3_patch": _basis(1)},
            "filename": "m1.png",
            "md5": "m1",
            "origin_name": "m1.png",
            "origin": {"importer": "test", "params": {}},
        },
        2: {
            "id": 2,
            "media_type": "image",
            "embedder": "siglip",
            "embeddings": {"siglip": _basis(1), "dinov3_patch": _basis(0)},
            "filename": "m2.png",
            "md5": "m2",
            "origin_name": "m2.png",
            "origin": {"importer": "test", "params": {}},
        },
    }


def _fires_on_dim1_mlp() -> nn.Module:
    """MLP whose logit is high on e1 and low on e0 (score >0.5 only for e1)."""
    linear = nn.Linear(DIM, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[0.0, 10.0, 0.0, 0.0]]))
        linear.bias.copy_(torch.tensor([-5.0]))
    return nn.Sequential(linear).eval()


class TestFindScoreEmbedderResolution:
    def test_score_embedder_is_patch_slot_not_primary(self):
        # A legacy/typeless detector routes via the dataset score precedence
        # (structural ▸ patch ▸ text), which names the patch slot here - not the
        # recorded primary (siglip).
        dc = {"embedder_type": ""}
        assert find_mod._find_score_embedder(dc, _trio_medias()) == "dinov3_patch"

    def test_typed_detector_routes_to_its_type(self):
        semantic = {"embedder_type": "semantic"}
        patch = {"embedder_type": "patch_semantic"}
        snap = _trio_medias()
        assert find_mod._find_score_embedder(semantic, snap) == "siglip"
        assert find_mod._find_score_embedder(patch, snap) == "dinov3_patch"


class TestSelectScorerUsesScoreEmbedder:
    def test_live_selected_when_live_embedder_matches_score_not_primary(self):
        # The live MLP trained in dinov3_patch (the patch slot / score embedder),
        # which differs from the dataset primary (siglip).  The old code compared
        # against the primary and would have fallen through to cold; the fix
        # compares against the score embedder and keeps the live MLP.
        dc = {
            "name": "d",
            "live_mlp": _fires_on_dim1_mlp(),
            "threshold": 0.5,
            "live_embedder": "dinov3_patch",
            "embedder_type": "patch_semantic",
            "detector_data": {"labelset": {"labels": [{}]}},
        }
        assert find_mod._select_scorer(dc, _trio_medias()) == "live"

    def test_cold_selected_when_live_embedder_mismatches(self):
        dc = {
            "name": "d",
            "live_mlp": _fires_on_dim1_mlp(),
            "threshold": 0.5,
            "live_embedder": "clap",  # a space this dataset does not bind
            "embedder_type": "patch_semantic",
            "detector_data": {"labelset": {"labels": [{}]}},
        }
        assert find_mod._select_scorer(dc, _trio_medias()) == "cold"


class TestScoreDatasetRoutesLiveMatrix:
    def test_live_mlp_scored_against_patch_slot_not_primary(self, client, monkeypatch):
        """`_score_dataset` builds the live matrix in the MLP's embedder space.

        Scored in dinov3_patch, media 1 (e1) is Good and media 2 (e0) is Bad. A
        matrix built from the primary (siglip) vectors would flip both verdicts,
        so the positive/negative split proves the routed space was used.
        """
        monkeypatch.setattr(find_mod, "_load_find_dataset_medias", lambda ds: _trio_medias())

        dc = {
            "name": "patch-det",
            "detector_id": "patch-det",
            "live_mlp": _fires_on_dim1_mlp(),
            "threshold": 0.5,
            "live_embedder": "dinov3_patch",
            "embedder_type": "patch_semantic",
        }

        with app_module.app.test_request_context("/api/find"):
            positives, negatives, _units, _added, _mt = find_mod._score_dataset(
                {"name": "trio", "pkl_path": "ignored"}, [dc], 0, 0
            )

        pos_ids = {p["id"] for p in positives}
        neg_ids = {n["id"] for n in negatives}
        assert pos_ids == {1}, (pos_ids, neg_ids)
        assert neg_ids == {2}, (pos_ids, neg_ids)
