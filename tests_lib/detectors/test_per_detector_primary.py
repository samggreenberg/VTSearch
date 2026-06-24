"""Per-detector primary embedder routing (docs/plans/patch-embedder.md).

A detector binds one *primary* embedder - the vector space it trains and
scores in - independent of the dataset-level score precedence.  These cover
the pure resolvers that route training/scoring through that choice:

* ``keying_embedder_for_snap`` - the primary when the snap supplies it, else
  the dataset score precedence (the legacy / cross-dataset-portability fallback).
* ``detector_score_embedder`` - the scoring-space name handed to the matrix
  layer (delegates to keying).
* ``_score_all_media`` region gating - region max-pool only when the detector
  scores in the dataset's *patch* slot, so a text-primary detector on a
  text+patch dataset scores text full-image vectors, not the patch tree.
* ``resolve_detector_primary`` - create-time validation against the active
  dataset's bound embedders.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from vtscore.detectors.training import _score_all_media, detector_score_embedder
from vtscore.embedding.binding import keying_embedder_for_snap
from vtscore.media.patch_embed import RegionVector

DIM = 4


def _basis(i: int) -> np.ndarray:
    return np.eye(DIM, dtype=np.float32)[i]


def _dual_snap() -> dict[int, dict]:
    """Two image medias bound to both a text (siglip) and a patch (dinov3) embedder.

    media 1's siglip vector is e1, media 2's is e2; both share a dinov3 patch
    region vector e3.  The recorded primary is the patch embedder.
    """
    snap: dict[int, dict] = {}
    for cid in (1, 2):
        snap[cid] = {
            "id": cid,
            "media_type": "image",
            "embedder": "dinov3_patch",
            "embeddings": {"siglip": _basis(cid), "dinov3_patch": _basis(0)},
            "patch_regions": [RegionVector(box=(0.0, 0.0, 1.0, 1.0), vec=_basis(3))],
        }
    return snap


def _det(primary: str = "") -> SimpleNamespace:
    return SimpleNamespace(primary_embedder=primary, embedder="")


class TestKeyingEmbedder:
    def test_primary_used_when_snap_supplies_it(self):
        snap = _dual_snap()
        # The detector's chosen text primary is bound on the dataset → used.
        assert keying_embedder_for_snap(_det("siglip"), snap) == "siglip"
        # A patch primary is likewise honoured.
        assert keying_embedder_for_snap(_det("dinov3_patch"), snap) == "dinov3_patch"

    def test_falls_back_to_precedence_when_primary_absent(self):
        snap = _dual_snap()
        # "clap" isn't bound here → precedence (structural ▸ patch ▸ text) picks
        # the patch slot.
        assert keying_embedder_for_snap(_det("clap"), snap) == "dinov3_patch"

    def test_no_primary_is_precedence(self):
        snap = _dual_snap()
        assert keying_embedder_for_snap(_det(""), snap) == "dinov3_patch"
        assert keying_embedder_for_snap(None, snap) == "dinov3_patch"

    def test_empty_snap_is_empty_string(self):
        assert keying_embedder_for_snap(_det("siglip"), {}) == ""
        assert keying_embedder_for_snap(_det("siglip"), None) == ""

    def test_single_embedder_dataset_returns_that_embedder(self):
        snap = {1: {"id": 1, "embedder": "clap", "embeddings": {"clap": _basis(0)}}}
        # No chosen primary → precedence names the one (text-capable) embedder
        # only if it's role-typed; clap is text-capable so it fills the text slot.
        # Either way a single-embedder dataset never diverges the detector's space.
        assert keying_embedder_for_snap(_det(""), snap) in ("clap", "")


class TestDetectorScoreEmbedder:
    def test_delegates_to_keying(self):
        snap = _dual_snap()
        assert detector_score_embedder(_det("siglip"), snap) == "siglip"
        assert detector_score_embedder(_det("dinov3_patch"), snap) == "dinov3_patch"

    def test_empty_snap_is_none(self):
        assert detector_score_embedder(_det("siglip"), {}) is None


class TestRegionGating:
    """``_score_all_media`` region max-pool is gated on the *patch* slot."""

    def _model(self) -> nn.Module:
        # Linear that fires on e1 (media 1's text vector) and not on e3 (the
        # shared patch region), so the chosen space changes which media wins.
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(DIM, 1)).eval()
        with torch.no_grad():
            m[0].weight.copy_(torch.tensor([[10.0, 0.0, 0.0, -10.0]]))
            m[0].bias.zero_()
        return m

    def test_text_primary_scores_text_vectors_no_region(self):
        snap = _dual_snap()
        model = cast(nn.Sequential, self._model())
        # Scoring in the text space: media 1 (e1) beats media 2 (e2); the patch
        # tree is bypassed, so no best_region is produced.
        all_ids, scores, best_region = _score_all_media(model, snap, "siglip")
        assert set(all_ids) == {1, 2}
        s = dict(zip(all_ids, scores))
        assert s[1] > s[2]
        # All winning regions are 0 (single full-image row per media; no tree).
        assert set(best_region) == {0}

    def test_patch_primary_uses_region_tree(self):
        snap = _dual_snap()
        model = cast(nn.Sequential, self._model())
        # Scoring in the patch space: every media's region vector is e3, so the
        # region path runs (one region row per media) and the scores tie.
        all_ids, scores, _best = _score_all_media(model, snap, "dinov3_patch")
        assert set(all_ids) == {1, 2}
        assert abs(scores[0] - scores[1]) < 1e-6

    def test_none_embedder_is_legacy_region_detection(self):
        # No explicit embedder → any patch_regions takes the region path
        # (byte-for-byte the pre-per-detector behaviour).
        snap = _dual_snap()
        model = cast(nn.Sequential, self._model())
        all_ids, scores, _best = _score_all_media(model, snap)
        assert set(all_ids) == {1, 2}
        assert abs(scores[0] - scores[1]) < 1e-6


class TestResolveDetectorPrimary:
    """Create-time resolution / validation against the active dataset."""

    def _activate(self, embedder_names: list[str]):
        from vtscore.state.core import DatasetContext, thread_dataset_context

        ctx = DatasetContext("ds-primary-test")
        ctx.medias[1] = {
            "id": 1,
            "media_type": "image",
            "embedder": embedder_names[0] if embedder_names else "",
            "embeddings": {name: _basis(0) for name in embedder_names},
        }
        return thread_dataset_context(ctx)

    def test_single_embedder_auto_resolves(self):
        from vtscore.detectors.primary_embedder import resolve_detector_primary

        with self._activate(["clap"]):
            primary, err = resolve_detector_primary("")
        assert err is None
        assert primary == "clap"

    def test_multi_embedder_requires_pick(self):
        from vtscore.detectors.primary_embedder import resolve_detector_primary

        with self._activate(["siglip", "dinov3_patch"]):
            primary, err = resolve_detector_primary("")
        assert primary == ""
        assert err is not None and "multiple" in err.lower()

    def test_explicit_pick_validated(self):
        from vtscore.detectors.primary_embedder import resolve_detector_primary

        with self._activate(["siglip", "dinov3_patch"]):
            primary, err = resolve_detector_primary("dinov3_patch")
            assert err is None and primary == "dinov3_patch"

            bad, bad_err = resolve_detector_primary("not_bound")
        assert bad == ""
        assert bad_err is not None and "not bound" in bad_err.lower()

    def test_no_active_dataset_leaves_empty(self):
        from vtscore.detectors.primary_embedder import resolve_detector_primary

        # No active dataset → can't validate, leave empty (resolved at first train).
        primary, err = resolve_detector_primary("")
        assert primary == "" and err is None
