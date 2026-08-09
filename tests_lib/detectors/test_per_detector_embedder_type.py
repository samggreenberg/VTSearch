"""Per-detector embedder *type* routing (docs/plans/patch-embedder.md).

A detector locks one embedder **type** - ``semantic`` / ``patch_semantic`` /
``structural`` - and trains/scores in whichever concrete embedder of that type
the active dataset binds, independent of the dataset-level score precedence.
These cover the pure resolvers that route training/scoring through that choice:

* ``embedder_type`` / ``embedder_of_type`` / ``dataset_supplied_types`` /
  ``detector_dataset_compatible`` - the type taxonomy.
* ``keying_embedder_for_snap`` - the dataset's concrete embedder of the
  detector's type when the snap supplies it, else the dataset score precedence
  (the legacy / cross-dataset-portability fallback).
* ``detector_score_embedder`` - the scoring-space name handed to the matrix
  layer (delegates to keying).
* ``_score_all_media`` region gating - region max-pool only when the detector
  scores in the dataset's *patch* slot.
* ``_build_vote_xy`` / ``populate_label_embeddings`` **training** gating - the
  Bad-vote flood and the Good-vote region pool are gated the same way, so a
  detector scoring in a non-patch space never trains on rows it will never be
  scored over (#2935).
* ``resolve_detector_embedder_type`` - create-time resolution: an explicit type
  (the user's declared intent) is accepted regardless of the active dataset's
  bound embedders; an empty request auto-resolves a single-type dataset and is
  rejected on a multi-type one.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from vtscore.detectors.training import (
    _build_vote_xy,
    _score_all_media,
    detector_score_embedder,
)
from vtscore.embedding.binding import (
    dataset_supplied_types,
    detector_dataset_compatible,
    embedder_of_type,
    embedder_type,
    keying_embedder_for_snap,
)

DIM = 4


def _basis(i: int) -> np.ndarray:
    return np.eye(DIM, dtype=np.float32)[i]


def _dual_snap() -> dict[int, dict]:
    """Two image medias bound to both a semantic (siglip) and a patch (dinov3) embedder.

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
            "patch_grid": _basis(3)[None, None, :].astype(np.float16),
        }
    return snap


def _det(embedder_type: str = "") -> SimpleNamespace:
    return SimpleNamespace(embedder_type=embedder_type, embedder="")


def _ragged_snap() -> dict[int, dict]:
    """A dual-embedder snap whose two spaces have **different** dimensions.

    siglip is 4-d, the dinov3 patch slot (and its grid) 6-d, so mixing rows
    from the two spaces into one ``X_list`` cannot be papered over - it raises
    at ``np.stack``.  Both medias carry an md5 so the labelset path can resolve
    them in-dataset.
    """
    snap: dict[int, dict] = {}
    for cid in (1, 2):
        semantic = np.zeros(DIM, dtype=np.float32)
        semantic[cid] = 1.0
        patch = np.zeros(6, dtype=np.float32)
        patch[cid] = 1.0
        snap[cid] = {
            "id": cid,
            "md5": f"md5-{cid}",
            "media_type": "image",
            "embedder": "dinov3_patch",
            "embeddings": {"siglip": semantic, "dinov3_patch": patch},
            "patch_grid": np.zeros((1, 2, 6), dtype=np.float16),
        }
    return snap


class TestEmbedderTypeTaxonomy:
    def test_partition(self):
        # structural ▸ patch ▸ semantic; each registered embedder lands in one.
        assert embedder_type("siglip") == "semantic"
        assert embedder_type("clip") == "semantic"
        assert embedder_type("dinov3_patch") == "patch_semantic"
        assert embedder_type("dinov2_patch") == "patch_semantic"
        assert embedder_type("sift_vlad") == "structural"

    def test_unknown_and_empty_have_no_type(self):
        assert embedder_type("") == ""
        assert embedder_type("not_a_real_embedder") == ""

    def test_embedder_of_type(self):
        names = ["siglip", "dinov3_patch"]
        assert embedder_of_type(names, "semantic") == "siglip"
        assert embedder_of_type(names, "patch_semantic") == "dinov3_patch"
        # Nothing structural is bound here.
        assert embedder_of_type(names, "structural") is None
        assert embedder_of_type(names, "") is None

    def test_dataset_supplied_types(self):
        assert dataset_supplied_types(["siglip", "dinov3_patch"]) == {"semantic", "patch_semantic"}
        # Two semantic embedders collapse to a single supplied type.
        assert dataset_supplied_types(["siglip", "clip"]) == {"semantic"}
        assert dataset_supplied_types([]) == set()

    def test_detector_dataset_compatible(self):
        # SigLIP-trained (semantic) detector is compatible with a CLIP dataset.
        assert detector_dataset_compatible("semantic", ["clip"]) is True
        # ...but not with a structural-only dataset.
        assert detector_dataset_compatible("semantic", ["sift_vlad"]) is False
        # A patch detector runs on any patch dataset (DinoV2 ↔ DinoV3).
        assert detector_dataset_compatible("patch_semantic", ["dinov2_patch"]) is True
        # A legacy/typeless detector is always compatible (resolved at train).
        assert detector_dataset_compatible("", ["sift_vlad"]) is True


class TestKeyingEmbedder:
    def test_concrete_of_type_used_when_snap_supplies_it(self):
        snap = _dual_snap()
        # The detector's semantic type → the dataset's semantic embedder (siglip).
        assert keying_embedder_for_snap(_det("semantic"), snap) == "siglip"
        # The patch type → the patch embedder.
        assert keying_embedder_for_snap(_det("patch_semantic"), snap) == "dinov3_patch"

    def test_falls_back_to_precedence_when_type_absent(self):
        snap = _dual_snap()
        # No structural embedder is bound → precedence (structural ▸ patch ▸
        # semantic) names the patch slot.  (The scoring route gates this pair.)
        assert keying_embedder_for_snap(_det("structural"), snap) == "dinov3_patch"

    def test_no_type_is_precedence(self):
        snap = _dual_snap()
        assert keying_embedder_for_snap(_det(""), snap) == "dinov3_patch"
        assert keying_embedder_for_snap(None, snap) == "dinov3_patch"

    def test_empty_snap_is_empty_string(self):
        assert keying_embedder_for_snap(_det("semantic"), {}) == ""
        assert keying_embedder_for_snap(_det("semantic"), None) == ""

    def test_siglip_to_clip_reembeds_in_clip(self):
        # A semantic detector trained on siglip, now on a clip-only dataset:
        # keying returns clip, so the cache-space compare invalidates and the
        # labelset re-embeds in clip - the same-type portability.
        clip_snap = {1: {"id": 1, "embedder": "clip", "embeddings": {"clip": _basis(0)}}}
        assert keying_embedder_for_snap(_det("semantic"), clip_snap) == "clip"


class TestDetectorScoreEmbedder:
    def test_delegates_to_keying(self):
        snap = _dual_snap()
        assert detector_score_embedder(_det("semantic"), snap) == "siglip"
        assert detector_score_embedder(_det("patch_semantic"), snap) == "dinov3_patch"

    def test_empty_snap_is_none(self):
        assert detector_score_embedder(_det("semantic"), {}) is None


class TestRegionGating:
    """``_score_all_media`` region max-pool is gated on the *patch* slot.

    Takes a concrete embedder *name* (the keyer already resolved the type), so
    this layer is unchanged by the type model.
    """

    def _model(self) -> nn.Module:
        torch.manual_seed(0)
        linear = nn.Linear(DIM, 1)
        with torch.no_grad():
            # Fires on e1 (dim 1, media 1's semantic vector); zero on e2 (dim 2)
            # and e3 (dim 3, the shared patch region) → the scored space decides.
            linear.weight.copy_(torch.tensor([[0.0, 10.0, 0.0, 0.0]]))
            linear.bias.zero_()
        return nn.Sequential(linear).eval()

    def test_semantic_space_scores_full_image_no_region(self):
        snap = _dual_snap()
        model = cast(nn.Sequential, self._model())
        all_ids, scores, best_region = _score_all_media(model, snap, "siglip")
        assert set(all_ids) == {1, 2}
        s = dict(zip(all_ids, scores))
        assert s[1] > s[2]
        assert set(best_region) == {0}

    def test_patch_space_uses_region_tree(self):
        snap = _dual_snap()
        model = cast(nn.Sequential, self._model())
        all_ids, scores, _best = _score_all_media(model, snap, "dinov3_patch")
        assert set(all_ids) == {1, 2}
        assert abs(scores[0] - scores[1]) < 1e-6

    def test_none_embedder_is_legacy_region_detection(self):
        snap = _dual_snap()
        model = cast(nn.Sequential, self._model())
        all_ids, scores, _best = _score_all_media(model, snap)
        assert set(all_ids) == {1, 2}
        assert abs(scores[0] - scores[1]) < 1e-6


class TestVoteTrainingGating:
    """``_build_vote_xy`` trains under the gate ``_score_all_media`` scores under.

    Regression for #2935: the training side used to flood every Bad vote with
    the media's patch stack (and pool a boxed Good vote out of the grid)
    regardless of which space the detector actually scores in, so a
    semantic-locked detector on a text+patch dataset trained on patch-space
    vectors it would never be scored over.
    """

    def test_semantic_detector_gets_one_negative_row_per_bad_vote(self):
        snap = _dual_snap()
        X, y, groups, score_rows = _build_vote_xy(snap, {1: None}, {2: None}, {}, "siglip")
        # No flood: one row per vote, each a siglip image-level vector.
        assert y == [1.0, 0.0]
        assert groups == [("g", 1), ("b", 2)]
        np.testing.assert_allclose(X[0], _basis(1))
        np.testing.assert_allclose(X[1], _basis(2))
        # The recorded calibration stacks are the rows the scorer pools for a
        # siglip detector: the single full-image row, not the patch grid.
        assert set(score_rows) == {("g", 1), ("b", 2)}
        for rows in score_rows.values():
            assert rows.shape == (1, DIM)

    def test_patch_detector_still_floods(self):
        snap = _dual_snap()
        X, y, groups, score_rows = _build_vote_xy(snap, {1: None}, {2: None}, {}, "dinov3_patch")
        # Image-level row + the grid's single cell = 2 negative rows, one bag.
        assert y == [1.0, 0.0, 0.0]
        assert groups == [("g", 1), ("b", 2), ("b", 2)]
        assert len(X) == 3
        assert score_rows[("b", 2)].shape == (2, DIM)

    def test_none_embedder_keeps_dataset_precedence(self):
        # The pre-per-detector path: any media with a grid floods, matching
        # ``_score_all_media(model, snap)`` with no explicit primary.
        snap = _dual_snap()
        _X, y, groups, _rows = _build_vote_xy(snap, {1: None}, {2: None}, {}, None)
        assert y == [1.0, 0.0, 0.0]
        assert groups == [("g", 1), ("b", 2), ("b", 2)]

    def test_semantic_detector_ignores_a_region_box(self):
        # The box designates a patch of the *patch* embedder's grid; a siglip
        # detector has no such row, so it trains on its own image-level vector.
        snap = _dual_snap()
        X, _y, _groups, _rows = _build_vote_xy(snap, {1: None}, {2: None}, {1: (0.0, 0.0, 0.5, 0.5)}, "siglip")
        np.testing.assert_allclose(X[0], _basis(1))

    def test_ragged_spaces_no_longer_blow_up_at_stack(self):
        # Pre-fix this mixed 4-d Good row with 6-d flooded Bad rows, so the
        # trainer's ``np.stack`` raised ValueError and every learned sort 500'd.
        snap = _ragged_snap()
        X, y, groups, _rows = _build_vote_xy(snap, {1: None}, {2: None}, {1: (0.0, 0.0, 0.5, 0.5)}, "siglip")
        assert y == [1.0, 0.0]
        assert groups == [("g", 1), ("b", 2)]
        assert np.stack(X).shape == (2, DIM)


class TestLabelsetTrainingGating:
    """The labelset (load-time / cross-dataset) path is gated identically.

    ``populate_label_embeddings`` resolves each element in-dataset when it can;
    without the gate a semantic detector's Bad element read the media's stored
    ``patch_grid`` - the patch embedder's space - into its negative cache.
    """

    def _labelset(self):
        from vtscore.datasets.labelset import LabeledElement, LabelSet

        good = LabeledElement(md5="md5-1", label="good")
        bad = LabeledElement(md5="md5-2", label="bad")
        return LabelSet([good, bad]), good, bad

    def _populate(self, embedder_type: str):
        from vtscore.detectors.labelset_elements import stable_element_id
        from vtscore.detectors.labelset_training import populate_label_embeddings
        from vtscore.state.core import DetectorContext

        snap = _ragged_snap()
        labelset, good, bad = self._labelset()
        det_ctx = DetectorContext("d-2935")
        det_ctx.embedder_type = embedder_type
        populate_label_embeddings(det_ctx, labelset, media_type="image", snap=snap)
        return det_ctx, stable_element_id(good), stable_element_id(bad)

    def test_semantic_detector_caches_no_patch_rows(self):
        det_ctx, gid, bid = self._populate("semantic")
        assert det_ctx.label_negative_regions == {}
        assert det_ctx.label_score_regions == {}
        # Both labels are the dataset's siglip vectors, so training stacks.
        assert det_ctx.label_embeddings[gid].shape == (DIM,)
        assert det_ctx.label_embeddings[bid].shape == (DIM,)

    def test_patch_detector_still_caches_the_flood(self):
        det_ctx, _gid, bid = self._populate("patch_semantic")
        # Image-level row + the grid's 2 cells, in the 6-d patch space.
        assert len(det_ctx.label_negative_regions[bid]) == 3
        assert det_ctx.label_negative_regions[bid][0].shape == (6,)


class TestResolveDetectorEmbedderType:
    """Create-time resolution / validation against the active dataset's types."""

    def _activate(self, embedder_names: list[str]):
        from vtscore.state.core import DatasetContext, thread_dataset_context

        ctx = DatasetContext("ds-type-test")
        ctx.medias[1] = {
            "id": 1,
            "media_type": "image",
            "embedder": embedder_names[0] if embedder_names else "",
            "embeddings": {name: _basis(0) for name in embedder_names},
        }
        return thread_dataset_context(ctx)

    def test_single_type_auto_resolves(self):
        from vtscore.detectors.embedder_type import resolve_detector_embedder_type

        with self._activate(["clap"]):
            resolved, err = resolve_detector_embedder_type("")
        assert err is None
        assert resolved == "semantic"

    def test_two_semantic_embedders_still_single_type(self):
        from vtscore.detectors.embedder_type import resolve_detector_embedder_type

        # siglip + clip are both semantic → one supplied type → auto-resolve,
        # no "must choose" (the SigLIP↔CLIP interchangeability).
        with self._activate(["siglip", "clip"]):
            resolved, err = resolve_detector_embedder_type("")
        assert err is None
        assert resolved == "semantic"

    def test_multi_type_requires_pick(self):
        from vtscore.detectors.embedder_type import resolve_detector_embedder_type

        with self._activate(["siglip", "dinov3_patch"]):
            resolved, err = resolve_detector_embedder_type("")
        assert resolved == ""
        assert err is not None and "multiple" in err.lower()

    def test_explicit_type_accepted(self):
        from vtscore.detectors.embedder_type import resolve_detector_embedder_type

        with self._activate(["siglip", "dinov3_patch"]):
            resolved, err = resolve_detector_embedder_type("patch_semantic")
            assert err is None and resolved == "patch_semantic"

            # An explicit, valid type the dataset doesn't bind is still accepted:
            # the type is the user's declared intent (a detector can be created
            # before any dataset exists), and it simply gates as incompatible on
            # datasets that don't supply it.
            other, other_err = resolve_detector_embedder_type("structural")
        assert other == "structural"
        assert other_err is None

    def test_unknown_type_rejected(self):
        from vtscore.detectors.embedder_type import resolve_detector_embedder_type

        with self._activate(["siglip"]):
            resolved, err = resolve_detector_embedder_type("nonsense")
        assert resolved == ""
        assert err is not None and "unknown" in err.lower()

    def test_explicit_type_accepted_without_dataset(self):
        from vtscore.detectors.embedder_type import resolve_detector_embedder_type

        # No active dataset: an explicit type is the user's declared intent and
        # is persisted as-is (resolved against whatever dataset is used later).
        with self._activate([]):
            resolved, err = resolve_detector_embedder_type("structural")
        assert resolved == "structural"
        assert err is None

    def test_concrete_name_is_classified(self):
        from vtscore.detectors.embedder_type import resolve_detector_embedder_type

        # A client that still sends a concrete embedder name gets it classified.
        with self._activate(["siglip", "dinov3_patch"]):
            resolved, err = resolve_detector_embedder_type("dinov3_patch")
        assert err is None and resolved == "patch_semantic"

    def test_no_bound_embedders_leaves_empty(self):
        from vtscore.detectors.embedder_type import resolve_detector_embedder_type

        with self._activate([]):
            resolved, err = resolve_detector_embedder_type("")
        assert resolved == "" and err is None


class TestMigrationFromLegacyPrimary:
    def test_legacy_primary_name_classified_to_type(self):
        from vtscore.detectors.embedder_type import detector_embedder_type_from_data

        # New field wins.
        assert detector_embedder_type_from_data({"embedder_type": "structural"}) == "structural"
        # Legacy primary name migration-reads to its type.
        assert detector_embedder_type_from_data({"primary_embedder": "siglip"}) == "semantic"
        assert detector_embedder_type_from_data({"primary_embedder": "dinov3_patch"}) == "patch_semantic"
        # Neither → empty.
        assert detector_embedder_type_from_data({}) == ""
