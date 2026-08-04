"""The HAC-internal flood gap, and why the old rationale for it was wrong.

Issue #2731's "Related, smaller" note: ``bad_negative_vecs`` floods the CLS node
plus the HAC **leaves**, while ``_score_all_media`` max-pools **every** region
node - internals included.  So a Bad vote leaves scored rows it never trains
down directly.

The gap was justified by calling internals "saliency-weighted pools of those
leaves", i.e. redundant.  That reasoning does not hold.
:func:`~vtscore.media.patch_embed.build_hac_tree` sets ``merged_vec =
_l2_normalize(sum_a + sum_b)``: the convex-hull point of the descendants is
projected **back onto the unit sphere**, scaling it by ``1 / ||hull point||``.
Under the linear production head that scales the logit by the same factor, so
an internal node can out-score every one of its own leaves - training the
leaves down does not pull the internal down with them.

The flood stays leaves-only anyway - but for a measured reason, not that one:
flooding internals costs ranking (paired AP -0.058 +/- 0.036 over 24 synthetic
patch detectors, ``scripts/probe_hac_internal_flood.py``).  These tests pin the
*real* shape of the gap so the refuted "dominated by its leaves" argument
cannot quietly come back.
"""

import numpy as np
import torch
import torch.nn as nn

from vtscore.detectors.training import bad_negative_vecs, inference_score_rows
from vtscore.media.patch_embed import PatchEmbedOutput, build_region_tree


DIM = 16
GRID = 8
K = 6
EMB = "dinov3_patch"


def _unit(v, axis=-1):
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12)


def _patch_media(cid: int, rng) -> dict:
    """A media whose region tree comes from the production builder."""
    grid = _unit(rng.standard_normal((GRID, GRID, DIM)).astype(np.float32))
    saliency = rng.random((GRID, GRID)).astype(np.float32) + 0.2
    saliency /= saliency.sum()
    # A real CLS token is not the saliency-weighted patch mean - if it were, it
    # would coincide with the HAC root and the root would be flooded after all.
    pooled = (grid * saliency[..., None]).reshape(-1, DIM).sum(axis=0)
    cls = _unit(pooled + 0.8 * rng.standard_normal(DIM).astype(np.float32))
    output = PatchEmbedOutput(cls_vec=cls, patch_grid=grid, patch_saliency=saliency)
    return {
        "id": cid,
        "md5": f"md5-{cid:04x}",
        "media_type": "image",
        "embedder": EMB,
        "embeddings": {EMB: cls},
        "patch_regions": build_region_tree(output, k=K),
    }


def _descendant_leaves(regions, i: int) -> list[int]:
    node = regions[i]
    if node.children is None:
        return [i]
    a, b = node.children
    return _descendant_leaves(regions, a) + _descendant_leaves(regions, b)


def _internal_indices(regions) -> list[int]:
    return [i for i, r in enumerate(regions) if r.children is not None]


def _linear_head(direction):
    linear = nn.Linear(DIM, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(np.asarray(direction, dtype=np.float32)[None, :]))
        linear.bias.zero_()
    model = nn.Sequential(linear)
    model.eval()
    return model


class TestInternalsAreNotDominatedByTheirLeaves:
    def test_merged_vectors_are_renormalised_off_the_leaf_hull(self):
        """Every internal node sits strictly *outside* its descendants' hull.

        The argument needs no reconstruction of the weighted sums: the leaves
        are unit vectors, so every point of their convex hull has norm <= 1 and
        the only hull points *at* norm 1 are the vertices themselves.  An
        internal node is unit-norm (``_l2_normalize`` at merge time) and equals
        no leaf, so it cannot be in the hull.  The unweighted leaf mean is
        reported as the illustrative size of that renormalisation gain.
        """
        regions = _patch_media(1, np.random.default_rng(0))["patch_regions"]
        internals = _internal_indices(regions)
        assert internals, "expected the tree to contain internal nodes"
        for i in internals:
            vec = np.asarray(regions[i].vec, dtype=np.float32)
            leaves = np.stack([np.asarray(regions[j].vec, dtype=np.float32) for j in _descendant_leaves(regions, i)])
            # Unit-norm, like its leaves...
            assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3
            assert np.allclose(np.linalg.norm(leaves, axis=1), 1.0, atol=1e-3)
            # ...but not one of them, hence not in their hull.
            assert float(np.abs(leaves - vec).sum(axis=1).min()) > 1e-3
            # The gain the renormalisation applies to the hull point.
            assert float(np.linalg.norm(leaves.mean(axis=0))) < 1.0 - 1e-6

    def test_an_internal_node_out_scores_every_one_of_its_leaves(self):
        """Refutes "internals are redundant with the leaves they pool".

        Read the internal node's own direction as the linear head - the head a
        detector learns when a Good vote snaps to that node.  The internal
        projects at 1.0 while every descendant leaf projects at its cosine to
        it, which is strictly less.  So no amount of training the leaves down
        constrains the internal.
        """
        regions = _patch_media(2, np.random.default_rng(1))["patch_regions"]
        for i in _internal_indices(regions):
            direction = np.asarray(regions[i].vec, dtype=np.float32)
            leaves = np.stack([np.asarray(regions[j].vec, dtype=np.float32) for j in _descendant_leaves(regions, i)])
            leaf_max = float((leaves @ direction).max())
            assert leaf_max < 1.0 - 1e-4
            assert float(direction @ direction) > leaf_max

    def test_internals_beat_their_leaves_under_generic_directions_too(self):
        """Not an artefact of choosing the node's own direction: it happens for
        plain random linear heads as well, just rarely."""
        rng = np.random.default_rng(2)
        directions = _unit(rng.standard_normal((64, DIM)).astype(np.float32), axis=1)
        exceed = 0
        for cid in range(4):
            regions = _patch_media(cid, np.random.default_rng(10 + cid))["patch_regions"]
            vecs = np.stack([np.asarray(r.vec, dtype=np.float32) for r in regions])
            proj = vecs @ directions.T
            for i in _internal_indices(regions):
                leaf_max = proj[_descendant_leaves(regions, i)].max(axis=0)
                exceed += int((proj[i] > leaf_max + 1e-6).sum())
        assert exceed > 0, "internals never beat their leaves - the redundancy argument would hold"


class TestTheFloodGapItself:
    def test_the_flood_omits_exactly_the_scored_internals(self):
        """``bad_negative_vecs`` vs ``inference_score_rows``: the shortfall is
        the internal nodes, no more and no less."""
        media = _patch_media(3, np.random.default_rng(3))
        regions = media["patch_regions"]
        flooded = np.stack(bad_negative_vecs(media, EMB))
        scored = inference_score_rows(media, EMB)

        assert scored is not None
        assert len(scored) == len(regions)
        uncovered = [i for i, row in enumerate(scored) if not any(np.allclose(row, f, atol=1e-6) for f in flooded)]
        assert uncovered, "expected the internals to be uncovered"
        assert set(uncovered) == set(_internal_indices(regions))
        assert len(flooded) == len(regions) - len(uncovered)

    def test_an_uncovered_internal_can_set_the_image_score(self):
        """The gap is reachable, not theoretical: pointing the head at an
        internal node makes that never-flooded row win the max-pool."""
        media = _patch_media(4, np.random.default_rng(4))
        regions = media["patch_regions"]
        target = _internal_indices(regions)[-1]
        model = _linear_head(np.asarray(regions[target].vec, dtype=np.float32) * 10.0)

        rows = inference_score_rows(media, EMB)
        with torch.no_grad():
            row_scores = torch.sigmoid(model(torch.from_numpy(rows))).numpy().ravel()
        assert int(row_scores.argmax()) == target

        flooded = np.stack(bad_negative_vecs(media, EMB))
        with torch.no_grad():
            flooded_max = float(torch.sigmoid(model(torch.from_numpy(flooded))).max())
        assert float(row_scores[target]) > flooded_max
