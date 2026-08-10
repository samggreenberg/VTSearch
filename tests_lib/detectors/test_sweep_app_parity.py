"""Parity tests: the SOD sweep vs the app's label construction and scoring.

The sweep (``vtscore/eval/region_sources.py`` + ``vtscore/eval/scoring_heads.py``)
re-derives the region-voting flood/snap and the region max-pool rather than calling
the live detector directly, so its numbers only transfer to the app while those
re-derivations stay in lockstep with the production primitives in
``vtscore/detectors/training.py`` and ``vtscore/embedding/matrix.py``.

**Label parity is BROKEN as of #2886, deliberately, on the production side.** That
change made production tree-free: there is no longer a ``patch_regions`` HAC tree for
the sweep's label construction to agree with, and both primitives were rewritten around
raw patches instead.

============================  ==============================  ============================
vote                          sweep (HAC arm)                 production after #2886
============================  ==============================  ============================
Bad -> ``bad_negative_vecs``  CLS + the K HAC leaves          the image vector + EVERY raw
                              (``k + 1`` rows)                patch (``media_score_rows``)
Good -> ``pool_box_from_media``  the snapped HAC node         the single raw patch nearest
                                                              the box; ``None`` without a
                                                              ``patch_grid``
============================  ==============================  ============================

The three label-parity tests below are therefore ``xfail(strict=True)``. They are kept
rather than deleted because they are the executable record of the divergence, and
because ``strict`` makes them fail loudly as **unexpectedly passing** if production ever
regains a region tree - at which point the sweep's numbers would transfer again and this
docstring is what needs re-reading. The consequence to keep in mind meanwhile: **HAC-arm
sweep results no longer predict app behaviour.** ``vtscore/eval/_hac_compat.py`` explains
where the tree the sweep still measures now lives.

What remains genuinely locked:

* **Flood rule** — the sweep floods exactly the childless nodes (CLS + HAC leaves) and
  never the internal merge nodes. Sweep-side only, so #2886 does not touch it.
* **Scorer parity** — ``scoring_heads`` per-image max-pool equals production
  ``segmented_max_pool`` (score, first-wins region, and the non-finite sentinel). Still a
  live gate: the scorer is shared, only label construction diverged.

Library tier: every production symbol imported here is Flask-clean, so this lives
in ``tests_lib`` and runs under ``./run-tests.sh detectors``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.detectors.training import bad_negative_vecs, pool_box_from_media
from vtscore.embedding.matrix import segmented_max_pool
from vtscore.eval._hac_compat import build_region_tree
from vtscore.eval.region_sources import _covering_box, _PatchSource
from vtscore.eval.scoring_heads import max_pool_over_images, max_pool_with_argmax
from vtscore.media.patch_embed import PatchEmbedOutput
from vtscore.utils.scores import NON_FINITE_SCORE_SENTINEL

#: Why the three label-parity tests are expected to fail (see the module docstring).
#: ``strict`` so that production regaining a region tree fails the suite as an
#: unexpected pass rather than quietly restoring a gate nobody re-read.
_TREE_FREE = pytest.mark.xfail(
    strict=True,
    reason=(
        "#2886 made production tree-free: bad_negative_vecs now floods the image vector plus "
        "every raw patch, and pool_box_from_media returns the nearest raw patch (None without a "
        "patch_grid), so the HAC sweep's label construction no longer mirrors the app"
    ),
)

# The tree knobs the parity test builds with. They must match the values the
# stubbed _PatchSource carries, so region_sources' in-prepare build_region_tree
# and the reference tree below are byte-identical (build_region_tree is
# deterministic for a fixed PatchEmbedOutput + knobs).
_TREE_KW = dict(k=6, alpha=0.5, seeding="spread", leaf_assign="feature", leaf_beta=None, pca_dims=None)


def _make_pe(h: int = 6, w: int = 6, d: int = 16, seed: int = 0) -> PatchEmbedOutput:
    """A deterministic synthetic patch-embedder output (L2-normed grid + CLS)."""
    rng = np.random.default_rng(seed)
    grid = rng.standard_normal((h, w, d)).astype(np.float32)
    grid /= np.linalg.norm(grid, axis=-1, keepdims=True)
    cls = rng.standard_normal(d).astype(np.float32)
    cls /= np.linalg.norm(cls)
    sal = rng.random((h, w)).astype(np.float32)
    sal /= sal.sum()
    return PatchEmbedOutput(cls_vec=cls, patch_grid=grid, patch_saliency=sal)


def _stub_patch_source(pe: PatchEmbedOutput) -> _PatchSource:
    """A region-voting HAC ``_PatchSource`` whose forward is stubbed to *pe*.

    Built via ``__new__`` to skip ``__init__``'s live-embedder dimensionality
    probe, then the per-image forward is replaced so ``prepare()`` runs the real
    label-construction code (tree build + leaf_mask + snap) with no model.
    """
    src = _PatchSource.__new__(_PatchSource)
    src._emb = None
    src.name = "dino"
    src._k = _TREE_KW["k"]
    src._alpha = _TREE_KW["alpha"]
    src._pca_dims = _TREE_KW["pca_dims"]
    src._seeding = _TREE_KW["seeding"]
    src._leaf_assign = _TREE_KW["leaf_assign"]
    src._leaf_beta = _TREE_KW["leaf_beta"]
    src._proposer = None
    src._region_voting = True
    src.supports_text = False
    src.input_dim = int(pe.cls_vec.shape[0])
    src._patch_forward = lambda _image: pe  # instance attr shadows the method
    return src


class TestFloodParity:
    """The sweep's leaf_mask flood vs production bad_negative_vecs (diverged, #2886)."""

    @_TREE_FREE
    def test_leaf_mask_selects_bad_negative_vecs(self):
        pe = _make_pe(seed=1)
        prep = _stub_patch_source(pe).prepare(object())
        sweep_flood = np.asarray(prep.vecs)[np.asarray(prep.leaf_mask)]

        tree = build_region_tree(pe, **_TREE_KW)
        prod_flood = np.asarray(bad_negative_vecs({"patch_regions": tree}), dtype=np.float32)

        assert sweep_flood.shape == prod_flood.shape, "flood row count diverged from bad_negative_vecs"
        np.testing.assert_array_equal(sweep_flood, prod_flood)

    def test_flood_is_exactly_the_childless_nodes(self):
        # Guards the *rule*, not just the vectors: the flooded rows must be the
        # CLS node + HAC leaves (children is None), never the internal merge nodes.
        pe = _make_pe(seed=4)
        tree = build_region_tree(pe, **_TREE_KW)
        prep = _stub_patch_source(pe).prepare(object())
        n_childless = sum(1 for r in tree if r.children is None)
        assert int(np.asarray(prep.leaf_mask).sum()) == n_childless
        assert n_childless == _TREE_KW["k"] + 1  # K HAC leaves + the CLS full-image node


class TestSnapParity:
    """The sweep's snapped Good-vote positive vs pool_box_from_media (diverged, #2886)."""

    @_TREE_FREE
    def test_snapped_positive_matches_pool_box_from_media(self):
        pe = _make_pe(seed=2)
        box = (0.2, 0.2, 0.6, 0.6)
        prep = _stub_patch_source(pe).prepare(object(), gt_boxes=[box])
        assert prep.exemplars.shape[0] == 1, "region-voting must yield one snapped positive per image"

        tree = build_region_tree(pe, **_TREE_KW)
        cover = _covering_box([box])
        prod = pool_box_from_media({"patch_regions": tree}, cover)
        assert prod is not None
        np.testing.assert_array_equal(prep.exemplars[0], np.asarray(prod, dtype=np.float32))

    @_TREE_FREE
    def test_multi_instance_covers_all_boxes(self):
        # The covering box spans every instance box (the documented modeling
        # choice), so the snap is against that union, matching pool_box_from_media.
        pe = _make_pe(seed=3)
        boxes = [(0.05, 0.05, 0.2, 0.2), (0.7, 0.7, 0.95, 0.95)]
        prep = _stub_patch_source(pe).prepare(object(), gt_boxes=boxes)
        tree = build_region_tree(pe, **_TREE_KW)
        prod = pool_box_from_media({"patch_regions": tree}, _covering_box(boxes))
        assert prod is not None
        np.testing.assert_array_equal(prep.exemplars[0], np.asarray(prod, dtype=np.float32))


def _linear_predict(dim: int, seed: int):
    """A deterministic row scorer (dot with a fixed vector); enough to exercise
    max-pool + argmax without training an MLP."""
    w = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return lambda x: np.asarray(x, dtype=np.float32) @ w


def _reference_pool(predict, mats, *, sanitise: bool) -> tuple[np.ndarray, np.ndarray]:
    """Production segmented_max_pool over the same flattened rows as scoring_heads."""
    nonempty = [m for m in mats if m.shape[0] > 0]
    flat = np.concatenate(nonempty, axis=0)
    flat_scores = np.asarray(predict(flat), dtype=np.float64).reshape(-1)
    if sanitise:
        flat_scores = np.where(np.isfinite(flat_scores), flat_scores, NON_FINITE_SCORE_SENTINEL)
    media_idx = np.concatenate([np.full(m.shape[0], j, dtype=np.int64) for j, m in enumerate(nonempty)])
    region_idx = np.concatenate([np.arange(m.shape[0], dtype=np.int64) for m in nonempty])
    scores, best = segmented_max_pool(flat_scores, media_idx, region_idx, len(nonempty))
    return np.asarray(scores), np.asarray(best)


class TestScorerParity:
    """scoring_heads max-pool == production segmented_max_pool."""

    def test_scores_and_argmax_match(self):
        rng = np.random.default_rng(0)
        mats = [rng.standard_normal((n, 16)).astype(np.float32) for n in (5, 1, 8, 3)]
        predict = _linear_predict(16, seed=9)

        scores, argmax = max_pool_with_argmax(predict, mats)
        ref_scores, ref_best = _reference_pool(predict, mats, sanitise=False)

        np.testing.assert_allclose(scores, ref_scores)
        np.testing.assert_array_equal(argmax, ref_best)
        # max_pool_over_images must agree with the argmax variant's scores.
        np.testing.assert_allclose(max_pool_over_images(predict, mats), scores)

    def test_first_wins_tie_break(self):
        # Duplicate maxima within an image: both the sweep and production must pick
        # the FIRST (lowest-index) row that attains the max.
        rng = np.random.default_rng(1)
        mat = rng.standard_normal((6, 16)).astype(np.float32)
        mat[3] = mat[1]  # rows 1 and 3 will score identically -> row 1 must win
        predict = _linear_predict(16, seed=2)
        _, argmax = max_pool_with_argmax(predict, [mat])
        _, ref_best = _reference_pool(predict, [mat], sanitise=False)
        assert int(argmax[0]) == int(ref_best[0])

    def test_non_finite_rows_are_sentinelled(self):
        # A NaN/inf row score must not poison its image's pooled score: the sweep
        # replaces it with NON_FINITE_SCORE_SENTINEL before the max, exactly like
        # production's sigmoid_to_finite_array -> segmented_max_pool.
        mats = [np.ones((3, 16), np.float32), np.ones((2, 16), np.float32)]

        def predict(x):
            out = np.arange(x.shape[0], dtype=np.float64)
            out[0] = np.nan  # image 0, row 0
            out[3] = np.inf  # image 1, row 0
            return out

        scores, argmax = max_pool_with_argmax(predict, mats)
        ref_scores, ref_best = _reference_pool(predict, mats, sanitise=True)
        np.testing.assert_allclose(scores, ref_scores)
        np.testing.assert_array_equal(argmax, ref_best)
        # Image 0: rows score [sentinel, 1, 2] -> row 2 wins with 2.0.
        assert scores[0] == 2.0 and int(argmax[0]) == 2
        # Image 1: rows score [sentinel, 4] -> row 1 wins with 4.0.
        assert scores[1] == 4.0 and int(argmax[1]) == 1

    def test_all_non_finite_image_resolves_to_region_zero(self):
        # An all-sentinel image resolves to region 0, matching production.
        mats = [np.ones((3, 16), np.float32)]

        def predict(x):
            return np.full(x.shape[0], np.nan, dtype=np.float64)

        scores, argmax = max_pool_with_argmax(predict, mats)
        assert scores[0] == NON_FINITE_SCORE_SENTINEL
        assert int(argmax[0]) == 0
