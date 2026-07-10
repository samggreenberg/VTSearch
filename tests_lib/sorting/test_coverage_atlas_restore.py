"""Tests for restoring a cached coverage atlas onto a dataset context.

Covers :func:`vtscore.state.coverage.restore_coverage_atlas_from_cache`, the
helper the reload path uses to skip the hierarchical k-means rebuild when a
dataset pickle already carries a matching atlas snapshot.
"""

import numpy as np

from vtscore.state.core import DatasetContext
from vtscore.state.coverage import restore_coverage_atlas_from_cache
from vtscore.state.coverage_atlas import CoverageAtlas


def _ctx_with_vectors(n, seed=7):
    rng = np.random.default_rng(seed)
    vecs = {i: rng.standard_normal(16).astype(np.float32) for i in range(n)}
    ctx = DatasetContext(f"ds-{seed}-{n}")
    ctx.medias = {i: {"embedding": vecs[i]} for i in range(n)}
    return ctx, vecs


def test_restore_adopts_matching_cache():
    ctx, vecs = _ctx_with_vectors(100)
    snap = CoverageAtlas(vecs, k=3, min_node_size=10).to_serializable()

    assert restore_coverage_atlas_from_cache(ctx, snap) is True
    assert ctx.coverage_atlas is not None
    assert ctx.coverage_atlas.vector_to_leaf.keys() == ctx.medias.keys()


def test_restore_rejected_when_cids_differ():
    """A media set that shifted since the cache was written must rebuild."""
    ctx, vecs = _ctx_with_vectors(100)
    snap = CoverageAtlas(vecs, k=3, min_node_size=10).to_serializable()

    # Drop one media so the loaded set no longer matches the cached vectors.
    ctx.medias.pop(next(iter(ctx.medias)))
    assert restore_coverage_atlas_from_cache(ctx, snap) is False
    assert ctx.coverage_atlas is None


def test_restore_handles_missing_and_garbage_cache():
    ctx, _ = _ctx_with_vectors(20)
    assert restore_coverage_atlas_from_cache(ctx, None) is False
    assert restore_coverage_atlas_from_cache(ctx, {}) is False
    assert restore_coverage_atlas_from_cache(ctx, {"format": 999}) is False
    assert restore_coverage_atlas_from_cache(ctx, "junk") is False
    assert ctx.coverage_atlas is None


def test_restore_rejects_old_diversity_tree_cache():
    """A pre-migration diversity-tree snapshot (format 1) falls through to a rebuild."""
    ctx, _ = _ctx_with_vectors(20)
    old_snap = {
        "format": 1,
        "k": 3,
        "max_depth": 10,
        "min_node_size": 20,
        "nodes": {},
        "vector_to_leaf": {},
        "nodes_by_depth": {},
    }
    assert restore_coverage_atlas_from_cache(ctx, old_snap) is False
    assert ctx.coverage_atlas is None


def test_restored_atlas_starts_with_clean_evidence_state():
    ctx, vecs = _ctx_with_vectors(80)
    snap = CoverageAtlas(vecs, k=3, min_node_size=10).to_serializable()

    assert restore_coverage_atlas_from_cache(ctx, snap) is True
    # No votes replayed yet: the restored atlas must look unlabeled.
    assert ctx.coverage_atlas.labeled_ids == set()
    assert all(n["n_pos"] == 0 and n["n_neg"] == 0 for n in ctx.coverage_atlas.nodes.values())
