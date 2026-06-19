"""Tests for restoring a cached diversity tree onto a dataset context.

Covers :func:`vtscore.state.diversity.restore_diversity_tree_from_cache`, the
helper the reload path uses to skip the hierarchical k-means rebuild when a
dataset pickle already carries a matching tree snapshot.
"""

import numpy as np

from vtscore.state.core import DatasetContext
from vtscore.state.diversity import restore_diversity_tree_from_cache
from vtscore.state.diversity_tree import DiversityTree


def _ctx_with_vectors(n, seed=7):
    rng = np.random.default_rng(seed)
    vecs = {i: rng.standard_normal(16).astype(np.float32) for i in range(n)}
    ctx = DatasetContext(f"ds-{seed}-{n}")
    ctx.medias = {i: {"embedding": vecs[i]} for i in range(n)}
    return ctx, vecs


def test_restore_adopts_matching_cache():
    ctx, vecs = _ctx_with_vectors(100)
    snap = DiversityTree(vecs, k=3, min_node_size=10).to_serializable()

    assert restore_diversity_tree_from_cache(ctx, snap) is True
    assert ctx.diversity_tree is not None
    assert ctx.diversity_tree.vector_to_leaf.keys() == ctx.medias.keys()


def test_restore_rejected_when_cids_differ():
    """A media set that shifted since the cache was written must rebuild."""
    ctx, vecs = _ctx_with_vectors(100)
    snap = DiversityTree(vecs, k=3, min_node_size=10).to_serializable()

    # Drop one media so the loaded set no longer matches the cached vectors.
    ctx.medias.pop(next(iter(ctx.medias)))
    assert restore_diversity_tree_from_cache(ctx, snap) is False
    assert ctx.diversity_tree is None


def test_restore_handles_missing_and_garbage_cache():
    ctx, _ = _ctx_with_vectors(20)
    assert restore_diversity_tree_from_cache(ctx, None) is False
    assert restore_diversity_tree_from_cache(ctx, {}) is False
    assert restore_diversity_tree_from_cache(ctx, {"format": 999}) is False
    assert restore_diversity_tree_from_cache(ctx, "junk") is False
    assert ctx.diversity_tree is None


def test_restored_tree_starts_with_clean_seen_state():
    ctx, vecs = _ctx_with_vectors(80)
    snap = DiversityTree(vecs, k=3, min_node_size=10).to_serializable()

    assert restore_diversity_tree_from_cache(ctx, snap) is True
    # No votes replayed yet: the restored tree must look unlabeled.
    assert ctx.diversity_tree.seen == set()
    assert ctx.diversity_tree.labeled_ids == set()
