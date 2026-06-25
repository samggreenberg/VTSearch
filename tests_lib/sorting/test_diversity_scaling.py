"""Library-tier tests for the Phase 2 diversity-tree scalability changes.

Covers the two open Phase 2.1 pieces from ``docs/plans/scalability-plan.md``:

- **Part A** (smarter k-means defaults): ``_n_init_for`` scales restarts down
  for large nodes while leaving small (test-sized) nodes at the full
  ``_N_INIT``, and ``auto_max_depth`` never caps *below* the natural splitting
  depth - so trees over normal datasets are structurally unchanged.
- **Part B** (defer large builds): ``should_auto_build_diversity_tree`` gates
  the threshold and ``_build_diversity_tree_stage`` skips the build above it.
"""

from __future__ import annotations

import numpy as np

from vtscore.state.core import DatasetContext
from vtscore.state.diversity import (
    DIVERSITY_TREE_AUTO_THRESHOLD,
    should_auto_build_diversity_tree,
)
from vtscore.state.diversity_tree import (
    DIVERSITY_TREE_MAX_DEPTH,
    DIVERSITY_TREE_MIN_NODE_SIZE,
    DiversityTree,
    _MAX_LEAVES,
    _n_init_for,
    auto_max_depth,
)


def _make_vectors(n: int, dim: int = 32, seed: int = 42) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {i + 1: rng.standard_normal(dim).astype(np.float32) for i in range(n)}


# ---------------------------------------------------------------------------
# Part A: _n_init_for
# ---------------------------------------------------------------------------


class TestNInitFor:
    def test_small_nodes_keep_full_restarts(self):
        # <=1000 stays at the small-node default so test-sized trees are
        # bit-for-bit unchanged.
        assert _n_init_for(1) == 10
        assert _n_init_for(1_000) == 10

    def test_medium_nodes_get_fewer_restarts(self):
        assert _n_init_for(1_001) == 5
        assert _n_init_for(10_000) == 5

    def test_large_nodes_get_fewest_restarts(self):
        assert _n_init_for(10_001) == 3
        assert _n_init_for(1_000_000) == 3


# ---------------------------------------------------------------------------
# Part A: auto_max_depth
# ---------------------------------------------------------------------------


class TestAutoMaxDepth:
    def test_never_exceeds_global_cap(self):
        assert auto_max_depth(10_000_000, k=3) <= DIVERSITY_TREE_MAX_DEPTH

    def test_always_at_least_one(self):
        assert auto_max_depth(0) == DIVERSITY_TREE_MAX_DEPTH  # degenerate guard
        assert auto_max_depth(1, k=3) >= 1

    def test_leaf_count_bounded_for_huge_n(self):
        # The cap term keeps k**depth from blowing past the soft leaf ceiling
        # by more than one level.
        depth = auto_max_depth(10_000_000, k=3)
        assert 3 ** (depth - 1) <= _MAX_LEAVES

    def test_does_not_cap_below_natural_depth(self):
        # For a small dataset the natural splitting depth (where _build_node
        # already stops via min_node_size) is below the global cap, so the
        # tree is structurally identical to one built with the default depth.
        vectors = _make_vectors(300)
        capped = DiversityTree(
            vectors, k=3, max_depth=auto_max_depth(len(vectors), k=3)
        )
        full = DiversityTree(vectors, k=3, max_depth=DIVERSITY_TREE_MAX_DEPTH)
        assert capped.vector_to_leaf == full.vector_to_leaf
        assert set(capped.nodes) == set(full.nodes)


# ---------------------------------------------------------------------------
# Part B: should_auto_build_diversity_tree
# ---------------------------------------------------------------------------


class TestShouldAutoBuild:
    def test_below_threshold_builds(self):
        assert should_auto_build_diversity_tree(0)
        assert should_auto_build_diversity_tree(DIVERSITY_TREE_AUTO_THRESHOLD)

    def test_above_threshold_skips(self):
        assert not should_auto_build_diversity_tree(DIVERSITY_TREE_AUTO_THRESHOLD + 1)


# ---------------------------------------------------------------------------
# Part B: finalize stage honours the threshold
# ---------------------------------------------------------------------------


class _NullTracker:
    """Minimal tracker stand-in: never cancels, swallows updates."""

    def check_cancelled(self) -> None:  # pragma: no cover - trivial
        pass

    def update(self, *args, **kwargs) -> None:  # pragma: no cover - trivial
        pass


def _ctx_with_medias(n: int) -> DatasetContext:
    ctx = DatasetContext("_scaling_test")
    rng = np.random.default_rng(7)
    for i in range(n):
        ctx.medias[i] = {
            "id": i,
            "embeddings": {"siglip": rng.standard_normal(16).astype(np.float32)},
            "embedder": "siglip",
        }
    return ctx


class TestFinalizeStageThreshold:
    def test_builds_below_threshold(self, monkeypatch):
        from vtscore.datasets.stages import finalize

        monkeypatch.setattr(finalize, "should_auto_build_diversity_tree", lambda n: n <= 30)
        ctx = _ctx_with_medias(25)
        finalize._build_diversity_tree_stage(ctx, _NullTracker())
        assert ctx.diversity_tree is not None

    def test_skips_above_threshold(self, monkeypatch):
        from vtscore.datasets.stages import finalize

        monkeypatch.setattr(finalize, "should_auto_build_diversity_tree", lambda n: n <= 30)
        ctx = _ctx_with_medias(40)
        finalize._build_diversity_tree_stage(ctx, _NullTracker())
        assert ctx.diversity_tree is None

    def test_skip_leaves_existing_tree_untouched(self, monkeypatch):
        from vtscore.datasets.stages import finalize

        monkeypatch.setattr(finalize, "should_auto_build_diversity_tree", lambda n: False)
        ctx = _ctx_with_medias(10)
        sentinel = object()
        ctx.diversity_tree = sentinel  # type: ignore[assignment]
        finalize._build_diversity_tree_stage(ctx, _NullTracker())
        assert ctx.diversity_tree is sentinel
