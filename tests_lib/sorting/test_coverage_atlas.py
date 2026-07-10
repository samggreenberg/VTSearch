"""Tests for the CoverageAtlas evidence-aware partition structure."""

import numpy as np
import pytest

from vtscore.state.coverage_atlas import CoverageAtlas, domain_shift_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vectors(n, dim=32, seed=42):
    """Create n random vectors keyed by 1..n."""
    rng = np.random.RandomState(seed)
    return {i + 1: rng.randn(dim).astype(np.float32) for i in range(n)}


def _make_clustered_vectors(cluster_sizes, dim=32, spread=0.1, seed=42):
    """Create vectors in well-separated clusters for deterministic k-means.

    Each cluster is centered at a distinct point along different axes.
    Returns dict keyed by 1..N.
    """
    rng = np.random.RandomState(seed)
    vectors = {}
    idx = 1
    for ci, size in enumerate(cluster_sizes):
        center = np.zeros(dim, dtype=np.float32)
        center[ci % dim] = 10.0 * (ci + 1)
        for _ in range(size):
            vectors[idx] = center + rng.randn(dim).astype(np.float32) * spread
            idx += 1
    return vectors


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_vectors(self):
        atlas = CoverageAtlas({}, k=2)
        assert atlas.nodes == {}
        assert atlas.coverage_level() == 0
        assert atlas.next_sample() is None

    def test_single_vector(self):
        vecs = _make_vectors(1)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        assert len(atlas.nodes) == 1
        assert atlas.vector_to_leaf[1] == "0"
        assert 1 in atlas.nodes["0"]["ids"]

    def test_small_set_stays_single_node(self):
        vecs = _make_vectors(10)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        # 10 < 20 so no splitting
        assert len(atlas.nodes) == 1
        for vid in vecs:
            assert atlas.vector_to_leaf[vid] == "0"

    def test_splits_when_above_min_size(self):
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        # Should have at least root + 2 children
        assert len(atlas.nodes) >= 3
        assert "0" in atlas.nodes
        assert atlas.nodes["0"]["children"]

    def test_all_vectors_assigned_to_leaves(self):
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=10)
        for vid in vecs:
            assert vid in atlas.vector_to_leaf
            leaf = atlas.vector_to_leaf[vid]
            assert leaf in atlas.nodes
            assert vid in atlas.nodes[leaf]["ids"]

    def test_disjoint_partitioning(self):
        """Every vector appears in exactly one leaf node."""
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        leaf_assignments = {}
        for name, node in atlas.nodes.items():
            if not node["children"]:  # leaf
                for vid in node["ids"]:
                    assert vid not in leaf_assignments, f"Vector {vid} in multiple leaves"
                    leaf_assignments[vid] = name
        assert set(leaf_assignments.keys()) == set(vecs.keys())

    def test_respects_max_depth(self):
        vecs = _make_vectors(500)
        atlas = CoverageAtlas(vecs, k=2, max_depth=3, min_node_size=2)
        for node in atlas.nodes.values():
            assert node["depth"] <= 3

    def test_node_has_ids(self):
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=10)
        for name, node in atlas.nodes.items():
            assert len(node["ids"]) > 0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be between 2 and 9"):
            CoverageAtlas(_make_vectors(10), k=1)
        with pytest.raises(ValueError, match="k must be between 2 and 9"):
            CoverageAtlas(_make_vectors(10), k=10)

    def test_invalid_max_depth_raises(self):
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            CoverageAtlas(_make_vectors(10), k=2, max_depth=0)

    def test_invalid_min_node_size_raises(self):
        with pytest.raises(ValueError, match="min_node_size must be >= 1"):
            CoverageAtlas(_make_vectors(10), k=2, min_node_size=0)

    def test_parent_child_consistency(self):
        vecs = _make_vectors(200)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=10)
        for name, node in atlas.nodes.items():
            for child_name in node["children"]:
                assert child_name in atlas.nodes
                assert atlas.nodes[child_name]["parent"] == name
        # Root has no parent
        assert atlas.nodes["0"]["parent"] is None

    def test_node_naming_convention(self):
        """Children names are parent name + single digit."""
        vecs = _make_vectors(200)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=10)
        for name, node in atlas.nodes.items():
            for child_name in node["children"]:
                assert child_name.startswith(name)
                assert len(child_name) == len(name) + 1

    def test_nodes_by_depth_consistent(self):
        vecs = _make_vectors(200)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        for depth, names in atlas.nodes_by_depth.items():
            for name in names:
                assert atlas.nodes[name]["depth"] == depth

    def test_children_ordered_largest_first(self):
        """Siblings are stored biggest-first so BFS covers big regions first."""
        vecs = _make_vectors(300)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=10)
        for node in atlas.nodes.values():
            sizes = [atlas.nodes[c]["n"] for c in node["children"]]
            assert sizes == sorted(sizes, reverse=True)

    def test_node_ids_sorted_most_typical_first(self):
        """ids[0] is the node's representative (max mu . x)."""
        vecs = _make_vectors(60)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        root = atlas.nodes["0"]
        queries = np.stack([vecs[i] for i in root["ids"]]).astype(np.float32)
        centered = queries - atlas.center
        centered /= np.linalg.norm(centered, axis=1, keepdims=True)
        t = centered @ root["mu"]
        assert np.all(np.diff(t) <= 1e-6), "ids must be ordered by descending typicality"

    def test_node_moments_present(self):
        vecs = _make_vectors(120)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=10)
        for node in atlas.nodes.values():
            assert node["n"] == len(node["ids"])
            assert node["mu"].shape == (32,)
            assert abs(float(np.linalg.norm(node["mu"])) - 1.0) < 1e-3
            assert 0.0 <= node["rbar"] <= 1.0 + 1e-6
            assert len(node["t_quantiles"]) == 21


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_lookup_returns_leaf(self):
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        for vid in vecs:
            leaf = atlas.lookup(vid)
            assert leaf in atlas.nodes
            # Leaf nodes have no children
            assert not atlas.nodes[leaf]["children"]

    def test_lookup_missing_id_raises(self):
        vecs = _make_vectors(10)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        with pytest.raises(KeyError):
            atlas.lookup(9999)


# ---------------------------------------------------------------------------
# Evidence channels
# ---------------------------------------------------------------------------


def _covered(atlas, name):
    node = atlas.nodes[name]
    return node["n_pos"] + node["n_neg"] > 0


class TestEvidence:
    def test_label_marks_leaf_and_ancestors(self):
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        vid = 1
        atlas.label(vid, good=True)
        leaf = atlas.lookup(vid)
        # Walk from leaf to root; all should carry positive evidence
        node = leaf
        while node is not None:
            assert atlas.nodes[node]["n_pos"] == 1
            node = atlas.nodes[node]["parent"]

    def test_initially_no_evidence(self):
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        for node in atlas.nodes.values():
            assert node["n_pos"] == 0
            assert node["n_neg"] == 0

    def test_label_tracks_labeled_ids(self):
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        atlas.label(1, good=True)
        atlas.label(5, good=False)
        assert atlas.labeled_ids == {1, 5}

    def test_label_idempotent(self):
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        atlas.label(1, good=True)
        atlas.label(1, good=True)
        assert atlas.nodes["0"]["n_pos"] == 1
        assert atlas.labeled_ids == {1}

    def test_relabel_moves_evidence_between_channels(self):
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        atlas.label(1, good=True)
        assert atlas.nodes["0"]["n_pos"] == 1
        assert atlas.nodes["0"]["n_neg"] == 0
        atlas.label(1, good=False)
        assert atlas.nodes["0"]["n_pos"] == 0
        assert atlas.nodes["0"]["n_neg"] == 1

    def test_per_class_counts_accumulate(self):
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        atlas.label(1, good=True)
        atlas.label(2, good=True)
        atlas.label(3, good=False)
        assert atlas.nodes["0"]["n_pos"] == 2
        assert atlas.nodes["0"]["n_neg"] == 1


# ---------------------------------------------------------------------------
# Unlabeling
# ---------------------------------------------------------------------------


class TestUnlabeling:
    def test_unlabel_single_removes_evidence(self):
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        atlas.label(1, good=True)
        assert _covered(atlas, "0")
        atlas.unlabel(1)
        assert not any(_covered(atlas, name) for name in atlas.nodes)
        assert atlas.labeled_ids == set()

    def test_unlabel_preserves_sibling_evidence(self):
        """If another labeled vector shares an ancestor, ancestor stays covered."""
        vecs = _make_clustered_vectors([30, 30], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)

        # Label one vector from each cluster child
        v1 = 1  # cluster 0
        v2 = 31  # cluster 1
        atlas.label(v1, good=True)
        atlas.label(v2, good=False)

        # Root should carry evidence from both
        assert atlas.nodes["0"]["n_pos"] == 1
        assert atlas.nodes["0"]["n_neg"] == 1

        atlas.unlabel(v1)

        # Root should still be covered (v2 keeps the other branch alive)
        assert _covered(atlas, "0")
        assert atlas.nodes["0"]["n_pos"] == 0
        assert atlas.nodes["0"]["n_neg"] == 1

    def test_unlabel_same_leaf_keeps_coverage(self):
        """If another labeled vector is in the same leaf, leaf stays covered."""
        vecs = _make_vectors(10, dim=8)  # Small, all in root
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        # All in root (single leaf)
        atlas.label(1, good=True)
        atlas.label(2, good=True)
        atlas.unlabel(1)
        assert _covered(atlas, "0")  # Vector 2 still keeps root covered
        atlas.unlabel(2)
        assert not _covered(atlas, "0")

    def test_unlabel_without_label_is_noop(self):
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        atlas.unlabel(1)  # Never labeled, should not error
        assert not _covered(atlas, "0")


# ---------------------------------------------------------------------------
# Bulk reset (used when votes are cleared or detector swaps)
# ---------------------------------------------------------------------------


class TestResetEvidence:
    def test_reset_evidence_clears_everything(self):
        vecs = _make_clustered_vectors([30, 30], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        atlas.label(1, good=True)
        atlas.label(31, good=False)
        assert len(atlas.labeled_ids) == 2

        atlas.reset_evidence()
        assert atlas.labeled_ids == set()
        assert atlas.coverage_level() == 0
        for node in atlas.nodes.values():
            assert node["n_pos"] == 0
            assert node["n_neg"] == 0

    def test_reset_then_relabel_picks_up_new_state(self):
        """After reset, label calls reproduce the evidence state from scratch."""
        vecs = _make_clustered_vectors([30, 30], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        atlas.label(1, good=True)
        leaf1 = atlas.lookup(1)
        assert _covered(atlas, leaf1)

        atlas.reset_evidence()
        # Label a different vector; only its leaf+ancestors should be covered.
        atlas.label(31, good=True)
        leaf31 = atlas.lookup(31)
        assert _covered(atlas, leaf31)
        if leaf1 != leaf31:
            assert not _covered(atlas, leaf1)

    def test_reset_on_empty_atlas_is_noop(self):
        atlas = CoverageAtlas({}, k=2)
        atlas.reset_evidence()  # must not raise
        assert atlas.labeled_ids == set()


# ---------------------------------------------------------------------------
# Coverage level
# ---------------------------------------------------------------------------


class TestCoverageLevel:
    def test_empty_atlas(self):
        atlas = CoverageAtlas({})
        assert atlas.coverage_level() == 0

    def test_no_labels_returns_zero(self):
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        assert atlas.coverage_level() == 0

    def test_root_covered_gives_one(self):
        """Labeling in a single-node atlas gives coverage level 1 (root covered)."""
        vecs = _make_vectors(10)  # All in root (single node)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        atlas.label(1, good=True)
        assert atlas.coverage_level() == 1

    def test_all_children_covered_advances(self):
        vecs = _make_clustered_vectors([30, 30], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)

        # Label one from each top-level child
        atlas.label(1, good=True)  # cluster 0
        atlas.label(31, good=False)  # cluster 1
        # Root + both children covered = at least 3
        assert atlas.coverage_level() >= 3

    def test_partial_children_caps_level(self):
        vecs = _make_clustered_vectors([30, 30], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)

        # Label only from one cluster
        atlas.label(1, good=True)
        # Root is covered + one child covered, but the other child is not
        level = atlas.coverage_level()
        assert level >= 1  # At least root
        assert level < atlas.total_nodes  # Not fully covered

    def test_coverage_level_increases_with_coverage(self):
        """More labels across clusters should increase coverage level."""
        vecs = _make_clustered_vectors([40, 40, 40], dim=32)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=10)

        # Start: nothing labeled
        assert atlas.coverage_level() == 0

        # Label one vector from cluster 0
        atlas.label(1, good=True)
        level_one = atlas.coverage_level()

        # Label vectors from clusters 1 and 2
        atlas.label(41, good=False)
        atlas.label(81, good=True)
        level_all = atlas.coverage_level()
        assert level_all >= level_one


# ---------------------------------------------------------------------------
# Span info
# ---------------------------------------------------------------------------


class TestSpanInfo:
    def test_span_info_diversity_matches_level(self):
        vecs = _make_clustered_vectors([30, 30], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        atlas.label(1, good=True)
        info = atlas.span_info()
        assert info["diversity_level"] == atlas.coverage_level()
        assert info["max_level"] == atlas.total_nodes

    def test_span_info_has_expected_keys(self):
        vecs = _make_clustered_vectors([30, 30], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        atlas.label(1, good=True)
        info = atlas.span_info()
        assert set(info.keys()) == {"level", "diversity_level", "depth", "max_level"}


# ---------------------------------------------------------------------------
# Total nodes
# ---------------------------------------------------------------------------


class TestTotalNodes:
    def test_empty_atlas(self):
        atlas = CoverageAtlas({})
        assert atlas.total_nodes == 0

    def test_single_node(self):
        vecs = _make_vectors(10)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        assert atlas.total_nodes == 1

    def test_multi_node(self):
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        assert atlas.total_nodes == len(atlas.nodes)
        assert atlas.total_nodes >= 3


# ---------------------------------------------------------------------------
# Next sample
# ---------------------------------------------------------------------------


class TestNextSample:
    def test_empty_atlas_returns_none(self):
        atlas = CoverageAtlas({})
        assert atlas.next_sample() is None

    def test_first_sample_is_from_root(self):
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        sample = atlas.next_sample()
        assert sample in atlas.nodes["0"]["ids"]

    def test_first_sample_is_most_typical(self):
        """Without scores, the pick is the node's representative element."""
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        assert atlas.next_sample() == atlas.nodes["0"]["ids"][0]

    def test_after_labeling_first_returns_child(self):
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        first = atlas.next_sample()
        assert first is not None
        atlas.label(first, good=True)

        # Next sample should not be the same element anymore
        # (root is covered now, so we descend to the first uncovered child)
        sample = atlas.next_sample()
        if sample is not None:
            assert sample != first or len(atlas.nodes) == 1

    def test_all_covered_returns_none(self):
        vecs = _make_vectors(10)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        # Single node atlas
        atlas.label(1, good=True)
        assert atlas.next_sample() is None

    def test_next_sample_returns_valid_vector_id(self):
        vecs = _make_vectors(200)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        sample = atlas.next_sample()
        assert sample in vecs

    def test_next_sample_bfs_order(self):
        """Next sample should follow BFS order through uncovered nodes."""
        vecs = _make_clustered_vectors([30, 30], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)

        # First sample is from root
        s1 = atlas.next_sample()
        assert s1 is not None
        assert s1 in atlas.nodes["0"]["ids"]

        # After labeling, next should be from the first uncovered child
        atlas.label(s1, good=True)
        s2 = atlas.next_sample()

        children = atlas.nodes["0"]["children"]
        uncovered_children = [c for c in children if not _covered(atlas, c)]
        if uncovered_children:
            assert s2 in atlas.nodes[uncovered_children[0]]["ids"]

    def test_bfs_prefers_larger_sibling(self):
        """The bigger unexplored region is offered before its smaller sibling."""
        vecs = _make_clustered_vectors([60, 20], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        s1 = atlas.next_sample()
        atlas.label(s1, good=True)
        s2 = atlas.next_sample()
        children = atlas.nodes["0"]["children"]
        uncovered = [c for c in children if not _covered(atlas, c)]
        if uncovered:
            # The first uncovered child in stored order is the largest one.
            assert atlas.nodes[uncovered[0]]["n"] == max(atlas.nodes[c]["n"] for c in uncovered)
            assert s2 in atlas.nodes[uncovered[0]]["ids"]

    def test_next_sample_with_scores(self):
        """When scores are provided, return the highest-scored element from the node."""
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        # Single node atlas; assign highest score to vector 50
        scores = {i: float(i) for i in range(1, 51)}
        sample = atlas.next_sample(scores=scores)
        # Vector 50 has the highest score
        assert sample == 50

    def test_next_sample_with_scores_picks_from_uncovered_node(self):
        """Scores should influence selection within the correct uncovered node."""
        vecs = _make_clustered_vectors([30, 30], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)

        # Give high scores to cluster-1 vectors, low to cluster-0
        scores = {}
        for vid in range(1, 31):
            scores[vid] = 0.1
        for vid in range(31, 61):
            scores[vid] = 0.9

        # First sample (from root) should be the highest-scored element overall
        s1 = atlas.next_sample(scores=scores)
        assert s1 in atlas.nodes["0"]["ids"]
        # Should pick a high-scoring element
        assert scores[s1] == 0.9

    def test_threshold_above_median_picks_lowest(self):
        """When the node's median score is >= threshold, return the lowest-scored element."""
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        # Single node atlas; scores 1..50
        scores = {i: float(i) for i in range(1, 51)}
        # Median of 1..50 is 25.5; set threshold below that
        sample = atlas.next_sample(scores=scores, threshold=20.0)
        # Median (25.5) >= 20.0, so we pick the lowest-scored element
        assert sample == 1

    def test_threshold_below_median_picks_highest(self):
        """When the node's median score is below threshold, return the highest-scored element."""
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        scores = {i: float(i) for i in range(1, 51)}
        # Median of 1..50 is 25.5; set threshold above that
        sample = atlas.next_sample(scores=scores, threshold=30.0)
        # Median (25.5) < 30.0, so we pick the highest-scored element
        assert sample == 50

    def test_threshold_none_always_picks_highest(self):
        """When threshold is None, always return the highest-scored element (original behavior)."""
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        scores = {i: float(i) for i in range(1, 51)}
        sample = atlas.next_sample(scores=scores, threshold=None)
        assert sample == 50

    def test_threshold_at_median_picks_lowest(self):
        """When threshold equals the median exactly, treat as above; pick lowest."""
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        scores = {i: float(i) for i in range(1, 51)}
        # Median of 1..50 is 25.5
        sample = atlas.next_sample(scores=scores, threshold=25.5)
        assert sample == 1


# ---------------------------------------------------------------------------
# Typicality and domain shift
# ---------------------------------------------------------------------------


def _two_block_vectors(n=200, dim=32, seed=7):
    """Unit vectors confined to the first dim//2 coordinates, two clusters."""
    rng = np.random.default_rng(seed)
    half = dim // 2
    base = rng.standard_normal((n, half)).astype(np.float32)
    base[: n // 2, 0] += 4.0
    base[n // 2 :, 1] += 4.0
    full = np.concatenate([base, np.zeros((n, half), dtype=np.float32)], axis=1)
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    return full


class TestTypicality:
    def test_empty_atlas_returns_fully_typical(self):
        atlas = CoverageAtlas({})
        pvals = atlas.typicality_pvalues(np.ones((3, 8), dtype=np.float32))
        assert pvals.shape == (3,)
        assert np.all(pvals == 1.0)

    def test_in_domain_queries_look_typical(self):
        """Fresh draws from the build distribution get bulk p-values."""
        train = _two_block_vectors(n=300, seed=7)
        atlas = CoverageAtlas({i: train[i] for i in range(300)}, k=3, max_depth=5)
        fresh = _two_block_vectors(n=200, seed=99)
        pvals = atlas.typicality_pvalues(fresh)
        assert 0.3 < float(np.median(pvals)) < 0.7
        assert float(np.mean(pvals < 0.05)) < 0.08

    def test_shifted_queries_look_atypical(self):
        """Directions the build data never had get small p-values."""
        train = _two_block_vectors(n=300, seed=7)
        atlas = CoverageAtlas({i: train[i] for i in range(300)}, k=3, max_depth=5)
        # Queries confined to the *other* half of the coordinates.
        rng = np.random.default_rng(11)
        out = rng.standard_normal((100, 16)).astype(np.float32)
        out[:, 0] += 4.0
        out_full = np.concatenate([np.zeros((100, 16), dtype=np.float32), out], axis=1)
        out_full /= np.linalg.norm(out_full, axis=1, keepdims=True)
        pvals = atlas.typicality_pvalues(out_full)
        assert float(np.median(pvals)) < 0.1
        assert float(np.mean(pvals < 0.05)) > 0.9

    def test_single_vector_helper_matches_batch(self):
        train = _two_block_vectors(n=100, seed=7)
        atlas = CoverageAtlas({i: train[i] for i in range(100)}, k=2)
        single = atlas.typicality_pvalue(train[0])
        batch = atlas.typicality_pvalues(train[:1])
        assert single == pytest.approx(float(batch[0]))

    def test_pvalues_survive_serialization(self):
        train = _two_block_vectors(n=150, seed=7)
        atlas = CoverageAtlas({i: train[i] for i in range(150)}, k=3)
        clone = CoverageAtlas.from_serializable(atlas.to_serializable())
        fresh = _two_block_vectors(n=50, seed=13)
        # float16 mu quantization perturbs p-values slightly; ranks must hold.
        p_orig = atlas.typicality_pvalues(fresh)
        p_clone = clone.typicality_pvalues(fresh)
        assert np.allclose(p_orig, p_clone, atol=0.05)


class TestDomainShiftReport:
    def test_no_shift_not_flagged(self):
        train = _two_block_vectors(n=300, seed=7)
        atlas = CoverageAtlas({i: train[i] for i in range(300)}, k=3, max_depth=5)
        report = domain_shift_report(atlas, _two_block_vectors(n=200, seed=23))
        assert report["n_items"] == 200
        assert not report["shifted"]
        assert report["frac_atypical"] < 2 * report["alpha"]

    def test_full_shift_flagged(self):
        train = _two_block_vectors(n=300, seed=7)
        atlas = CoverageAtlas({i: train[i] for i in range(300)}, k=3, max_depth=5)
        rng = np.random.default_rng(29)
        out = rng.standard_normal((100, 16)).astype(np.float32)
        out[:, 0] += 4.0
        out_full = np.concatenate([np.zeros((100, 16), dtype=np.float32), out], axis=1)
        out_full /= np.linalg.norm(out_full, axis=1, keepdims=True)
        report = domain_shift_report(atlas, out_full)
        assert report["shifted"]
        assert report["frac_atypical"] > 0.9
        assert report["median_pvalue"] < 0.05

    def test_partial_injection_flagged(self):
        """A foreign cluster hidden in mostly in-domain data still flags."""
        train = _two_block_vectors(n=300, seed=7)
        atlas = CoverageAtlas({i: train[i] for i in range(300)}, k=3, max_depth=5)
        fresh = _two_block_vectors(n=160, seed=31)
        rng = np.random.default_rng(37)
        out = rng.standard_normal((40, 16)).astype(np.float32)
        out[:, 0] += 4.0
        out_full = np.concatenate([np.zeros((40, 16), dtype=np.float32), out], axis=1)
        out_full /= np.linalg.norm(out_full, axis=1, keepdims=True)
        report = domain_shift_report(atlas, np.concatenate([fresh, out_full]))
        assert report["shifted"]
        # frac_atypical approximates the injected proportion (20%).
        assert 0.1 < report["frac_atypical"] < 0.35

    def test_empty_matrix(self):
        train = _two_block_vectors(n=100, seed=7)
        atlas = CoverageAtlas({i: train[i] for i in range(100)}, k=2)
        report = domain_shift_report(atlas, np.zeros((0, 32), dtype=np.float32))
        assert report["n_items"] == 0
        assert not report["shifted"]


# ---------------------------------------------------------------------------
# Integration / workflow
# ---------------------------------------------------------------------------


class TestWorkflow:
    def test_label_unlabel_cycle(self):
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)

        atlas.label(1, good=True)
        assert atlas.coverage_level() >= 1
        atlas.unlabel(1)
        assert atlas.coverage_level() == 0
        assert atlas.next_sample() in atlas.nodes["0"]["ids"]

    def test_progressive_labeling(self):
        """Label vectors progressively and verify coverage level grows."""
        vecs = _make_clustered_vectors([50, 50], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)

        levels = []
        # Label one vector at a time using next_sample
        for i in range(20):
            sample = atlas.next_sample()
            if sample is None:
                break
            atlas.label(sample, good=i % 2 == 0)
            levels.append(atlas.coverage_level())

        # Coverage level should be non-decreasing
        for i in range(1, len(levels)):
            assert levels[i] >= levels[i - 1]

    def test_large_atlas_k3(self):
        """Smoke test with larger k and more vectors."""
        vecs = _make_vectors(500, dim=16)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=15, max_depth=5)

        assert len(atlas.nodes) > 1
        assert all(vid in atlas.vector_to_leaf for vid in vecs)

        # Label several and check state
        for vid in range(1, 20):
            atlas.label(vid, good=vid % 2 == 0)
        assert atlas.coverage_level() >= 1

        sample = atlas.next_sample()
        if sample is not None:
            assert sample in vecs

    def test_depth_method(self):
        vecs = _make_vectors(200)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        assert atlas.depth() >= 1
        assert atlas.depth() <= 10

    def test_depth_empty_atlas(self):
        atlas = CoverageAtlas({})
        assert atlas.depth() == -1


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestKmeansProgress:
    """Progress should advance after each k-means call, not only at leaf placement."""

    def test_progress_called_during_build(self):
        """on_progress should fire multiple times, not just at the end."""
        vecs = _make_vectors(200)
        calls = []
        CoverageAtlas(vecs, k=3, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        assert len(calls) >= 3, f"Expected multiple progress calls, got {len(calls)}"

    def test_progress_starts_at_zero(self):
        vecs = _make_vectors(100)
        calls = []
        CoverageAtlas(vecs, k=2, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        assert calls[0][0] == 0, "First progress call should have current=0"

    def test_progress_ends_at_total(self):
        vecs = _make_vectors(100)
        calls = []
        CoverageAtlas(vecs, k=2, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        last_current, last_total = calls[-1]
        assert last_current == last_total, "Final progress call should have current == total"

    def test_progress_monotonically_increases(self):
        vecs = _make_vectors(300)
        calls = []
        CoverageAtlas(vecs, k=3, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        currents = [c for c, _ in calls]
        for i in range(1, len(currents)):
            assert currents[i] >= currents[i - 1], (
                f"Progress went backwards at index {i}: {currents[i - 1]} -> {currents[i]}"
            )

    def test_progress_advances_after_root_kmeans(self):
        """After the first (most expensive) k-means, progress should be > 0."""
        vecs = _make_vectors(200)
        calls = []
        CoverageAtlas(vecs, k=2, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        # calls[0] is (0, total), calls[1] should be after root k-means
        assert len(calls) >= 2
        assert calls[1][0] > 0, "Progress should advance after root k-means"

    def test_progress_total_reflects_estimated_work(self):
        """Total should be based on num_levels × num_vectors × N_INIT."""
        vecs = _make_vectors(200)
        calls = []
        CoverageAtlas(vecs, k=2, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        total = calls[0][1]
        from vtscore.state.coverage_atlas import _N_INIT

        # With k=2, 200 vectors, min_node_size=10 → 5 levels.
        # Estimated total = 5 * 200 * _N_INIT = 10000.
        assert total == 5 * 200 * _N_INIT, f"Expected {5 * 200 * _N_INIT}, got {total}"

    def test_no_progress_for_small_input(self):
        """Vectors below min_node_size should not trigger k-means progress."""
        vecs = _make_vectors(5)
        calls = []
        CoverageAtlas(vecs, k=2, min_node_size=20, on_progress=lambda c, t: calls.append((c, t)))
        # Should still get start (0) and end (total) calls
        assert len(calls) >= 2
        assert calls[-1][0] == calls[-1][1]

    def test_progress_not_called_without_callback(self):
        """Building without on_progress should not raise."""
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        assert atlas.depth() >= 1


class TestSerialization:
    """Round-trip the cached atlas structure used by dataset pickles."""

    _SNAP_KEYS = {"format", "k", "max_depth", "min_node_size", "center", "nodes", "vector_to_leaf", "nodes_by_depth"}

    def test_to_serializable_is_plain_types(self):
        """Snapshot must contain only plain containers and numpy arrays.

        The dataset pickle is read back through the restricted unpickler,
        which rejects anything beyond plain Python types + numpy arrays, so
        the snapshot must not embed the atlas object itself.  Evidence
        counts are session state and must not be serialized.
        """
        atlas = CoverageAtlas(_make_vectors(120), k=3, min_node_size=10)
        snap = atlas.to_serializable()

        def _assert_plain(obj):
            assert isinstance(obj, (int, str, type(None))), type(obj)

        assert set(snap) == self._SNAP_KEYS
        assert isinstance(snap["center"], np.ndarray)
        for node in snap["nodes"].values():
            assert set(node) == {"ids", "children", "depth", "parent", "n", "mu", "rbar", "t_quantiles"}
            for vid in node["ids"]:
                _assert_plain(vid)
            for child in node["children"]:
                _assert_plain(child)
            _assert_plain(node["depth"])
            _assert_plain(node["parent"])
            assert isinstance(node["mu"], np.ndarray)
            assert node["mu"].dtype == np.float16
            assert all(isinstance(q, float) for q in node["t_quantiles"])
        for vid, leaf in snap["vector_to_leaf"].items():
            _assert_plain(vid)
            _assert_plain(leaf)

    def test_survives_restricted_unpickler(self):
        """A snapshot nested in a pickle must load via safe_pickle_load."""
        import io
        import pickle

        from vtscore.security.pickle import safe_pickle_load

        atlas = CoverageAtlas(_make_vectors(120), k=3, min_node_size=10)
        buf = io.BytesIO()
        pickle.dump({"coverage_atlas": atlas.to_serializable()}, buf, protocol=5)
        buf.seek(0)
        restored = safe_pickle_load(buf)["coverage_atlas"]
        assert restored["format"] == "coverage-atlas/1"

    def test_round_trip_preserves_structure(self):
        """from_serializable rebuilds an equivalent atlas without k-means."""
        vecs = _make_vectors(150)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=10)
        clone = CoverageAtlas.from_serializable(atlas.to_serializable())

        assert clone.k == atlas.k
        assert clone.max_depth == atlas.max_depth
        assert clone.min_node_size == atlas.min_node_size
        assert set(clone.nodes) == set(atlas.nodes)
        for name, node in atlas.nodes.items():
            restored = clone.nodes[name]
            assert restored["ids"] == node["ids"]
            assert restored["children"] == node["children"]
            assert restored["depth"] == node["depth"]
            assert restored["parent"] == node["parent"]
            assert restored["n"] == node["n"]
            assert restored["rbar"] == pytest.approx(node["rbar"])
            assert restored["t_quantiles"] == node["t_quantiles"]
            assert np.allclose(restored["mu"], node["mu"], atol=1e-3)  # float16 round-trip
        assert clone.vector_to_leaf == atlas.vector_to_leaf
        assert clone.nodes_by_depth == atlas.nodes_by_depth
        assert clone.total_nodes == atlas.total_nodes
        assert np.allclose(clone.center, atlas.center)
        assert clone.next_sample() == atlas.next_sample()

    def test_restored_atlas_is_label_clean(self):
        """Evidence counts are session state and must start empty after restore."""
        vecs = _make_vectors(120)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=10)
        first = next(iter(vecs))
        atlas.label(first, good=True)
        assert atlas.labeled_ids  # original now has evidence state

        clone = CoverageAtlas.from_serializable(atlas.to_serializable())
        assert clone.labeled_ids == set()
        assert all(n["n_pos"] == 0 and n["n_neg"] == 0 for n in clone.nodes.values())
        # Labeling still works against the restored topology.
        clone.label(first, good=True)
        assert clone.coverage_level() >= 1

    def test_from_serializable_rejects_bad_format(self):
        with pytest.raises(ValueError, match="format"):
            CoverageAtlas.from_serializable({"format": 999})
        with pytest.raises(ValueError):
            CoverageAtlas.from_serializable("not a dict")

    def test_from_serializable_rejects_old_diversity_tree_cache(self):
        """A pre-migration diversity-tree cache (format 1) must be rejected."""
        with pytest.raises(ValueError, match="format"):
            CoverageAtlas.from_serializable({"format": 1, "k": 3})

    def test_from_serializable_rejects_incomplete(self):
        with pytest.raises(ValueError, match="missing"):
            CoverageAtlas.from_serializable({"format": "coverage-atlas/1", "k": 3})
