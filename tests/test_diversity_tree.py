"""Tests for the DiversityTree k-means clustering structure."""

import numpy as np
import pytest

from vtsearch.models.diversity_tree import DiversityTree


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
        tree = DiversityTree({}, k=2)
        assert tree.nodes == {}
        assert tree.diversity_level() == -1
        assert tree.next_sample() is None

    def test_single_vector(self):
        vecs = _make_vectors(1)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        assert len(tree.nodes) == 1
        assert tree.vector_to_leaf[1] == "0"
        assert 1 in tree.nodes["0"]["ids"]

    def test_small_set_stays_single_node(self):
        vecs = _make_vectors(10)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        # 10 < 20 so no splitting
        assert len(tree.nodes) == 1
        for vid in vecs:
            assert tree.vector_to_leaf[vid] == "0"

    def test_splits_when_above_min_size(self):
        vecs = _make_vectors(50)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        # Should have at least root + 2 children
        assert len(tree.nodes) >= 3
        assert "0" in tree.nodes
        assert tree.nodes["0"]["children"]

    def test_all_vectors_assigned_to_leaves(self):
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=3, min_node_size=10)
        for vid in vecs:
            assert vid in tree.vector_to_leaf
            leaf = tree.vector_to_leaf[vid]
            assert leaf in tree.nodes
            assert vid in tree.nodes[leaf]["ids"]

    def test_disjoint_partitioning(self):
        """Every vector appears in exactly one leaf node."""
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        leaf_assignments = {}
        for name, node in tree.nodes.items():
            if not node["children"]:  # leaf
                for vid in node["ids"]:
                    assert vid not in leaf_assignments, f"Vector {vid} in multiple leaves"
                    leaf_assignments[vid] = name
        assert set(leaf_assignments.keys()) == set(vecs.keys())

    def test_respects_max_depth(self):
        vecs = _make_vectors(500)
        tree = DiversityTree(vecs, k=2, max_depth=3, min_node_size=2)
        for node in tree.nodes.values():
            assert node["depth"] <= 3

    def test_node_has_ids(self):
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=3, min_node_size=10)
        for name, node in tree.nodes.items():
            assert len(node["ids"]) > 0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be between 2 and 9"):
            DiversityTree(_make_vectors(10), k=1)
        with pytest.raises(ValueError, match="k must be between 2 and 9"):
            DiversityTree(_make_vectors(10), k=10)

    def test_invalid_max_depth_raises(self):
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            DiversityTree(_make_vectors(10), k=2, max_depth=0)

    def test_invalid_min_node_size_raises(self):
        with pytest.raises(ValueError, match="min_node_size must be >= 1"):
            DiversityTree(_make_vectors(10), k=2, min_node_size=0)

    def test_parent_child_consistency(self):
        vecs = _make_vectors(200)
        tree = DiversityTree(vecs, k=3, min_node_size=10)
        for name, node in tree.nodes.items():
            for child_name in node["children"]:
                assert child_name in tree.nodes
                assert tree.nodes[child_name]["parent"] == name
        # Root has no parent
        assert tree.nodes["0"]["parent"] is None

    def test_node_naming_convention(self):
        """Children names are parent name + single digit."""
        vecs = _make_vectors(200)
        tree = DiversityTree(vecs, k=3, min_node_size=10)
        for name, node in tree.nodes.items():
            for child_name in node["children"]:
                assert child_name.startswith(name)
                assert len(child_name) == len(name) + 1

    def test_nodes_by_depth_consistent(self):
        vecs = _make_vectors(200)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        for depth, names in tree.nodes_by_depth.items():
            for name in names:
                assert tree.nodes[name]["depth"] == depth


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_lookup_returns_leaf(self):
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        for vid in vecs:
            leaf = tree.lookup(vid)
            assert leaf in tree.nodes
            # Leaf nodes have no children
            assert not tree.nodes[leaf]["children"]

    def test_lookup_missing_id_raises(self):
        vecs = _make_vectors(10)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        with pytest.raises(KeyError):
            tree.lookup(9999)


# ---------------------------------------------------------------------------
# Labeling and seen tracking
# ---------------------------------------------------------------------------


class TestLabeling:
    def test_label_marks_leaf_and_ancestors(self):
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        vid = 1
        tree.label(vid)
        leaf = tree.lookup(vid)
        # Walk from leaf to root; all should be seen
        node = leaf
        while node is not None:
            assert node in tree.seen
            node = tree.nodes[node]["parent"]

    def test_initially_nothing_seen(self):
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        assert len(tree.seen) == 0

    def test_label_tracks_labeled_ids(self):
        vecs = _make_vectors(50)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        tree.label(1)
        tree.label(5)
        assert tree.labeled_ids == {1, 5}

    def test_label_idempotent(self):
        vecs = _make_vectors(50)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        tree.label(1)
        seen_before = set(tree.seen)
        tree.label(1)
        assert tree.seen == seen_before
        assert tree.labeled_ids == {1}


# ---------------------------------------------------------------------------
# Unlabeling
# ---------------------------------------------------------------------------


class TestUnlabeling:
    def test_unlabel_single_removes_seen(self):
        vecs = _make_vectors(50)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        tree.label(1)
        assert len(tree.seen) > 0
        tree.unlabel(1)
        assert len(tree.seen) == 0
        assert tree.labeled_ids == set()

    def test_unlabel_preserves_sibling_seen(self):
        """If another labeled vector shares an ancestor, ancestor stays seen."""
        vecs = _make_clustered_vectors([30, 30], dim=32)
        tree = DiversityTree(vecs, k=2, min_node_size=10)

        # Label one vector from each cluster child
        v1 = 1  # cluster 0
        v2 = 31  # cluster 1
        tree.label(v1)
        tree.label(v2)

        # Root should be seen
        assert "0" in tree.seen

        # Unlabel v1
        tree.unlabel(v1)

        # Root should still be seen (v2 keeps the other branch alive)
        assert "0" in tree.seen

    def test_unlabel_same_leaf_keeps_seen(self):
        """If another labeled vector is in the same leaf, leaf stays seen."""
        vecs = _make_vectors(10, dim=8)  # Small, all in root
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        # All in root (single leaf)
        tree.label(1)
        tree.label(2)
        tree.unlabel(1)
        assert "0" in tree.seen  # Vector 2 still keeps root seen
        tree.unlabel(2)
        assert "0" not in tree.seen

    def test_unlabel_without_label_is_noop(self):
        vecs = _make_vectors(50)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        tree.unlabel(1)  # Never labeled, should not error
        assert len(tree.seen) == 0


# ---------------------------------------------------------------------------
# Diversity level
# ---------------------------------------------------------------------------


class TestDiversityLevel:
    def test_empty_tree(self):
        tree = DiversityTree({})
        assert tree.diversity_level() == -1

    def test_no_labels_returns_negative(self):
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        assert tree.diversity_level() == -1

    def test_root_seen_gives_level_zero(self):
        vecs = _make_vectors(10)  # All in root (single node)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        tree.label(1)
        assert tree.diversity_level() == 0

    def test_all_children_seen_gives_level_one(self):
        vecs = _make_clustered_vectors([30, 30], dim=32)
        tree = DiversityTree(vecs, k=2, min_node_size=10)

        # Label one from each top-level child
        tree.label(1)  # cluster 0
        tree.label(31)  # cluster 1
        assert tree.diversity_level() >= 1

    def test_partial_children_caps_level(self):
        vecs = _make_clustered_vectors([30, 30], dim=32)
        tree = DiversityTree(vecs, k=2, min_node_size=10)

        # Label only from one cluster
        tree.label(1)
        # Root is seen but not all children
        level = tree.diversity_level()
        assert level == 0  # Only root fully covered

    def test_diversity_level_increases_with_coverage(self):
        """More labels across clusters should increase diversity level."""
        vecs = _make_clustered_vectors([40, 40, 40], dim=32)
        tree = DiversityTree(vecs, k=3, min_node_size=10)

        # Start: nothing labeled
        assert tree.diversity_level() == -1

        # Label one vector from cluster 0
        tree.label(1)
        level_one = tree.diversity_level()

        # Label vectors from clusters 1 and 2
        tree.label(41)
        tree.label(81)
        level_all = tree.diversity_level()
        assert level_all >= level_one


# ---------------------------------------------------------------------------
# Fractional diversity level
# ---------------------------------------------------------------------------


class TestFractionalDiversityLevel:
    def test_empty_tree(self):
        tree = DiversityTree({})
        assert tree.fractional_diversity_level() == -1.0

    def test_no_labels_returns_negative(self):
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        assert tree.fractional_diversity_level() == -1.0

    def test_single_node_labeled_returns_zero(self):
        vecs = _make_vectors(10)  # All in root (single node, depth=0)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        tree.label(1)
        assert tree.fractional_diversity_level() == 0.0

    def test_partial_children_gives_fractional(self):
        """Labeling one of two children should give a fractional level between 0 and 1."""
        vecs = _make_clustered_vectors([30, 30], dim=32)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        tree.label(1)  # cluster 0 only
        frac = tree.fractional_diversity_level()
        # Root is fully seen (level 0), one of two children seen -> 0 + 0.5 = 0.5
        assert 0 < frac < 1.0

    def test_all_children_gives_integer(self):
        """Labeling from all top-level children should give >= 1.0."""
        vecs = _make_clustered_vectors([30, 30], dim=32)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        tree.label(1)  # cluster 0
        tree.label(31)  # cluster 1
        frac = tree.fractional_diversity_level()
        assert frac >= 1.0

    def test_fully_covered_equals_depth(self):
        vecs = _make_vectors(10)  # Single node tree (depth 0)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        tree.label(1)
        assert tree.fractional_diversity_level() == float(tree.depth())

    def test_monotonically_increasing_with_coverage(self):
        """Fractional level should not decrease as more clusters are covered."""
        vecs = _make_clustered_vectors([40, 40, 40], dim=32)
        tree = DiversityTree(vecs, k=3, min_node_size=10)

        levels = []
        for _ in range(30):
            sample = tree.next_sample()
            if sample is None:
                break
            tree.label(sample)
            levels.append(tree.fractional_diversity_level())

        for i in range(1, len(levels)):
            assert levels[i] >= levels[i - 1], f"Fractional level decreased: {levels[i - 1]} -> {levels[i]} at step {i}"

    def test_span_info_includes_fractional_level(self):
        vecs = _make_clustered_vectors([30, 30], dim=32)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        tree.label(1)
        info = tree.span_info()
        assert "fractional_level" in info
        assert isinstance(info["fractional_level"], float)
        assert info["fractional_level"] > -1.0


# ---------------------------------------------------------------------------
# Next sample
# ---------------------------------------------------------------------------


class TestNextSample:
    def test_empty_tree_returns_none(self):
        tree = DiversityTree({})
        assert tree.next_sample() is None

    def test_first_sample_is_from_root(self):
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        sample = tree.next_sample()
        assert sample in tree.nodes["0"]["ids"]

    def test_after_labeling_first_returns_child(self):
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        first = tree.next_sample()
        tree.label(first)

        # Next sample should not be the same element anymore
        # (root is seen now, so we descend to first unseen child)
        sample = tree.next_sample()
        if sample is not None:
            assert sample != first or len(tree.nodes) == 1

    def test_all_seen_returns_none(self):
        vecs = _make_vectors(10)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        # Single node tree
        tree.label(1)
        assert tree.next_sample() is None

    def test_next_sample_returns_valid_vector_id(self):
        vecs = _make_vectors(200)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        sample = tree.next_sample()
        assert sample in vecs

    def test_next_sample_bfs_order(self):
        """Next sample should follow BFS order through unseen nodes."""
        vecs = _make_clustered_vectors([30, 30], dim=32)
        tree = DiversityTree(vecs, k=2, min_node_size=10)

        # First sample is from root
        s1 = tree.next_sample()
        assert s1 in tree.nodes["0"]["ids"]

        # After labeling, next should be from first unseen child
        tree.label(s1)
        s2 = tree.next_sample()

        children = tree.nodes["0"]["children"]
        unseen_children = [c for c in children if c not in tree.seen]
        if unseen_children:
            assert s2 in tree.nodes[unseen_children[0]]["ids"]

    def test_next_sample_with_scores(self):
        """When scores are provided, return the highest-scored element from the node."""
        vecs = _make_vectors(50)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        # Single node tree; assign highest score to vector 5
        scores = {i: float(i) for i in range(1, 51)}
        sample = tree.next_sample(scores=scores)
        # Vector 50 has the highest score
        assert sample == 50

    def test_next_sample_with_scores_picks_from_unseen_node(self):
        """Scores should influence selection within the correct unseen node."""
        vecs = _make_clustered_vectors([30, 30], dim=32)
        tree = DiversityTree(vecs, k=2, min_node_size=10)

        # Give high scores to cluster-1 vectors, low to cluster-0
        scores = {}
        for vid in range(1, 31):
            scores[vid] = 0.1
        for vid in range(31, 61):
            scores[vid] = 0.9

        # First sample (from root) should be the highest-scored element overall
        s1 = tree.next_sample(scores=scores)
        assert s1 in tree.nodes["0"]["ids"]
        # Should pick a high-scoring element
        assert scores[s1] == 0.9

    def test_threshold_above_median_picks_lowest(self):
        """When the node's median score is >= threshold, return the lowest-scored element."""
        vecs = _make_vectors(50)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        # Single node tree; scores 1..50
        scores = {i: float(i) for i in range(1, 51)}
        # Median of 1..50 is 25.5 — set threshold below that
        sample = tree.next_sample(scores=scores, threshold=20.0)
        # Median (25.5) >= 20.0, so we pick the lowest-scored element
        assert sample == 1

    def test_threshold_below_median_picks_highest(self):
        """When the node's median score is below threshold, return the highest-scored element."""
        vecs = _make_vectors(50)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        scores = {i: float(i) for i in range(1, 51)}
        # Median of 1..50 is 25.5 — set threshold above that
        sample = tree.next_sample(scores=scores, threshold=30.0)
        # Median (25.5) < 30.0, so we pick the highest-scored element
        assert sample == 50

    def test_threshold_none_always_picks_highest(self):
        """When threshold is None, always return the highest-scored element (original behavior)."""
        vecs = _make_vectors(50)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        scores = {i: float(i) for i in range(1, 51)}
        sample = tree.next_sample(scores=scores, threshold=None)
        assert sample == 50

    def test_threshold_at_median_picks_lowest(self):
        """When threshold equals the median exactly, treat as above — pick lowest."""
        vecs = _make_vectors(50)
        tree = DiversityTree(vecs, k=2, min_node_size=20)
        scores = {i: float(i) for i in range(1, 51)}
        # Median of 1..50 is 25.5
        sample = tree.next_sample(scores=scores, threshold=25.5)
        assert sample == 1


# ---------------------------------------------------------------------------
# Integration / workflow
# ---------------------------------------------------------------------------


class TestWorkflow:
    def test_label_unlabel_cycle(self):
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=2, min_node_size=10)

        tree.label(1)
        assert tree.diversity_level() >= 0
        tree.unlabel(1)
        assert tree.diversity_level() == -1
        assert tree.next_sample() in tree.nodes["0"]["ids"]

    def test_progressive_labeling(self):
        """Label vectors progressively and verify diversity level grows."""
        vecs = _make_clustered_vectors([50, 50], dim=32)
        tree = DiversityTree(vecs, k=2, min_node_size=10)

        levels = []
        # Label one vector at a time using next_sample
        for _ in range(20):
            sample = tree.next_sample()
            if sample is None:
                break
            tree.label(sample)
            levels.append(tree.diversity_level())

        # Diversity level should be non-decreasing
        for i in range(1, len(levels)):
            assert levels[i] >= levels[i - 1]

    def test_large_tree_k3(self):
        """Smoke test with larger k and more vectors."""
        vecs = _make_vectors(500, dim=16)
        tree = DiversityTree(vecs, k=3, min_node_size=15, max_depth=5)

        assert len(tree.nodes) > 1
        assert all(vid in tree.vector_to_leaf for vid in vecs)

        # Label several and check state
        for vid in range(1, 20):
            tree.label(vid)
        assert tree.diversity_level() >= 0

        sample = tree.next_sample()
        if sample is not None:
            assert sample in vecs

    def test_depth_method(self):
        vecs = _make_vectors(200)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        assert tree.depth() >= 1
        assert tree.depth() <= 10

    def test_depth_empty_tree(self):
        tree = DiversityTree({})
        assert tree.depth() == -1


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestKmeansProgress:
    """Progress should advance after each k-means call, not only at leaf placement."""

    def test_progress_called_during_build(self):
        """on_progress should fire multiple times, not just at the end."""
        vecs = _make_vectors(200)
        calls = []
        DiversityTree(vecs, k=3, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        assert len(calls) >= 3, f"Expected multiple progress calls, got {len(calls)}"

    def test_progress_starts_at_zero(self):
        vecs = _make_vectors(100)
        calls = []
        DiversityTree(vecs, k=2, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        assert calls[0][0] == 0, "First progress call should have current=0"

    def test_progress_ends_at_total(self):
        vecs = _make_vectors(100)
        calls = []
        DiversityTree(vecs, k=2, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        last_current, last_total = calls[-1]
        assert last_current == last_total, "Final progress call should have current == total"

    def test_progress_monotonically_increases(self):
        vecs = _make_vectors(300)
        calls = []
        DiversityTree(vecs, k=3, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        currents = [c for c, _ in calls]
        for i in range(1, len(currents)):
            assert currents[i] >= currents[i - 1], (
                f"Progress went backwards at index {i}: {currents[i - 1]} -> {currents[i]}"
            )

    def test_progress_advances_after_root_kmeans(self):
        """After the first (most expensive) k-means, progress should be > 0."""
        vecs = _make_vectors(200)
        calls = []
        DiversityTree(vecs, k=2, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        # calls[0] is (0, total), calls[1] should be after root k-means
        assert len(calls) >= 2
        assert calls[1][0] > 0, "Progress should advance after root k-means"

    def test_progress_total_reflects_estimated_work(self):
        """Total should be based on N * num_levels, not just N."""
        vecs = _make_vectors(200)
        calls = []
        DiversityTree(vecs, k=2, min_node_size=10, on_progress=lambda c, t: calls.append((c, t)))
        total = calls[0][1]
        n = len(vecs)
        assert total > n, f"Estimated total work ({total}) should exceed vector count ({n})"

    def test_no_progress_for_small_input(self):
        """Vectors below min_node_size should not trigger k-means progress."""
        vecs = _make_vectors(5)
        calls = []
        DiversityTree(vecs, k=2, min_node_size=20, on_progress=lambda c, t: calls.append((c, t)))
        # Should still get start (0) and end (total) calls
        assert len(calls) >= 2
        assert calls[-1][0] == calls[-1][1]

    def test_progress_not_called_without_callback(self):
        """Building without on_progress should not raise."""
        vecs = _make_vectors(100)
        tree = DiversityTree(vecs, k=2, min_node_size=10)
        assert tree.depth() >= 1
