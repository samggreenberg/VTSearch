"""Tests for the CoverageAtlas evidence-aware partition structure."""

import numpy as np
import pytest

from vtscore.coverage.atlas import CoverageAtlas


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
            # The typicality calibration grid went with #3348.
            assert "t_quantiles" not in node


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
    return atlas.n_pos(name) + atlas.n_neg(name) > 0


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
            assert atlas.n_pos(node) == 1
            node = atlas.nodes[node]["parent"]

    def test_initially_no_evidence(self):
        vecs = _make_vectors(100)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        for name in atlas.nodes:
            assert atlas.n_pos(name) == 0
            assert atlas.n_neg(name) == 0

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
        assert atlas.n_pos("0") == 1
        assert atlas.labeled_ids == {1}

    def test_relabel_moves_evidence_between_channels(self):
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        atlas.label(1, good=True)
        assert atlas.n_pos("0") == 1
        assert atlas.n_neg("0") == 0
        atlas.label(1, good=False)
        assert atlas.n_pos("0") == 0
        assert atlas.n_neg("0") == 1

    def test_per_class_counts_accumulate(self):
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        atlas.label(1, good=True)
        atlas.label(2, good=True)
        atlas.label(3, good=False)
        assert atlas.n_pos("0") == 2
        assert atlas.n_neg("0") == 1


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
        assert atlas.n_pos("0") == 1
        assert atlas.n_neg("0") == 1

        atlas.unlabel(v1)

        # Root should still be covered (v2 keeps the other branch alive)
        assert _covered(atlas, "0")
        assert atlas.n_pos("0") == 0
        assert atlas.n_neg("0") == 1

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


class TestResetLabeled:
    def test_reset_labeled_clears_everything(self):
        vecs = _make_clustered_vectors([30, 30], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        atlas.label(1, good=True)
        atlas.label(31, good=False)
        assert len(atlas.labeled_ids) == 2

        atlas.reset_labeled()
        assert atlas.labeled_ids == set()
        assert atlas.coverage_level() == 0
        for name in atlas.nodes:
            assert atlas.n_pos(name) == 0
            assert atlas.n_neg(name) == 0

    def test_reset_then_relabel_picks_up_new_state(self):
        """After reset, label calls reproduce the evidence state from scratch."""
        vecs = _make_clustered_vectors([30, 30], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        atlas.label(1, good=True)
        leaf1 = atlas.lookup(1)
        assert _covered(atlas, leaf1)

        atlas.reset_labeled()
        # Label a different vector; only its leaf+ancestors should be covered.
        atlas.label(31, good=True)
        leaf31 = atlas.lookup(31)
        assert _covered(atlas, leaf31)
        if leaf1 != leaf31:
            assert not _covered(atlas, leaf1)

    def test_reset_on_empty_atlas_is_noop(self):
        atlas = CoverageAtlas({}, k=2)
        atlas.reset_labeled()  # must not raise
        assert atlas.labeled_ids == set()


class TestStructuralClone:
    def test_clone_shares_structure_by_reference(self):
        vecs = _make_vectors(120)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=10)
        clone = atlas.structural_clone()

        # The expensive, immutable maps are shared by reference (no re-fit).
        assert clone.nodes is atlas.nodes
        assert clone.vector_to_leaf is atlas.vector_to_leaf
        assert clone.nodes_by_depth is atlas.nodes_by_depth
        assert clone.center is atlas.center
        assert clone.k == atlas.k
        assert clone.max_depth == atlas.max_depth
        assert clone.min_node_size == atlas.min_node_size

    def test_clone_has_independent_empty_overlay(self):
        vecs = _make_vectors(120)
        atlas = CoverageAtlas(vecs, k=3, min_node_size=10)
        first = next(iter(vecs))
        atlas.label(first, good=True)

        clone = atlas.structural_clone()
        # Clone starts label-clean even though the original carries evidence.
        assert clone.labeled_ids == set()
        assert clone.coverage_level() == 0
        assert all(clone.n_pos(name) == 0 and clone.n_neg(name) == 0 for name in clone.nodes)

        # Labeling the clone must not disturb the original, and vice versa.
        second = next(i for i in vecs if i != first)
        clone.label(second, good=False)
        assert clone.labeled_ids == {second}
        assert atlas.labeled_ids == {first}
        assert atlas.n_pos("0") == 1 and atlas.n_neg("0") == 0
        assert clone.n_pos("0") == 0 and clone.n_neg("0") == 1


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
        assert s1 is not None
        atlas.label(s1, good=True)
        s2 = atlas.next_sample()
        children = atlas.nodes["0"]["children"]
        uncovered = [c for c in children if not _covered(atlas, c)]
        if uncovered:
            # The first uncovered child in stored order is the largest one.
            assert atlas.nodes[uncovered[0]]["n"] == max(atlas.nodes[c]["n"] for c in uncovered)
            assert s2 in atlas.nodes[uncovered[0]]["ids"]

    def test_next_sample_with_scores(self):
        """When scores are provided, return the highest-scored element from the node.

        The root's direction is degenerate after centering (rbar ~ 0), so no
        typical-half tempering applies there: the probe spans the whole node.
        """
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        assert atlas.nodes["0"]["rbar"] < 0.1  # degenerate root → no tempering
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

        # First sample (from root) should be a highest-scored element
        s1 = atlas.next_sample(scores=scores)
        assert s1 in atlas.nodes["0"]["ids"]
        # Should pick a high-scoring element
        assert scores[s1] == 0.9

    def _concentrated_uncovered_node(self):
        """Cover the root via the smaller cluster, leaving the larger cluster
        (a concentrated k-means cell, rbar >= 0.1) as the first uncovered node.
        """
        vecs = _make_clustered_vectors([40, 20], dim=32)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=10)
        # Label an item from the smaller cluster: its leaf-to-root path is
        # covered, so BFS lands on the larger cluster's (uncovered) child.
        atlas.label(41, good=True)
        children = atlas.nodes["0"]["children"]
        target = next(c for c in children if not (atlas.n_pos(c) + atlas.n_neg(c)))
        node = atlas.nodes[target]
        assert node["rbar"] >= 0.1, "cluster cell should have a concentrated direction"
        return atlas, node

    def test_surprise_pool_excludes_atypical_outlier(self):
        """An extreme score on an atypical item must not win the probe.

        In a node with a concentrated direction, the surprise extremum is
        drawn from the typical half: an oddball whose flip says nothing about
        the region would otherwise monopolise the pick.
        """
        atlas, node = self._concentrated_uncovered_node()
        ids = node["ids"]
        pool = ids[: (len(ids) + 1) // 2]
        # Rank-based scores: most typical → 1, least typical → n.
        scores = {vid: float(i + 1) for i, vid in enumerate(ids)}
        sample = atlas.next_sample(scores=scores)
        # The global maximum (ids[-1], the least typical item) is excluded;
        # the pick is the highest score within the typical half.
        assert sample != ids[-1]
        assert sample == pool[-1]

    def test_surprise_pool_threshold_flip_stays_typical(self):
        """The lowest-scored probe in a presumed-good region is also typical-half."""
        atlas, node = self._concentrated_uncovered_node()
        ids = node["ids"]
        pool = ids[: (len(ids) + 1) // 2]
        # High scores everywhere; the global minimum sits on the least
        # typical item, the typical-half minimum on the most typical one.
        scores = {vid: 0.9 for vid in ids}
        scores[ids[-1]] = 0.1  # atypical outlier, global minimum
        scores[ids[0]] = 0.5  # typical-half minimum
        sample = atlas.next_sample(scores=scores, threshold=0.5)
        # Median 0.9 >= 0.5 → presumed good → probe the lowest, but within
        # the typical half.
        assert sample == ids[0]
        assert sample in pool

    def test_median_spans_whole_node_not_just_pool(self):
        """The probe direction (median vs threshold) considers every element.

        Rank-based scores over an n-item node: the whole-node median is
        (n+1)/2 while the typical-half median is about half that.  A
        threshold between the two must still read the region as
        above-threshold (→ pick the lowest), which would flip if the median
        were pool-only.
        """
        atlas, node = self._concentrated_uncovered_node()
        ids = node["ids"]
        n = len(ids)
        scores = {vid: float(i + 1) for i, vid in enumerate(ids)}
        threshold = (n + 1) / 2 - n / 8  # between pool median and node median
        sample = atlas.next_sample(scores=scores, threshold=threshold)
        assert sample == ids[0]  # lowest — the region presumes above-threshold

    def test_threshold_above_median_picks_lowest(self):
        """When the node's median score is >= threshold, return the lowest-scored element."""
        vecs = _make_vectors(50)
        atlas = CoverageAtlas(vecs, k=2, min_node_size=20)
        assert atlas.nodes["0"]["rbar"] < 0.1  # degenerate root → whole-node probe
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
        """When threshold is None, always return the highest-scored element."""
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
        from vtscore.coverage.atlas import _N_INIT

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
            assert set(node) == {"ids", "children", "depth", "parent", "n", "mu", "rbar"}
            for vid in node["ids"]:
                _assert_plain(vid)
            for child in node["children"]:
                _assert_plain(child)
            _assert_plain(node["depth"])
            _assert_plain(node["parent"])
            assert isinstance(node["mu"], np.ndarray)
            assert node["mu"].dtype == np.float16
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

    def test_reads_a_pre_3348_snapshot(self):
        """A cache written while nodes still carried ``t_quantiles`` loads, the grid ignored."""
        atlas = CoverageAtlas(_make_vectors(120), k=3, min_node_size=10)
        snap = atlas.to_serializable()
        for node in snap["nodes"].values():
            node["t_quantiles"] = [i / 20.0 for i in range(21)]
        clone = CoverageAtlas.from_serializable(snap)
        assert set(clone.nodes) == set(atlas.nodes)
        assert all("t_quantiles" not in node for node in clone.nodes.values())

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
        assert all(clone.n_pos(name) == 0 and clone.n_neg(name) == 0 for name in clone.nodes)
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
