"""Diversity Tree: hierarchical k-means clustering for diverse sampling.

A k-Diversity Tree recursively partitions a set of vectors using k-means
clustering. Each node tracks a "seen" flag driven by label activity.

The tree supports:
- Lookup: given a vector ID, return the name of its deepest (leaf) node.
- Labeling: when a vector is labeled, mark its leaf and all ancestors as seen.
- Unlabeling: when a label is removed, propagate unseen status upward.
- Diversity level: the deepest tree level at which every node is seen.
- Next sample: a surprise-maximising element of the first unseen node in BFS order.
"""

from __future__ import annotations

from collections import deque

import numpy as np

DIVERSITY_TREE_DEFAULT_K = 2
DIVERSITY_TREE_MAX_DEPTH = 10
DIVERSITY_TREE_MIN_NODE_SIZE = 20


class DiversityTree:
    """Hierarchical k-means tree for tracking label diversity.

    Parameters
    ----------
    vectors : dict[int, np.ndarray]
        Mapping of vector ID to embedding array.
    k : int
        Number of clusters per split (must be 2-9).
    max_depth : int
        Maximum tree depth (number of splitting layers).
    min_node_size : int
        Minimum number of vectors in a node to allow splitting.
    """

    def __init__(
        self,
        vectors: dict[int, np.ndarray],
        k: int = DIVERSITY_TREE_DEFAULT_K,
        max_depth: int = DIVERSITY_TREE_MAX_DEPTH,
        min_node_size: int = DIVERSITY_TREE_MIN_NODE_SIZE,
    ):
        if k < 2 or k > 9:
            raise ValueError(f"k must be between 2 and 9, got {k}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if min_node_size < 1:
            raise ValueError(f"min_node_size must be >= 1, got {min_node_size}")

        self.k = k
        self.max_depth = max_depth
        self.min_node_size = min_node_size

        # Node storage: name -> {ids, children, depth, parent}
        self.nodes: dict[str, dict] = {}
        # Vector ID -> leaf node name
        self.vector_to_leaf: dict[int, str] = {}
        # Nodes grouped by depth for fast diversity_level queries
        self.nodes_by_depth: dict[int, list[str]] = {}
        # Tracking state
        self.seen: set[str] = set()
        self._labeled: set[int] = set()

        if vectors:
            ids = list(vectors.keys())
            vecs = np.array([vectors[i] for i in ids], dtype=np.float32)
            self._build_node("0", ids, vecs, depth=0, parent=None)

    def _build_node(
        self,
        name: str,
        ids: list[int],
        vecs: np.ndarray,
        depth: int,
        parent: str | None,
    ) -> None:
        """Recursively build a tree node and its children."""
        self.nodes[name] = {
            "ids": ids,
            "children": [],
            "depth": depth,
            "parent": parent,
        }
        self.nodes_by_depth.setdefault(depth, []).append(name)

        # Stop splitting if node is too small or max depth reached
        if len(ids) < self.min_node_size or depth >= self.max_depth:
            for vid in ids:
                self.vector_to_leaf[vid] = name
            return

        actual_k = min(self.k, len(ids))
        if actual_k < 2:
            for vid in ids:
                self.vector_to_leaf[vid] = name
            return

        # Run k-means
        from sklearn.cluster import KMeans  # noqa: PLC0415

        kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vecs)

        children = []
        for ci in range(actual_k):
            mask = labels == ci
            child_ids = [ids[j] for j in range(len(ids)) if mask[j]]
            child_vecs = vecs[mask]
            if len(child_ids) == 0:
                continue
            child_name = name + str(ci)
            children.append(child_name)
            self._build_node(child_name, child_ids, child_vecs, depth + 1, parent=name)

        self.nodes[name]["children"] = children

        # Edge case: all vectors ended up in one cluster or no children
        if not children:
            for vid in ids:
                self.vector_to_leaf[vid] = name

    def lookup(self, vector_id: int) -> str:
        """Return the name of the deepest (leaf) node containing this vector."""
        return self.vector_to_leaf[vector_id]

    def label(self, vector_id: int) -> None:
        """Mark a vector as labeled: update seen status for its leaf and ancestors."""
        self._labeled.add(vector_id)
        leaf = self.vector_to_leaf[vector_id]
        node = leaf
        while node is not None:
            self.seen.add(node)
            node = self.nodes[node]["parent"]

    def unlabel(self, vector_id: int) -> None:
        """Remove a vector's label and propagate unseen status upward."""
        self._labeled.discard(vector_id)
        leaf = self.vector_to_leaf[vector_id]

        # Check if the leaf still has any labeled vectors
        leaf_ids = self.nodes[leaf]["ids"]
        if any(vid in self._labeled for vid in leaf_ids):
            return  # Leaf still seen, ancestors unchanged

        # Leaf is no longer seen
        self.seen.discard(leaf)

        # Walk up ancestors: unmark if no children are seen
        node = self.nodes[leaf]["parent"]
        while node is not None:
            children = self.nodes[node]["children"]
            if any(c in self.seen for c in children):
                break  # At least one child still seen; this ancestor stays seen
            self.seen.discard(node)
            node = self.nodes[node]["parent"]

    def diversity_level(self) -> int:
        """Return the deepest level where all nodes at that level and above are seen.

        Returns -1 if even the root is unseen.
        """
        if not self.nodes:
            return -1

        max_depth = max(self.nodes_by_depth.keys())
        result = -1
        for level in range(max_depth + 1):
            nodes_at_level = self.nodes_by_depth.get(level, [])
            if not nodes_at_level:
                continue
            if all(name in self.seen for name in nodes_at_level):
                result = level
            else:
                break
        return result

    def fractional_diversity_level(self) -> float:
        """Return the diversity level as a float with fractional progress.

        The integer part is the deepest fully-covered level.  The fractional
        part represents progress toward the next level: ``seen / total`` at
        that level.  Returns -1.0 when nothing is seen.  When the tree is
        fully covered, returns ``depth`` (as a float).
        """
        level = self.diversity_level()
        d = self.depth()

        if level < 0:
            # Nothing seen yet — check if any nodes at level 0 are seen
            nodes_at_zero = self.nodes_by_depth.get(0, [])
            if not nodes_at_zero:
                return -1.0
            seen_count = sum(1 for name in nodes_at_zero if name in self.seen)
            if seen_count == 0:
                return -1.0
            # Partial progress toward level 0
            return -1.0 + seen_count / len(nodes_at_zero)

        if level >= d:
            return float(d)

        next_level = level + 1
        nodes_at_next = self.nodes_by_depth.get(next_level, [])
        if not nodes_at_next:
            return float(level)

        seen_count = sum(1 for name in nodes_at_next if name in self.seen)
        return level + seen_count / len(nodes_at_next)

    def next_sample(
        self,
        scores: dict[int, float] | None = None,
        threshold: float | None = None,
    ) -> int | None:
        """Return an element from the first unseen node in BFS order.

        When *scores* is provided, the selection depends on the node's median
        score relative to *threshold*:

        - If *threshold* is given and the median score in the node is **at or
          above** it, the **lowest**-scored element is returned (the one most
          likely to surprise the user in a predominantly-good region).
        - Otherwise the **highest**-scored element is returned (the one most
          likely to surprise the user in a predominantly-bad region).

        When *scores* is ``None``, returns the first element in the node's ID
        list.

        Returns ``None`` if all nodes have been seen.
        """
        if not self.nodes:
            return None

        queue = deque(["0"])
        while queue:
            name = queue.popleft()
            if name not in self.seen:
                ids = self.nodes[name]["ids"]
                if scores is not None:
                    node_scores = [scores.get(i, 0.0) for i in ids]
                    if threshold is not None:
                        median = float(np.median(node_scores))
                        if median >= threshold:
                            return min(ids, key=lambda i: scores.get(i, 0.0))
                    return max(ids, key=lambda i: scores.get(i, 0.0))
                return ids[0]
            children = self.nodes[name]["children"]
            queue.extend(children)

        return None

    @property
    def labeled_ids(self) -> set[int]:
        """Return the current set of labeled vector IDs."""
        return set(self._labeled)

    def depth(self) -> int:
        """Return the maximum depth of the tree."""
        if not self.nodes_by_depth:
            return -1
        return max(self.nodes_by_depth.keys())

    def span_info(self) -> dict:
        """Return span level details for the labeling progress indicator.

        Returns a dict with:
        - level: deepest fully-seen level (-1 if none)
        - fractional_level: float diversity level with partial progress
        - depth: max tree depth
        - next_level_seen: how many nodes at the next incomplete level are seen
        - next_level_total: total nodes at the next incomplete level
        """
        level = self.diversity_level()
        frac = self.fractional_diversity_level()
        d = self.depth()

        next_level = level + 1
        if next_level > d:
            # All levels fully covered
            return {
                "level": level,
                "fractional_level": round(frac, 4),
                "diversity_level": round(frac, 4),
                "depth": d,
                "max_level": d,
                "next_level_seen": 0,
                "next_level_total": 0,
            }

        nodes_at_next = self.nodes_by_depth.get(next_level, [])
        seen_count = sum(1 for name in nodes_at_next if name in self.seen)

        return {
            "level": level,
            "fractional_level": round(frac, 4),
            "diversity_level": round(frac, 4),
            "depth": d,
            "max_level": d,
            "next_level_seen": seen_count,
            "next_level_total": len(nodes_at_next),
        }
