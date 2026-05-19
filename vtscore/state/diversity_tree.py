"""Diversity Tree: hierarchical k-means clustering for diverse sampling.

A k-Diversity Tree recursively partitions a set of vectors using k-means
clustering. Each node tracks a "seen" flag driven by label activity.

The tree supports:
- Lookup: given a vector ID, return the name of its deepest (leaf) node.
- Labeling: when a vector is labeled, mark its leaf and all ancestors as seen.
- Unlabeling: when a label is removed, propagate unseen status upward.
- Diversity level: the number of consecutive seen nodes in BFS order.
- Next sample: a surprise-maximising element of the first unseen node in BFS order.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable

import numpy as np

# Imported eagerly so the ~1s sklearn cold-import is paid *before* the
# progress bar reports `(0, estimated_total_work)`.  When the import was
# lazy (inside `_build_node`) the bar appeared stuck at `(0, N)` while
# sklearn loaded on the first `_build_node` call.
from sklearn.cluster import KMeans

DIVERSITY_TREE_DEFAULT_K = 2
DIVERSITY_TREE_MAX_DEPTH = 10
DIVERSITY_TREE_MIN_NODE_SIZE = 20
_N_INIT = 10  # number of k-means initialisations per node


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
        on_progress: Callable[[int, int], None] | None = None,
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
            total = len(ids)
            vecs = np.array([vectors[i] for i in ids], dtype=np.float32)
            self._on_progress = on_progress
            # Estimate total k-means work, weighted by vector count.
            # K-means cost is proportional to the number of vectors being
            # clustered, so we weight each fit by len(vectors).  At each
            # level of a balanced k-ary tree the total vectors processed
            # equals *total* (all vectors are partitioned among nodes at
            # that level), so estimated_total ≈ num_levels * total * _N_INIT.
            # This keeps the progress bar proportional to wall-clock time
            # instead of sitting near 0% during the expensive root clustering.
            if total >= min_node_size and total >= 2 * k:
                num_levels = max(1, math.ceil(math.log(total / min_node_size, k)))
                num_levels = min(num_levels, max_depth)
            else:
                num_levels = 0
            self._estimated_total_work = max(num_levels * total * _N_INIT, 1)
            self._work_done = 0
            if on_progress:
                on_progress(0, self._estimated_total_work)
            self._build_node("0", ids, vecs, depth=0, parent=None)
            # Ensure we hit 100% at the end
            if on_progress:
                on_progress(self._estimated_total_work, self._estimated_total_work)
            self._on_progress = None

    def _build_node(  # noqa: C901
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

        # Run k-means in individual inits for granular progress reporting.
        # Each of the N_INIT runs reports progress separately, so the UI
        # updates during the expensive root clustering instead of sitting
        # at 0% until it finishes.
        best_labels = None
        best_inertia = float("inf")
        for init_i in range(_N_INIT):
            km = KMeans(
                n_clusters=actual_k,
                random_state=42 + init_i,
                n_init=1,  # pyright: ignore[reportArgumentType]
            )
            candidate_labels = km.fit_predict(vecs)
            inertia = km.inertia_
            if inertia is not None and inertia < best_inertia:
                best_inertia = inertia
                best_labels = candidate_labels

            # Report progress after each init, weighted by vector count
            # (k-means cost is proportional to the number of vectors).
            self._work_done += len(ids)
            if self._on_progress:
                self._on_progress(
                    min(self._work_done, self._estimated_total_work),
                    self._estimated_total_work,
                )

        labels = best_labels

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
        """Return the number of consecutive seen nodes in BFS order.

        Traverses the tree in breadth-first order and counts how many nodes
        are seen before hitting the first unseen node.  Returns 0 when nothing
        is seen or the tree is empty.
        """
        if not self.nodes:
            return 0

        queue = deque(["0"])
        count = 0
        while queue:
            name = queue.popleft()
            if name not in self.seen:
                break
            count += 1
            queue.extend(self.nodes[name]["children"])
        return count

    @property
    def total_nodes(self) -> int:
        """Return the total number of nodes in the tree."""
        return len(self.nodes)

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
        - level: number of consecutive BFS-order seen nodes
        - diversity_level: same as level
        - depth: total number of nodes (the maximum diversity level)
        - max_level: alias for depth
        """
        level = self.diversity_level()
        total = self.total_nodes

        return {
            "level": level,
            "diversity_level": level,
            "depth": total,
            "max_level": total,
        }
