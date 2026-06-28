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
# sklearn loaded on the first `_build_node` call.  The estimator itself is
# fitted via ``kmeans_fit_predict`` (see below), which routes to
# ``cuml.cluster.KMeans`` on a usable GPU and falls back to this sklearn
# import otherwise.
from sklearn.cluster import KMeans  # noqa: F401 — warms the CPU cold-import; kmeans_fit_predict is the fallback

from vtscore.gpu_backends import kmeans_fit_predict

DIVERSITY_TREE_DEFAULT_K = 2
DIVERSITY_TREE_MAX_DEPTH = 10
DIVERSITY_TREE_MIN_NODE_SIZE = 20
_N_INIT = 10  # number of k-means initialisations per node (small nodes)

# Soft ceiling on leaf count used by :func:`auto_max_depth` so a huge dataset
# can't explode the tree into tens of thousands of nodes.
_MAX_LEAVES = 4_000


def _n_init_for(node_size: float) -> int:
    """Number of k-means restarts to run for a node of *node_size* vectors.

    Large nodes converge reliably from a single init, so spending 10 restarts
    on them (the small-node default) is wasted compute that dominates build
    time at scale.  Nodes of 1 000 or fewer keep the full ``_N_INIT`` restarts,
    so trees over the test-sized datasets are bit-for-bit unchanged.
    """
    if node_size > 10_000:
        return 3
    if node_size > 1_000:
        return 5
    return _N_INIT


def auto_max_depth(
    n: int,
    k: int = DIVERSITY_TREE_DEFAULT_K,
    min_node_size: int = DIVERSITY_TREE_MIN_NODE_SIZE,
) -> int:
    """Depth cap that bounds the leaf count at ``_MAX_LEAVES`` for large *n*.

    ``min_node_size`` already bounds the leaf count at ``n / min_node_size``
    (a node smaller than it is never split), so when that bound is within the
    ``_MAX_LEAVES`` budget the cap is unnecessary and the full
    ``DIVERSITY_TREE_MAX_DEPTH`` is returned - a structural no-op for normal
    datasets, including the skewed k-means splits whose dense branches reach
    deeper than a balanced tree would.  Only once ``n / min_node_size`` exceeds
    the budget (very large datasets) is the depth clamped so ``k**depth`` stays
    under ``_MAX_LEAVES``.  Always returns at least 1.
    """
    if n <= 0:
        return DIVERSITY_TREE_MAX_DEPTH
    if n / max(min_node_size, 1) <= _MAX_LEAVES:
        return DIVERSITY_TREE_MAX_DEPTH
    base = max(k, 2)
    cap = int(math.floor(math.log(_MAX_LEAVES, base)))
    return max(1, min(DIVERSITY_TREE_MAX_DEPTH, cap))


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
            # Per-level node size shrinks by ~k each level, so the k-means
            # restart count (see ``_n_init_for``) drops as we descend.  Weight
            # the estimate by the per-level restart count so the progress bar
            # tracks wall-clock now that large nodes run fewer restarts.
            estimated_work = 0.0
            node_size = float(total)
            for _ in range(num_levels):
                estimated_work += total * _n_init_for(node_size)
                node_size /= k
            self._estimated_total_work = max(int(estimated_work), 1)
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
        n_init = _n_init_for(len(ids))
        for init_i in range(n_init):
            candidate_labels, inertia = kmeans_fit_predict(
                vecs,
                n_clusters=actual_k,
                random_state=42 + init_i,
                n_init=1,
            )
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

    def reset_seen(self) -> None:
        """Mark every node as un-seen and forget every labeled vector ID.

        Used when the labeled set is replaced wholesale - e.g. votes are
        cleared, or the active detector is swapped on the same dataset and
        the tree's per-leaf seen state needs to be rebuilt from the new
        detector's votes (see :func:`resync_diversity_tree_to_detector`).
        """
        self.seen.clear()
        self._labeled.clear()

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

    # ------------------------------------------------------------------
    # Persistence
    #
    # The tree's *structure* (cluster topology + per-vector leaf assignment)
    # is expensive to rebuild - hierarchical k-means over every embedding - so
    # a dataset pickle caches it to skip the rebuild on reload.  Only plain
    # Python containers are emitted (ints, strs, lists, dicts), never the
    # ``DiversityTree`` class itself, so the restricted unpickler in
    # ``vtscore.security.pickle`` (allowlist: plain types + numpy) accepts the
    # cache without widening its class allowlist.
    #
    # ``seen`` / ``_labeled`` are deliberately *not* persisted: they are
    # per-session label state, repopulated from the active detector's votes via
    # ``resync_diversity_tree_to_detector`` after a restore.
    # ------------------------------------------------------------------

    _SERIAL_FORMAT = 1

    def to_serializable(self) -> dict:
        """Return a plain-dict snapshot of the tree structure for caching.

        Contains only ints, strs, lists and dicts so it survives the
        restricted unpickler.  Pair with :meth:`from_serializable`.
        """
        return {
            "format": self._SERIAL_FORMAT,
            "k": self.k,
            "max_depth": self.max_depth,
            "min_node_size": self.min_node_size,
            "nodes": self.nodes,
            "vector_to_leaf": self.vector_to_leaf,
            "nodes_by_depth": self.nodes_by_depth,
        }

    @classmethod
    def from_serializable(cls, data: object) -> DiversityTree:
        """Reconstruct a tree from a :meth:`to_serializable` snapshot.

        Bypasses ``__init__`` (and thus the expensive k-means build).  Raises
        ``ValueError`` when the snapshot is missing keys or carries an
        unrecognised format version, so callers can fall back to a rebuild.
        """
        if not isinstance(data, dict) or data.get("format") != cls._SERIAL_FORMAT:
            got = data.get("format") if isinstance(data, dict) else type(data).__name__
            raise ValueError(f"Unrecognised diversity-tree cache format: {got!r}")
        tree = cls.__new__(cls)
        try:
            tree.k = data["k"]
            tree.max_depth = data["max_depth"]
            tree.min_node_size = data["min_node_size"]
            tree.nodes = data["nodes"]
            tree.vector_to_leaf = data["vector_to_leaf"]
            tree.nodes_by_depth = data["nodes_by_depth"]
        except KeyError as exc:
            raise ValueError(f"Incomplete diversity-tree cache: missing {exc}") from exc
        tree.seen = set()
        tree._labeled = set()
        return tree

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
