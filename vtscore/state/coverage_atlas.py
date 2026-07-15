"""Coverage Atlas: hierarchical partition with evidence channels and calibrated typicality.

The atlas recursively partitions a set of embeddings with k-means, like the
Diversity Tree it replaces, but keeps what the tree threw away
(see ``docs/plans/coverage-atlas.md``):

- **Geometry**: vectors are mean-centered and re-normalized to the unit
  sphere before partitioning.  Contrastive embeddings concentrate in a
  narrow cone, so raw cosines are uniformly high; centering restores
  contrast.  The centering vector is part of the structure.
- **Evidence channels**: each node counts labeled evidence *per class*
  (``n_pos`` / ``n_neg``) instead of a binary "seen" flag, so samplers and
  future queries can distinguish "verified good here" from "verified bad
  here" from "never exercised".
- **Moments**: each node stores its mean direction ``mu`` and resultant
  length ``rbar`` — the sufficient statistics of a von Mises–Fisher
  component — so the atlas read at any depth is a multiresolution mixture
  model.
- **Calibration**: each node stores a quantile grid of its own points'
  typicality ``t(x) = mu . x``, so a query returns a p-value ("what fraction of the
  build data looks less typical than x?"), never a raw distance.  This is
  what powers domain-shift detection when a detector trained on dataset A
  is pointed at dataset B.

The atlas supports:

- Lookup: given a vector ID, return the name of its deepest (leaf) node.
- Evidence: when a vector is labeled good/bad, count it in its leaf and all
  ancestors; unlabeling decrements.
- Coverage level: the number of consecutive evidence-bearing nodes in BFS
  order (the autopilot's span progress).
- Next sample: a surprise-maximising element of the first evidence-free
  node in BFS order — drawn from the node's typical half when the node has
  a concentrated direction, so the surprise is a representative
  counterexample rather than a lone oddball — with siblings visited
  largest-first so each click covers the biggest unexplored region.
- Typicality: calibrated p-values for arbitrary query vectors, and a
  dataset-level domain-shift report built on them.
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

COVERAGE_ATLAS_DEFAULT_K = 3
COVERAGE_ATLAS_MAX_DEPTH = 10
COVERAGE_ATLAS_MIN_NODE_SIZE = 20
_N_INIT = 10  # number of k-means initialisations per node (small nodes)

# Soft ceiling on leaf count used by :func:`auto_max_depth` so a huge dataset
# can't explode the tree into tens of thousands of nodes.
_MAX_LEAVES = 4_000

# Minimum node population for typicality calibration: routing scores a query
# at every ancestor with at least this many points, so sparse regions
# terminate shallow — which *is* the adaptive bandwidth.  Matches the split
# floor (min_node_size) so every splittable node calibrates: with leave-one-out
# quantiles even 20 points give usable tail estimates, and requiring more
# would leave small datasets calibrating only at the root, whose mean
# direction is degenerate after centering (a single isotropic cluster has no
# preferred direction left, so root-only typicality carries no signal).
_CALIBRATION_MIN_NODE = 20

# Minimum resultant length (rbar = ||sum of unit vectors|| / n) for a node to
# calibrate.  The root sits at rbar ~ 0 by construction — the build subtracts
# the dataset mean, so the centered vectors' resultant vanishes — and a node
# with no concentrated direction has (a) a meaningless mu and (b) leave-one-out
# calibration scores systematically far below full-mu query scores (removing
# one point from a near-zero resultant flips it), which reads every query as
# "more typical than everything".  K-means cells are cohesive and land well
# above this floor.
_CALIBRATION_MIN_RBAR = 0.1

# Quantile grid stored per node for typicality calibration.  21 points
# (ventiles) rather than 11 halves the interpolation error in the tails,
# where the p-values that matter for domain-shift detection live.
_CALIBRATION_GRID = [i / 20.0 for i in range(21)]

_EPS = 1e-12


def _n_init_for(node_size: float) -> int:
    """Number of k-means restarts to run for a node of *node_size* vectors.

    Large nodes converge reliably from a single init, so spending 10 restarts
    on them (the small-node default) is wasted compute that dominates build
    time at scale.  Nodes of 1 000 or fewer keep the full ``_N_INIT`` restarts,
    so atlases over the test-sized datasets are bit-for-bit unchanged.
    """
    if node_size > 10_000:
        return 3
    if node_size > 1_000:
        return 5
    return _N_INIT


def auto_max_depth(
    n: int,
    k: int = COVERAGE_ATLAS_DEFAULT_K,
    min_node_size: int = COVERAGE_ATLAS_MIN_NODE_SIZE,
) -> int:
    """Depth cap that bounds the leaf count at ``_MAX_LEAVES`` for large *n*.

    ``min_node_size`` already bounds the leaf count at ``n / min_node_size``
    (a node smaller than it is never split), so when that bound is within the
    ``_MAX_LEAVES`` budget the cap is unnecessary and the full
    ``COVERAGE_ATLAS_MAX_DEPTH`` is returned - a structural no-op for normal
    datasets, including the skewed k-means splits whose dense branches reach
    deeper than a balanced tree would.  Only once ``n / min_node_size`` exceeds
    the budget (very large datasets) is the depth clamped so ``k**depth`` stays
    under ``_MAX_LEAVES``.  Always returns at least 1.
    """
    if n <= 0:
        return COVERAGE_ATLAS_MAX_DEPTH
    if n / max(min_node_size, 1) <= _MAX_LEAVES:
        return COVERAGE_ATLAS_MAX_DEPTH
    base = max(k, 2)
    cap = int(math.floor(math.log(_MAX_LEAVES, base)))
    return max(1, min(COVERAGE_ATLAS_MAX_DEPTH, cap))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return *matrix* with each row scaled to unit norm (zero rows stay zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.where(norms > _EPS, matrix / np.maximum(norms, _EPS), 0.0).astype(np.float32)


class CoverageAtlas:
    """Hierarchical evidence-aware partition over centered spherical embeddings.

    Parameters
    ----------
    vectors : dict[int, np.ndarray]
        Mapping of vector ID to embedding array (unit-normalized at ingest).
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
        k: int = COVERAGE_ATLAS_DEFAULT_K,
        max_depth: int = COVERAGE_ATLAS_MAX_DEPTH,
        min_node_size: int = COVERAGE_ATLAS_MIN_NODE_SIZE,
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

        # Node storage: name -> {ids, children, depth, parent, n, mu, rbar,
        # t_quantiles}.  ``ids`` is sorted most-typical-first (descending
        # mu . x) so ids[0] is the node's representative element.  Node records
        # are *immutable* once built — evidence lives in the separate overlay
        # dicts below — so :meth:`structural_clone` can share this map (and the
        # two below) by reference across atlases with independent labels.
        self.nodes: dict[str, dict] = {}
        # Vector ID -> leaf node name
        self.vector_to_leaf: dict[int, str] = {}
        # Nodes grouped by depth for fast level queries
        self.nodes_by_depth: dict[int, list[str]] = {}
        # Geometry: dataset mean direction subtracted before renormalizing.
        self.center: np.ndarray = np.zeros(0, dtype=np.float32)
        # Label overlay (session state, not persisted; per-instance so a
        # structural clone's labels never touch the original's):
        #   node name -> positive / negative evidence count (absent = 0),
        #   vector ID -> is_good.
        self._n_pos: dict[str, int] = {}
        self._n_neg: dict[str, int] = {}
        self._labeled: dict[int, bool] = {}

        if vectors:
            ids = list(vectors.keys())
            total = len(ids)
            raw = np.array([vectors[i] for i in ids], dtype=np.float32)
            self.center = raw.mean(axis=0).astype(np.float32)
            vecs = _normalize_rows(raw - self.center)
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
        node = self._make_node(ids, vecs, depth, parent)
        self.nodes[name] = node
        ids = node["ids"]  # re-sorted most-typical-first by _make_node
        # Keep the vector matrix aligned with the re-sorted ids so cluster
        # masks below index the right rows.
        vecs = vecs[node.pop("_order")]
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
            elif best_labels is None:
                # The backend contract allows ``inertia=None`` (a backend
                # that doesn't report it).  Keep the labels as a fallback
                # candidate so ``best_labels`` can't stay None and crash the
                # ``labels == ci`` mask below; a later init with a real
                # inertia still wins.
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

        clusters = []
        for ci in range(actual_k):
            mask = labels == ci
            child_ids = [ids[j] for j in range(len(ids)) if mask[j]]
            if len(child_ids) == 0:
                continue
            clusters.append((child_ids, vecs[mask]))

        # Largest cluster first, so BFS traversal (coverage level, next
        # sample) reaches the biggest unexplored regions before their
        # smaller siblings — best coverage gain per click.
        clusters.sort(key=lambda c: len(c[0]), reverse=True)

        children = []
        for ci, (child_ids, child_vecs) in enumerate(clusters):
            child_name = name + str(ci)
            children.append(child_name)
            self._build_node(child_name, child_ids, child_vecs, depth + 1, parent=name)

        node["children"] = children

        # Edge case: all vectors ended up in one cluster or no children
        if not children:
            for vid in ids:
                self.vector_to_leaf[vid] = name

    def _make_node(self, ids: list[int], vecs: np.ndarray, depth: int, parent: str | None) -> dict:
        """Return a node record with moments, calibration quantiles, and evidence.

        Sorts ``ids`` most-typical-first (descending ``mu . x``) and stashes the
        sort permutation under ``"_order"`` so the caller can reorder its
        vector matrix to match before splitting.
        """
        resultant = vecs.sum(axis=0)
        rlen = float(np.linalg.norm(resultant))
        n = len(ids)
        if rlen > _EPS:
            mu = (resultant / rlen).astype(np.float32)
            t = vecs @ mu
            # Calibration scores are leave-one-out: each build point is scored
            # against the mean direction of the *other* points, via the closed
            # form mu_loo(x) . x = (R.x - 1) / ||R - x|| for unit x.  Scoring a
            # point against a mean it helped shape is optimistic, which made
            # fresh in-domain queries read as systematically atypical.
            r_dot = vecs @ resultant
            denom = np.sqrt(np.maximum(rlen * rlen - 2.0 * r_dot + 1.0, _EPS))
            t_loo = (r_dot - 1.0) / denom
        else:
            # Degenerate node (e.g. antipodal pair): no preferred direction.
            mu = np.zeros(vecs.shape[1] if vecs.ndim == 2 else 0, dtype=np.float32)
            t = np.zeros(n, dtype=np.float32)
            t_loo = t
        order = np.argsort(-t, kind="stable")
        return {
            "ids": [ids[j] for j in order],
            "children": [],
            "depth": depth,
            "parent": parent,
            "n": n,
            "mu": mu,
            "rbar": rlen / n if n else 0.0,
            "t_quantiles": [float(q) for q in np.quantile(t_loo, _CALIBRATION_GRID)] if n else [],
            "_order": order,
        }

    def lookup(self, vector_id: int) -> str:
        """Return the name of the deepest (leaf) node containing this vector."""
        return self.vector_to_leaf[vector_id]

    # ------------------------------------------------------------------
    # Evidence channels
    # ------------------------------------------------------------------

    def label(self, vector_id: int, good: bool) -> None:
        """Count a good/bad label as evidence in the vector's leaf and ancestors.

        Re-labeling with the other class moves the evidence between channels.
        """
        prev = self._labeled.get(vector_id)
        if prev == good:
            return
        if prev is not None:
            self._shift_evidence(vector_id, prev, -1)
        self._labeled[vector_id] = good
        self._shift_evidence(vector_id, good, +1)

    def unlabel(self, vector_id: int) -> None:
        """Remove a vector's label and decrement evidence along its path."""
        prev = self._labeled.pop(vector_id, None)
        if prev is None:
            return
        self._shift_evidence(vector_id, prev, -1)

    def _shift_evidence(self, vector_id: int, good: bool, delta: int) -> None:
        overlay = self._n_pos if good else self._n_neg
        node = self.vector_to_leaf[vector_id]
        while node is not None:
            count = overlay.get(node, 0) + delta
            if count:
                overlay[node] = count
            else:
                overlay.pop(node, None)
            node = self.nodes[node]["parent"]

    def n_pos(self, name: str) -> int:
        """Return the positive-evidence count accumulated at node *name*."""
        return self._n_pos.get(name, 0)

    def n_neg(self, name: str) -> int:
        """Return the negative-evidence count accumulated at node *name*."""
        return self._n_neg.get(name, 0)

    def reset_labeled(self) -> None:
        """Rewind the label overlay: drop every evidence count and labeled ID.

        Used when the labeled set is replaced wholesale - e.g. votes are
        cleared, or the active detector is swapped on the same dataset and
        the atlas's evidence state needs to be rebuilt from the new
        detector's votes (see :func:`resync_coverage_atlas_to_detector`) - or
        when the per-step labeling-progress cache is truncated and its atlas
        is rewound and replayed instead of rebuilt.  The shared node structure
        is untouched, so it is safe to call on a :meth:`structural_clone`.
        """
        self._n_pos.clear()
        self._n_neg.clear()
        self._labeled.clear()

    def _covered(self, name: str) -> bool:
        return self._n_pos.get(name, 0) + self._n_neg.get(name, 0) > 0

    def coverage_level(self) -> int:
        """Return the number of consecutive evidence-bearing nodes in BFS order.

        Traverses the tree in breadth-first order and counts how many nodes
        carry labeled evidence before hitting the first evidence-free node.
        Returns 0 when nothing is labeled or the atlas is empty.
        """
        if not self.nodes:
            return 0

        queue = deque(["0"])
        count = 0
        while queue:
            name = queue.popleft()
            if not self._covered(name):
                break
            count += 1
            queue.extend(self.nodes[name]["children"])
        return count

    @property
    def total_nodes(self) -> int:
        """Return the total number of nodes in the atlas."""
        return len(self.nodes)

    def next_sample(
        self,
        scores: dict[int, float] | None = None,
        threshold: float | None = None,
    ) -> int | None:
        """Return an element from the first evidence-free node in BFS order.

        Siblings are stored largest-first, so the BFS frontier reaches the
        biggest unexplored regions before their smaller siblings.

        When *scores* is provided, the selection depends on the node's median
        score relative to *threshold*:

        - If *threshold* is given and the median score in the node is **at or
          above** it, the **lowest**-scored element is returned (the one most
          likely to surprise the user in a predominantly-good region).
        - Otherwise the **highest**-scored element is returned (the one most
          likely to surprise the user in a predominantly-bad region).

        The surprise extremum is taken over the node's **typical half**
        (``ids`` is sorted by descending ``mu . x``), not all of it: an
        extreme score on an atypical item is disproportionately often a lone
        oddball — a corrupt file, a weird crop — whose flip says nothing
        about the region.  A flip on a *typical* item is evidence of a real
        hidden pocket, and when the typical extremum does not flip, the
        node's presumption has been stress-tested at its (representative)
        weakest point.  The median that decides the probe direction still
        spans the whole node — the presumption is about the region.  Nodes
        with no concentrated direction (``rbar`` below the calibration
        floor — notably the root, whose resultant vanishes by construction
        after centering) probe the whole node: their typicality ordering is
        noise, and tempering by it would just hide an arbitrary half.

        When *scores* is ``None``, returns the node's most typical element,
        so an unscored click still lands on a representative of the
        unexplored region.

        Returns ``None`` if all nodes carry evidence.
        """
        if not self.nodes:
            return None

        queue = deque(["0"])
        while queue:
            name = queue.popleft()
            if not self._covered(name):
                node = self.nodes[name]
                ids = node["ids"]
                if scores is not None:
                    if node["rbar"] >= _CALIBRATION_MIN_RBAR:
                        pool = ids[: max(1, math.ceil(len(ids) / 2))]
                    else:
                        pool = ids
                    if threshold is not None:
                        median = float(np.median([scores.get(i, 0.0) for i in ids]))
                        if median >= threshold:
                            return min(pool, key=lambda i: scores.get(i, 0.0))
                    return max(pool, key=lambda i: scores.get(i, 0.0))
                return ids[0]
            queue.extend(self.nodes[name]["children"])

        return None

    @property
    def labeled_ids(self) -> set[int]:
        """Return the current set of labeled vector IDs."""
        return set(self._labeled)

    def depth(self) -> int:
        """Return the maximum depth of the atlas."""
        if not self.nodes_by_depth:
            return -1
        return max(self.nodes_by_depth.keys())

    # ------------------------------------------------------------------
    # Typicality (Q1): calibrated p-values against the build distribution
    # ------------------------------------------------------------------

    def typicality_pvalues(self, matrix: np.ndarray) -> np.ndarray:
        """Return one calibrated typicality p-value per row of *matrix*.

        Each query is centered into the atlas frame, routed down the tree
        (cosine-nearest child), and scored at the deepest node with at least
        ``_CALIBRATION_MIN_NODE`` build points — sparse regions terminate
        shallow, which is the adaptive bandwidth.  The p-value is the
        fraction of that node's own points whose typicality ``t = mu . x``
        is below the query's, read off the stored quantiles: small p means
        "almost nothing the atlas was built on looks this atypical here".

        Returns an array of 1.0 (fully typical) for an empty atlas.
        """
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        n_items = matrix.shape[0]
        if not self.nodes or n_items == 0:
            return np.ones(n_items, dtype=np.float32)

        queries = _normalize_rows(matrix - self.center)
        p_sums = np.zeros(n_items, dtype=np.float64)
        p_counts = np.zeros(n_items, dtype=np.int64)

        def _score(name: str, idx: np.ndarray) -> None:
            node = self.nodes[name]
            t = queries[idx] @ node["mu"]
            grid = np.asarray(node["t_quantiles"], dtype=np.float64)
            p = np.interp(t, grid, _CALIBRATION_GRID, left=0.0, right=1.0)
            # Floor at the resolution one build point contributes, so a
            # far-field query reads "less typical than everything" without
            # returning an exact impossible zero.
            p_sums[idx] += np.maximum(p, 0.5 / max(node["n"], 1))
            p_counts[idx] += 1

        def _route(name: str, idx: np.ndarray) -> None:
            node = self.nodes[name]
            # Score at every calibrated node along the path, not only the
            # deepest one: a hard partition has boundary artifacts (a fresh
            # in-domain query near a k-means cell edge reads atypical at leaf
            # scale), and averaging the path's p-values smooths them the same
            # way a tree ensemble would, at zero extra build cost.
            if node["n"] >= _CALIBRATION_MIN_NODE and node["rbar"] >= _CALIBRATION_MIN_RBAR and node["t_quantiles"]:
                _score(name, idx)
            children = node["children"]
            if not children:
                return
            mus = np.stack([self.nodes[c]["mu"] for c in children])
            nearest = np.argmax(queries[idx] @ mus.T, axis=1)
            for ci in range(len(children)):
                sub = idx[nearest == ci]
                if sub.size:
                    _route(children[ci], sub)

        _route("0", np.arange(n_items))
        # Items whose whole path was too sparse to calibrate (tiny atlases)
        # are fully typical by convention.
        return np.where(p_counts > 0, p_sums / np.maximum(p_counts, 1), 1.0).astype(np.float32)

    def typicality_pvalue(self, vector: np.ndarray) -> float:
        """Return the calibrated typicality p-value of a single query vector."""
        return float(self.typicality_pvalues(np.asarray(vector)[None, :])[0])

    # ------------------------------------------------------------------
    # Persistence
    #
    # The atlas's *structure* (cluster topology, per-vector leaf assignment,
    # geometry, moments, calibration) is expensive to rebuild - hierarchical
    # k-means over every embedding - so a dataset pickle caches it to skip the
    # rebuild on reload.  Only plain Python containers and numpy arrays are
    # emitted (ints, strs, floats, lists, dicts, ndarrays), never the
    # ``CoverageAtlas`` class itself, so the restricted unpickler in
    # ``vtscore.security.pickle`` (allowlist: plain types + numpy) accepts the
    # cache without widening its class allowlist.
    #
    # Evidence state (``n_pos`` / ``n_neg`` / ``_labeled``) is deliberately
    # *not* persisted: it is per-session label state, repopulated from the
    # active detector's votes via ``resync_coverage_atlas_to_detector`` after
    # a restore.
    # ------------------------------------------------------------------

    _SERIAL_FORMAT = "coverage-atlas/1"

    def to_serializable(self) -> dict:
        """Return a plain-dict snapshot of the atlas structure for caching.

        Contains only ints, strs, floats, lists, dicts and numpy arrays so it
        survives the restricted unpickler.  Node ``mu`` vectors are stored as
        float16 to halve the cache footprint.  Pair with
        :meth:`from_serializable`.
        """
        nodes = {}
        for name, node in self.nodes.items():
            nodes[name] = {
                "ids": node["ids"],
                "children": node["children"],
                "depth": node["depth"],
                "parent": node["parent"],
                "n": node["n"],
                "mu": node["mu"].astype(np.float16),
                "rbar": node["rbar"],
                "t_quantiles": node["t_quantiles"],
            }
        return {
            "format": self._SERIAL_FORMAT,
            "k": self.k,
            "max_depth": self.max_depth,
            "min_node_size": self.min_node_size,
            "center": self.center,
            "nodes": nodes,
            "vector_to_leaf": self.vector_to_leaf,
            "nodes_by_depth": self.nodes_by_depth,
        }

    @classmethod
    def from_serializable(cls, data: object) -> CoverageAtlas:
        """Reconstruct an atlas from a :meth:`to_serializable` snapshot.

        Bypasses ``__init__`` (and thus the expensive k-means build).  Raises
        ``ValueError`` when the snapshot is missing keys or carries an
        unrecognised format version (including old diversity-tree caches), so
        callers can fall back to a rebuild.
        """
        if not isinstance(data, dict) or data.get("format") != cls._SERIAL_FORMAT:
            got = data.get("format") if isinstance(data, dict) else type(data).__name__
            raise ValueError(f"Unrecognised coverage-atlas cache format: {got!r}")
        atlas = cls.__new__(cls)
        try:
            atlas.k = data["k"]
            atlas.max_depth = data["max_depth"]
            atlas.min_node_size = data["min_node_size"]
            atlas.center = np.asarray(data["center"], dtype=np.float32)
            atlas.nodes = {
                name: {
                    "ids": node["ids"],
                    "children": node["children"],
                    "depth": node["depth"],
                    "parent": node["parent"],
                    "n": node["n"],
                    "mu": np.asarray(node["mu"], dtype=np.float32),
                    "rbar": node["rbar"],
                    "t_quantiles": node["t_quantiles"],
                }
                for name, node in data["nodes"].items()
            }
            atlas.vector_to_leaf = data["vector_to_leaf"]
            atlas.nodes_by_depth = data["nodes_by_depth"]
        except KeyError as exc:
            raise ValueError(f"Incomplete coverage-atlas cache: missing {exc}") from exc
        atlas._n_pos = {}
        atlas._n_neg = {}
        atlas._labeled = {}
        return atlas

    def structural_clone(self) -> CoverageAtlas:
        """Return a new atlas sharing this atlas's structure, with empty labels.

        The clone reuses the node table, per-vector leaf assignment, depth
        index, geometry, and build parameters *by reference* - none of which
        the evidence path ever rewrites - while starting with a fresh, empty
        label overlay (``_n_pos`` / ``_n_neg`` / ``_labeled``).  Building the
        structure is the expensive part (hierarchical k-means over every
        embedding); cloning skips it entirely, so a second atlas over the same
        id set - e.g. the labeling-progress per-step atlas mirroring the
        dataset context's - costs a few dict allocations instead of a full
        re-fit.  Labeling either atlas never affects the other because the
        overlay dicts are per-instance.
        """
        clone = self.__class__.__new__(self.__class__)
        clone.k = self.k
        clone.max_depth = self.max_depth
        clone.min_node_size = self.min_node_size
        clone.center = self.center
        clone.nodes = self.nodes
        clone.vector_to_leaf = self.vector_to_leaf
        clone.nodes_by_depth = self.nodes_by_depth
        clone._n_pos = {}
        clone._n_neg = {}
        clone._labeled = {}
        return clone

    def span_info(self) -> dict:
        """Return span level details for the labeling progress indicator.

        Returns a dict with:
        - level: number of consecutive BFS-order evidence-bearing nodes
        - diversity_level: same as level
        - depth: total number of nodes (the maximum coverage level)
        - max_level: alias for depth
        """
        level = self.coverage_level()
        total = self.total_nodes

        return {
            "level": level,
            "diversity_level": level,
            "depth": total,
            "max_level": total,
        }


def domain_shift_report(atlas: CoverageAtlas, matrix: np.ndarray, alpha: float = 0.05) -> dict:
    """Compare a dataset's embeddings against an atlas and report domain shift.

    Under the null "the queried items are drawn from the atlas's build
    distribution", typicality p-values are roughly uniform, so about *alpha*
    of them fall below *alpha*.  A large excess means the atlas's detector is
    being pointed at data unlike what it was trained on and shouldn't be
    trusted without hands-on verification.

    Returns a dict with ``n_items``, ``alpha``, ``frac_atypical`` (observed
    fraction with p < alpha), ``expected_atypical`` (= alpha), ``z_score``
    (binomial z of the excess), ``median_pvalue``, and ``shifted`` (True when
    the excess is both statistically clear, z > 3, and practically large,
    at least double the expected rate).
    """
    pvals = atlas.typicality_pvalues(matrix)
    n = int(pvals.shape[0])
    if n == 0:
        return {
            "n_items": 0,
            "alpha": alpha,
            "frac_atypical": 0.0,
            "expected_atypical": alpha,
            "z_score": 0.0,
            "median_pvalue": 1.0,
            "shifted": False,
        }
    frac = float(np.mean(pvals < alpha))
    se = math.sqrt(alpha * (1.0 - alpha) / n)
    z = (frac - alpha) / se if se > 0 else 0.0
    return {
        "n_items": n,
        "alpha": alpha,
        "frac_atypical": frac,
        "expected_atypical": alpha,
        "z_score": z,
        "median_pvalue": float(np.median(pvals)),
        "shifted": bool(z > 3.0 and frac >= 2.0 * alpha),
    }
