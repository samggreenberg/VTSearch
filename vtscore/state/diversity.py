"""Diversity tree management for diverse media sampling.

Operates on the active :class:`DatasetContext` (for the tree itself) and the
active :class:`DetectorContext` (for the labels that get replayed into it).
Functions resolve the contexts themselves — no module-level proxy names are
imported.  See Phase 3 of ``../docs/architecture.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vtscore.state.core import (
    DatasetContext,
    _state_lock,
    get_active_context,
    get_active_detector_context,
)


def build_diversity_tree(
    media_dict: dict[int, dict[str, Any]] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Build a 3-Diversity Tree from media embeddings on the active dataset context.

    Uses the active :class:`DatasetContext`'s medias by default, or an explicit
    *media_dict* if provided.  Existing labels on the active detector context
    (``good_votes`` and ``bad_votes``) are replayed into the new tree so the
    seen state stays accurate.

    *on_progress*, when provided, is called as ``on_progress(current, total)``
    to report how many k-means clustering fits have been completed so far.
    """
    import numpy as np

    from vtscore.state.diversity_tree import DiversityTree

    with _state_lock:
        ds_ctx = get_active_context()
        source = media_dict if media_dict is not None else ds_ctx.medias
        vectors: dict[int, np.ndarray] = {}
        for cid, media in source.items():
            emb = media.get("embedding")
            if emb is not None:
                vectors[cid] = np.asarray(emb, dtype=np.float32)

        if not vectors:
            ds_ctx.diversity_tree = None
            return

        tree = DiversityTree(vectors, k=3, on_progress=on_progress)
        ds_ctx.diversity_tree = tree

        # Replay existing labels so the tree reflects the current vote state.
        det_ctx = get_active_detector_context()
        for cid in det_ctx.good_votes:
            if cid in tree.vector_to_leaf:
                tree.label(cid)
        for cid in det_ctx.bad_votes:
            if cid in tree.vector_to_leaf:
                tree.label(cid)


def build_diversity_tree_for_context(
    ctx: DatasetContext,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Build a diversity tree and store it on a specific context.

    Unlike :func:`build_diversity_tree` this does not touch the active
    context — it operates entirely on *ctx*.  Used by parallel dataset
    loading where the new dataset is not yet active.
    """
    import numpy as np

    from vtscore.state.diversity_tree import DiversityTree

    vectors: dict[int, np.ndarray] = {}
    for cid, media in ctx.medias.items():
        emb = media.get("embedding")
        if emb is not None:
            vectors[cid] = np.asarray(emb, dtype=np.float32)

    if not vectors:
        ctx.diversity_tree = None
        return

    tree = DiversityTree(vectors, k=3, on_progress=on_progress)
    ctx.diversity_tree = tree
    # Fresh context has no votes — nothing to replay.


def get_diversity_tree():
    """Return the active dataset context's DiversityTree instance, or ``None``."""
    with _state_lock:
        return get_active_context().diversity_tree


def diversity_tree_next_sample(
    scores: dict[int, float] | None = None,
    threshold: float | None = None,
) -> int | None:
    """Return the next diverse sample ID, or ``None`` if unavailable.

    When *scores* and *threshold* are provided, the selection picks the
    element most likely to surprise the user: the lowest-scored element
    if the node's median score is above the threshold, or the highest-
    scored element otherwise.
    """
    with _state_lock:
        tree = get_active_context().diversity_tree
        if tree is None:
            return None
        return tree.next_sample(scores=scores, threshold=threshold)


def diversity_tree_label(media_id: int) -> None:
    """Mark *media_id* as labeled in the active dataset's diversity tree."""
    with _state_lock:
        tree = get_active_context().diversity_tree
        if tree is not None and media_id in tree.vector_to_leaf:
            tree.label(media_id)


def diversity_tree_unlabel(media_id: int) -> None:
    """Remove *media_id*'s label from the active dataset's diversity tree."""
    with _state_lock:
        tree = get_active_context().diversity_tree
        if tree is not None and media_id in tree.vector_to_leaf:
            tree.unlabel(media_id)
