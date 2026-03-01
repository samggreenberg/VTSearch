"""Diversity tree management for diverse media sampling."""

from __future__ import annotations

from typing import Any

import vtsearch.utils.state_core as _core
from vtsearch.utils.state_core import _state_lock, bad_votes, good_votes, medias


def build_diversity_tree(media_dict: dict[int, dict[str, Any]] | None = None) -> None:
    """Build a 3-Diversity Tree from media embeddings.

    Uses the global ``medias`` dict by default, or an explicit *media_dict*
    if provided.  Existing labels in ``good_votes`` and ``bad_votes`` are
    replayed into the new tree so the seen state stays accurate.
    """
    import numpy as np

    from vtsearch.models.diversity_tree import DiversityTree

    with _state_lock:
        source = media_dict if media_dict is not None else medias
        vectors: dict[int, np.ndarray] = {}
        for cid, media in source.items():
            emb = media.get("embedding")
            if emb is not None:
                vectors[cid] = np.asarray(emb, dtype=np.float32)

        if not vectors:
            _core._diversity_tree = None
            return

        _core._diversity_tree = DiversityTree(vectors, k=3)

        # Replay existing labels so the tree reflects the current vote state.
        for cid in good_votes:
            if cid in _core._diversity_tree.vector_to_leaf:
                _core._diversity_tree.label(cid)
        for cid in bad_votes:
            if cid in _core._diversity_tree.vector_to_leaf:
                _core._diversity_tree.label(cid)


def get_diversity_tree():
    """Return the current DiversityTree instance, or ``None``."""
    with _state_lock:
        return _core._diversity_tree


def diversity_tree_next_sample(scores: dict[int, float] | None = None) -> int | None:
    """Return the next diverse sample ID, or ``None`` if unavailable.

    When *scores* is provided, the highest-scored element in the next
    unseen node is returned (so the sort mode influences selection).
    """
    with _state_lock:
        if _core._diversity_tree is None:
            return None
        return _core._diversity_tree.next_sample(scores=scores)


def diversity_tree_label(media_id: int) -> None:
    """Mark *media_id* as labeled in the diversity tree."""
    with _state_lock:
        if _core._diversity_tree is not None and media_id in _core._diversity_tree.vector_to_leaf:
            _core._diversity_tree.label(media_id)


def diversity_tree_unlabel(media_id: int) -> None:
    """Remove *media_id*'s label from the diversity tree."""
    with _state_lock:
        if _core._diversity_tree is not None and media_id in _core._diversity_tree.vector_to_leaf:
            _core._diversity_tree.unlabel(media_id)
