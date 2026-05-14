"""Lazy, cached contiguous embedding matrix on :class:`DatasetContext`.

Building ``np.array([medias[cid]["embedding"] for cid in sorted(...)])``
per request copies 10k+ arrays twice (once into a list, once into the
``np.array(...)`` allocation) and another copy when wrapping with
``torch.tensor(...)``.  The matrix changes only when the set of loaded
media IDs changes, so we cache one contiguous ``(N, D)`` float32 array
on the active dataset context and hand callers a ``torch.from_numpy``
view (zero-copy) when they need a tensor.

Cache invalidation is keyed on ``sorted(ctx.medias.keys())``: when that
list differs from the cached one, the matrix is rebuilt.  Callers that
mutate ``ctx.medias`` don't need to do anything — the next access will
detect the new key set and rebuild.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vtsearch.utils.state_core import _state_lock

if TYPE_CHECKING:
    from vtsearch.utils.state_core import DatasetContext


def get_embedding_matrix(ctx: "DatasetContext") -> tuple[list[int], np.ndarray]:
    """Return ``(sorted_ids, (N, D) float32 matrix)`` for *ctx*'s medias.

    The matrix is cached on the context and rebuilt only when the set of
    media IDs changes.  Convert to a tensor with ``torch.from_numpy(matrix)``
    for a zero-copy view.

    Returns ``([], np.empty((0, 0), dtype=np.float32))`` when the dataset is
    empty.
    """
    with _state_lock:
        sorted_ids = sorted(ctx.medias.keys())
        cached_ids = ctx._emb_matrix_ids
        cached_matrix = ctx._emb_matrix
        if cached_matrix is not None and cached_ids == sorted_ids:
            return list(sorted_ids), cached_matrix

        if not sorted_ids:
            ctx._emb_matrix_ids = []
            ctx._emb_matrix = np.empty((0, 0), dtype=np.float32)
            return [], ctx._emb_matrix

        medias = ctx.medias
        first_emb = np.asarray(medias[sorted_ids[0]]["embedding"], dtype=np.float32)
        dim = int(first_emb.shape[-1])
        matrix = np.empty((len(sorted_ids), dim), dtype=np.float32)
        for i, cid in enumerate(sorted_ids):
            matrix[i] = medias[cid]["embedding"]

        ctx._emb_matrix_ids = sorted_ids
        ctx._emb_matrix = matrix
        return list(sorted_ids), matrix


def invalidate_embedding_matrix(ctx: "DatasetContext") -> None:
    """Drop the cached matrix on *ctx*; next access rebuilds it."""
    with _state_lock:
        ctx._emb_matrix_ids = None
        ctx._emb_matrix = None


def get_embedding_matrix_for_snap(
    snap: dict,
) -> tuple[list[int], np.ndarray]:
    """Return ``(sorted_ids, matrix)`` for *snap*.

    When *snap*'s key set matches the active :class:`DatasetContext`'s
    medias, the cached matrix is reused.  Otherwise (a different snapshot
    or a temp dict from cross-dataset Find) the matrix is built fresh
    without populating the cache.
    """
    from vtsearch.utils.state_core import get_active_context

    sorted_ids = sorted(snap.keys())
    if not sorted_ids:
        return [], np.empty((0, 0), dtype=np.float32)

    ctx = get_active_context()
    with _state_lock:
        cached_ids = ctx._emb_matrix_ids
        cached_matrix = ctx._emb_matrix
    if cached_matrix is not None and cached_ids == sorted_ids:
        return sorted_ids, cached_matrix

    # Populate the cache when *snap* matches the active dataset's medias
    # (the common case: `snap = snapshot_medias()`).
    if sorted_ids == sorted(ctx.medias.keys()):
        return get_embedding_matrix(ctx)

    # Temp dict / cross-dataset case: build fresh, don't cache.
    first_emb = np.asarray(snap[sorted_ids[0]]["embedding"], dtype=np.float32)
    dim = int(first_emb.shape[-1])
    matrix = np.empty((len(sorted_ids), dim), dtype=np.float32)
    for i, cid in enumerate(sorted_ids):
        matrix[i] = snap[cid]["embedding"]
    return sorted_ids, matrix
