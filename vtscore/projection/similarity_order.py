"""A 1-D similarity order over a dataset's items: look-alikes side by side.

The Browse canvas lays a dataset out in 2-D (:mod:`.umap_projection`); a list
has one axis, so a scroll of every item — New Detector's *Pick from Dataset*
grid (#5465) — wants an order in which neighbours look alike.  This fits UMAP
with ``n_components=1`` on the same embedding matrix the 2-D layout uses and
sorts the ids by that coordinate, so the faces of one person run together
instead of sitting wherever the import happened to put them.

**Which datasets get one.**  Only the media types in :data:`ORDERED_MEDIA_TYPES`
(Face): a Face dataset is a few thousand crops at most and can't be searched by
text, so its picker is the main way to start a detector, and the fit costs a
second or two next to the face detection the build already paid for.  It is fit
when the dataset is built (the load pipeline's similarity-order stage, in
:mod:`vtscore.datasets.stages.projection`) and, for a
Face dataset built before that existed, the first time the picker asks.  Every
other type keeps dataset order.

**Persistence.**  The order is a list of media ids — no vectors — stored
beside the dataset in its container (``similarity_order.json``) and cached on
the :class:`~vtscore.state.DatasetContext` (``_similarity_order``).  It is a
derived cache under the carve-out the 2-D projection uses (see
:mod:`.store`): a pure function of the pickle's medias, checked against the
live id set on every read, swept with the container, and costing only a re-fit
if lost.  An order that still names every current item serves even after items
were removed (the gone ids are skipped); one that misses an item is re-fit.

Like :func:`~vtscore.projection.umap_projection.fit_projection`, the fit is
unseeded by default and seeded by :data:`~vtscore.config.PROJECTION_SEED` when
set, below ``min_n_for_umap`` points it falls back to PCA-1, and below three it
keeps the ids as given.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vtscore.state import DatasetContext

logger = logging.getLogger(__name__)

#: Media types whose datasets are given a similarity order (module docstring).
ORDERED_MEDIA_TYPES = frozenset({"face"})

#: ``(current, total, message)`` progress sink, as the load stages take it.
ProgressFn = Callable[[int, int, str], None]


@dataclass(frozen=True)
class SimilarityOrder:
    """A dataset's media ids in 1-D similarity order.

    ``method`` records how it was produced: ``"umap"``, ``"pca"`` (too few
    points for a neighbour graph), or ``"trivial"`` (fewer than three, kept as
    given).
    """

    ids: list[int]
    method: str

    def covers(self, ids: Iterable[int]) -> bool:
        """Whether every one of *ids* has a place in this order."""
        return set(ids) <= set(self.ids)


def fit_similarity_order(
    matrix: np.ndarray,
    ids: list[int],
    *,
    n_neighbors: int,
    min_dist: float,
    random_state: int | None = None,
    min_n_for_umap: int = 10,
    on_heartbeat: Callable[[int], None] | None = None,
) -> SimilarityOrder:
    """Order *ids* by a 1-D UMAP of the ``(N, d)`` embedding *matrix*.

    ``ids[i]`` labels row ``i``.  ``n_neighbors`` is clamped to ``N - 1``.
    ``on_heartbeat(elapsed_seconds)`` ticks while the UMAP fit runs, which is
    one opaque blocking call.  Ties keep their input order (a stable sort).
    """
    from vtscore.projection.umap_projection import _umap_layout  # noqa: PLC0415

    # An owned, writable copy: *matrix* may be a read-only mmap of the
    # embedding-matrix sidecar, and neither UMAP nor PCA promise not to write.
    mat = np.array(matrix, dtype=np.float32, copy=True, order="C")
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2-D (N, d), got shape {mat.shape}")
    n = mat.shape[0]
    if len(ids) != n:
        raise ValueError(f"ids length {len(ids)} != matrix rows {n}")
    if n <= 2 or mat.shape[1] < 1:
        return SimilarityOrder([int(i) for i in ids], "trivial")

    if n < min_n_for_umap:
        from sklearn.decomposition import PCA  # noqa: PLC0415

        coord = PCA(n_components=1).fit_transform(mat)[:, 0]
        method = "pca"
    else:
        coord = _umap_layout(
            mat,
            n_neighbors=min(n_neighbors, n - 1),
            min_dist=min_dist,
            random_state=random_state,
            on_heartbeat=on_heartbeat or (lambda _elapsed: None),
            n_components=1,
        )[:, 0]
        method = "umap"
    order = np.argsort(coord, kind="stable")
    return SimilarityOrder([int(ids[i]) for i in order], method)


def ordered_ids(order: SimilarityOrder | None, ids: Iterable[int]) -> list[int]:
    """*ids* in *order*, any it doesn't place appended in their given order.

    Ids the order names but *ids* lacks (items removed since the fit) are
    skipped, so a stale-but-covering order still serves.
    """
    ids = list(ids)
    if order is None:
        return ids
    present = set(ids)
    placed = [i for i in order.ids if i in present]
    seen = set(placed)
    return placed + [i for i in ids if i not in seen]


# ----------------------------------------------------------------------
# Per-dataset lifecycle: memory → container → fit
# ----------------------------------------------------------------------

#: One fit at a time per dataset: the build stage and a picker request racing
#: for the same order wait on each other instead of fitting twice.
_fit_locks: dict[str, threading.Lock] = {}
_fit_locks_guard = threading.Lock()


def _fit_lock(dataset_id: str) -> threading.Lock:
    with _fit_locks_guard:
        lock = _fit_locks.get(dataset_id)
        if lock is None:
            lock = _fit_locks[dataset_id] = threading.Lock()
        return lock


def wants_similarity_order(ctx: DatasetContext) -> bool:
    """Whether *ctx*'s media type is one :data:`ORDERED_MEDIA_TYPES` orders."""
    from vtscore.projection.service import media_type_for  # noqa: PLC0415

    return media_type_for(ctx) in ORDERED_MEDIA_TYPES


def _matrix_for(ctx: DatasetContext) -> tuple[list[int], np.ndarray] | None:
    """The ``(sorted_ids, matrix)`` the 2-D layout clusters in, or ``None``."""
    from vtscore.embedding.matrix import get_embedding_matrix  # noqa: PLC0415

    try:
        # The score embedder's space (patch-else-text), as the 2-D layout uses.
        sorted_ids, matrix = get_embedding_matrix(ctx, ctx.routed_embedder("score"))
    except ValueError:
        return None
    if matrix.size == 0:
        return None
    return list(sorted_ids), matrix


def _stored_order(ctx: DatasetContext, ids: list[int]) -> SimilarityOrder | None:
    """The order cached on *ctx* or persisted beside it, if it covers *ids*."""
    cached = ctx._similarity_order
    if cached is not None and cached.covers(ids):
        return cached
    from vtscore.projection.store import pkl_path_for  # noqa: PLC0415

    pkl_path = pkl_path_for(ctx.dataset_id)
    if pkl_path is None:
        return None
    from vtscore.datasets.container import read_similarity_order  # noqa: PLC0415

    loaded = read_similarity_order(pkl_path)
    if loaded is None:
        return None
    order = SimilarityOrder(*loaded)
    if not order.covers(ids):
        logger.info("Persisted similarity order misses items of %s; will re-fit.", ctx.dataset_id)
        return None
    ctx._similarity_order = order
    return order


def _persist(dataset_id: str, order: SimilarityOrder) -> None:
    """Best-effort save of *order* into the dataset's container."""
    from vtscore.projection.store import pkl_path_for  # noqa: PLC0415

    pkl_path = pkl_path_for(dataset_id)
    if pkl_path is None:
        return
    try:
        from vtscore.datasets.container import append_similarity_order  # noqa: PLC0415

        append_similarity_order(pkl_path, order.ids, order.method)
    except Exception:
        logger.warning("Failed to persist similarity order for %s", dataset_id, exc_info=True)


def similarity_order_for(
    ctx: DatasetContext,
    *,
    fit: bool = True,
    on_progress: ProgressFn | None = None,
) -> SimilarityOrder | None:
    """*ctx*'s similarity order: cached, else persisted, else (with *fit*) fit now.

    ``None`` for a dataset of a type :func:`wants_similarity_order` leaves in
    dataset order, one with no usable embeddings, or (``fit=False``) one with
    nothing stored yet.  A fresh fit is cached on *ctx* and persisted.
    """
    if not wants_similarity_order(ctx):
        return None
    found = _matrix_for(ctx)
    if found is None:
        return None
    sorted_ids, matrix = found
    stored = _stored_order(ctx, sorted_ids)
    if stored is not None or not fit:
        return stored
    with _fit_lock(ctx.dataset_id):
        # Whoever held the lock may have just fit it.
        stored = _stored_order(ctx, sorted_ids)
        if stored is not None:
            return stored
        from vtscore.projection.params import resolve_projection_params  # noqa: PLC0415

        params = resolve_projection_params(ctx)
        n = len(sorted_ids)
        message = f"Ordering {n} items by similarity…"

        def _heartbeat(elapsed: int) -> None:
            if on_progress is not None:
                on_progress(0, 0, f"{message} {elapsed}s")

        if on_progress is not None:
            on_progress(0, 0, message)
        order = fit_similarity_order(
            matrix,
            sorted_ids,
            n_neighbors=params.n_neighbors,
            min_dist=params.min_dist,
            random_state=params.random_state,
            on_heartbeat=_heartbeat,
        )
        ctx._similarity_order = order
        _persist(ctx.dataset_id, order)
        return order


__all__ = [
    "ORDERED_MEDIA_TYPES",
    "SimilarityOrder",
    "fit_similarity_order",
    "ordered_ids",
    "similarity_order_for",
    "wants_similarity_order",
]
