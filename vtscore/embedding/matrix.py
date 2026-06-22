"""Lazy, cached contiguous embedding matrix on :class:`DatasetContext`.

Building ``np.array([media_embedding(medias[cid]) for cid in sorted(...)])``
per request copies 10k+ arrays twice (once into a list, once into the
``np.array(...)`` allocation) and another copy when wrapping with
``torch.tensor(...)``.  The matrix changes only when the set of loaded
media IDs changes, so we cache one contiguous ``(N, D)`` float32 array
on the active dataset context and hand callers a ``torch.from_numpy``
view (zero-copy) when they need a tensor.

Cache invalidation is keyed on ``sorted(ctx.medias.keys())``: when that
list differs from the cached one, the matrix is rebuilt.  Callers that
mutate ``ctx.medias`` don't need to do anything - the next access will
detect the new key set and rebuild.

Any media whose ``embedding`` is ``None`` causes the builder to raise
``ValueError`` instead of silently filling the row with NaN - the bug
described as M11 in ``docs/plans/logical-bug-audit.md``.  On numpy 2.x
``matrix[i] = None`` quietly stores ``nan`` and the resulting score
propagates through every downstream consumer (always-False threshold
compares, NaN-poisoned sort, JSON ``NaN`` in the response).  Raising
turns that into a loud, locatable failure naming the offending cid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vtscore.embedding.media_vectors import media_embedding, primary_embedder_name
from vtscore.state.core import _state_lock

if TYPE_CHECKING:
    from vtscore.state.core import DatasetContext


def _collapse_to_primary(medias: dict[int, dict[str, Any]], embedder_name: str | None) -> str | None:
    """Map a routed embedder name to ``None`` when it is the dataset's primary.

    The routing layer (``DatasetContext.routed_embedder``) hands callers an
    explicit embedder name even for the common single-embedder dataset, where
    that name *is* the primary mirror.  The named matrix path builds fresh on
    every call; the primary path is cached on the context.  Since a name equal
    to the primary yields byte-for-byte the same vectors as the primary path
    (the singular ``embedding`` mirrors the recorded embedder's vector), we
    collapse it to ``None`` here so the cache is reused - keeping the
    single-embedder hot path unchanged after routing threads names through.

    A name that differs from the primary (a genuine second bound embedder)
    passes through unchanged and takes the uncached named path.
    """
    if embedder_name is None or not medias:
        return embedder_name
    first = next(iter(medias.values()))
    if embedder_name == primary_embedder_name(first):
        return None
    return embedder_name


def _require_embedding(cid: int, media: dict[str, Any], embedder_name: str | None = None) -> Any:
    emb = media_embedding(media, embedder_name)
    if emb is None:
        suffix = f" for embedder {embedder_name!r}" if embedder_name else ""
        raise ValueError(
            f"media {cid!r} has no embedding{suffix} (embedding=None); "
            "scoring/sorting require every media to have a vector. "
            "This usually means an importer or re-embed step silently failed."
        )
    return emb


def _stack_embeddings(
    sorted_ids: list[int],
    source: dict[int, dict[str, Any]],
    embedder_name: str | None,
) -> np.ndarray:
    """Build a contiguous ``(N, D)`` float32 matrix of *embedder_name*'s vectors.

    Rows follow *sorted_ids* order, pulling each media's vector via
    :func:`_require_embedding` (which routes through the dict-keyed accessor).
    Raises ``ValueError`` naming the first media that lacks a vector.
    """
    first_emb = np.asarray(_require_embedding(sorted_ids[0], source[sorted_ids[0]], embedder_name), dtype=np.float32)
    dim = int(first_emb.shape[-1])
    matrix = np.empty((len(sorted_ids), dim), dtype=np.float32)
    for i, cid in enumerate(sorted_ids):
        matrix[i] = _require_embedding(cid, source[cid], embedder_name)
    return matrix


def get_embedding_matrix(ctx: "DatasetContext", embedder_name: str | None = None) -> tuple[list[int], np.ndarray]:
    """Return ``(sorted_ids, (N, D) float32 matrix)`` for *ctx*'s medias.

    With *embedder_name* unset the matrix is built from each media's *primary*
    embedder and cached on the context, rebuilt only when the set of media IDs
    changes.  Pass an explicit *embedder_name* (one of a multi-embedder
    dataset's bound slots) to build a matrix from that embedder's vectors
    instead; the named path builds fresh on every call and never touches the
    cache, since the cache is reserved for the hot primary path.  Convert to a
    tensor with ``torch.from_numpy(matrix)`` for a zero-copy view.

    Returns ``([], np.empty((0, 0), dtype=np.float32))`` when the dataset is
    empty.  Raises ``ValueError`` if any media lacks the requested vector.
    """
    # Phase 1 (locked): snapshot the media refs and serve a cache hit. The
    # expensive _stack_embeddings build runs OUTSIDE the lock (phase 2) so a
    # large primary or named-embedder build cannot hold _state_lock across the
    # numpy stack and stall every other request's before_request state-sync.
    with _state_lock:
        sorted_ids = sorted(ctx.medias.keys())
        # A routed name equal to the primary collapses to the cached path.
        embedder_name = _collapse_to_primary(ctx.medias, embedder_name)
        if embedder_name is None:
            cached_matrix = ctx._emb_matrix
            if cached_matrix is not None and ctx._emb_matrix_ids == sorted_ids:
                return list(sorted_ids), cached_matrix

        if not sorted_ids:
            if embedder_name is None:
                ctx._emb_matrix_ids = []
                ctx._emb_matrix = np.empty((0, 0), dtype=np.float32)
                return [], ctx._emb_matrix
            return [], np.empty((0, 0), dtype=np.float32)

        # Shallow ref-copy so the build below reads a stable view even if
        # ctx.medias is reassigned concurrently (cheap: pointers, not vectors).
        medias_snapshot = dict(ctx.medias)

    # Phase 2 (unlocked): the heavy contiguous (N, D) build.
    matrix = _stack_embeddings(sorted_ids, medias_snapshot, embedder_name)

    # Phase 3 (locked): repopulate the primary cache, double-checking the id
    # set still matches so a media mutation during the unlocked build cannot
    # cache a stale matrix. The named path never touches the cache.
    if embedder_name is None:
        with _state_lock:
            if sorted(ctx.medias.keys()) == sorted_ids:
                ctx._emb_matrix_ids = sorted_ids
                ctx._emb_matrix = matrix
    return list(sorted_ids), matrix


def get_embedding_submatrix(
    ctx: "DatasetContext", ids: list[int], embedder_name: str | None = None
) -> tuple[list[int], np.ndarray]:
    """Return ``(sorted_ids, (N, D) float32 matrix)`` for a *subset* of *ctx*'s medias.

    Unlike :func:`get_embedding_matrix`, this builds a fresh matrix over only
    the requested *ids* (intersected with the dataset's current medias) and
    never populates the context-wide cache - subset projections (e.g. the
    positives of a Find run) are ephemeral.  The returned id list is sorted and
    de-duplicated; ids absent from the dataset are dropped silently.  Pass
    *embedder_name* to source the rows from a specific bound embedder.

    Returns ``([], np.empty((0, 0), dtype=np.float32))`` when nothing matches.
    Raises ``ValueError`` if any requested media lacks the requested vector.
    """
    # Snapshot just the requested rows under the lock, then build the matrix
    # outside it so a large subset stack does not hold _state_lock across the
    # numpy build (see get_embedding_matrix for the rationale).
    with _state_lock:
        medias = ctx.medias
        sorted_ids = sorted({cid for cid in ids if cid in medias})
        if not sorted_ids:
            return [], np.empty((0, 0), dtype=np.float32)
        medias_snapshot = {cid: medias[cid] for cid in sorted_ids}

    return sorted_ids, _stack_embeddings(sorted_ids, medias_snapshot, embedder_name)


def invalidate_embedding_matrix(ctx: "DatasetContext") -> None:
    """Drop the cached matrices on *ctx*; next access rebuilds them.

    Clears both the per-media embedding matrix and the flattened
    per-region matrix (used by patch-region scoring), since both are keyed
    on the media-id set and become stale together when the dataset's media
    change.
    """
    with _state_lock:
        ctx._emb_matrix_ids = None
        ctx._emb_matrix = None
        ctx._region_matrix_ids = None
        ctx._region_matrix = None
        ctx._region_media_index = None
        ctx._region_index_per_row = None


def _build_region_arrays(
    snap: dict,
    sorted_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten every media's ``patch_regions`` into one ``(R, D)`` matrix.

    Returns ``(region_matrix, media_index_per_row, region_index_per_row)``:

    * ``region_matrix`` - ``(R, D)`` float32, one row per (media, region) pair.
    * ``media_index_per_row`` - ``int64 (R,)``, the index into *sorted_ids*
      that each row belongs to.  Non-decreasing and contiguous per media.
    * ``region_index_per_row`` - ``int64 (R,)``, the region's index within its
      media's ``patch_regions`` list (the winning value surfaces as the UI's
      best-match overlay).

    Media that expose no ``patch_regions`` contribute a single row from their
    image-level ``embedding`` (region index 0), so every media has at least
    one row - keeping the downstream segmented max-pool free of empty groups.
    """
    flat_vecs: list[np.ndarray] = []
    media_index_per_row: list[int] = []
    region_index_per_row: list[int] = []
    for mi, cid in enumerate(sorted_ids):
        media = snap[cid]
        regions = media.get("patch_regions")
        if regions:
            for ri, r in enumerate(regions):
                flat_vecs.append(np.asarray(r.vec, dtype=np.float32))
                media_index_per_row.append(mi)
                region_index_per_row.append(ri)
        else:
            flat_vecs.append(np.asarray(_require_embedding(cid, media), dtype=np.float32))
            media_index_per_row.append(mi)
            region_index_per_row.append(0)
    region_matrix = np.stack(flat_vecs).astype(np.float32, copy=False)
    return (
        region_matrix,
        np.asarray(media_index_per_row, dtype=np.int64),
        np.asarray(region_index_per_row, dtype=np.int64),
    )


def get_region_matrix_for_snap(
    snap: dict,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Return the cached flattened region matrix for *snap*.

    Returns ``(sorted_ids, region_matrix, media_index_per_row,
    region_index_per_row)`` - see :func:`_build_region_arrays` for the
    array shapes.  When *snap*'s key set matches the active
    :class:`DatasetContext`'s medias (the common per-vote case), the matrix
    is built once and cached on the context, then reused across subsequent
    votes; only the MLP weights change between votes, never the region
    vectors, so the cache is valid until the media-id set changes.  A
    cross-dataset / subset *snap* builds fresh without populating the cache.

    Returns empty arrays when *snap* is empty.  Raises ``ValueError`` if a
    region-less media has ``embedding=None``.
    """
    from vtscore.state.core import get_active_context

    sorted_ids = sorted(snap.keys())
    if not sorted_ids:
        empty_vecs = np.empty((0, 0), dtype=np.float32)
        empty_idx = np.empty((0,), dtype=np.int64)
        return [], empty_vecs, empty_idx, empty_idx

    ctx = get_active_context()
    with _state_lock:
        if (
            ctx._region_matrix is not None
            and ctx._region_matrix_ids == sorted_ids
            and ctx._region_media_index is not None
            and ctx._region_index_per_row is not None
        ):
            return (
                list(sorted_ids),
                ctx._region_matrix,
                ctx._region_media_index,
                ctx._region_index_per_row,
            )

    region_matrix, media_index, region_index = _build_region_arrays(snap, sorted_ids)

    # Populate the cache only when *snap* matches the active dataset's medias
    # (the common case: ``snap = snapshot_medias()``).  Subset / cross-dataset
    # dicts are ephemeral and must not clobber the active cache.
    if sorted_ids == sorted(ctx.medias.keys()):
        with _state_lock:
            ctx._region_matrix_ids = sorted_ids
            ctx._region_matrix = region_matrix
            ctx._region_media_index = media_index
            ctx._region_index_per_row = region_index
    return list(sorted_ids), region_matrix, media_index, region_index


def get_embedding_matrix_for_snap(
    snap: dict,
    embedder_name: str | None = None,
) -> tuple[list[int], np.ndarray]:
    """Return ``(sorted_ids, matrix)`` for *snap*.

    With *embedder_name* unset and *snap*'s key set matching the active
    :class:`DatasetContext`'s medias, the cached primary matrix is reused.
    Otherwise (a different snapshot, a temp dict from cross-dataset Find, or an
    explicit non-primary embedder) the matrix is built fresh without populating
    the cache.  Raises ``ValueError`` if any entry in *snap* lacks the
    requested vector.
    """
    from vtscore.state.core import get_active_context

    sorted_ids = sorted(snap.keys())
    if not sorted_ids:
        return [], np.empty((0, 0), dtype=np.float32)

    # A routed name equal to the snapshot's primary collapses to the cached
    # primary path (matching get_embedding_matrix), so single-embedder
    # snapshots keep reusing the context cache after routing names through.
    embedder_name = _collapse_to_primary(snap, embedder_name)

    ctx = get_active_context()
    matches_active = sorted_ids == sorted(ctx.medias.keys())

    if embedder_name is None:
        with _state_lock:
            cached_ids = ctx._emb_matrix_ids
            cached_matrix = ctx._emb_matrix
        if cached_matrix is not None and cached_ids == sorted_ids:
            return sorted_ids, cached_matrix

    # When *snap* matches the active dataset's medias (the common case:
    # `snap = snapshot_medias()`), delegate to the context builder so the
    # primary path populates / reuses the cache; the named path builds fresh
    # there too.
    if matches_active:
        return get_embedding_matrix(ctx, embedder_name)

    # Temp dict / cross-dataset case: build fresh, don't cache.
    return sorted_ids, _stack_embeddings(sorted_ids, snap, embedder_name)
