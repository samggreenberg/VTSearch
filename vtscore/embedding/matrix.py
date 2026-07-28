"""Lazy, cached contiguous embedding matrix on :class:`DatasetContext`.

Building ``np.array([media_embedding(medias[cid]) for cid in sorted(...)])``
per request copies 10k+ arrays twice (once into a list, once into the
``np.array(...)`` allocation) and another copy when wrapping with
``torch.tensor(...)``.  The matrix changes only when the set of loaded
media IDs changes, so we cache one contiguous ``(N, D)`` float32 array
on the active dataset context and hand callers a ``torch.from_numpy``
view (zero-copy) when they need a tensor.

Cache invalidation is keyed on ``ctx.media_revision``: when the counter
differs from the one the cached matrix was built at, the matrix is
rebuilt.  Structural mutations of ``ctx.medias`` (add / remove / replace
an entry) bump the counter automatically via
:class:`~vtscore.state.core.MediasDict`, so callers don't need to do
anything.  An *in-place* rewrite of an existing media's vector (re-embed /
clip) is invisible to a dict subclass, so those stages call
``invalidate_embedding_matrix`` (which bumps the counter) after the
rewrite - see the ``media_revision`` root-cause pattern (logical-bug-audit
Pattern #4).

Any media whose ``embedding`` is ``None`` causes the builder to raise
``ValueError`` instead of silently filling the row with NaN - the bug
described as logical-bug-audit M11.  On numpy 2.x
``matrix[i] = None`` quietly stores ``nan`` and the resulting score
propagates through every downstream consumer (always-False threshold
compares, NaN-poisoned sort, JSON ``NaN`` in the response).  Raising
turns that into a loud, locatable failure naming the offending cid.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vtscore.embedding.media_vectors import media_embedder_names, media_embedding, primary_embedder_name
from vtscore.state.core import _state_lock

if TYPE_CHECKING:
    from vtscore.state.core import DatasetContext

logger = logging.getLogger(__name__)


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


def _registered_pkl_path(dataset_id: str) -> str | None:
    """Return the on-disk pkl path registered for *dataset_id*, or ``None``.

    Datasets built purely in memory (tests, ephemeral browse contexts,
    positives-map previews) have no registry entry and get no sidecar - the
    mmap cache (S1, ``docs/plans/scalability.md``) is opportunistic only for
    datasets actually backed by a saved pickle file.
    """
    from vtscore.datasets.registry import get_dataset  # noqa: PLC0415

    try:
        entry = get_dataset(dataset_id)
    except Exception:
        return None
    return (entry or {}).get("pkl_path") or None


def _emb_sidecar_paths(pkl_path: str) -> tuple[Path, Path]:
    """Return ``(ids_path, matrix_path)`` sidecar paths for *pkl_path*.

    Both share the pkl's stem (``ds_<uuid>``) followed by a dot, so
    ``registry.unregister_dataset``'s stem-glob sweep deletes them alongside
    the pkl on both age-off expiry and manual delete - no separate cleanup
    bookkeeping needed.
    """
    p = Path(pkl_path)
    return p.parent / f"{p.stem}.embids.npy", p.parent / f"{p.stem}.embmat.npy"


def _try_load_matrix_sidecar(pkl_path: str, sorted_ids: list[int], probe_dim: int) -> np.ndarray | None:
    """Return the mmap'd primary embedding matrix for *pkl_path* if valid, else ``None``.

    Valid means: both sidecar files exist, the persisted id list matches
    *sorted_ids* exactly (order and content), and the persisted matrix's
    column count matches *probe_dim* (a live vector's dimension, guarding
    against a same-id-set re-embed to a different dimension - the drift the
    id-list check alone can't see). Any mismatch, missing file, or read error
    returns ``None``; the caller always has a safe, correct fallback: rebuild
    from live ``ctx.medias``.
    """
    ids_path, mat_path = _emb_sidecar_paths(pkl_path)
    if not (ids_path.is_file() and mat_path.is_file()):
        return None
    try:
        sidecar_ids = np.load(ids_path)
        if sidecar_ids.shape != (len(sorted_ids),) or not np.array_equal(
            sidecar_ids, np.asarray(sorted_ids, dtype=np.int64)
        ):
            return None
        matrix = np.load(mat_path, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[0] != len(sorted_ids) or matrix.shape[1] != probe_dim:
            return None
        return matrix
    except Exception:
        logger.warning("Failed to load embedding-matrix sidecar for %s", pkl_path, exc_info=True)
        return None


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """Write *arr* to *path* as ``.npy`` bytes via write-to-temp + atomic rename.

    Mirrors the tmp-file idiom used by ``vtscore.datasets.container.write_container``:
    a crash mid-write leaves an orphan ``.tmp`` file, never a truncated file at
    the real name, so a concurrent reader never observes a partial array.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _maybe_persist_matrix_sidecar(pkl_path: str, sorted_ids: list[int], matrix: np.ndarray) -> None:
    """Best-effort write of the primary embedding matrix as a mmap-able sidecar.

    Lets a future cold load of the same pkl (a fresh process / DatasetContext)
    skip rebuilding the matrix from per-item embeddings - see S1,
    ``docs/plans/scalability.md``. A pure derived cache of data already
    durably persisted in the dataset pickle: always regenerable from
    ``ctx.medias``, deterministic, and swept alongside the pkl by
    ``registry.unregister_dataset``. Any failure (read-only filesystem, full
    disk, concurrent writer) is logged and swallowed - the sidecar is an
    optimization, never a dependency.

    Always writes (no "ids already match, skip" shortcut): the caller only
    reaches here on a genuine cache-miss rebuild, which happens either on the
    first-ever build for this context or after an explicit invalidation - and
    an in-place vector rewrite (re-embed/clip) leaves the id set unchanged
    while the *content* legitimately changed, so an ids-only check would skip
    a write that must happen and silently entrench a stale sidecar.
    """
    ids_path, mat_path = _emb_sidecar_paths(pkl_path)
    try:
        # Matrix first: if a crash lands between the two atomic renames, a
        # mismatched pair is caught by the shape/dim checks in
        # ``_try_load_matrix_sidecar`` on the next read, never served as-is.
        _atomic_save_npy(mat_path, matrix)
        _atomic_save_npy(ids_path, np.asarray(sorted_ids, dtype=np.int64))
    except OSError:
        logger.info("Could not persist embedding-matrix sidecar for %s (read-only filesystem?)", pkl_path)
    except Exception:
        logger.warning("Failed to persist embedding-matrix sidecar for %s", pkl_path, exc_info=True)


def _try_primary_sidecar(
    ctx: "DatasetContext", sorted_ids: list[int], medias_snapshot: dict[int, dict[str, Any]]
) -> tuple[str | None, np.ndarray | None]:
    """Return ``(pkl_path, matrix)`` for the primary-path on-disk mmap sidecar.

    *matrix* is ``None`` when there is no registered pkl, the sidecar-disabled
    latch is set (see ``invalidate_embedding_matrix``), or the sidecar is
    missing/stale - the caller always falls back to ``_stack_embeddings`` in
    that case. *pkl_path* is returned even on a sidecar miss (``None`` matrix)
    since the caller still needs it to persist a freshly-built matrix.
    """
    pkl_path = _registered_pkl_path(ctx.dataset_id)
    if not pkl_path or ctx._emb_sidecar_disabled:
        return pkl_path, None
    probe_dim = int(np.asarray(_require_embedding(sorted_ids[0], medias_snapshot[sorted_ids[0]])).shape[-1])
    return pkl_path, _try_load_matrix_sidecar(pkl_path, sorted_ids, probe_dim)


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
        # Snapshot the revision under the lock; the cache is valid iff it still
        # matches after the unlocked build (phase 3). Keying on the counter
        # rather than an id-list compare also catches an in-place vector
        # rewrite that leaves the id set unchanged (root-cause Pattern #4).
        revision = ctx.media_revision
        # A routed name equal to the primary collapses to the cached path.
        embedder_name = _collapse_to_primary(ctx.medias, embedder_name)
        if embedder_name is None:
            cached_matrix = ctx._emb_matrix
            # A revision match guarantees the id set is unchanged, so the live
            # sorted_ids equals the cached ids the matrix rows correspond to.
            if cached_matrix is not None and ctx._emb_matrix_revision == revision:
                return list(sorted_ids), cached_matrix

        if not sorted_ids:
            if embedder_name is None:
                ctx._emb_matrix_ids = []
                ctx._emb_matrix = np.empty((0, 0), dtype=np.float32)
                ctx._emb_matrix_revision = revision
                return [], ctx._emb_matrix
            return [], np.empty((0, 0), dtype=np.float32)

        # Shallow ref-copy so the build below reads a stable view even if
        # ctx.medias is reassigned concurrently (cheap: pointers, not vectors).
        medias_snapshot = dict(ctx.medias)

    # Phase 2 (unlocked): try a matching on-disk mmap sidecar first (S1,
    # docs/plans/scalability.md), else the heavy contiguous (N, D) build.
    # ``pkl_path`` is resolved even on a sidecar miss (used again in phase 4
    # to persist a fresh build).
    pkl_path: str | None = None
    matrix: np.ndarray | None = None
    if embedder_name is None:
        pkl_path, matrix = _try_primary_sidecar(ctx, sorted_ids, medias_snapshot)
    used_sidecar = matrix is not None
    if matrix is None:
        matrix = _stack_embeddings(sorted_ids, medias_snapshot, embedder_name)

    # Phase 3 (locked): repopulate the primary cache, double-checking the
    # revision still matches so a media mutation during the unlocked build
    # cannot cache a stale matrix. The named path never touches the cache.
    cache_populated = False
    if embedder_name is None:
        with _state_lock:
            if ctx.media_revision == revision:
                ctx._emb_matrix_ids = sorted_ids
                ctx._emb_matrix = matrix
                ctx._emb_matrix_revision = revision
                cache_populated = True

    # Phase 4 (unlocked, best-effort): persist a freshly-built primary matrix
    # as a sidecar so a future cold load of this pkl can mmap it instead of
    # rebuilding. Skipped when we just read a valid sidecar (already on disk,
    # nothing to refresh) or when phase 3 lost the race (that matrix no
    # longer matches the live id set and must not be written as this pkl's
    # cache).
    if cache_populated and not used_sidecar and pkl_path:
        _maybe_persist_matrix_sidecar(pkl_path, sorted_ids, matrix)

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
    on ``media_revision`` and become stale together when the dataset's media
    change.  Also bumps ``media_revision`` so this stands in as the explicit
    "vectors changed in place" signal at the embed / clip stages: an
    in-place rewrite of existing media dicts is invisible to
    :class:`~vtscore.state.core.MediasDict`, so those stages call this to
    advance the counter (and free the cached arrays' RAM immediately).
    """
    with _state_lock:
        ctx._emb_matrix_ids = None
        ctx._emb_matrix = None
        ctx._emb_matrix_revision = None
        # An in-place rewrite can leave the id set (and dimension) unchanged,
        # which the sidecar's validity check alone cannot detect - permanently
        # stop trusting the on-disk mmap sidecar for this context so every
        # later rebuild reads live ``ctx.medias``, never a stale cached file.
        ctx._emb_sidecar_disabled = True
        ctx._region_matrix_ids = None
        ctx._region_matrix = None
        ctx._region_matrix_revision = None
        ctx._region_media_index = None
        ctx._region_index_per_row = None
        ctx.bump_media_revision()


def _patch_embedder_for_region_snap(snap: dict[int, dict[str, Any]]) -> str | None:
    """Return the patch-slot embedder name that produced *snap*'s region rows.

    Derived from a media that actually carries ``patch_regions`` (rather than
    just the first media in the dict, which may be a region-less item that
    never had a vector for the patch embedder at all - the mixed-media-type
    case).  That media's own bound embedder names are role-typed via
    :func:`~vtscore.embedding.binding.derive_binding_from_names`; the patch
    slot is the space the region vectors live in.  Returns ``None`` when no
    media in *snap* has regions, or when the region-bearing media's embedders
    don't role-type to a patch slot (unexpected; the fallback path then keeps
    the pre-fix behaviour of reading the primary vector).
    """
    from vtscore.embedding.binding import derive_binding_from_names  # noqa: PLC0415

    for media in snap.values():
        if media.get("patch_regions"):
            _text, patch, _structural = derive_binding_from_names(media_embedder_names(media))
            return patch
    return None


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

    Media that expose no ``patch_regions`` contribute a single row (region
    index 0) so every media has at least one row - keeping the downstream
    segmented max-pool free of empty groups.  That fallback row is read from
    the *patch-slot* embedder shared by the rest of the snapshot's region rows
    (:func:`_patch_embedder_for_region_snap`), not unconditionally the primary
    vector: on a dataset that mixes patch-capable and patch-less media (e.g. a
    combined dataset, or a media type the patch embedder can't process), the
    primary can be a different embedder than the one that produced the region
    vectors, and stacking its vector alongside them would silently mix
    embedding spaces in one matrix.  If the region-less media has no vector
    under that patch embedder either, :func:`_require_embedding` raises rather
    than falling back further - a loud, locatable failure instead of a
    silently meaningless score.
    """
    patch_embedder_name = _patch_embedder_for_region_snap(snap)
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
            flat_vecs.append(np.asarray(_require_embedding(cid, media, patch_embedder_name), dtype=np.float32))
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
        # Snapshot the revision so a mutation during the unlocked build below
        # can't cache a stale matrix. The id-list compare still guards against
        # a *different* snap with a coincidentally equal cached id set; the
        # revision compare additionally catches an in-place vector rewrite
        # under the same id set (root-cause Pattern #4).
        revision = ctx.media_revision
        if (
            ctx._region_matrix is not None
            and ctx._region_matrix_ids == sorted_ids
            and ctx._region_matrix_revision == revision
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
    # (the common case: ``snap = snapshot_medias()``) and no mutation landed
    # during the build.  Subset / cross-dataset dicts are ephemeral and must
    # not clobber the active cache.
    if sorted_ids == sorted(ctx.medias.keys()):
        with _state_lock:
            if ctx.media_revision == revision:
                ctx._region_matrix_ids = sorted_ids
                ctx._region_matrix = region_matrix
                ctx._region_matrix_revision = revision
                ctx._region_media_index = media_index
                ctx._region_index_per_row = region_index
    return list(sorted_ids), region_matrix, media_index, region_index


def segmented_max_pool(
    flat_scores: np.ndarray,
    media_index_per_row: np.ndarray,
    region_index_per_row: np.ndarray,
    n_media: int,
) -> tuple[list[float], list[int]]:
    """Max-pool per-row scores down to one score + winning region per media.

    Shared by the MLP scoring path (:func:`vtscore.detectors.training._score_all_media`)
    and the region-aware cosine sort (:func:`vtscore.training.region_similarity.cosine_sort_with_boxes`),
    both of which flatten every ``(media, region)`` pair into one ``(R,)`` score
    vector (via :func:`get_region_matrix_for_snap`) and need to reduce it back to
    one score + winning region index per media.

    *media_index_per_row* is non-decreasing and contiguous (every media owns
    a single run of rows), and every media has at least one row, so each
    media's rows form one ``reduceat`` segment.  Returns ``(scores,
    best_region)`` as plain Python lists, where ``best_region[m]`` is the
    region index of the *first* row achieving media ``m``'s max - matching
    the strict-``>`` "first wins" tie-break of the original scalar loop.

    Fully vectorised so the scoring tail holds the GIL for microseconds
    rather than iterating hundreds of thousands of rows in Python.
    """
    # Start of each media's contiguous run of rows.
    seg_starts = np.searchsorted(media_index_per_row, np.arange(n_media))
    seg_max = np.maximum.reduceat(flat_scores, seg_starts)

    # First row per media that reaches its segment max (region 0 - the
    # CLS/full-image node - is always row 0 of a segment, so an all-sentinel
    # media resolves to region 0, exactly as the old -1.0-seeded loop did).
    is_max = flat_scores >= seg_max[media_index_per_row]
    cand_rows = np.flatnonzero(is_max)
    cand_media = media_index_per_row[cand_rows]
    first_cand = np.searchsorted(cand_media, np.arange(n_media))
    winning_rows = cand_rows[first_cand]
    best_region = region_index_per_row[winning_rows]

    return seg_max.tolist(), best_region.tolist()


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
            cache_current = ctx._emb_matrix_revision == ctx.media_revision
        # The id-list compare guards against a *different* snap whose ids
        # happen to equal the cached set; the revision compare additionally
        # rejects a cache stale from an in-place vector rewrite under the same
        # id set (root-cause Pattern #4).
        if cached_matrix is not None and cached_ids == sorted_ids and cache_current:
            return sorted_ids, cached_matrix

    # When *snap* matches the active dataset's medias (the common case:
    # `snap = snapshot_medias()`), delegate to the context builder so the
    # primary path populates / reuses the cache; the named path builds fresh
    # there too.
    if matches_active:
        return get_embedding_matrix(ctx, embedder_name)

    # Temp dict / cross-dataset case: build fresh, don't cache.
    return sorted_ids, _stack_embeddings(sorted_ids, snap, embedder_name)
