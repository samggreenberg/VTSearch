"""The people in a Face dataset: its face crops merged into discrete people (#5586).

The first slice of Co-face Browse (``docs/plans/co-face-browse.md``): a
partition of a Face dataset's crops into **people**, each a list of face ids,
plus the faces placed in no person.  Every face is in at most one person, and
a person with one face is a person.  It is a pure function of the dataset's
own face vectors, so nothing about it is ever saved but ids.

**The cut is the face ball's.**  A face belongs to a person when it passes the
test Find's face ball applies (:mod:`vtscore.detectors.centroid_head`, #5478):
cosine to the person's unit centroid at or above the ball's beta 1 floor in the
dataset's face space (:func:`people_cutoff`).  The Goods' spread that moves the
ball's floor plays no part, since a partition has no Goods.  A space the ball
has no floor for gets no partition at all: :func:`request_people` answers
``not_calibrated`` with the ball's own wording, and nothing plausible-but-wrong
is shown (the owner's call, 2026-11-05).

**The fit** (:func:`fit_people`), on the face embedder's matrix
(:func:`~vtscore.embedding.matrix.get_embedding_matrix`, unit vectors, so a
cosine is a dot product):

1. *A neighbour graph.*  Each face's :data:`PEOPLE_NEIGHBOURS` nearest faces by
   cosine, from chunked dot products (never an ``N x N`` matrix), keeping the
   edges at or above the cut.  Its connected components are the candidates.
2. *A split within each component.*  A threshold graph chains (A like B, B like
   C, A unlike C), so two people who share a lookalike land in one component.
   Each is re-cut by agglomerative clustering with **centroid linkage on the
   unit sphere**: the closest pair of clusters by the cosine between their
   unit centroids merges, while that cosine clears the cut.  Then each face
   goes to the cluster of its component whose centroid it clears the cut to
   best, or stands as a person of its own.
3. *Hard views last.*  A face under its space's ``HARD_VIEW_QUALITY``
   (:mod:`vtscore.media.face.imagekit_bridge`, #5372) is poor evidence of
   identity, as the trained head treats it (#5413), so it takes no part in
   steps 1 and 2.  It then joins the person whose centroid it clears the cut
   to best, else it is unplaced.

The linkage is written here rather than taken from ``scipy``, whose centroid
linkage is Euclidean between the clusters' *raw* means.  A person's mean of
unit vectors has a norm well under 1 (about 0.7 on buffalo_s, where a person's
own photos sit about 0.5 apart), and between such means the floor's Euclidean
equivalent, ``sqrt(2 - 2 * cut)``, admits clusters whose centroids are almost
at right angles.  The face ball measures a cosine to a *unit* centroid
(:func:`~vtscore.detectors.centroid_head.goods_centroid`), so the linkage does.

The fit is deterministic (no seed, no sampling): the same faces give the same
people.  People are listed by face count, largest first.  Each lists its
**representative** first, the face with the highest Face Quality (ties to the
one closest to the centroid), then the rest from the closest to the centroid
out.

**Persistence.**  The partition is ids and the stamps it was cut under (the
method, the linkage, the cut and the space), with no vector, stored beside the
dataset in its container (``people.json``) and cached on the
:class:`~vtscore.state.DatasetContext` (``_people``).  It is a derived cache
under the carve-out the 1-D similarity order uses
(:mod:`vtscore.projection.similarity_order`): a pure function of the pickle's
medias, checked against the live id set on every read, swept with the
container, and costing only a fit if lost.  A partition that still places every
current face serves after items were removed (:meth:`PeoplePartition.restricted_to`
drops the gone ones).  One that misses a face, or was cut under another cut,
space, method or linkage, is fit again.

**When it is fit.**  When a Face dataset is built (the load pipeline's people
stage, :mod:`vtscore.datasets.stages.projection`), and for a Face dataset built
before that existed, on the first :func:`request_people`.  That runs the fit on
the layout builds' runner and is answered with its progress until a later ask
finds the partition stored, as a list sorting by similarity is answered
(:func:`~vtscore.projection.similarity_order.request_similarity_order`).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vtscore.state import DatasetContext

logger = logging.getLogger(__name__)

#: Media types whose datasets have people: faces only.
PEOPLE_MEDIA_TYPES = frozenset({"face"})

#: How a partition is fit (module docstring); stamped on every partition, and
#: a stored one fit another way is fit again.
PEOPLE_METHOD = "knn_agglomerative"

#: The agglomerative step's linkage: the cosine between unit centroids.  #5591
#: measures it against average linkage.
PEOPLE_LINKAGE = "centroid"

#: How many nearest faces each face's neighbour-graph edges are drawn from.
PEOPLE_NEIGHBOURS = 20

#: Rows of the matrix multiplied at a time while drawing the neighbour graph and
#: attaching hard views: bounds the scratch to ``_CHUNK x N`` cosines.
_CHUNK = 1024

#: The feature name the uncalibrated answer gives (:func:`not_calibrated_message`).
PEOPLE_FEATURE = "People"

#: ``(current, total, message)`` progress sink, as the load stages take it.
ProgressFn = Callable[[int, int, str], None]


class NothingToPartition(ValueError):
    """There are no people to find: not a Face dataset, or no face vectors.

    A ``ValueError`` as :class:`~vtscore.projection.service.NothingToProject`
    is, for the same reason: the embedding-matrix builder signals a dataset
    without vectors with a plain ``ValueError``, and a caller treats the two
    alike (the HTTP layer with a 409 carrying the message).
    """


# ----------------------------------------------------------------------
# The cut
# ----------------------------------------------------------------------


def people_cutoff(compat: str | None) -> float | None:
    """The cosine a face must clear to a person's centroid in *compat*'s space, or ``None``.

    The face ball's beta 1 floor for the space (:data:`~vtscore.detectors.centroid_head.FACE_BALL_FLOORS`,
    fitted in :data:`~vtscore.detectors.centroid_head.FACE_BALL_COMPAT`), unmoved by a spread.
    ``None`` for a space the ball has no floor for, which gets no partition.  The one place the cut is
    read, so #5497's per-compat floors change this body and nothing else.
    """
    from vtscore.detectors.centroid_head import FACE_BALL_COMPAT, FACE_BALL_FLOORS  # noqa: PLC0415

    if not compat or compat != FACE_BALL_COMPAT:
        return None
    return float(FACE_BALL_FLOORS[1.0])


def not_calibrated_message(compat: str | None) -> str:
    """What a dataset whose face space has no floor is told: the face ball's wording, for People."""
    from vtscore.detectors.centroid_head import face_uncalibrated_message  # noqa: PLC0415

    return face_uncalibrated_message(PEOPLE_FEATURE, compat)


# ----------------------------------------------------------------------
# The partition
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PeoplePartition:
    """A Face dataset's faces merged into people, and the stamps it was cut under.

    ``people`` lists each person's face ids, the largest person first; a
    person's first id is its representative.  ``unplaced`` holds the faces in
    no person (hard views that cleared no person's centroid), in id order.
    ``cutoff`` is the cosine a face cleared to its person's centroid,
    ``compat`` the face space (``custom_metadata["Face Compat"]``) it was cut
    in, and ``method`` / ``linkage`` how (:data:`PEOPLE_METHOD`,
    :data:`PEOPLE_LINKAGE`).
    """

    people: tuple[tuple[int, ...], ...]
    unplaced: tuple[int, ...]
    method: str
    linkage: str
    cutoff: float
    compat: str

    @property
    def face_ids(self) -> frozenset[int]:
        """Every face the partition knows: placed in a person or unplaced."""
        return frozenset(i for person in self.people for i in person) | frozenset(self.unplaced)

    def covers(self, ids: Iterable[int]) -> bool:
        """Whether every one of *ids* has a place here, in a person or unplaced."""
        return set(ids) <= self.face_ids

    def cut_as(self, cutoff: float, compat: str) -> bool:
        """Whether this partition was cut at *cutoff* in *compat*'s space, by today's method and linkage."""
        return (
            self.method == PEOPLE_METHOD
            and self.linkage == PEOPLE_LINKAGE
            and self.compat == compat
            and math.isclose(self.cutoff, cutoff, rel_tol=0.0, abs_tol=1e-9)
        )

    def person_of(self) -> dict[int, int]:
        """``{face id: index into people}`` for every placed face."""
        return {i: rank for rank, person in enumerate(self.people) for i in person}

    def restricted_to(self, ids: Iterable[int]) -> PeoplePartition:
        """This partition over *ids* alone: gone faces dropped, and any person left with none.

        What a partition that still covers the live faces serves once items
        were removed.  The people keep their order (a person that lost faces is
        not re-ranked), and one whose representative went is led by its next
        face, the one that sat closest to its centroid.
        """
        live = set(ids)
        people = tuple(kept for kept in (tuple(i for i in p if i in live) for p in self.people) if kept)
        unplaced = tuple(i for i in self.unplaced if i in live)
        return PeoplePartition(people, unplaced, self.method, self.linkage, self.cutoff, self.compat)

    def to_payload(self) -> dict[str, Any]:
        """The JSON-shaped dict the container stores: ids and stamps, no vectors."""
        return {
            "method": self.method,
            "linkage": self.linkage,
            "cutoff": self.cutoff,
            "compat": self.compat,
            "people": [list(p) for p in self.people],
            "unplaced": list(self.unplaced),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PeoplePartition:
        """The partition :meth:`to_payload` stored (as :func:`~vtscore.datasets.container.read_people` reads it)."""
        return cls(
            people=tuple(tuple(int(i) for i in p) for p in payload["people"]),
            unplaced=tuple(int(i) for i in payload["unplaced"]),
            method=str(payload["method"]),
            linkage=str(payload["linkage"]),
            cutoff=float(payload["cutoff"]),
            compat=str(payload["compat"]),
        )


# ----------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """*matrix* as an owned float64 array with every non-zero row scaled to unit length."""
    rows = np.array(matrix, dtype=np.float64, copy=True)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _neighbour_components(
    unit: np.ndarray, cutoff: float, neighbours: int, on_chunk: Callable[[int, int], None]
) -> np.ndarray:
    """Step 1: a component label per row of *unit*, from its nearest-neighbour edges that clear *cutoff*."""
    from scipy.sparse import coo_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

    n = unit.shape[0]
    k = min(neighbours, n - 1)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    n_chunks = max(1, math.ceil(n / _CHUNK))
    for c, start in enumerate(range(0, n, _CHUNK)):
        stop = min(start + _CHUNK, n)
        if k > 0:
            sims = unit[start:stop] @ unit.T
            sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(sims, top, axis=1)
            keep = vals >= cutoff
            rows.append(np.nonzero(keep)[0] + start)
            cols.append(top[keep])
        on_chunk(c + 1, n_chunks)
    src = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    dst = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    graph = coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n))
    _count, labels = connected_components(graph, directed=False)
    return labels


def _centroid_linkage(unit: np.ndarray, cutoff: float) -> list[list[int]]:
    """Step 2's clustering of one component: merge the closest unit centroids while their cosine clears *cutoff*.

    *unit* holds the component's unit rows; the clusters come back as lists of
    row indices.  Centroid linkage is not monotone (a merged centroid can sit
    closer to a third cluster than either half did), so each cluster's best
    partner is tracked and re-read only where a merge could have lowered it.
    Ties go to the lowest index pair, so the result is a function of the rows.
    """
    m = unit.shape[0]
    members: list[list[int]] = [[i] for i in range(m)]
    if m < 2:
        return members
    sums = unit.copy()
    cents = unit.copy()
    active = np.ones(m, dtype=bool)
    cos = cents @ cents.T
    np.fill_diagonal(cos, -np.inf)
    best_j = np.argmax(cos, axis=1)
    best = cos[np.arange(m), best_j]
    while True:
        i = int(np.argmax(np.where(active, best, -np.inf)))
        if not active[i] or best[i] < cutoff:
            break
        j = int(best_j[i])
        a, b = min(i, j), max(i, j)
        # Merge b into a.
        sums[a] += sums[b]
        cents[a] = _unit(sums[a])
        members[a].extend(members[b])
        members[b] = []
        active[b] = False
        cos[b, :] = -np.inf
        cos[:, b] = -np.inf
        row = cents @ cents[a]
        row[~active] = -np.inf
        row[a] = -np.inf
        cos[a, :] = row
        cos[:, a] = row
        best[b] = -np.inf
        best_j[a] = int(np.argmax(cos[a]))
        best[a] = cos[a, best_j[a]]
        # Every other cluster: one whose best partner was a or b re-reads its
        # row; any other takes the merged centroid if it is now closer.
        others = active.copy()
        others[a] = False
        stale = others & ((best_j == a) | (best_j == b))
        if stale.any():
            best_j[stale] = np.argmax(cos[stale], axis=1)
            best[stale] = cos[stale, best_j[stale]]
        closer = others & ~stale & (row > best)
        best_j[closer] = a
        best[closer] = row[closer]
    return [sorted(g) for g in members if g]


def _reassign(unit: np.ndarray, clusters: list[list[int]], cutoff: float) -> list[list[int]]:
    """Step 2's last pass: each row goes to the cluster whose unit centroid it clears *cutoff* to best, or alone."""
    if len(clusters) < 2 and all(len(g) < 2 for g in clusters):
        return clusters
    cents = np.stack([_unit(unit[g].sum(axis=0)) for g in clusters])
    sims = unit @ cents.T
    pick = np.argmax(sims, axis=1)
    clears = sims[np.arange(unit.shape[0]), pick] >= cutoff
    out: dict[int, list[int]] = {}
    alone: list[list[int]] = []
    for r in range(unit.shape[0]):
        if clears[r]:
            out.setdefault(int(pick[r]), []).append(r)
        else:
            alone.append([r])
    return [out[c] for c in sorted(out)] + alone


def _quality(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    q = float(value)
    return q if math.isfinite(q) else None


def fit_people(
    matrix: np.ndarray,
    ids: Sequence[int],
    *,
    cutoff: float,
    compat: str,
    hard: Sequence[bool] | None = None,
    quality: Sequence[Any] | None = None,
    neighbours: int = PEOPLE_NEIGHBOURS,
    on_progress: ProgressFn | None = None,
) -> PeoplePartition:
    """Merge the faces of the ``(N, d)`` face *matrix* into people (module docstring).

    ``ids[i]`` labels row ``i``.  *hard* flags the rows that are hard views
    (left out of the graph and the clustering, then attached or left unplaced),
    and *quality* gives each row's Face Quality (``None`` or a non-number where
    it has none), which picks a person's representative; both default to none.
    *cutoff* and *compat* are the cut and the space it is in
    (:func:`people_cutoff`), and are stamped on the result.  Rows need not be
    unit length: each is normalised first, and an all-zero row is a person of
    its own (it clears no one).
    """
    unit = _unit_rows(matrix)
    if unit.ndim != 2:
        raise ValueError(f"matrix must be 2-D (N, d), got shape {unit.shape}")
    n = unit.shape[0]
    if len(ids) != n:
        raise ValueError(f"ids length {len(ids)} != matrix rows {n}")
    hard_mask = np.zeros(n, dtype=bool) if hard is None else np.asarray(hard, dtype=bool)
    if hard_mask.shape != (n,):
        raise ValueError(f"hard has {hard_mask.size} flags for {n} rows")
    if quality is not None and len(quality) != n:
        raise ValueError(f"quality has {len(quality)} values for {n} rows")
    qualities = [None] * n if quality is None else [_quality(q) for q in quality]

    def _report(current: int, total: int, message: str) -> None:
        if on_progress is not None:
            on_progress(current, total, message)

    core = np.nonzero(~hard_mask)[0]
    message = f"Finding people among {n} faces…"
    _report(0, 0, message)

    # Steps 1 and 2 over the faces that are not hard views.
    groups: list[list[int]] = []  # row indices into the full matrix
    if core.size:
        core_unit = unit[core]
        labels = _neighbour_components(
            core_unit, cutoff, neighbours, lambda done, total: _report(done, total, message)
        )
        order = np.argsort(labels, kind="stable")
        bounds = np.flatnonzero(np.diff(labels[order])) + 1
        for comp in np.split(order, bounds):
            local = core_unit[comp]
            clusters = _reassign(local, _centroid_linkage(local, cutoff), cutoff)
            groups.extend([[int(core[comp[r]]) for r in g] for g in clusters])

    cents = np.stack([_unit(unit[g].sum(axis=0)) for g in groups]) if groups else np.empty((0, unit.shape[1]))

    # Step 3: each hard view joins the person whose centroid it clears best.
    unplaced: list[int] = []
    attached: list[list[int]] = [[] for _ in groups]
    hard_rows = np.nonzero(hard_mask)[0]
    for start in range(0, hard_rows.size, _CHUNK):
        chunk = hard_rows[start : start + _CHUNK]
        if not groups:
            unplaced.extend(int(r) for r in chunk)
            continue
        sims = unit[chunk] @ cents.T
        pick = np.argmax(sims, axis=1)
        for r, p, s in zip(chunk, pick, sims[np.arange(chunk.size), pick]):
            (attached[int(p)] if s >= cutoff else unplaced).append(int(r))

    people: list[tuple[int, ...]] = []
    for g, extra, cent in zip(groups, attached, cents):
        rows = g + extra
        closeness = unit[rows] @ cent
        # Closest to the centroid first, ties by id; the representative leads.
        by_closeness = sorted(range(len(rows)), key=lambda r: (-closeness[r], ids[rows[r]]))
        rep = max(
            by_closeness,
            key=lambda r: (qualities[rows[r]] is not None, qualities[rows[r]] or 0.0),
        )
        ordered = [rep] + [r for r in by_closeness if r != rep]
        people.append(tuple(int(ids[rows[r]]) for r in ordered))
    people.sort(key=lambda p: (-len(p), min(p)))
    _report(1, 1, message)
    return PeoplePartition(
        people=tuple(people),
        unplaced=tuple(sorted(int(ids[r]) for r in unplaced)),
        method=PEOPLE_METHOD,
        linkage=PEOPLE_LINKAGE,
        cutoff=float(cutoff),
        compat=str(compat),
    )


# ----------------------------------------------------------------------
# Per-dataset lifecycle: memory → container → fit
# ----------------------------------------------------------------------

#: One fit at a time per dataset: the build stage and a request racing for the
#: same partition wait on each other instead of fitting twice.
_fit_locks: dict[str, threading.Lock] = {}
_fit_locks_guard = threading.Lock()

#: Serialises a request's look at the in-flight job with its start of a new one,
#: so two asks at once start one fit.
_request_lock = threading.Lock()


def _fit_lock(dataset_id: str) -> threading.Lock:
    with _fit_locks_guard:
        lock = _fit_locks.get(dataset_id)
        if lock is None:
            lock = _fit_locks[dataset_id] = threading.Lock()
        return lock


def wants_people(ctx: DatasetContext) -> bool:
    """Whether *ctx* is a dataset of a type that has people (:data:`PEOPLE_MEDIA_TYPES`)."""
    from vtscore.projection.service import media_type_for  # noqa: PLC0415

    return media_type_for(ctx) in PEOPLE_MEDIA_TYPES


def dataset_compat(ctx: DatasetContext) -> str | None:
    """The face space *ctx*'s faces were embedded in (their ``Face Compat`` stamp), or ``None``.

    ``None`` when no face carries one, or when they carry more than one: the
    face embedder refuses to mix spaces in a dataset, so a mixed one has no
    space a cut could be in.
    """
    from vtscore.media.face.imagekit_bridge import COMPAT_METADATA_KEY  # noqa: PLC0415

    tags = {
        str(tag)
        for media in list(ctx.medias.values())
        if (tag := (media.get("custom_metadata") or {}).get(COMPAT_METADATA_KEY))
    }
    return next(iter(tags)) if len(tags) == 1 else None


def _face_matrix(ctx: DatasetContext) -> tuple[list[int], np.ndarray]:
    """The ``(sorted_ids, matrix)`` of *ctx*'s face vectors; raises :class:`NothingToPartition` without any."""
    from vtscore.embedding.matrix import get_embedding_matrix  # noqa: PLC0415
    from vtscore.media.face.imagekit_bridge import EMBEDDER_NAME  # noqa: PLC0415

    if not wants_people(ctx):
        raise NothingToPartition("People are found in Face datasets only.")
    try:
        sorted_ids, matrix = get_embedding_matrix(ctx, EMBEDDER_NAME)
    except ValueError as exc:
        raise NothingToPartition(f"This dataset's faces have no vectors to find people by: {exc}") from exc
    if matrix.size == 0:
        raise NothingToPartition("This dataset has no faces to find people among.")
    return list(sorted_ids), matrix


def _face_signals(ctx: DatasetContext, sorted_ids: list[int]) -> tuple[list[bool], list[Any]]:
    """Each face's hard-view flag and Face Quality, in *sorted_ids* order."""
    from vtscore.media.face.imagekit_bridge import QUALITY_METADATA_KEY, media_hard_view  # noqa: PLC0415

    medias = dict(ctx.medias)
    hard: list[bool] = []
    quality: list[Any] = []
    for cid in sorted_ids:
        media = medias.get(cid) or {}
        hard.append(bool(media_hard_view(media)))
        quality.append((media.get("custom_metadata") or {}).get(QUALITY_METADATA_KEY))
    return hard, quality


def _stored_people(ctx: DatasetContext, ids: list[int], cutoff: float, compat: str) -> PeoplePartition | None:
    """The partition cached on *ctx* or persisted beside it, if it covers *ids* and was cut as asked."""
    cached = ctx._people
    if cached is not None and cached.covers(ids) and cached.cut_as(cutoff, compat):
        return cached
    from vtscore.projection.store import pkl_path_for  # noqa: PLC0415

    pkl_path = pkl_path_for(ctx.dataset_id)
    if pkl_path is None:
        return None
    from vtscore.datasets.container import read_people  # noqa: PLC0415

    payload = read_people(pkl_path)
    if payload is None:
        return None
    try:
        partition = PeoplePartition.from_payload(payload)
    except (KeyError, TypeError, ValueError):
        logger.warning("Persisted people of %s are unreadable; will re-fit.", ctx.dataset_id, exc_info=True)
        return None
    if not partition.cut_as(cutoff, compat):
        logger.info("Persisted people of %s were cut another way; will re-fit.", ctx.dataset_id)
        return None
    if not partition.covers(ids):
        logger.info("Persisted people of %s miss faces; will re-fit.", ctx.dataset_id)
        return None
    ctx._people = partition
    return partition


def _persist(dataset_id: str, partition: PeoplePartition) -> None:
    """Best-effort save of *partition* into the dataset's container."""
    from vtscore.projection.store import pkl_path_for  # noqa: PLC0415

    pkl_path = pkl_path_for(dataset_id)
    if pkl_path is None:
        return
    try:
        from vtscore.datasets.container import append_people  # noqa: PLC0415

        append_people(pkl_path, partition.to_payload())
    except Exception:
        logger.warning("Failed to persist people for %s", dataset_id, exc_info=True)


def ensure_people(
    ctx: DatasetContext,
    sorted_ids: list[int],
    matrix: np.ndarray,
    *,
    cutoff: float,
    compat: str,
    force: bool = False,
    on_progress: ProgressFn | None = None,
) -> PeoplePartition:
    """*ctx*'s people over *sorted_ids*: the stored partition, else a fit of *matrix* now.

    *matrix* is the face embedder's, row ``i`` being ``sorted_ids[i]``.  A
    fresh fit is cached on *ctx* and persisted.  With *force*, the stored
    partition is fit again.  One fit at a time per dataset: a second caller
    waits, then finds the first one's partition (a forced caller fits anyway).
    """
    if not force:
        stored = _stored_people(ctx, sorted_ids, cutoff, compat)
        if stored is not None:
            return stored
    with _fit_lock(ctx.dataset_id):
        if not force:
            # Whoever held the lock may have just fit it.
            stored = _stored_people(ctx, sorted_ids, cutoff, compat)
            if stored is not None:
                return stored
        hard, quality = _face_signals(ctx, sorted_ids)
        partition = fit_people(
            matrix,
            sorted_ids,
            cutoff=cutoff,
            compat=compat,
            hard=hard,
            quality=quality,
            on_progress=on_progress,
        )
        ctx._people = partition
        _persist(ctx.dataset_id, partition)
        return partition


def people_for(
    ctx: DatasetContext,
    *,
    fit: bool = True,
    on_progress: ProgressFn | None = None,
) -> PeoplePartition | None:
    """*ctx*'s people: cached, else persisted, else (with *fit*) fit now.

    ``None`` for a dataset that has none to find (not Face, no face vectors), a
    face space with no cut (:func:`people_cutoff`), or (``fit=False``) one with
    nothing stored yet.  What the load pipeline's people stage calls.
    """
    try:
        sorted_ids, matrix = _face_matrix(ctx)
    except NothingToPartition:
        return None
    compat = dataset_compat(ctx)
    cutoff = people_cutoff(compat)
    if compat is None or cutoff is None:
        return None
    if not fit:
        return _stored_people(ctx, sorted_ids, cutoff, compat)
    return ensure_people(ctx, sorted_ids, matrix, cutoff=cutoff, compat=compat, on_progress=on_progress)


def request_people(ctx: DatasetContext, *, force: bool = False) -> dict:
    """What a view of *ctx*'s people is told (#5586).

    - ``{"status": "ready", "people": PeoplePartition}`` once a partition that
      covers the live faces is stored.  It may name faces removed since the
      fit; :meth:`PeoplePartition.restricted_to` drops them.
    - ``{"status": "building", ...}`` (as
      :func:`~vtscore.projection.service.build_progress` words it) while a fit
      runs.  The first ask with nothing stored starts one on the layout
      builds' runner, and so does an ask with *force* (Rebuild people), unless
      a fit is already running.
    - ``{"status": "error", "error": ...}`` once for a fit that failed; then
      it is forgotten, so the next ask (a new request from the user, not a
      poll) starts over rather than every poll re-running a failing fit.
    - ``{"status": "not_calibrated", "compat": ..., "message": ...}`` for a
      face space with no cut (:func:`people_cutoff`): no partition is fit.

    Raises :class:`NothingToPartition` for a dataset with no people to find:
    not a Face dataset, or no face vectors.
    """
    from vtscore.concurrency.async_jobs import projection_jobs  # noqa: PLC0415
    from vtscore.projection import service  # noqa: PLC0415

    sorted_ids, matrix = _face_matrix(ctx)
    compat = dataset_compat(ctx)
    cutoff = people_cutoff(compat)
    if compat is None or cutoff is None:
        return {"status": "not_calibrated", "compat": compat, "message": not_calibrated_message(compat)}

    with _request_lock:
        job = projection_jobs.get(ctx._people_job_id) if ctx._people_job_id else None
        if job is not None:
            if job.status in ("running", "pending"):
                return service.build_progress(job)
            ctx._people_job_id = None
            if job.status == "error":
                return {"status": "error", "error": job.error or "Finding people failed."}
            # Done (its partition is stored unless the faces changed under it)
            # or cancelled: read what is stored, else start over below.
        if not force:
            stored = _stored_people(ctx, sorted_ids, cutoff, compat)
            if stored is not None:
                return {"status": "ready", "people": stored}
        sig = (ctx.dataset_id, "people", tuple(sorted_ids), cutoff, compat, force)
        job = projection_jobs.start(
            sig,
            _people_runner(ctx, sorted_ids, matrix, cutoff=cutoff, compat=compat, force=force),
            dataset_id=ctx.dataset_id,
        )
        ctx._people_job_id = job.job_id
        return service.build_progress(job)


def _people_runner(
    ctx: DatasetContext, sorted_ids: list[int], matrix: np.ndarray, *, cutoff: float, compat: str, force: bool
):
    """The worker-thread body of a people fit, shaped as the layout build's runner is.

    Snapshots the matrix and ids so the job is independent of any later change
    to the dataset, and re-binds the worker thread to *ctx*.
    """
    mat_copy = np.array(matrix, copy=True)
    ids_copy = list(sorted_ids)

    def _run(job):
        from vtscore.state.core import thread_dataset_context  # noqa: PLC0415

        with thread_dataset_context(ctx):
            job.set_phase(1, 1, "finding people")
            job.result = ensure_people(
                ctx, ids_copy, mat_copy, cutoff=cutoff, compat=compat, force=force, on_progress=job.update_progress
            )

    return _run


__all__ = [
    "PEOPLE_LINKAGE",
    "PEOPLE_MEDIA_TYPES",
    "PEOPLE_METHOD",
    "PEOPLE_NEIGHBOURS",
    "NothingToPartition",
    "PeoplePartition",
    "dataset_compat",
    "ensure_people",
    "fit_people",
    "not_calibrated_message",
    "people_cutoff",
    "people_for",
    "request_people",
    "wants_people",
]
