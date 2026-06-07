"""Stage 1 of the VTSBrowse browse canvas: project the embedding matrix to 2-D.

Runs UMAP on the cached ``(N, d)`` embedding matrix
(:func:`vtscore.embedding.matrix.get_embedding_matrix`) to produce an
``(N, 2)`` layout.  Because embeddings are L2-normalized at ingest (see
*§Prerequisite* in ``docs/plans/vtsbrowse.md``), Euclidean
distance on the unit sphere is monotonic in cosine distance, so UMAP uses the
plain ``"euclidean"`` metric with no per-fit normalization.

Determinism follows the locked design decision: the fit is **unseeded by
default** (``random_state=None``), which keeps UMAP's numba parallelism on.
That is safe only because the projection is computed exactly once per dataset
and then frozen/persisted — it never re-runs, so its non-reproducibility never
surfaces.  Callers (e.g. tests) may pass a ``random_state`` for a reproducible
fit at the cost of parallelism.

Tiny datasets can't support a neighbor graph (UMAP needs ``n_neighbors < N``),
so below ``min_n_for_umap`` this falls back to a deterministic PCA-2 layout —
or a trivial layout for the degenerate ``N ≤ 1`` cases — rather than failing.

This module is Flask-free and imports ``umap`` lazily so importing the package
never pulls in numba's JIT until an actual fit runs.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Matches the ingest progress-callback shape (status, message, current, total).
ProgressCallback = Callable[[str, str, int, int], None]

# How often the UMAP fit emits a "still working" heartbeat.  UMAP's
# ``fit_transform`` is one opaque blocking call (the epoch loop runs inside
# numba with no per-epoch Python callback), so there's no real fraction to
# report; instead we tick elapsed seconds on this cadence so the UI shows a
# live counter and an animated bar instead of dead airtime.  The browse view
# polls ``/api/projection/meta`` once a second, so a 1 s heartbeat lines up.
_HEARTBEAT_SECONDS = 1.0


@dataclass(frozen=True)
class Projection:
    """A frozen 2-D layout of a dataset's embedding matrix.

    ``coords[i]`` is the projected point for media id ``ids[i]``.  ``method``
    records how it was produced (``"umap"``, ``"pca"``, or ``"trivial"``) and
    ``projection_id`` is minted at the one-time fit so tiles derived from it can
    be namespaced/cached against it (see *§Tile-cache invalidation*).
    """

    projection_id: str
    ids: list[int]
    coords: np.ndarray  # (N, 2) float32
    method: str

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` of the layout; zeros when empty."""
        if self.coords.shape[0] == 0:
            return (0.0, 0.0, 0.0, 0.0)
        xmin, ymin = self.coords.min(axis=0)
        xmax, ymax = self.coords.max(axis=0)
        return (float(xmin), float(ymin), float(xmax), float(ymax))


def _trivial_layout(n: int) -> np.ndarray:
    """A deterministic layout for ``n`` points when there's nothing to project.

    One point sits at the origin; a handful spread along a line so they don't
    all collapse to a single hex.
    """
    if n <= 1:
        return np.zeros((n, 2), dtype=np.float32)
    xs = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.stack([xs, np.zeros(n, dtype=np.float32)], axis=1)


def _pca_layout(matrix: np.ndarray) -> np.ndarray:
    """PCA-2 fallback for small-N datasets, padded to 2 columns when ``d < 2``."""
    from sklearn.decomposition import PCA

    n, d = matrix.shape
    n_components = min(2, n, d)
    if n_components < 1:
        return _trivial_layout(n)
    coords = PCA(n_components=n_components).fit_transform(matrix)
    if coords.shape[1] < 2:
        pad = np.zeros((n, 2 - coords.shape[1]), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)
    return np.ascontiguousarray(coords, dtype=np.float32)


def fit_projection(
    matrix: np.ndarray,
    ids: list[int],
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_n_for_umap: int = 10,
    random_state: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> Projection:
    """Project the ``(N, d)`` embedding *matrix* to a frozen 2-D :class:`Projection`.

    ``ids[i]`` labels row ``i`` of *matrix* (as returned by
    :func:`~vtscore.embedding.matrix.get_embedding_matrix`).  ``n_neighbors`` is
    clamped to ``N - 1`` so it stays valid on small datasets.  Below
    ``min_n_for_umap`` points the layout falls back to PCA-2 (deterministic).

    ``random_state`` defaults to ``None`` (unseeded, parallel — the production
    path); pass an int for a reproducible fit.  ``on_progress`` receives coarse
    ``(status, message, current, total)`` milestones if provided.
    """
    mat = np.ascontiguousarray(matrix, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2-D (N, d), got shape {mat.shape}")
    n = mat.shape[0]
    if len(ids) != n:
        raise ValueError(f"ids length {len(ids)} != matrix rows {n}")

    projection_id = uuid.uuid4().hex

    def _progress(status: str, message: str, current: int, total: int) -> None:
        if on_progress is not None:
            on_progress(status, message, current, total)

    if n == 0:
        return Projection(projection_id, [], np.empty((0, 2), dtype=np.float32), "trivial")

    if n <= 2 or mat.shape[1] < 2:
        # Too few points (or scalar embeddings) for either UMAP or PCA-2.
        _progress("projecting", "trivial layout", n, n)
        return Projection(projection_id, list(ids), _trivial_layout(n), "trivial")

    if n < min_n_for_umap:
        _progress("projecting", f"PCA fallback ({n} points)", 0, n)
        coords = _pca_layout(mat)
        _progress("projecting", "PCA fallback done", n, n)
        return Projection(projection_id, list(ids), coords, "pca")

    # total=0 keeps the frontend bar in its animated *indeterminate* state for
    # the whole fit (a non-zero total would freeze it at 0/N — the dead airtime
    # this code exists to kill); the message carries the live elapsed-seconds.
    _progress("projecting", f"UMAP fit ({n} points)", 0, 0)
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, n - 1),
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )

    # UMAP's fit is a single blocking call with no per-epoch hook, so we run it
    # on a worker thread and tick an elapsed-seconds heartbeat from here while
    # it's alive.  Anything the worker raises is re-raised in this thread.
    fit_result: list[np.ndarray] = []
    fit_error: list[Exception] = []

    def _fit() -> None:
        try:
            fit_result.append(reducer.fit_transform(mat))
        except Exception as exc:  # surfaced via fit_error below
            fit_error.append(exc)

    worker = threading.Thread(target=_fit, name="umap-fit", daemon=True)
    started = time.monotonic()
    worker.start()
    while True:
        worker.join(timeout=_HEARTBEAT_SECONDS)
        if not worker.is_alive():
            break
        elapsed = int(time.monotonic() - started)
        _progress("projecting", f"UMAP fit ({n} points) — {elapsed}s elapsed", 0, 0)

    if fit_error:
        raise fit_error[0]

    coords = np.ascontiguousarray(fit_result[0], dtype=np.float32)
    _progress("projecting", "UMAP fit done", n, n)
    return Projection(projection_id, list(ids), coords, "umap")
