"""Where a dataset's Browse layout lives on disk, and whether it is still fresh.

The persistence half of the projection lifecycle: resolving a dataset id to
its container path, saving a freshly-fit layout into it, restoring one back
out, and deciding whether a restored layout was fit under the knobs the
active configuration would use today.

This is the carve-out from the "No Persisted Vectors or MLPs" rule (see
CLAUDE.md): a projection is 2-D coordinates derived from the pickle's own
medias, written beside them in the same container, and rebuilt from scratch
when the id set or the fit parameters no longer match.

Everything here is best-effort by contract.  A missing container, an
unwritable pickle, or a layout fit under stale parameters costs only the time
of a re-fit, never correctness -- so failures are logged and reported as
"nothing stored", never raised at the caller.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vtscore.projection.pyramid import Pyramid
    from vtscore.projection.umap_projection import Projection
    from vtscore.state import DatasetContext

logger = logging.getLogger(__name__)

#: Bin shapes a dataset can be tiled with.  Mirrors
#: :data:`vtscore.projection.pyramid.BIN_SHAPES`, kept as a local literal so
#: probing the container for persisted coordinates does not import the
#: (numba-pulling) binning modules.
_VALID_SHAPES = ("hex", "square")


def pkl_path_for(dataset_id: str) -> str | None:
    """Return the container path registered for *dataset_id*, or ``None``.

    The single seam every projection-persistence path resolves its container
    through -- the lazy Browse build, the ingest pre-build stage, and the
    signpost writer all used to inline their own copy of this registry
    lookup.  Being one function also makes it one monkeypatch target for
    tests that want to point persistence at a temp file.
    """
    from vtscore.datasets.registry import get_dataset  # noqa: PLC0415

    entry = get_dataset(dataset_id)
    if entry is None:
        return None
    return entry.get("pkl_path") or None


def persist_projection(dataset_id: str, proj: Projection, pyr: Pyramid) -> None:
    """Best-effort save of *proj* (for ``pyr``'s bin shape) into the container.

    Failures are swallowed and logged: the layout is already cached on the
    context, so a missing on-disk copy costs the next process a re-fit and
    nothing else.
    """
    pkl_path = pkl_path_for(dataset_id)
    if pkl_path is None:
        return
    try:
        from vtscore.datasets.container import append_projection  # noqa: PLC0415

        append_projection(pkl_path, proj, pyr)
    except Exception:
        logger.warning("Failed to persist projection for %s", dataset_id, exc_info=True)


def remove_persisted_projections(dataset_id: str) -> None:
    """Best-effort drop of every persisted layout for *dataset_id*.

    Called by a forced re-projection: the coordinates are shared across bin
    shapes, so leaving any stored shape behind would let a later load (or the
    not-yet-rebuilt other shape) resurrect the arrangement the user just
    asked to replace.
    """
    pkl_path = pkl_path_for(dataset_id)
    if pkl_path is None:
        return
    try:
        from vtscore.datasets.container import remove_projections  # noqa: PLC0415

        remove_projections(pkl_path)
    except Exception:
        logger.warning("Failed to clear persisted projection for %s", dataset_id, exc_info=True)


def projection_params_match(proj: Any, ctx: DatasetContext | None = None) -> bool:
    """Whether a persisted projection was fit under the active params.

    Non-UMAP layouts (the PCA / trivial fallbacks for tiny datasets) ignore
    these knobs, so they always match.  A legacy projection with no stamped
    UMAP params is assumed to have used the config defaults, so it only
    mismatches once an operator has changed a setting away from the default --
    exactly when its layout must be recomputed.  An unstamped ``compact``
    reads as ``True``: compaction was on by default for every layout written
    before it was recorded, so those layouts correctly fail against today's
    ``compact=False`` and get refit.
    """
    if getattr(proj, "method", None) != "umap":
        return True
    from vtscore.config import PROJECTION_MIN_DIST, PROJECTION_N_NEIGHBORS  # noqa: PLC0415
    from vtscore.projection.params import resolve_projection_params  # noqa: PLC0415

    stored_n = proj.n_neighbors if proj.n_neighbors is not None else PROJECTION_N_NEIGHBORS
    stored_d = proj.min_dist if proj.min_dist is not None else PROJECTION_MIN_DIST
    stored_c = getattr(proj, "compact", None)
    stored_c = True if stored_c is None else bool(stored_c)
    want = resolve_projection_params(ctx)
    return (
        stored_n == want.n_neighbors
        and math.isclose(stored_d, want.min_dist, abs_tol=1e-9)
        and stored_c == want.compact
    )


def load_persisted_layout(
    ctx: DatasetContext,
    sorted_ids: list[int],
    bin_shape: str,
) -> tuple[Any, Any] | None:
    """Restore the *bin_shape* layout from the container, or ``None``.

    Returns the ``(projection, pyramid)`` pair only when it is genuinely
    usable for *ctx* right now: the stored id set must match *sorted_ids*
    (the dataset has not gained or lost items) and the stored fit parameters
    must match what :func:`projection_params_match` would resolve today.
    Does **not** install anything on the context -- that is the caller's
    decision.
    """
    pkl_path = pkl_path_for(ctx.dataset_id)
    if pkl_path is None:
        return None
    from vtscore.datasets.container import read_projection  # noqa: PLC0415

    loaded = read_projection(pkl_path, bin_shape)
    if loaded is None:
        return None
    proj, pyr = loaded
    if set(proj.ids) != set(sorted_ids):
        logger.info("Persisted %s projection ids mismatch; will recompute.", bin_shape)
        return None
    if not projection_params_match(proj, ctx):
        logger.info("Persisted %s projection UMAP params changed; will recompute.", bin_shape)
        return None
    return proj, pyr


def load_any_persisted_layout(
    ctx: DatasetContext,
    sorted_ids: list[int],
    *,
    prefer: str | None = None,
) -> tuple[Any, Any] | None:
    """Restore a usable layout from *any* persisted bin shape, or ``None``.

    The 2-D coordinates are shared across bin shapes, so any stored pyramid
    yields them -- letting a shape that was never persisted be re-binned from
    the frozen layout instead of re-fitting UMAP.  The pyramid that supplied
    the coordinates is returned alongside, since it was deserialized anyway,
    and *prefer* is probed first so a caller that wants one particular shape
    gets its own pyramid back rather than a re-bin of the other one.

    Each shape is probed at most once, so a caller that wants "this shape,
    else any" makes one pass and logs one line per genuinely stale shape.
    """
    shapes = _VALID_SHAPES if prefer is None else (prefer, *(s for s in _VALID_SHAPES if s != prefer))
    for shape in shapes:
        loaded = load_persisted_layout(ctx, sorted_ids, shape)
        if loaded is not None:
            return loaded
    return None


__all__ = [
    "load_any_persisted_layout",
    "load_persisted_layout",
    "persist_projection",
    "pkl_path_for",
    "projection_params_match",
    "remove_persisted_projections",
]
