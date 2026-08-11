"""The one resolver for the knobs a Browse projection is fit under.

Two code paths fit the VTSBrowse projection — the on-demand
``POST /api/projection/build`` route and the opt-in ingest-time
``build_projection`` load stage — and they must agree.  When they don't, the
ingest fit is either thrown away on the first Browse open (its stamped params
fail the persisted-layout check, UMAP re-runs, and the whole point of the
pre-build is lost) or silently kept under knobs nobody chose.  So neither path
resolves anything itself; both call :func:`resolve_projection_params`.

Resolution order for ``n_neighbors`` / ``min_dist``:

1. an explicit ``ServerSettings`` override, read off the library-tier
   :class:`~vtscore.config.CoreConfig` (the app populates its
   ``projection_n_neighbors`` / ``projection_min_dist`` fields from the
   settings file).  "Explicit" means a value that *differs* from the global
   config default, so an operator who deliberately picks the global value
   simply gets it;
2. the per-embedder tuned default
   (:data:`~vtscore.config.PROJECTION_DEFAULTS_BY_EMBEDDER`), keyed off the
   embedder whose vectors the projection is actually fit on;
3. the global config default.

``compact`` is not a user-facing setting at all: it is always
:data:`~vtscore.config.PROJECTION_COMPACT_DEFAULT` (off since the Part 1 sweep
— see ``docs/plans/vtsbrowse-empirical-tuning.md``).  It is resolved here
anyway so that "the params this layout was fit under" is a single value both
paths thread into ``fit_projection`` and stamp onto the frozen
:class:`~vtscore.projection.umap_projection.Projection`.

Reading ``CoreConfig`` is best-effort: a library-only process with no app
config builder installed falls back to the tuned/global defaults rather than
failing a fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vtscore.config import (
    PROJECTION_COMPACT_DEFAULT,
    PROJECTION_DEFAULTS_BY_EMBEDDER,
    PROJECTION_MIN_DIST,
    PROJECTION_N_NEIGHBORS,
)

if TYPE_CHECKING:
    from vtscore.state.core import DatasetContext


@dataclass(frozen=True)
class ProjectionParams:
    """The resolved knobs one Browse projection is fit under.

    Passed straight into :func:`~vtscore.projection.umap_projection.fit_projection`
    and stamped onto the resulting layout, so a persisted projection can be
    compared field-for-field against what the active configuration would
    produce today.
    """

    n_neighbors: int
    min_dist: float
    compact: bool


def projection_embedder_for(ctx: DatasetContext | None) -> str | None:
    """The embedder whose vectors this dataset's projection is fit on.

    The projection clusters in the score embedder's space (patch-else-text; the
    v3 routing table), which for the common single-embedder dataset resolves to
    the dataset's primary embedder.  This is the key
    :data:`~vtscore.config.PROJECTION_DEFAULTS_BY_EMBEDDER` is looked up under,
    so it must name the embedder that actually produced the matrix — not
    whichever one happens to be listed first.
    """
    if ctx is None:
        return None
    try:
        routed = ctx.routed_embedder("score")
        if routed:
            return routed
        medias = getattr(ctx, "medias", None)
        if not medias:
            return None
        from vtscore.embedding.media_vectors import primary_embedder_name  # noqa: PLC0415

        return primary_embedder_name(next(iter(medias.values())))
    except Exception:  # pragma: no cover - defensive; fall back to the globals
        return None


def _settings_overrides() -> tuple[int | None, float | None]:
    """The operator's explicit ``(n_neighbors, min_dist)`` overrides, if any.

    ``None`` in a slot means "left at the global default", i.e. no override —
    which is what lets the per-embedder tuned value apply.  A missing or
    un-built :class:`~vtscore.config.CoreConfig` reads as "no override".
    """
    try:
        from vtscore.config import CoreConfig  # noqa: PLC0415

        cfg = CoreConfig.from_settings()
        n = int(cfg.projection_n_neighbors)
        d = float(cfg.projection_min_dist)
    except Exception:
        return None, None
    return (
        n if n != PROJECTION_N_NEIGHBORS else None,
        d if d != PROJECTION_MIN_DIST else None,
    )


def resolve_projection_params(ctx: DatasetContext | None = None) -> ProjectionParams:
    """Resolve the UMAP knobs *ctx*'s Browse projection should be fit under.

    See the module docstring for the resolution order.  Safe to call from
    either tier and from a background worker thread; it only reads
    configuration.
    """
    embedder = projection_embedder_for(ctx)
    tuned_n, tuned_d = PROJECTION_DEFAULTS_BY_EMBEDDER.get(
        embedder or "", (PROJECTION_N_NEIGHBORS, PROJECTION_MIN_DIST)
    )
    override_n, override_d = _settings_overrides()
    return ProjectionParams(
        n_neighbors=override_n if override_n is not None else tuned_n,
        min_dist=override_d if override_d is not None else tuned_d,
        compact=PROJECTION_COMPACT_DEFAULT,
    )


__all__ = ["ProjectionParams", "projection_embedder_for", "resolve_projection_params"]
