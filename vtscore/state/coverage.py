"""Coverage Atlas management for evidence-aware diverse sampling.

Operates on the active :class:`DatasetContext` (for the atlas itself) and the
active :class:`DetectorContext` (for the labels that get replayed into it).
Functions resolve the contexts themselves - no module-level proxy names are
imported.  See Phase 3 of ``../docs/architecture.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vtscore.embedding.media_vectors import media_embedding
from vtscore.state.core import (
    DatasetContext,
    DetectorContext,
    _state_lock,
    get_active_context,
    get_active_detector_context,
)

# Above this item count the coverage atlas is *not* built automatically at
# dataset load: hierarchical k-means over millions of vectors costs minutes and
# gigabytes, and the atlas only feeds the labeling autopilot (which degrades
# gracefully to score-only sampling when the atlas is absent - see the
# ``atlas is None`` guards in this module).  Large datasets can build it
# on demand via ``POST /api/datasets/registry/<id>/coverage-atlas``.
COVERAGE_ATLAS_AUTO_THRESHOLD = 50_000


def should_auto_build_coverage_atlas(n: int) -> bool:
    """Whether a dataset of *n* items should build its coverage atlas at load.

    Returns ``False`` past :data:`COVERAGE_ATLAS_AUTO_THRESHOLD` so large
    datasets load fast; the atlas can then be built on demand.
    """
    return n <= COVERAGE_ATLAS_AUTO_THRESHOLD


def resync_coverage_atlas_to_detector(
    ds_ctx: DatasetContext,
    det_ctx: DetectorContext,
) -> None:
    """Rebuild *ds_ctx*'s coverage-atlas evidence state from *det_ctx*'s votes.

    Clears the atlas's evidence channels and replays every cid in
    ``det_ctx.good_votes`` / ``det_ctx.bad_votes`` whose leaf is known to the
    atlas.  Used when the labeled set is replaced wholesale - votes are
    cleared, or the active detector is swapped on the same dataset - so the
    atlas continues to reflect the *current* detector's labels instead of
    whatever it last observed.

    Callers must already hold ``_state_lock``.  No-op when the dataset has
    no coverage atlas.
    """
    atlas = ds_ctx.coverage_atlas
    if atlas is None:
        return
    atlas.reset_labeled()
    for cid in det_ctx.good_votes:
        if cid in atlas.vector_to_leaf:
            atlas.label(cid, good=True)
    for cid in det_ctx.bad_votes:
        if cid in atlas.vector_to_leaf:
            atlas.label(cid, good=False)


def build_coverage_atlas(
    media_dict: dict[int, dict[str, Any]] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Build a 3-way Coverage Atlas from media embeddings on the active dataset context.

    Uses the active :class:`DatasetContext`'s medias by default, or an explicit
    *media_dict* if provided.  Existing labels on the active detector context
    (``good_votes`` and ``bad_votes``) are replayed into the new atlas so the
    evidence state stays accurate.

    *on_progress*, when provided, is called as ``on_progress(current, total)``
    to report how many k-means clustering fits have been completed so far.
    """
    import numpy as np

    from vtscore.coverage.atlas import CoverageAtlas, auto_max_depth

    with _state_lock:
        ds_ctx = get_active_context()
        source = media_dict if media_dict is not None else ds_ctx.medias
        vectors: dict[int, np.ndarray] = {}
        for cid, media in source.items():
            emb = media_embedding(media)
            if emb is not None:
                vectors[cid] = np.asarray(emb, dtype=np.float32)

        if not vectors:
            ds_ctx.coverage_atlas = None
            return

        atlas = CoverageAtlas(
            vectors,
            k=3,
            max_depth=auto_max_depth(len(vectors), k=3),
            on_progress=on_progress,
        )
        ds_ctx.coverage_atlas = atlas

        # Replay existing labels so the atlas reflects the current vote state.
        resync_coverage_atlas_to_detector(ds_ctx, get_active_detector_context())


def build_coverage_atlas_for_context(
    ctx: DatasetContext,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Build a coverage atlas and store it on a specific context.

    Unlike :func:`build_coverage_atlas` this does not touch the active
    context - it operates entirely on *ctx*.  Used by parallel dataset
    loading where the new dataset is not yet active.
    """
    import numpy as np

    from vtscore.coverage.atlas import CoverageAtlas, auto_max_depth

    with _state_lock:
        medias_snapshot = dict(ctx.medias)

    vectors: dict[int, np.ndarray] = {}
    for cid, media in medias_snapshot.items():
        emb = media_embedding(media)
        if emb is not None:
            vectors[cid] = np.asarray(emb, dtype=np.float32)

    if not vectors:
        ctx.coverage_atlas = None
        return

    atlas = CoverageAtlas(
        vectors,
        k=3,
        max_depth=auto_max_depth(len(vectors), k=3),
        on_progress=on_progress,
    )
    ctx.coverage_atlas = atlas
    # Fresh context has no votes - nothing to replay.


def build_coverage_atlas_serializable(
    medias: dict[int, dict[str, Any]],
    on_progress: Callable[[int, int], None] | None = None,
) -> dict | None:
    """Build a coverage atlas over *medias* and return its cache payload.

    Returns the :meth:`CoverageAtlas.to_serializable` dict for embedding in a
    dataset pickle, or ``None`` when there are no usable vectors.  Unlike
    :func:`build_coverage_atlas_for_context` this touches no context and
    replays no votes — it exists for *save* paths (e.g. dataset promote) that
    want to cache the atlas at creation so reloads restore it instead of
    paying the hierarchical-k-means rebuild every time.
    """
    import numpy as np

    from vtscore.coverage.atlas import CoverageAtlas, auto_max_depth

    vectors: dict[int, np.ndarray] = {}
    for cid, media in medias.items():
        emb = media_embedding(media)
        if emb is not None:
            vectors[cid] = np.asarray(emb, dtype=np.float32)

    if not vectors:
        return None

    atlas = CoverageAtlas(
        vectors,
        k=3,
        max_depth=auto_max_depth(len(vectors), k=3),
        on_progress=on_progress,
    )
    return atlas.to_serializable()


def restore_coverage_atlas_from_cache(ctx: DatasetContext, cached: object) -> bool:
    """Adopt a cached coverage atlas onto *ctx* when it matches the medias.

    *cached* is the ``"coverage_atlas"`` payload read from a dataset pickle (or
    ``None``).  The cache is adopted only when it deserialises cleanly **and**
    its vector set exactly matches *ctx*'s current media IDs.  A mismatch means
    the medias were remapped, deduplicated, or partially dropped since the
    cache was written, so the caller must rebuild from scratch instead.
    Old diversity-tree caches fail the format check and rebuild the same way.

    Returns ``True`` when the atlas was restored (the caller should skip the
    rebuild), ``False`` otherwise.  The caller remains responsible for
    replaying the active detector's votes via
    :func:`resync_coverage_atlas_to_detector` once the atlas is in place.
    """
    if not cached:
        return False

    from vtscore.coverage.atlas import CoverageAtlas

    try:
        atlas = CoverageAtlas.from_serializable(cached)
    except (ValueError, TypeError):
        return False
    # dict_keys compare as sets; require an exact 1:1 with the loaded medias so
    # a remapped / deduped / dropped media set falls through to a rebuild.
    if atlas.vector_to_leaf.keys() != ctx.medias.keys():
        return False
    ctx.coverage_atlas = atlas
    return True


def get_coverage_atlas():
    """Return the active dataset context's CoverageAtlas instance, or ``None``."""
    with _state_lock:
        return get_active_context().coverage_atlas


def coverage_atlas_next_sample(
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
        atlas = get_active_context().coverage_atlas
        if atlas is None:
            return None
        return atlas.next_sample(scores=scores, threshold=threshold)


def coverage_atlas_label(media_id: int, good: bool) -> None:
    """Count *media_id* as good/bad evidence in the active dataset's atlas."""
    with _state_lock:
        atlas = get_active_context().coverage_atlas
        if atlas is not None and media_id in atlas.vector_to_leaf:
            atlas.label(media_id, good=good)


def coverage_atlas_unlabel(media_id: int) -> None:
    """Remove *media_id*'s evidence from the active dataset's atlas."""
    with _state_lock:
        atlas = get_active_context().coverage_atlas
        if atlas is not None and media_id in atlas.vector_to_leaf:
            atlas.unlabel(media_id)
