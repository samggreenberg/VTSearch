"""Signpost prep orchestration: texts → Toponymy fit → cache (+ persist).

The one entry point every build path calls after it has a frozen layout in
hand (see ``docs/plans/vtsbrowse-toponymy.md``):

* the **ingest** projection stage (``vtscore.datasets.stages.projection``) —
  the user is already watching a progress bar, so prep rides along and the
  first Browse opens lettered;
* the **lazy Browse build** (``vtscore.projection.service.start_full_build``)
  — datasets ingested before/without prep get signs on their first Browse;
* the **Find→Browse subset build** (``service.start_subset_build``) — signs are
  re-fit over just the subset (contrastive keyphrases recompute against the
  subset's own siblings, which the image study showed beats filtering
  dataset-level signs), reusing the per-media texts cached at ingest so the
  re-fit is interactive.

Split of work between the stages (what must be computed when):

* **Per-media texts** are the only full-corpus model cost — Toponymy's
  keyphrase mining reads *every* object's text, not just sampled exemplars —
  and they are clustering-independent, so they are computed once and cached
  on the media dicts (:mod:`vtscore.projection.signpost_texts`).
* **Clustering + naming** are layout-scoped and cheap (a ~5-D UMAP + the
  fit), so they run fresh per layout, full or subset.

Everything is best-effort: any missing prerequisite returns ``None`` and the
map simply stays unlettered.  The prerequisites are not equal, though.  A
missing text-capable embedder or a media type with no provider is a routine,
data-dependent skip (silent).  A missing ``toponymy`` install is *not*:
``scripts/install.sh`` installs it unconditionally, so its absence is a broken
environment, and the build paths gate on :func:`~vtscore.projection.signpost_build.require_signposting`
(which logs a one-time error) rather than the silent probe.  The serve /
signature paths still use the quiet probe, since a ``None`` there is expected
on every poll.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from vtscore.projection.signpost_build import (
    _MIN_POINTS,
    build_region_labels,
    require_signposting,
    signposting_available,
    toponymy_version,
)
from vtscore.projection.signpost_texts import ensure_signpost_texts, provider_for

if TYPE_CHECKING:
    from vtscore.projection.labels import RegionLabelSet
    from vtscore.projection.umap_projection import Projection

logger = logging.getLogger(__name__)

#: Progress callback shape: ``(current, total, message)``.
ProgressFn = Callable[[int, int, str], None]

#: Minimum fraction of the fitted set that must have a usable text — below
#: this the keyphrase stage is mining mostly-empty documents and the signs
#: would be noise, so we skip labeling instead.
_MIN_TEXT_COVERAGE = 0.5

#: Toponymy prompt vocabulary per media type (``object_description``).
_OBJECT_DESCRIPTIONS = {
    "audio": "audio clips",
    "image": "images",
    "video": "video clips",
    "document": "documents",
    "text": "text documents",
}


def _text_embedder(ctx: Any) -> Any | None:
    """The dataset's text-capable embedder instance, or ``None``.

    Keyphrase strings must be embedded into the same space as the media
    matrix used for alignment, so signposting requires the text slot of the
    v3 routing table (CLAP / SigLIP / X-CLIP / E5); patch- or structural-only
    datasets go unlettered.
    """
    name = ctx.routed_embedder("text")
    if not name:
        return None
    try:
        from vtscore.media import get_embedder  # noqa: PLC0415

        embedder = get_embedder(name)
    except Exception:
        return None
    if not getattr(embedder, "supports_text", False):
        return None
    return embedder


def labeler_signature(ctx: Any) -> str | None:
    """The signature the active pipeline would stamp on freshly built signs.

    Persisted label sets carry the signature they were built under; a set
    whose signature no longer matches is stale (different provider, embedder,
    or toponymy version) and is not served.  ``None`` when signposting isn't
    possible for this dataset at all.
    """
    from vtscore.projection.signpost_texts import texts_signature  # noqa: PLC0415

    if not signposting_available():
        return None
    embedder = _text_embedder(ctx)
    if embedder is None:
        return None
    texts_sig = texts_signature(ctx.medias, embedder)
    if texts_sig is None:
        return None
    return f"keyphrase|{texts_sig}|toponymy={toponymy_version()}"


def ensure_texts_for_dataset(ctx: Any, on_progress: ProgressFn | None = None) -> None:
    """Compute + cache the per-media signpost texts for *ctx*'s whole dataset.

    The ingest pipeline calls this **before the dataset pickle is written**, so
    the texts — the only full-corpus model cost in the sign pipeline — persist
    with the dataset and every later browse or Find→Browse re-fit is free of
    text-model calls.  No-op when signposting isn't possible for this dataset
    (no toponymy install, no text-capable embedder, no provider).
    """
    if not require_signposting():
        return
    embedder = _text_embedder(ctx)
    if embedder is None:
        return
    if not ctx.medias:
        return
    first = next(iter(ctx.medias.values()))
    if provider_for(first.get("media_type", "")) is None:
        return
    from vtscore.embedding.matrix import get_embedding_matrix  # noqa: PLC0415

    try:
        ids, matrix = get_embedding_matrix(ctx, getattr(embedder, "name", None))
    except ValueError:
        return
    if matrix.size == 0:
        return
    ensure_signpost_texts(ctx.medias, ids, matrix, embedder, on_progress)


def prep_signposts(
    ctx: Any,
    proj: "Projection",
    *,
    subset: bool,
    on_progress: ProgressFn | None = None,
) -> "RegionLabelSet | None":
    """Build region signposts for *proj* and cache them on *ctx*.

    Returns the built :class:`RegionLabelSet` (assigned to
    ``ctx._subset_region_labels`` or ``ctx._region_labels``), or ``None``
    when signposting isn't available for this dataset.  Full-dataset sets are
    also persisted into the dataset container next to the projection; subset
    sets are in-memory only, like the subset layout itself.

    Raises nothing in normal operation — callers still wrap it best-effort so
    a labeling failure can never take down the build that produced the layout.
    """
    prerequisites = _prep_prerequisites(ctx, proj)
    if prerequisites is None:
        return None
    ids, media_type, embedder = prerequisites

    matrices = _aligned_matrices(ctx, ids, embedder)
    if matrices is None:
        return None
    score_matrix, embed_matrix = matrices

    if on_progress is not None:
        on_progress(0, 0, "Preparing signpost texts…")
    texts_map = ensure_signpost_texts(ctx.medias, ids, embed_matrix, embedder, on_progress)
    if not texts_map or len(texts_map) < _MIN_TEXT_COVERAGE * len(ids):
        return None
    texts = [texts_map.get(mid, "") for mid in ids]

    object_description = _OBJECT_DESCRIPTIONS.get(media_type, "items")
    label_set = build_region_labels(
        proj,
        score_matrix,
        embed_matrix,
        texts,
        embedder,
        object_description=object_description,
        corpus_description=f"a collection of {object_description}",
        on_progress=on_progress,
    )
    if not label_set.labels:
        # Nothing worth lettering (tiny corpus, degenerate tree).  Leave the
        # context slot alone rather than pinning an empty set — an empty set
        # would suppress the lazy ground-truth fallback for datasets that
        # ship a category hierarchy (see ``signpost_serve.label_set_for``).
        return None

    if subset:
        ctx._subset_region_labels = label_set
    else:
        ctx._region_labels = label_set
        _persist_region_labels(ctx, label_set)
    return label_set


def _prep_prerequisites(ctx: Any, proj: "Projection") -> tuple[list[int], str, Any] | None:
    """Gate a lettering run; return ``(ids, media_type, embedder)`` or ``None``.

    ``None`` means signposting isn't possible or worthwhile here: no toponymy
    install, no text-capable embedder, a layout too small for a non-degenerate
    topic tree (bail *before* paying for texts), or a media type with no
    registered text provider.
    """
    if not require_signposting():
        return None
    embedder = _text_embedder(ctx)
    if embedder is None:
        return None
    ids = [int(mid) for mid in proj.ids]
    if len(ids) < _MIN_POINTS:
        return None
    first = ctx.medias.get(ids[0])
    if not first:
        return None
    media_type = first.get("media_type", "")
    if provider_for(media_type) is None:
        return None
    return ids, media_type, embedder


def _aligned_matrices(ctx: Any, ids: list[int], embedder: Any) -> tuple[Any, Any] | None:
    """The score + text embedding matrices, both row-aligned with *ids*.

    The clustering space must match the frozen layout's (the score slot),
    while keyphrase alignment lives in the text embedder's space; for the
    common single cross-modal embedder these are the same matrix.  Row order
    matters — cluster member indices index into texts/matrix rows AND into
    ``proj.coords`` rows — and the submatrix helper returns sorted ids while
    every projection is fit on sorted ids, so an order mismatch means medias
    changed under the layout (anchors would lie): return ``None`` and skip.
    """
    from vtscore.embedding.matrix import get_embedding_submatrix  # noqa: PLC0415

    score_name = ctx.routed_embedder("score")
    text_name = getattr(embedder, "name", None)
    try:
        sub_ids, score_matrix = get_embedding_submatrix(ctx, ids, score_name)
        if sub_ids != ids:
            return None
        if score_name == text_name:
            return score_matrix, score_matrix
        embed_ids, embed_matrix = get_embedding_submatrix(ctx, ids, text_name)
        if embed_ids != ids:
            return None
        return score_matrix, embed_matrix
    except ValueError:
        return None


def _persist_region_labels(ctx: Any, label_set: "RegionLabelSet") -> None:
    """Best-effort save of a full-dataset label set into the dataset container."""
    signature = labeler_signature(ctx)
    if signature is None:
        return
    try:
        from vtscore.datasets.registry import get_dataset  # noqa: PLC0415

        entry = get_dataset(ctx.dataset_id)
        pkl_path = (entry or {}).get("pkl_path")
        if not pkl_path:
            return
        from vtscore.datasets.container import append_region_labels  # noqa: PLC0415

        append_region_labels(pkl_path, label_set, signature)
    except Exception:
        logger.warning("Failed to persist region labels for %s", ctx.dataset_id, exc_info=True)


__all__ = ["ensure_texts_for_dataset", "labeler_signature", "prep_signposts"]
