"""Finalisation stages: drop failed embeds, collapse duplicates, coverage atlas.

These run after embedding to hand the registry/projection stages a clean
``medias`` dict: media that finished without an embedding are dropped,
exact-duplicate media are collapsed, and the coverage atlas is built.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from vtscore.embedding.matrix import invalidate_embedding_matrix
from vtscore.embedding.media_vectors import media_embedding
from vtscore.state import (
    build_coverage_atlas_for_context,
    collapse_duplicates,
    collapse_near_duplicates,
    should_auto_build_coverage_atlas,
)

from vtscore.datasets.stages._common import _TOTAL_LOAD_STEPS

if TYPE_CHECKING:
    from vtscore.datasets.stages.embedding import EmbedFailures
    from vtscore.state import DatasetContext

#: Reason given for a dropped item the embed stage recorded nothing about -
#: e.g. one that reached the finalize stage without a media type to embed by.
_UNKNOWN_DROP_REASON = "No embedder produced a vector"


def _dropped_item_label(mid: int, media: dict[str, Any]) -> str:
    """Name a dropped media the way the user would look for it.

    The file it came from, not the internal id (which the user cannot look up,
    and which is gone the moment the item is dropped).  A clip adds where in
    that file it sits, since every clip of one file shares its name.
    """
    name = str(media.get("origin_name") or media.get("filename") or media.get("media_path") or f"item {mid}")
    start, end = media.get("clip_start"), media.get("clip_end")
    if start is not None and end is not None:
        try:
            return f"{name} [{float(start):g}–{float(end):g}s]"
        except (TypeError, ValueError):
            pass
    index = media.get("clip_index")
    if index is not None:
        return f"{name} (clip {index})"
    return name


def _drop_summary(n_dropped: int, n_total: int, reasons: Counter[str]) -> str:
    """The toast's detail line: how many were dropped, and why, grouped by reason."""
    lead = f"{n_dropped:,} of {n_total:,} imported item(s) had no vector after the embed step, so they were left out."
    if len(reasons) == 1:
        return f"{lead} {next(iter(reasons))}."
    return " ".join([lead, *(f"{reason} ({count:,} item(s))." for reason, count in reasons.most_common())])


def _drop_none_embeddings_stage(ctx: DatasetContext, tracker, failures: EmbedFailures | None = None) -> None:
    """Drop any media that finished the clipper stage without an embedding.

    ``_fixup_clip_md5_and_embeddings`` is best-effort: when its bulk
    re-embed call fails (no embedder, vector ``None``, exception) the
    clip is left in ``ctx.medias`` with ``embedding=None``.  Letting
    those through poisons every downstream consumer; the matrix builder
    in ``vtscore/embedding/matrix.py`` raises (M11 fix); sort/score
    aggregations get wrong-length lists.  Drop them here so the rest of
    the load pipeline (dedup, coverage atlas, registry) sees a clean
    dict, and surface the count to the progress tracker so the user
    knows N is lower than the importer reported.

    *failures* is what the embed stage recorded about why each item has no
    vector (see :func:`~vtscore.datasets.stages.embedding._embed_missing_stage`).
    The warning toast states those reasons and lists every dropped item by
    name, since a GUI user has no other way to find out which ones (#4232).
    """
    none_ids = [cid for cid, media in ctx.medias.items() if media_embedding(media) is None]
    if not none_ids:
        return

    failures = failures or {}
    reason_of = {cid: failures.get(cid, _UNKNOWN_DROP_REASON) for cid in none_ids}
    reasons = Counter(reason_of.values())
    # Grouped by reason (commonest first) so a mixed failure reads as blocks.
    rank = {reason: i for i, (reason, _) in enumerate(reasons.most_common())}
    ordered = sorted(none_ids, key=lambda cid: rank[reason_of[cid]])
    labels = [_dropped_item_label(cid, ctx.medias[cid]) for cid in ordered]
    if len(reasons) > 1:
        # One reason is already the detail line; several need saying per item.
        labels = [f"{label} — {reason_of[cid]}" for label, cid in zip(labels, ordered)]

    for cid in none_ids:
        del ctx.medias[cid]

    import logging  # noqa: PLC0415

    from vtscore.concurrency.notifications import notify  # noqa: PLC0415

    logging.getLogger(__name__).warning(
        "Dropped %d media item(s) with embedding=None (importer or re-embed step failed)",
        len(none_ids),
    )
    # The tracker message below is overwritten by the next stage within
    # seconds, so a load that silently shrank was indistinguishable from one
    # that didn't (issue #3798).  A warning toast stays until dismissed.
    notify(
        f"Dropped {len(none_ids)} item(s) whose embedding failed",
        level="warning",
        detail=_drop_summary(len(none_ids), len(none_ids) + len(ctx.medias), reasons),
        items=labels,
        source="Dataset import",
    )
    tracker.update(
        "loading",
        f"Dropped {len(none_ids)} item(s) with failed embedding…",
        current=0,
        total=0,
        step=_TOTAL_LOAD_STEPS,
        total_steps=_TOTAL_LOAD_STEPS,
    )
    invalidate_embedding_matrix(ctx)


def _collapse_duplicates_stage(ctx: DatasetContext, tracker) -> None:
    def _progress(current: int, total: int) -> None:
        tracker.check_cancelled()
        tracker.update(
            "loading",
            "Removing duplicates…",
            current=current,
            total=total,
            step=_TOTAL_LOAD_STEPS,
            total_steps=_TOTAL_LOAD_STEPS,
        )

    _progress(0, 0)
    collapse_duplicates(ctx.medias, on_progress=_progress)
    invalidate_embedding_matrix(ctx)


def _collapse_near_duplicates_stage(ctx: DatasetContext, tracker) -> None:
    """Opt-in pass: collapse near-duplicate images/text after exact dedup.

    Gated on the transient ``ctx.merge_near_duplicates`` create-time flag
    (set from the importer modal's "Merge near-duplicates" checkbox).  No-op
    on every reload path, where the flag defaults off and the grouping is
    already baked into origins.
    """
    if not getattr(ctx, "merge_near_duplicates", False):
        return

    def _progress(current: int, total: int) -> None:
        tracker.check_cancelled()
        tracker.update(
            "loading",
            "Merging near-duplicates…",
            current=current,
            total=total,
            step=_TOTAL_LOAD_STEPS,
            total_steps=_TOTAL_LOAD_STEPS,
        )

    _progress(0, 0)
    collapse_near_duplicates(ctx.medias, on_progress=_progress)
    invalidate_embedding_matrix(ctx)


def _build_coverage_atlas_stage(ctx: DatasetContext, tracker) -> None:
    # An upstream step (e.g. a pickle restore) may have already populated the
    # tree; skip the expensive hierarchical k-means rebuild when so.
    if ctx.coverage_atlas is not None:
        return

    # Past the auto-build threshold the tree is deferred: the build would cost
    # minutes/GBs and the autopilot degrades gracefully without it.  The user
    # can trigger it later via POST /api/datasets/registry/<id>/coverage-atlas.
    if not should_auto_build_coverage_atlas(len(ctx.medias)):
        import logging  # noqa: PLC0415

        logging.getLogger(__name__).info(
            "Skipping automatic coverage-atlas build for %d items (> threshold); "
            "build on demand via the coverage-atlas endpoint.",
            len(ctx.medias),
        )
        return

    def _progress(current: int, total: int) -> None:
        tracker.check_cancelled()
        tracker.update(
            "loading",
            "Building diversity index…",
            current=current,
            total=total,
            step=_TOTAL_LOAD_STEPS,
            total_steps=_TOTAL_LOAD_STEPS,
        )

    _progress(0, 0)
    build_coverage_atlas_for_context(ctx, on_progress=_progress)
