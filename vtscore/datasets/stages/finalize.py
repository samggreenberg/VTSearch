"""Finalisation stages: drop failed embeds, collapse duplicates, diversity tree.

These run after embedding to hand the registry/projection stages a clean
``medias`` dict: media that finished without an embedding are dropped,
exact-duplicate media are collapsed, and the diversity index is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vtscore.embedding.matrix import invalidate_embedding_matrix
from vtscore.embedding.media_vectors import media_embedding
from vtscore.state import (
    build_diversity_tree_for_context,
    collapse_duplicates,
    collapse_near_duplicates,
    should_auto_build_diversity_tree,
)

from vtscore.datasets.stages._common import _TOTAL_LOAD_STEPS

if TYPE_CHECKING:
    from vtscore.state import DatasetContext


def _drop_none_embeddings_stage(ctx: DatasetContext, tracker) -> None:
    """Drop any media that finished the clipper stage without an embedding.

    ``_fixup_clip_md5_and_embeddings`` is best-effort: when its bulk
    re-embed call fails (no embedder, vector ``None``, exception) the
    clip is left in ``ctx.medias`` with ``embedding=None``.  Letting
    those through poisons every downstream consumer; the matrix builder
    in ``vtscore/embedding/matrix.py`` raises (M11 fix); sort/score
    aggregations get wrong-length lists.  Drop them here so the rest of
    the load pipeline (dedup, diversity tree, registry) sees a clean
    dict, and surface the count to the progress tracker so the user
    knows N is lower than the importer reported.
    """
    none_ids = [cid for cid, media in ctx.medias.items() if media_embedding(media) is None]
    if not none_ids:
        return

    for cid in none_ids:
        del ctx.medias[cid]

    import logging  # noqa: PLC0415

    logging.getLogger(__name__).warning(
        "Dropped %d media item(s) with embedding=None (importer or re-embed step failed)",
        len(none_ids),
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


def _build_diversity_tree_stage(ctx: DatasetContext, tracker) -> None:
    # An upstream step (e.g. a pickle restore) may have already populated the
    # tree; skip the expensive hierarchical k-means rebuild when so.
    if ctx.diversity_tree is not None:
        return

    # Past the auto-build threshold the tree is deferred: the build would cost
    # minutes/GBs and the autopilot degrades gracefully without it.  The user
    # can trigger it later via POST /api/datasets/registry/<id>/diversity-tree.
    if not should_auto_build_diversity_tree(len(ctx.medias)):
        import logging  # noqa: PLC0415

        logging.getLogger(__name__).info(
            "Skipping automatic diversity-tree build for %d items (> threshold); "
            "build on demand via the diversity-tree endpoint.",
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
    build_diversity_tree_for_context(ctx, on_progress=_progress)
