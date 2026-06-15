"""Finalisation stages: drop failed embeds, collapse duplicates, diversity tree.

These run after embedding to hand the registry/projection stages a clean
``medias`` dict: media that finished without an embedding are dropped,
exact-duplicate media are collapsed, and the diversity index is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vtscore.embedding.matrix import invalidate_embedding_matrix
from vtscore.state import build_diversity_tree_for_context, collapse_duplicates

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
    none_ids = [cid for cid, media in ctx.medias.items() if media.get("embedding") is None]
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


def _build_diversity_tree_stage(ctx: DatasetContext, tracker) -> None:
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
