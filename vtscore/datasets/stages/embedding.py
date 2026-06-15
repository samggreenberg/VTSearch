"""Embed-missing stage: embed media items the importer left unembedded.

Importers emit media dicts and optionally pre-populate ``embedding`` from
``content_vectors`` / ``custom_metadata_map``; anything still at ``None``
after the importer finishes is bulk-embedded here in a single
``embed_media_bulk`` call, with patch-region tensors attached for embedders
that report ``supports_patch_regions``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from vtscore.embedding.matrix import invalidate_embedding_matrix

from vtscore.datasets.stages._common import _STATUS_TO_STEP, _TOTAL_LOAD_STEPS


def embed_missing(  # noqa: C901
    medias: dict[int, dict[str, Any]],
    embedder_name: str = "",
    on_progress: Callable[[str, str, int, int], None] | None = None,
) -> None:
    """Embed media items in *medias* that don't already have an embedding.

    Importers are not responsible for calling the embedder; they emit
    media dicts and optionally pre-populate ``embedding`` from
    ``content_vectors`` / ``custom_metadata_map``.  Items still at
    ``None`` after the importer finishes go through this function, which
    resolves a single embedder (the user's pick when given, otherwise the
    default for the media type of the unembedded items) and bulk-embeds
    them in one ``embed_media_bulk`` call.

    Items whose bulk-embed call returns ``None`` stay at ``None``; the
    load pipeline drops them via :func:`_drop_none_embeddings_stage`.
    Patch-region tensors are also attached here for embedders that
    report ``supports_patch_regions``.
    """
    missing = [(mid, m) for mid, m in medias.items() if m.get("embedding") is None]
    if not missing:
        return

    media_type = ""
    for _, m in missing:
        mt = m.get("media_type")
        if mt:
            media_type = mt
            break
    if not media_type:
        return

    from vtscore.media import embedders_for_type, get_embedder  # noqa: PLC0415

    emb = None
    if embedder_name:
        try:
            emb = get_embedder(embedder_name)
        except KeyError:
            emb = None
    if emb is None:
        avail = embedders_for_type(media_type)
        emb = avail[0] if avail else None
    if emb is None:
        return

    if on_progress is None:

        def _noop_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
            return None

        on_progress = _noop_progress

    if getattr(emb, "_model", None) is None:
        on_progress("loading", "Loading embedding model…", 0, 0)
        emb.load_models()

    total = len(missing)
    on_progress("embedding", f"Embedding {total} item(s)…", 0, total)

    inputs = [m for _, m in missing]
    original_cb = emb._on_progress
    emb._on_progress = on_progress
    try:
        vectors = emb.embed_media_bulk(inputs)
    except Exception:
        logging.getLogger(__name__).exception("Bulk embed failed for media_type=%s (%d items)", media_type, total)
        return
    finally:
        emb._on_progress = original_cb

    embedder_id = emb.name
    for (mid, _), vec in zip(missing, vectors):
        if vec is None:
            continue
        media = medias.get(mid)
        if media is None:
            continue
        media["embedding"] = vec
        if not media.get("embedder"):
            media["embedder"] = embedder_id

    # Patch-region pass for embedders that support it (DINOv2/v3/EUPE).
    if getattr(emb, "supports_patch_regions", False) is not True:
        return

    patch_inputs = [m for _, m in missing if m.get("patch_regions") is None]
    if not patch_inputs:
        return

    emb._on_progress = on_progress
    try:
        outputs = emb.patch_forward_bulk(patch_inputs)
    except Exception:
        logging.getLogger(__name__).exception(
            "Bulk patch-forward failed for media_type=%s (%d items)", media_type, len(patch_inputs)
        )
        outputs = [None] * len(patch_inputs)
    finally:
        emb._on_progress = original_cb

    for media, patch_out in zip(patch_inputs, outputs):
        if patch_out is None:
            continue
        _attach_patch_regions_to_media(media, patch_out)


def _attach_patch_regions_to_media(media: dict, patch_out) -> None:
    """Attach HAC patch-region tree to *media* (mirrors ``loader_folder._attach_patch_regions``)."""
    import numpy as np  # noqa: PLC0415

    from vtscore.media.patch_embed import build_region_tree, to_fp16  # noqa: PLC0415

    regions = build_region_tree(patch_out, k=12, alpha=0.5)
    media["patch_regions"] = to_fp16(regions)
    media["patch_grid"] = patch_out.patch_grid.astype(np.float16, copy=False)


def _embed_missing_stage(
    ctx,
    tracker,
    embedder_name: str,
) -> None:
    """Pipeline wrapper around :func:`embed_missing` with tracker-routed progress."""

    def _emb_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        tracker.check_cancelled()
        step = _STATUS_TO_STEP.get(status, _STATUS_TO_STEP["embedding"])
        tracker.update(status, message, current, total, step=step, total_steps=_TOTAL_LOAD_STEPS)

    embed_missing(ctx.medias, embedder_name, on_progress=_emb_progress)
    invalidate_embedding_matrix(ctx)
