"""Embed-missing stage: embed media items the importer left unembedded.

Importers emit media dicts and optionally pre-populate ``embedding`` from
``content_vectors`` / ``custom_metadata_map``; anything still at ``None``
after the importer finishes is bulk-embedded here in a single
``embed_media_bulk`` call, with patch-region tensors attached for embedders
that report ``supports_patch_regions``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable

from vtscore.embedding.matrix import invalidate_embedding_matrix
from vtscore.embedding.media_vectors import (
    ensure_embeddings_dict,
    media_embedder_names,
    media_embedding,
    set_media_embedding,
)

from vtscore.datasets.stages._common import _STATUS_TO_STEP, _TOTAL_LOAD_STEPS

if TYPE_CHECKING:
    from vtscore.state import DatasetContext


def _first_media_type(items: Iterable[tuple[int, dict[str, Any]]]) -> str:
    """Return the first non-empty ``media_type`` among *items*, or ``""``."""
    for _, m in items:
        mt = m.get("media_type")
        if mt:
            return mt
    return ""


def _first_stored_embedder(medias: dict[int, dict[str, Any]]) -> str:
    """Return the first non-empty ``embedder`` recorded on a media, or ``""``.

    Already-embedded media (a pickle reload or content-vector importer) carry
    the name of the embedder that produced their vectors.  When no explicit
    embedder was requested for this load, we resolve against that stored name
    rather than the media-type default so a patch-embedded dataset re-binds to
    its patch embedder - the single-vector default would otherwise report
    ``supports_patch_regions=False`` and silently skip the region back-fill.
    """
    for m in medias.values():
        name = m.get("embedder")
        if name:
            return name
    return ""


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

    Multi-embedder note: "missing" and the patch / structural back-fills are
    keyed to *this* embedder's per-media vector (``media["embeddings"][name]``,
    via :func:`media_embedding`).  So a second bound embedder run over an
    already-text-embedded dataset still embeds every item (it has no vector
    under its own key yet) without disturbing the first embedder's vectors.
    """
    # Resolve the media type from any media so the patch-region back-fill
    # below can run even when every image already carries an embedding (e.g. a
    # pre-embedded pickle or content-vector importer bound to a patch
    # embedder).  Homogeneous datasets share one media type.
    media_type = _first_media_type(medias.items())
    if not media_type:
        return

    from vtscore.media import embedders_for_type, get_embedder  # noqa: PLC0415

    emb = None
    if embedder_name:
        try:
            emb = get_embedder(embedder_name)
        except KeyError:
            emb = None
    if emb is None and not embedder_name:
        # No explicit embedder for this load: honour the one the media were
        # already embedded with (set on a pickle reload / content-vector
        # importer) before falling back to the media-type default.  Without
        # this a patch-embedded dataset reloaded with no embedder pick would
        # resolve to the single-vector default and skip the patch-region
        # back-fill below, so the best-match highlight and region voting have
        # no region data to work with.
        stored = _first_stored_embedder(medias)
        if stored:
            try:
                emb = get_embedder(stored)
            except KeyError:
                emb = None
    if emb is None:
        avail = embedders_for_type(media_type)
        emb = avail[0] if avail else None
    if emb is None:
        return

    # Which items still need *this* embedder's vector.
    #
    # When the caller named an embedder explicitly (the bound-set driver and
    # the reload path both do), "missing" is keyed to that embedder's own
    # per-media entry, so a second bound embedder embeds items the first
    # already covered (their singular vector belongs to the first model).
    #
    # When no embedder was named (bare default-resolution call), keep the
    # legacy contract: only items with *no* vector at all are embedded, so a
    # dataset already populated by some other source isn't re-embedded under
    # the resolved default.
    if embedder_name:
        missing = [(mid, m) for mid, m in medias.items() if media_embedding(m, emb.name) is None]
    else:
        missing = [(mid, m) for mid, m in medias.items() if media_embedding(m) is None]

    # Patch-capable embedders attach a per-image HAC region tree + patch grid
    # alongside the CLS ``embedding``.  That side-channel must exist for any
    # image *this* embedder produced but that lacks ``patch_regions`` - not
    # only the images we embed in this call.  Without it the best-match
    # highlight, region voting, and region-aware scoring have no region data,
    # which is exactly what happens to an already-embedded dataset (pickle /
    # content-vector importer) that never ran the patch pass: the Highlight
    # toggle shows (the embedder reports the capability) but draws nothing.
    # Re-deriving from the source file at load keeps ``patch_regions`` an
    # in-memory artifact, consistent with the no-persisted-vectors rule.
    patch_capable = getattr(emb, "supports_patch_regions", False) is True

    def _needs_patch(m: dict[str, Any]) -> bool:
        # This embedder must have produced a CLS vector for the image (so we
        # never re-pool regions for an image it never embedded), but key off
        # its own per-embedder entry rather than the singular mirror: in a
        # multi-embedder dataset the mirror may belong to a *different* model.
        return media_embedding(m, emb.name) is not None and m.get("patch_regions") is None

    # Structural embedders (SIFT/VLAD) attach a per-image keypoint+descriptor set
    # alongside the VLAD ``embedding``, on the same back-fill terms as patch
    # regions: any image this embedder produced that lacks ``local_features``
    # (e.g. a pre-embedded pickle reload) is re-derived from its source file at
    # load, keeping local features an in-memory artifact.
    structural_capable = getattr(emb, "supports_geometric_verification", False) is True

    def _needs_local_features(m: dict[str, Any]) -> bool:
        return media_embedding(m, emb.name) is not None and m.get("local_features") is None

    has_patch_backfill = patch_capable and any(_needs_patch(m) for m in medias.values())
    has_structural_backfill = structural_capable and any(_needs_local_features(m) for m in medias.values())

    if not missing and not has_patch_backfill and not has_structural_backfill:
        return

    if on_progress is None:

        def _noop_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
            return None

        on_progress = _noop_progress

    if getattr(emb, "_model", None) is None:
        on_progress("loading", "Loading embedding model…", 0, 0)
        emb.load_models()

    original_cb = emb._on_progress

    if missing:
        total = len(missing)
        on_progress("embedding", f"Embedding {total} item(s)…", 0, total)

        inputs = [m for _, m in missing]
        emb._on_progress = on_progress
        try:
            vectors = emb.embed_media_bulk(inputs)
        except Exception:
            logging.getLogger(__name__).exception("Bulk embed failed for media_type=%s (%d items)", media_type, total)
            vectors = None
        finally:
            emb._on_progress = original_cb

        if vectors is not None:
            embedder_id = emb.name
            for (mid, _), vec in zip(missing, vectors):
                if vec is None:
                    continue
                media = medias.get(mid)
                if media is None:
                    continue
                set_media_embedding(media, embedder_id, vec)

    # Patch-region pass for embedders that support it (DINOv2/v3/EUPE).  Runs
    # over every patch-capable image still lacking a region tree, including
    # ones that arrived already-embedded - not just the items embedded above.
    if patch_capable:
        patch_inputs = [m for m in medias.values() if _needs_patch(m)]
        if patch_inputs:
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

    # Local-features pass for structural embedders (SIFT/VLAD).  Same back-fill
    # shape as the patch pass: detect+describe every structural image still
    # missing ``local_features`` and store the compact (fp16/uint8) form.
    if structural_capable:
        structural_inputs = [m for m in medias.values() if _needs_local_features(m)]
        if structural_inputs:
            emb._on_progress = on_progress
            try:
                features = emb.local_features_forward_bulk(structural_inputs)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Bulk local-feature detection failed for media_type=%s (%d items)",
                    media_type,
                    len(structural_inputs),
                )
                features = [None] * len(structural_inputs)
            finally:
                emb._on_progress = original_cb

            for media, feats in zip(structural_inputs, features):
                if feats is None:
                    continue
                media["local_features"] = feats.compact()

    # Re-key any legacy single-vector media into the dict-keyed representation
    # and drop the singular media["embedding"] mirror (Phase 2c): afterward
    # media["embeddings"] is the sole per-media vector store.
    for media in medias.values():
        ensure_embeddings_dict(media)


def _attach_patch_regions_to_media(media: dict, patch_out) -> None:
    """Attach HAC patch-region tree to *media* (mirrors ``loader_folder._attach_patch_regions``)."""
    import numpy as np  # noqa: PLC0415

    from vtscore.media.patch_embed import build_region_tree, to_fp16  # noqa: PLC0415

    regions = build_region_tree(patch_out, k=12, alpha=0.5)
    media["patch_regions"] = to_fp16(regions)
    media["patch_grid"] = patch_out.patch_grid.astype(np.float16, copy=False)


def _ordered_load_embedders(medias: dict[int, dict[str, Any]], requested: list[str]) -> list[str]:
    """Resolve the ordered set of embedders to run over *medias* at load.

    The *requested* embedders (the create-time picks, if any) lead in order,
    followed by any embedders already present on the medias that they don't
    cover — the reload case, where a v3 pickle restores one vector per bound
    embedder under ``media["embeddings"]`` and each must have its in-memory
    patch / structural side-channels re-derived.

    *requested* is the create-time embedder list (v3 trio: text / patch /
    structural picks).  A single-embedder create passes a one-element list; a
    bare ``[""]`` (or empty list) with no embedders present on the medias falls
    back to ``[""]``, which lets :func:`embed_missing` resolve the media-type
    default — the single-embedder create path, unchanged.
    """
    names: list[str] = []
    for r in requested:
        if r and r not in names:
            names.append(r)
    for m in medias.values():
        present = media_embedder_names(m)
        if present:
            for name in present:
                if name not in names:
                    names.append(name)
            break
    return names or [""]


def _embed_missing_stage(
    ctx: DatasetContext,
    tracker,
    requested_embedders: list[str],
) -> None:
    """Run every bound embedder over the context's medias (tracker-routed progress).

    Single-embedder datasets resolve to one name and behave exactly as before;
    a v3 trio (text + patch + structural picks) runs each in turn, so
    ``media["embeddings"]`` carries a per-embedder vector, the patch embedder
    also populates ``patch_regions`` / ``patch_grid``, and the structural
    embedder populates ``local_features``.
    """

    def _emb_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        tracker.check_cancelled()
        step = _STATUS_TO_STEP.get(status, _STATUS_TO_STEP["embedding"])
        tracker.update(status, message, current, total, step=step, total_steps=_TOTAL_LOAD_STEPS)

    for name in _ordered_load_embedders(ctx.medias, requested_embedders):
        embed_missing(ctx.medias, name, on_progress=_emb_progress)
    invalidate_embedding_matrix(ctx)
