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

from vtscore.concurrency.progress import noop_progress
from vtscore.embedding.binding import expected_dim_for_embedder
from vtscore.embedding.matrix import invalidate_embedding_matrix
from vtscore.embedding.media_vectors import (
    EMBEDDINGS_KEY,
    UNKNOWN_EMBEDDER_KEY,
    ensure_embeddings_dict,
    media_embedder_names,
    media_embedding,
    set_media_embedding,
)
from vtscore.embedding.precomputed import require_dim

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


def _resolve_embedder(
    medias: dict[int, dict[str, Any]],
    embedder_name: str,
    media_type: str,
):
    """Resolve the embedder for this load: explicit pick → stored → default.

    Returns the resolved embedder, or ``None`` when no embedder is available.

    An explicit *embedder_name* is looked up directly; if it resolves, that
    wins.  Otherwise (only when no explicit name was given) we honour the
    embedder the media were already embedded with (a pickle reload /
    content-vector importer records it on the media) so a patch-embedded
    dataset re-binds to its patch embedder rather than the single-vector
    media-type default — the default would report
    ``supports_patch_regions=False`` and silently skip the region back-fill.
    Failing both, fall back to the first embedder registered for *media_type*.
    """
    from vtscore.media import embedders_for_type, get_embedder  # noqa: PLC0415

    emb = None
    if embedder_name:
        try:
            emb = get_embedder(embedder_name)
        except KeyError:
            emb = None
    if emb is None and not embedder_name:
        stored = _first_stored_embedder(medias)
        if stored:
            try:
                emb = get_embedder(stored)
            except KeyError:
                emb = None
    if emb is None:
        avail = embedders_for_type(media_type)
        emb = avail[0] if avail else None
    return emb


def _stamp_requested_embedder(medias: dict[int, dict[str, Any]], embedder_name: str) -> None:
    """Stamp the requested *embedder_name* onto pre-embedded media whose name is blank.

    An npz/sidecar importer that ships a pre-computed vector but not its
    producing embedder stores it under the blank sentinel key
    (:data:`UNKNOWN_EMBEDDER_KEY`) with no recorded ``media["embedder"]``.  When
    the caller named an embedder for this load, that nameless vector *is* that
    embedder's vector — the archive just didn't carry the name — so re-key it
    under *embedder_name* and record the primary.  This lets
    ``media_embedding(m, embedder_name)`` resolve, so the named-missing check
    below won't needlessly re-embed the media and downstream binding won't fall
    back to the media-type default (a dimension mismatch when the pick differs).

    Only media whose embedder name is blank are touched; an importer-set name is
    never overwritten.  A no-op when *embedder_name* is blank (no pick to stamp).

    **The stamp is checked, not assumed.**  Re-keying is an assertion that the
    nameless vector belongs to *embedder_name*'s space, and a manifest whose
    vectors came from a different model would make that assertion false - after
    which the media is indistinguishable from a correctly-labelled one, and the
    mismatch only surfaces as a broadcast error deep in the matrix builder.  So
    when the embedder declares a width, a vector that disagrees with it is
    rejected here, naming both widths.
    """
    if not embedder_name:
        return
    expected_dim = expected_dim_for_embedder(embedder_name)
    for mid, m in medias.items():
        if m.get("embedder"):
            continue
        embs = m.get(EMBEDDINGS_KEY)
        if not isinstance(embs, dict) or UNKNOWN_EMBEDDER_KEY not in embs:
            continue
        vec = embs[UNKNOWN_EMBEDDER_KEY]
        if expected_dim is not None:
            require_dim(
                vec,
                expected_dim,
                label=f"pre-computed vector for media {mid} ({m.get('origin_name') or m.get('filename') or '?'})",
                expected_source=f"the width declared by embedder {embedder_name!r}",
            )
        embs.pop(UNKNOWN_EMBEDDER_KEY)
        embs.setdefault(embedder_name, vec)
        m["embedder"] = embedder_name


def _ensure_model_loaded(emb, on_progress: Callable[[str, str, int, int], None]) -> None:
    """Load the embedder's models if not already loaded, announcing progress.

    The load runs inside :meth:`~vtscore.media.embedder.MediaEmbedder.progress_scope`
    so the model's own "Loading … processor…" ticks land on *this* load's
    tracker, next to every other phase of the import.  Unscoped they fall
    through to the embedder's process-wide default sink, which resolves
    per-thread — so on a worker that bound no tracker of its own they would be
    dropped entirely rather than surfacing next to the import they belong to.
    """
    if getattr(emb, "_model", None) is None:
        on_progress("loading", "Loading embedding model…", 0, 0)
        with emb.progress_scope(on_progress):
            emb.load_models()


def _missing_for_embedder(
    medias: dict[int, dict[str, Any]],
    emb,
    embedder_name: str,
) -> list[tuple[int, dict[str, Any]]]:
    """Return the ``(mid, media)`` items still needing *this* embedder's vector.

    When the caller named an embedder explicitly (the bound-set driver and the
    reload path both do), "missing" is keyed to that embedder's own per-media
    entry, so a second bound embedder embeds items the first already covered
    (their singular vector belongs to the first model).

    When no embedder was named (bare default-resolution call), keep the legacy
    contract: only items with *no* vector at all are embedded, so a dataset
    already populated by some other source isn't re-embedded under the resolved
    default.
    """
    if embedder_name:
        return [(mid, m) for mid, m in medias.items() if media_embedding(m, emb.name) is None]
    return [(mid, m) for mid, m in medias.items() if media_embedding(m) is None]


def _needs_side_channel(m: dict[str, Any], embedder_name: str, key: str) -> bool:
    """Whether *m* was embedded by *embedder_name* but lacks side-channel *key*.

    The embedder must have produced a CLS/VLAD vector for the image (so we
    never re-derive a side channel for an image it never embedded), keyed off
    its own per-embedder entry rather than the singular mirror: in a
    multi-embedder dataset the mirror may belong to a *different* model.
    """
    return media_embedding(m, embedder_name) is not None and m.get(key) is None


def _run_embed_pass(
    emb,
    medias: dict[int, dict[str, Any]],
    media_type: str,
    missing: list[tuple[int, dict[str, Any]]],
    on_progress: Callable[[str, str, int, int], None],
) -> None:
    """Bulk-embed the *missing* items and attach each non-``None`` vector.

    Announces progress, routes the embedder's callback through *on_progress*
    for the duration of the call (via the thread-scoped
    :meth:`~vtscore.media.embedder.MediaEmbedder.progress_scope`, so a
    concurrent load on this singleton embedder keeps its own tracker), and on a
    bulk-embed failure logs and attaches nothing (items stay at ``None`` for the
    drop-none stage).  Items whose media vanished from *medias* during the call
    are skipped.
    """
    if not missing:
        return
    total = len(missing)
    on_progress("embedding", f"Embedding {total} item(s)…", 0, total)

    inputs = [m for _, m in missing]
    try:
        with emb.progress_scope(on_progress):
            vectors = emb.embed_media_bulk(inputs)
    except Exception:
        logging.getLogger(__name__).exception("Bulk embed failed for media_type=%s (%d items)", media_type, total)
        vectors = None

    if vectors is None:
        return
    embedder_id = emb.name
    for (mid, _), vec in zip(missing, vectors):
        if vec is None:
            continue
        media = medias.get(mid)
        if media is None:
            continue
        set_media_embedding(media, embedder_id, vec)


def _run_backfill_pass(
    emb,
    medias: dict[int, dict[str, Any]],
    media_type: str,
    on_progress: Callable[[str, str, int, int], None],
    *,
    needs: Callable[[dict[str, Any]], bool],
    forward: Callable[[list[dict[str, Any]]], list],
    attach: Callable[[dict[str, Any], Any], None],
    fail_message: str,
) -> None:
    """Run one side-channel back-fill pass over every media still needing it.

    Filters *medias* by *needs*, bulk-forwards them through *forward* (routing
    progress through *on_progress* for the duration of the call, scoped to this
    thread so a concurrent load keeps its own tracker), and attaches each
    non-``None`` output via *attach*.  On a *forward* failure the pass logs
    *fail_message* and attaches nothing (every output is treated as ``None``).
    Runs over every matching media, including ones that arrived
    already-embedded — not only the items embedded in this load.
    """
    inputs = [m for m in medias.values() if needs(m)]
    if not inputs:
        return
    try:
        with emb.progress_scope(on_progress):
            outputs = forward(inputs)
    except Exception:
        logging.getLogger(__name__).exception(fail_message, media_type, len(inputs))
        outputs = [None] * len(inputs)

    for media, out in zip(inputs, outputs):
        if out is None:
            continue
        attach(media, out)


def embed_missing(
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

    emb = _resolve_embedder(medias, embedder_name, media_type)
    if emb is None:
        return

    # A pre-embedded media whose producing embedder the archive didn't record
    # (npz/sidecar import → vector under the blank sentinel key) carries the
    # caller's named embedder when one was given: stamp that name so the vector
    # resolves under it rather than being re-embedded or leaving downstream
    # binding to fall back to the media-type default (dimension mismatch).
    _stamp_requested_embedder(medias, embedder_name)

    # Which items still need *this* embedder's vector.
    missing = _missing_for_embedder(medias, emb, embedder_name)

    # Patch-capable embedders attach the raw per-image patch grid alongside
    # the CLS ``embedding``.  That side-channel must exist for any image *this*
    # embedder produced but that lacks ``patch_grid`` - not only the images we
    # embed in this call.  Without it the best-match highlight, region voting,
    # and patch-aware scoring have no patch data, which is exactly what happens
    # to an already-embedded dataset (pickle / content-vector importer) that
    # never ran the patch pass: the Highlight toggle shows (the embedder reports
    # the capability) but draws nothing.  Re-deriving from the source file at
    # load keeps ``patch_grid`` an in-memory artifact, consistent with the
    # no-persisted-vectors rule.
    patch_capable = getattr(emb, "supports_patch_regions", False) is True

    def _needs_patch(m: dict[str, Any]) -> bool:
        return _needs_side_channel(m, emb.name, "patch_grid")

    # Structural embedders (SIFT/VLAD) attach a per-image keypoint+descriptor set
    # alongside the VLAD ``embedding``, on the same back-fill terms as patch
    # regions: any image this embedder produced that lacks ``local_features``
    # (e.g. a pre-embedded pickle reload) is re-derived from its source file at
    # load, keeping local features an in-memory artifact.
    structural_capable = getattr(emb, "supports_geometric_verification", False) is True

    def _needs_local_features(m: dict[str, Any]) -> bool:
        return _needs_side_channel(m, emb.name, "local_features")

    has_patch_backfill = patch_capable and any(_needs_patch(m) for m in medias.values())
    has_structural_backfill = structural_capable and any(_needs_local_features(m) for m in medias.values())

    if not missing and not has_patch_backfill and not has_structural_backfill:
        return

    if on_progress is None:
        on_progress = noop_progress

    _ensure_model_loaded(emb, on_progress)

    _run_embed_pass(emb, medias, media_type, missing, on_progress)

    # Patch-grid pass for embedders that support it (DINOv2/v3/EUPE).  Runs
    # over every patch-capable image still lacking a grid, including ones that
    # arrived already-embedded - not just the items embedded above.
    if patch_capable:
        _run_backfill_pass(
            emb,
            medias,
            media_type,
            on_progress,
            needs=_needs_patch,
            forward=emb.patch_forward_bulk,
            attach=_attach_patch_grid_to_media,
            fail_message="Bulk patch-forward failed for media_type=%s (%d items)",
        )

    # Local-features pass for structural embedders (SIFT/VLAD).  Same back-fill
    # shape as the patch pass: detect+describe every structural image still
    # missing ``local_features`` and store the compact (fp16/uint8) form.
    if structural_capable:
        _run_backfill_pass(
            emb,
            medias,
            media_type,
            on_progress,
            needs=_needs_local_features,
            forward=emb.local_features_forward_bulk,
            attach=lambda media, feats: media.__setitem__("local_features", feats.compact()),
            fail_message="Bulk local-feature detection failed for media_type=%s (%d items)",
        )

    # Re-key any legacy single-vector media into the dict-keyed representation
    # and drop the singular media["embedding"] mirror (Phase 2c): afterward
    # media["embeddings"] is the sole per-media vector store.
    for media in medias.values():
        ensure_embeddings_dict(media)


def _attach_patch_grid_to_media(media: dict, patch_out) -> None:
    """Attach the raw ``(H, W, D)`` patch grid to *media*, float16.

    That is the *whole* patch side-channel now.  Ingest used to also build a
    24-node HAC region tree per image here (``build_region_tree(patch_out,
    k=12, alpha=0.5)``); #2886 dropped it after the Max-Patch study found raw
    patches beat every tree variant at the operating point, so the payload gets
    strictly **smaller** - the grid was already being stored alongside the tree
    - and ingest sheds the tree's ``O(k^3)`` agglomerative merge.  The
    per-patch saliency ``patch_out`` also carries is not stored: nothing
    downstream reads it now that leaf pooling is gone.
    """
    import numpy as np  # noqa: PLC0415

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


class EmbedLoopProgress:
    """Tracker proxy that spreads a multi-embedder ingest loop across the embed step.

    :func:`_embed_missing_stage` runs each bound embedder (v3 trio: text / patch
    / structural picks) in turn, and every :func:`embed_missing` call reports its
    own ``current``/``total`` starting from 0.  Forwarded straight to the tracker
    those restart the within-step fraction each embedder, so the first embedder
    fills the embed slice and the tracker's monotonic clamp then pins the unified
    bar for the 2nd/3rd embedders — no backslide, but a long static stretch.

    This proxy gives each embedder an equal, ordered sub-range of the embed step
    (step 3) and rewrites the within-embedder fraction into the active range
    before forwarding, so the bar fills once, cumulatively, across the whole loop
    instead of per embedder.  An embedder that does no work (e.g. a reload whose
    side-channels are already present) simply emits nothing and leaves its slice
    unfilled; the next embedder jumps the bar forward, which is monotonic.

    The **first** embedder's model-load keeps the dedicated model-load step
    (step 2) so its weighted slice of the bar still fills; **later** embedders'
    model loads fold into the embed step at their sub-range floor, because a step
    number going *backwards* (3 → 2) reads as a brand-new job and would reset the
    overall clock (see :meth:`ProgressTracker._compute_overall`).

    Single-embedder loads (``n_embedders <= 1``) forward every update unchanged,
    so their pacing — download / model-load / embed / finalize — is untouched.
    """

    #: Resolution of the synthetic within-step counter forwarded to the real
    #: tracker (the embed fraction 0..1 is reported as ``current`` out of this).
    _SCALE = 1000

    def __init__(self, tracker, n_embedders: int) -> None:
        self._tracker = tracker
        self._n = max(int(n_embedders), 1)
        self._idx = 0

    def begin(self, idx: int) -> None:
        """Activate embedder *idx*; subsequent calls map into its sub-range."""
        self._idx = idx

    def check_cancelled(self) -> None:
        self._tracker.check_cancelled()

    def __call__(self, status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        self._tracker.check_cancelled()
        if self._n <= 1:
            step = _STATUS_TO_STEP.get(status, _STATUS_TO_STEP["embedding"])
            self._tracker.update(status, message, current, total, step=step, total_steps=_TOTAL_LOAD_STEPS)
            return
        # First embedder's model-load keeps the model-load step so its bar slice
        # fills; everything else folds into the embed step's cumulative range.
        if status == "loading" and self._idx == 0:
            self._tracker.update(
                status, message, current, total, step=_STATUS_TO_STEP["loading"], total_steps=_TOTAL_LOAD_STEPS
            )
            return
        within = current / total if total and total > 0 else 0.0
        within = min(max(within, 0.0), 1.0)
        frac = (self._idx + within) / self._n
        self._tracker.update(
            status,
            message,
            int(frac * self._SCALE),
            self._SCALE,
            step=_STATUS_TO_STEP["embedding"],
            total_steps=_TOTAL_LOAD_STEPS,
        )


def _embed_missing_stage(
    ctx: DatasetContext,
    tracker,
    requested_embedders: list[str],
) -> None:
    """Run every bound embedder over the context's medias (tracker-routed progress).

    Single-embedder datasets resolve to one name and behave exactly as before;
    a v3 trio (text + patch + structural picks) runs each in turn, so
    ``media["embeddings"]`` carries a per-embedder vector, the patch embedder
    also populates ``patch_grid``, and the structural
    embedder populates ``local_features``.

    Progress is routed through :class:`EmbedLoopProgress` so a multi-embedder
    loop reports cumulative progress across the embed step rather than restarting
    the bar at 0 for each embedder.
    """
    names = _ordered_load_embedders(ctx.medias, requested_embedders)
    progress = EmbedLoopProgress(tracker, len(names))
    for idx, name in enumerate(names):
        progress.begin(idx)
        embed_missing(ctx.medias, name, on_progress=progress)
    invalidate_embedding_matrix(ctx)
