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
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

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

#: Why each media left the embed stage without a vector, keyed by media id.
#: :func:`embed_missing` fills it in when handed one; the finalize stage's drop
#: toast reads it, because that toast is the only place a GUI user learns what
#: happened - the embed stage is the only place that knows (issue #4232).
EmbedFailures = dict[int, str]

#: Longest exception message quoted in a failure reason.  The reason ends up in
#: a toast, where a stack-trace-sized ``str(exc)`` would bury everything else.
_MAX_ERROR_CHARS = 200


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


def _would_leave_mixed(medias: dict[int, dict[str, Any]], embedder_name: str) -> bool:
    """Whether leaving nameless vectors un-stamped would split *medias* by name.

    The no-pick load path (the CLI never names an embedder; a GUI import need
    not) resolves *embedder_name* itself and embeds every item with no vector
    under it.  A partially pre-embedded import - an importer that shipped
    ``content_vectors`` for some files and left the rest to the framework -
    then holds two kinds of media: the framework-embedded ones keyed under
    *embedder_name*, and the shipped ones under the blank sentinel with no
    recorded ``embedder``.  One space in fact, two by name; and the first
    request for the routed embedder's matrix (Browse, Train, text sort) raises
    ``has no embedding for embedder`` on every shipped media (issue #3798).

    ``True`` when *medias* holds at least one nameless vector **and** at least
    one media that is (or is about to be) keyed under *embedder_name*: an item
    with no vector at all, which this load will embed, or one already carrying
    *embedder_name*'s vector (the same dataset reloaded from a pickle written
    in the split shape).  ``False`` for a dataset that is nameless throughout
    with nothing to embed - a manifest of vectors from a model this process
    does not know is a supported import, and stamping the media-type default
    onto it would assert a space nobody asked for.
    """
    has_nameless = False
    has_named_or_missing = False
    for m in medias.values():
        embs = m.get(EMBEDDINGS_KEY)
        if not m.get("embedder") and isinstance(embs, dict) and UNKNOWN_EMBEDDER_KEY in embs:
            has_nameless = True
        elif media_embedding(m) is None or media_embedding(m, embedder_name) is not None:
            has_named_or_missing = True
        if has_nameless and has_named_or_missing:
            return True
    return False


def _stamp_load_embedder(medias: dict[int, dict[str, Any]], requested: str, resolved: str) -> None:
    """Stamp the load's embedder onto nameless pre-computed vectors when it applies.

    A pre-embedded media whose producing embedder the archive didn't record
    (npz/sidecar/``content_vectors`` import → vector under the blank sentinel
    key) is stamped with *resolved* - the embedder this load runs - so the
    vector resolves under that name rather than being re-embedded or leaving
    the dataset split by name.  Always when the caller *requested* a pick (the
    pre-existing contract; *resolved* is that pick whenever it exists, else
    the default it fell back to, which is the embedder that embeds the rest);
    when no pick was named, only if leaving the sentinel in place would split
    the dataset (issue #3798, :func:`_would_leave_mixed`).
    """
    if requested or _would_leave_mixed(medias, resolved):
        _stamp_requested_embedder(medias, resolved)


def _embedder_label(emb) -> str:
    """The embedder's friendly name for a user-facing reason, e.g. ``"SigLIP"``.

    Falls back to the registry slug when an embedder has no usable
    ``display_name`` (the base class defaults it to the slug anyway).
    """
    label = getattr(emb, "display_name", None)
    return label if isinstance(label, str) and label else str(emb.name)


def _record_failures(failures: EmbedFailures | None, mids: Iterable[int], reason: str) -> None:
    """Record *reason* for every media in *mids* that has none recorded yet.

    First reason wins: in a multi-embedder load an item that every bound
    embedder declined is described by the first, which is the one whose
    vector would have been its primary.
    """
    if failures is None:
        return
    for mid in mids:
        failures.setdefault(mid, reason)


def _warn_no_embedder(
    medias: dict[int, dict[str, Any]],
    media_type: str,
    failures: EmbedFailures | None = None,
) -> None:
    """Log that no embedder resolved for *media_type* and how many items it costs.

    Returning silently here is how an import "silently" shrinks: nothing
    embeds, and the finalize stage drops every vector-less item with one
    generic line that names neither the media type nor the reason.
    """
    unembedded = [mid for mid, m in medias.items() if media_embedding(m) is None]
    logging.getLogger(__name__).warning(
        "No embedder is registered for media_type=%r; %d item(s) left unembedded "
        "(they will be dropped at the end of the load)",
        media_type,
        len(unembedded),
    )
    _record_failures(failures, unembedded, f"No embedder is installed for {media_type} media")


def _stamp_requested_embedder(medias: dict[int, dict[str, Any]], embedder_name: str) -> None:
    """Stamp *embedder_name* onto pre-embedded media whose embedder name is blank.

    An npz/sidecar/``content_vectors`` importer that ships a pre-computed
    vector but not its producing embedder stores it under the blank sentinel
    key (:data:`UNKNOWN_EMBEDDER_KEY`) with no recorded ``media["embedder"]``.
    When the load resolved an embedder - the caller's pick, or the one
    :func:`embed_missing` resolved for a load that would otherwise be split by
    name (see :func:`_would_leave_mixed`) - that nameless vector *is* that
    embedder's vector, the archive just didn't carry the name, so re-key it
    under *embedder_name* and record the primary.  This lets
    ``media_embedding(m, embedder_name)`` resolve, so the named-missing check
    below won't needlessly re-embed the media and downstream binding won't fall
    back to the media-type default (a dimension mismatch when the pick differs).

    Only media whose embedder name is blank are touched; an importer-set name is
    never overwritten.  A no-op when *embedder_name* is blank (nothing to stamp).

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
    failures: EmbedFailures | None = None,
) -> None:
    """Bulk-embed the *missing* items and attach each non-``None`` vector.

    Announces progress, routes the embedder's callback through *on_progress*
    for the duration of the call (via the thread-scoped
    :meth:`~vtscore.media.embedder.MediaEmbedder.progress_scope`, so a
    concurrent load on this singleton embedder keeps its own tracker), and on a
    bulk-embed failure logs and attaches nothing (items stay at ``None`` for the
    drop-none stage).  Items whose media vanished from *medias* during the call
    are skipped.  Every item left without a vector gets a reason in *failures*.
    """
    if not missing:
        return
    total = len(missing)
    on_progress("embedding", f"Embedding {total} item(s)…", 0, total)
    who = f"The {_embedder_label(emb)} embedder"
    missing_ids = [mid for mid, _ in missing]

    inputs = [m for _, m in missing]
    try:
        with emb.progress_scope(on_progress):
            vectors = emb.embed_media_bulk(inputs)
    except Exception as exc:
        logging.getLogger(__name__).exception("Bulk embed failed for media_type=%s (%d items)", media_type, total)
        error = str(exc).strip()
        if len(error) > _MAX_ERROR_CHARS:
            error = error[: _MAX_ERROR_CHARS - 1].rstrip() + "…"
        _record_failures(
            failures,
            missing_ids,
            f"{who} failed on the whole batch ({type(exc).__name__}{': ' + error if error else ''})",
        )
        return

    if vectors is None:
        _record_failures(failures, missing_ids, f"{who} returned nothing for the whole batch")
        return
    embedder_id = emb.name
    # A wrong-length answer cannot be paired with its inputs: ``zip`` would
    # silently truncate and attach vectors to the wrong media, so attach none
    # and say so.  The contract is one entry per input, ``None`` for a failure.
    if len(vectors) != total:
        logging.getLogger(__name__).warning(
            "Embedder %r returned %d vector(s) for %d item(s) (media_type=%s); attaching none. "
            "embed_media_bulk must return one entry per input media, None where an item could not be embedded.",
            embedder_id,
            len(vectors),
            total,
            media_type,
        )
        _record_failures(
            failures,
            missing_ids,
            f"{who} returned {len(vectors)} vector(s) for {total} item(s), so none could be matched to its item",
        )
        return
    n_failed = 0
    declined = f"{who} returned no vector (the item may be unreadable, empty, or in a format it cannot decode)"
    for (mid, _), vec in zip(missing, vectors):
        if vec is None:
            n_failed += 1
            _record_failures(failures, (mid,), declined)
            continue
        media = medias.get(mid)
        if media is None:
            continue
        set_media_embedding(media, embedder_id, vec)
    if n_failed:
        # Items the embedder could not embed stay at ``None`` and are dropped
        # by the finalize stage; that drop is announced there, but this is the
        # only place that knows *which embedder* declined and how often.
        logging.getLogger(__name__).warning(
            "Embedder %r produced no vector for %d of %d item(s) (media_type=%s); "
            "they will be dropped at the end of the load",
            embedder_id,
            n_failed,
            total,
            media_type,
        )


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


class _SideChannel(NamedTuple):
    """One per-media side channel an embedder attaches beside its vector, and how to back-fill it."""

    needs: Callable[[dict[str, Any]], bool]
    forward: Callable[[list[dict[str, Any]]], list]
    attach: Callable[[dict[str, Any], Any], None]
    fail_message: str


def _side_channel_passes(emb) -> list[_SideChannel]:
    """The side channels *emb* attaches, in the order they must be derived.

    Every one is an in-memory artifact, re-derived at load for any media this
    embedder produced but that lacks it (e.g. a pickle reload), consistent with
    the no-persisted-vectors rule:

    * **patch grid** (``patch_grid``): patch-capable embedders (DINOv2/v3/EUPE).
      Without it the best-match highlight, region voting and patch-aware scoring
      have no patch data, which is what an already-embedded dataset that never
      ran the patch pass would otherwise get;
    * **local features** (``local_features``): structural embedders (SIFT/VLAD),
      the keypoints Stage 2 verifies, stored compact (fp16/uint8);
    * **tile vectors** (``tile_vectors``): document structural embedders, the
      tiled Stage 1 (#3928), derived from the local features and so after them.
      A key of their own, not ``patch_grid``: the tiles overlap, where
      ``patch_grid`` is a regular grid of the dataset's one patch embedder, and
      reusing it would switch on the patch-space training path.
    """
    passes: list[_SideChannel] = []
    if getattr(emb, "supports_patch_regions", False) is True:
        passes.append(
            _SideChannel(
                needs=lambda m: _needs_side_channel(m, emb.name, "patch_grid"),
                forward=emb.patch_forward_bulk,
                attach=_attach_patch_grid_to_media,
                fail_message="Bulk patch-forward failed for media_type=%s (%d items)",
            )
        )
    if getattr(emb, "supports_geometric_verification", False) is True:
        passes.append(
            _SideChannel(
                needs=lambda m: _needs_side_channel(m, emb.name, "local_features"),
                forward=emb.local_features_forward_bulk,
                attach=lambda media, feats: media.__setitem__("local_features", feats.compact()),
                fail_message="Bulk local-feature detection failed for media_type=%s (%d items)",
            )
        )
        if getattr(emb, "supports_tiled_stage1", False) is True:
            passes.append(
                _SideChannel(
                    needs=lambda m: (
                        m.get("local_features") is not None and _needs_side_channel(m, emb.name, "tile_vectors")
                    ),
                    forward=emb.tile_vectors_forward_bulk,
                    attach=lambda media, tiles: media.__setitem__("tile_vectors", tiles),
                    fail_message="Tile-vector derivation failed for media_type=%s (%d items)",
                )
            )
    return passes


def embed_missing(
    medias: dict[int, dict[str, Any]],
    embedder_name: str = "",
    on_progress: Callable[[str, str, int, int], None] | None = None,
    failures: EmbedFailures | None = None,
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
    load pipeline drops them via :func:`_drop_none_embeddings_stage`.  Pass a
    *failures* dict to learn why: each such item's id is mapped to a
    one-sentence, user-facing reason naming the embedder and what went wrong
    (declined the item, failed the whole batch, returned the wrong number of
    vectors, or none is installed for the media type).
    Patch-region tensors are also attached here for embedders that
    report ``supports_patch_regions``.

    A partially pre-embedded import (some items shipped with a nameless
    vector, the rest left for the framework) leaves this function keyed under
    **one** embedder name throughout: the resolved embedder is stamped onto the
    nameless vectors whenever leaving them un-stamped would split the dataset
    by name (issue #3798; see :func:`_would_leave_mixed`).  A fully nameless
    dataset with nothing to embed is left as it arrived.

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
        _warn_no_embedder(medias, media_type, failures)
        return

    _stamp_load_embedder(medias, embedder_name, emb.name)

    # Which items still need *this* embedder's vector.
    missing = _missing_for_embedder(medias, emb, embedder_name)

    passes = _side_channel_passes(emb)
    if not missing and not any(p.needs(m) for p in passes for m in medias.values()):
        return

    if on_progress is None:
        on_progress = noop_progress

    _ensure_model_loaded(emb, on_progress)

    _run_embed_pass(emb, medias, media_type, missing, on_progress, failures)

    # Each side channel back-fills every media still missing it, including ones
    # that arrived already-embedded - not just the items embedded above.  In
    # order: a later pass may read an earlier one's output (tiles read local
    # features).
    for p in passes:
        _run_backfill_pass(
            emb,
            medias,
            media_type,
            on_progress,
            needs=p.needs,
            forward=p.forward,
            attach=p.attach,
            fail_message=p.fail_message,
        )

    # Re-key any legacy single-vector media into the dict-keyed representation
    # and drop the singular media["embedding"] mirror (Phase 2c): afterward
    # media["embeddings"] is the sole per-media vector store.
    for media in medias.values():
        ensure_embeddings_dict(media)


def _attach_patch_grid_to_media(media: dict, patch_out) -> None:
    """Attach the raw ``(H, W, D)`` patch grid to *media*, as ``PATCH_ROW_DTYPE``.

    That dtype (:data:`vtscore.embedding.matrix.PATCH_ROW_DTYPE`) is float16;
    #3159 measured what the cast costs the region path.

    That is the *whole* patch side-channel now.  Ingest used to also build a
    24-node HAC region tree per image here (``build_region_tree(patch_out,
    k=12, alpha=0.5)``); #2886 dropped it after the Max-Patch study found raw
    patches beat every tree variant at the operating point, so the payload gets
    strictly **smaller** - the grid was already being stored alongside the tree
    - and ingest sheds the tree's ``O(k^3)`` agglomerative merge.  The
    per-patch saliency ``patch_out`` also carries is not stored: nothing
    downstream reads it now that leaf pooling is gone.
    """
    from vtscore.embedding import matrix  # noqa: PLC0415

    media["patch_grid"] = patch_out.patch_grid.astype(matrix.PATCH_ROW_DTYPE, copy=False)


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
) -> EmbedFailures:
    """Run every bound embedder over the context's medias (tracker-routed progress).

    Single-embedder datasets resolve to one name and behave exactly as before;
    a v3 trio (text + patch + structural picks) runs each in turn, so
    ``media["embeddings"]`` carries a per-embedder vector, the patch embedder
    also populates ``patch_grid``, and the structural
    embedder populates ``local_features``.

    Progress is routed through :class:`EmbedLoopProgress` so a multi-embedder
    loop reports cumulative progress across the embed step rather than restarting
    the bar at 0 for each embedder.

    Returns why each item that came out without a vector did so (see
    :data:`EmbedFailures`), for :func:`_drop_none_embeddings_stage` to show.
    """
    names = _ordered_load_embedders(ctx.medias, requested_embedders)
    progress = EmbedLoopProgress(tracker, len(names))
    failures: EmbedFailures = {}
    for idx, name in enumerate(names):
        progress.begin(idx)
        embed_missing(ctx.medias, name, on_progress=progress, failures=failures)
    invalidate_embedding_matrix(ctx)
    return failures
