"""Resolve and validate a detector's immutable *embedder type*.

A detector locks exactly one **embedder type** at create time - one of
``"semantic"``, ``"patch_semantic"``, ``"structural"`` (see
``docs/plans/patch-embedder.md`` → "Per-detector embedder type").  It is the
*kind* of vector space the detector's MLP trains and scores in, not a specific
embedder name: the detector is compatible with any dataset (of the same
``media_type``) that binds an embedder of that type, and the MLP re-derives
against whichever concrete embedder that dataset supplies.  Only the type is
persisted on the detector JSON - never a vector or MLP - so it satisfies the
"No Persisted Vectors or MLPs" rule.

This module holds the pure resolution / validation used by the detector-create
routes.  It is library-tier (no Flask): the route turns a returned error string
into an HTTP 400.
"""

from __future__ import annotations

from vtscore.embedding.binding import (
    EMBEDDER_TYPE_LABELS,
    dataset_supplied_types,
    embedder_type,
)


def active_dataset_bound_embedders() -> list[str]:
    """Return the active dataset's bound embedder names, or ``[]``.

    These are the embedders the active dataset has vectors for - the keys of
    ``media["embeddings"]`` - which is what the create flow classifies into
    supplied *types*.  Empty when no dataset is loaded (the create flow then
    can't classify, and leaves the choice to first-train migration).
    """
    from vtscore.embedding.media_vectors import media_embedder_names
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    if not ctx.dataset_id or not ctx.medias:
        return []
    first = next(iter(ctx.medias.values()), {})
    return media_embedder_names(first)


def detector_embedder_type_from_data(data: dict) -> str:
    """Read a detector JSON's locked embedder type, migrating legacy records.

    Prefers the persisted ``embedder_type``; when absent, classifies a legacy
    ``primary_embedder`` *name* (pre-type detectors stored a concrete embedder)
    so existing detector JSONs keep their compatibility behaviour without a
    rewrite.  Returns ``""`` for a detector that has neither (resolved at first
    train).
    """
    if not data:
        return ""
    t = (data.get("embedder_type", "") or "").strip()
    if t:
        return t
    return embedder_type((data.get("primary_embedder", "") or "").strip())


def _labels(types: set[str]) -> str:
    """Human-facing list of type labels, in display precedence, for an error."""
    order = ["semantic", "patch_semantic", "structural"]
    return ", ".join(EMBEDDER_TYPE_LABELS[t] for t in order if t in types)


def resolve_detector_embedder_type(requested: str) -> tuple[str, str | None]:
    """Resolve the embedder type to persist for a new detector.

    Returns ``(embedder_type, error)`` - *error* is ``None`` on success, else a
    message the route surfaces as HTTP 400.

    * non-empty *requested* (a type name, or a concrete embedder name we
      classify) → accepted as long as it is one of the three valid types,
      **regardless of whether the active dataset binds it**.  The type is the
      user's *declared intent*: a detector can be created before any dataset is
      loaded (the create modal can't guess the type), and one bound to a type
      the active dataset doesn't supply simply gates as incompatible there
      (greyed "Sort by Learned", find-label 409) the same way it does on any
      other dataset that lacks the type.  Only a string that maps to no known
      type is rejected;
    * empty *requested* on a dataset supplying exactly one type → that type
      (zero-friction default);
    * empty *requested* on a dataset supplying more than one type → an error
      (the client must choose);
    * empty *requested* with no active dataset → ``("", None)``: left empty and
      resolved at first train via the legacy score precedence (migration).
    """
    requested = (requested or "").strip()
    if requested:
        # Accept a type name directly, or a concrete embedder name (classify it)
        # so a client that still sends an embedder name keeps working.
        resolved = requested if requested in EMBEDDER_TYPE_LABELS else embedder_type(requested)
        if not resolved:
            return "", f"unknown embedder type {requested!r}"
        return resolved, None
    supplied = dataset_supplied_types(active_dataset_bound_embedders())
    if len(supplied) == 1:
        return next(iter(supplied)), None
    if len(supplied) > 1:
        return "", f"This dataset binds multiple embedder types; choose one of: {_labels(supplied)}"
    return "", None
