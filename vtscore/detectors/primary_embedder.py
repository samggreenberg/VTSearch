"""Resolve and validate a detector's per-detector *primary* embedder.

A detector binds exactly one **primary embedder** at create time - the single
vector space its MLP is trained and scored in (see
``docs/plans/patch-embedder.md`` → "Per-detector primary embedder").  It is a
persisted *name* on the detector JSON, never a vector or MLP, so it satisfies
the "No Persisted Vectors or MLPs" rule.

This module holds the pure resolution / validation used by the detector-create
routes.  It is library-tier (no Flask): the route turns a returned error string
into an HTTP 400.
"""

from __future__ import annotations


def active_dataset_bound_embedders() -> list[str]:
    """Return the active dataset's bound embedder names, or ``[]``.

    The eligible primaries for a detector are exactly the embedders the active
    dataset has vectors for - the keys of ``media["embeddings"]`` - which is
    broader than the three role slots: a single-vector embedder (e.g.
    ``dinov2_single``) fills no role slot but is still a valid scoring space.
    Empty when no dataset is loaded (the create flow then can't validate, and
    leaves the choice to first-train migration).
    """
    from vtscore.embedding.media_vectors import media_embedder_names
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    if not ctx.dataset_id or not ctx.medias:
        return []
    first = next(iter(ctx.medias.values()), {})
    return media_embedder_names(first)


def resolve_detector_primary(requested: str) -> tuple[str, str | None]:
    """Resolve the primary embedder to persist for a new detector.

    Returns ``(primary, error)`` - *error* is ``None`` on success, else a
    message the route surfaces as HTTP 400.  Validation/resolution against the
    active dataset's bound embedders:

    * non-empty *requested* → must be one of the active dataset's bound
      embedders (when a dataset is loaded), else an error;
    * empty *requested* on a single-embedder dataset → that one embedder
      (zero-friction default, identical UX to today);
    * empty *requested* on a multi-embedder dataset → an error (the client must
      choose: there is no safe default once more than one space exists);
    * empty *requested* with no active dataset → ``("", None)``: left empty and
      resolved at first train via the legacy score precedence (migration).
    """
    requested = (requested or "").strip()
    bound = active_dataset_bound_embedders()
    if requested:
        if bound and requested not in bound:
            return "", f"embedder '{requested}' is not bound to this dataset"
        return requested, None
    if len(bound) == 1:
        return bound[0], None
    if len(bound) > 1:
        return "", "This dataset has multiple embedders; choose one as the detector's embedder"
    return "", None
