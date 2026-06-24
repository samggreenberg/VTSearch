"""Role-typed embedder binding for a dataset.

A dataset binds **up to one text-capable embedder**, **up to one
patch-capable embedder**, and **up to one structural (geometric
verification) embedder** (the v3 "three-slot" trio in
``docs/plans/patch-embedder.md``).  Text sort runs against the text
embedder; region similarity, region voting, and the detector MLP run
against the patch embedder; instance retrieval + geometric re-rank run
against the structural embedder; all three can coexist on one dataset.

This module is the library-tier source of truth for two pure operations
on that binding:

* :func:`derive_binding` - map a single (legacy) embedder name onto the
  ``(text_embedder, patch_embedder, structural_embedder)`` triple, by
  role-typing it against the embedder's declared capabilities.  This is how
  a pre-v3 dataset (one ``embedder`` name) resolves into the three-slot
  model on load.
* :func:`validate_binding` - reject an explicit binding whose slot points
  at an embedder lacking that role's capability.

Neither function holds state; the binding itself lives on
:class:`vtscore.state.core.DatasetContext`.  Capability lookups go
through the embedder registry, so an unknown name resolves to "no
capabilities" (and is therefore ineligible for any slot).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vtscore.embedding.media_vectors import media_embedder_names


def _capabilities(embedder_name: str) -> tuple[bool, bool, bool]:
    """Return ``(supports_text, supports_patch_regions, supports_geometric_verification)``.

    An unregistered name resolves to ``(False, False, False)`` so it can
    fill no role - the caller treats that as "ineligible", not "raise".
    """
    from vtscore.media import get_embedder  # noqa: PLC0415 - avoid import cycle

    try:
        emb = get_embedder(embedder_name)
    except KeyError:
        return (False, False, False)
    return (
        bool(emb.supports_text),
        bool(emb.supports_patch_regions),
        bool(emb.supports_geometric_verification),
    )


def derive_binding(embedder_name: str | None) -> tuple[str | None, str | None, str | None]:
    """Map a single (legacy) embedder name to the role-typed slot triple.

    Returns ``(text_embedder, patch_embedder, structural_embedder)``.  A
    text-capable embedder fills the text slot; a patch-capable embedder
    fills the patch slot; a geometric-verification-capable embedder fills
    the structural slot.  A single-vector, non-text embedder (e.g.
    ``dinov2_single``) fills **no** slot (all ``None``): it still drives
    cosine sort and the detector MLP via each media's primary vector, which
    the routing layer reads directly rather than through a role slot.

    ``None`` / empty in → ``(None, None, None)``.
    """
    return derive_binding_from_names([embedder_name] if embedder_name else [])


def derive_binding_from_names(
    embedder_names: Iterable[str | None],
) -> tuple[str | None, str | None, str | None]:
    """Role-type a *set* of embedder names into the slot triple.

    Returns ``(text_embedder, patch_embedder, structural_embedder)``.  This
    is the multi-embedder generalisation of :func:`derive_binding`: a v3
    dataset carries one vector per bound embedder under
    ``media["embeddings"]``, so its binding is recovered by role-typing the
    full set of embedder names present (the dict keys), not just one legacy
    name.  Each role slot takes the **first** name that advertises that
    capability; a multi-capability embedder can fill more than one slot.
    Single-vector, non-text names (e.g. ``dinov2_single``) fill no slot and
    are skipped.

    Empty / all-unknown in → ``(None, None, None)``.
    """
    text_embedder: str | None = None
    patch_embedder: str | None = None
    structural_embedder: str | None = None
    for name in embedder_names:
        if not name:
            continue
        supports_text, supports_patch, supports_structural = _capabilities(name)
        if supports_text and text_embedder is None:
            text_embedder = name
        if supports_patch and patch_embedder is None:
            patch_embedder = name
        if supports_structural and structural_embedder is None:
            structural_embedder = name
    return (text_embedder, patch_embedder, structural_embedder)


def score_marker_embedder(media: dict[str, Any]) -> str:
    """Concrete embedder name a detector MLP is keyed to for *media*.

    This is the v3 model-keying marker (the "embedder" component of the
    ``(detector, dataset, embedder)`` MLP key in ``docs/plans/patch-embedder.md``).
    It resolves to the **score** embedder (structural ▸ patch ▸ text, the
    routing table) when a role slot is bound, and falls back to *media*'s
    primary embedder *name* for a slot-less single-vector dataset (e.g.
    ``dinov2_single``).  ``""`` only when *media* carries no embedder at all.

    Unlike :meth:`DatasetContext.routed_embedder`, which returns ``None`` for
    a slot-less dataset so the matrix layer collapses to the cached primary
    path, this always returns a concrete name so it can be stamped on and
    compared against ``DetectorContext.embedder`` (a ``str``).  The two never
    disagree about which vector space a model was trained against: for a
    single-embedder dataset both name the same embedder; for a multi-embedder
    dataset both pick the structural-else-patch-else-text slot; only the
    slot-less fallback differs (a name here vs ``None`` there), and that name
    *is* the primary the matrix layer reads in that case.
    """
    text, patch, structural = derive_binding_from_names(media_embedder_names(media))
    return structural or patch or text or media.get("embedder", "") or ""


def score_marker_embedder_for_snap(snap: dict[int, dict[str, Any]] | None) -> str:
    """:func:`score_marker_embedder` for the first media in a *snap* dict.

    Returns ``""`` when *snap* is empty.  Used by the model-invalidation and
    cross-dataset training paths, which derive the marker from the medias they
    are about to score rather than from the active context's binding (a
    non-active snapshot resolves correctly this way).
    """
    if not snap:
        return ""
    return score_marker_embedder(next(iter(snap.values()), {}))


def keying_embedder_for_snap(det_ctx: Any, snap: dict[int, dict[str, Any]] | None) -> str:
    """The marker to compare against ``det_ctx.embedder`` for cache invalidation.

    Under the *per-detector primary embedder* model (see
    ``docs/plans/patch-embedder.md`` → "Per-detector primary embedder"), a
    detector scores in its **own** explicit primary (``det_ctx.embedder``),
    not the dataset's score precedence.  So the model/label caches stay valid
    as long as the active dataset can still *supply* that primary embedder:

    * primary set **and** present among the snap's bound embedders → return
      the primary unchanged, so the compare against ``det_ctx.embedder`` is
      equal and nothing is invalidated (no per-request retrain thrash on a
      multi-embedder dataset whose precedence differs from the primary);
    * primary set but **absent** from the snap (the dataset can't score this
      detector) → return the snap's score precedence, which differs from the
      primary and so invalidates the stale cache (the cold path then re-embeds
      the labelset against the new dataset, or the detector is gated);
    * no explicit primary yet (legacy / pre-first-train ``det_ctx``) → the
      score precedence, the legacy-migration default.

    For every single-embedder dataset the primary *is* that one embedder and is
    always present, so this returns the primary unchanged - byte-for-byte the
    pre-per-detector behaviour, where the marker equalled the precedence.
    """
    primary = (getattr(det_ctx, "embedder", "") or "") if det_ctx is not None else ""
    if primary and snap:
        first = next(iter(snap.values()), {})
        if primary in media_embedder_names(first):
            return primary
    return score_marker_embedder_for_snap(snap)


def validate_binding(
    text_embedder: str | None,
    patch_embedder: str | None,
    structural_embedder: str | None = None,
) -> None:
    """Raise ``ValueError`` if a slot points at an embedder lacking its role.

    The slots are role-typed: ``text_embedder`` must name an embedder with
    ``supports_text``; ``patch_embedder`` must name one with
    ``supports_patch_regions``; ``structural_embedder`` must name one with
    ``supports_geometric_verification``.  An unregistered name is rejected
    for any slot it is placed in (it advertises no capabilities).  ``None``
    slots are always valid.

    Note this does **not** enforce "at least one slot is set" - a
    single-vector dataset legitimately binds no role.  That higher-level
    rule (no usable sort/search) belongs to the dataset-create flow, not the
    per-slot type check.
    """
    if text_embedder is not None:
        supports_text, _, _ = _capabilities(text_embedder)
        if not supports_text:
            raise ValueError(
                f"text_embedder {text_embedder!r} does not support text queries (no supports_text capability)"
            )
    if patch_embedder is not None:
        _, supports_patch, _ = _capabilities(patch_embedder)
        if not supports_patch:
            raise ValueError(
                f"patch_embedder {patch_embedder!r} does not produce patch regions "
                "(no supports_patch_regions capability)"
            )
    if structural_embedder is not None:
        _, _, supports_structural = _capabilities(structural_embedder)
        if not supports_structural:
            raise ValueError(
                f"structural_embedder {structural_embedder!r} does not support geometric "
                "verification (no supports_geometric_verification capability)"
            )
