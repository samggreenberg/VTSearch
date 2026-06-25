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

# The three immutable detector *embedder types*.  Each maps to one capability
# bucket on :class:`~vtscore.media.embedder.MediaEmbedder`; the buckets
# partition the registry (no embedder advertises more than one of the flags),
# with precedence ``structural ▸ patch ▸ semantic`` so the classification is a
# total function that matches the score-routing precedence.  A detector locks
# one of these at create time and is compatible with any dataset that binds an
# embedder of the same type - the labels (and, for ``patch_semantic``, the
# region boxes) re-derive against whichever concrete embedder that dataset
# supplies.
EMBEDDER_TYPE_STRUCTURAL = "structural"
EMBEDDER_TYPE_PATCH_SEMANTIC = "patch_semantic"
EMBEDDER_TYPE_SEMANTIC = "semantic"

# Human-facing labels (UI pickers + 400 error messages).
EMBEDDER_TYPE_LABELS: dict[str, str] = {
    EMBEDDER_TYPE_SEMANTIC: "Semantic",
    EMBEDDER_TYPE_PATCH_SEMANTIC: "Patch Semantic",
    EMBEDDER_TYPE_STRUCTURAL: "Structural",
}


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


def embedder_type(embedder_name: str | None) -> str:
    """Classify *embedder_name* into one of the three detector embedder types.

    Returns :data:`EMBEDDER_TYPE_STRUCTURAL` for a geometric-verification
    embedder, :data:`EMBEDDER_TYPE_PATCH_SEMANTIC` for a patch-region embedder,
    else :data:`EMBEDDER_TYPE_SEMANTIC` for any *registered* global
    single-vector embedder (text-capable like ``siglip``/``clip`` or not, like
    ``dinov2_single``/``ast``).  Precedence is ``structural ▸ patch ▸ semantic``.

    ``None`` / empty / **unregistered** → ``""`` (no type): an unknown name
    fills no bucket and gates as incompatible, rather than silently claiming the
    semantic bucket.
    """
    if not embedder_name:
        return ""
    from vtscore.media import get_embedder  # noqa: PLC0415 - avoid import cycle

    try:
        emb = get_embedder(embedder_name)
    except KeyError:
        return ""
    if emb.supports_geometric_verification:
        return EMBEDDER_TYPE_STRUCTURAL
    if emb.supports_patch_regions:
        return EMBEDDER_TYPE_PATCH_SEMANTIC
    return EMBEDDER_TYPE_SEMANTIC


def embedder_of_type(embedder_names: Iterable[str | None], target_type: str) -> str | None:
    """The concrete embedder among *embedder_names* of *target_type*, or ``None``.

    This is the scoring-space resolver for a type-locked detector: given the
    embedders a dataset binds (the keys of ``media["embeddings"]``) and the
    detector's locked type, return the single concrete embedder the MLP should
    train/score in, or ``None`` when the dataset supplies nothing of that type
    (the detector is then incompatible with the dataset).

    A dataset binds at most one patch and one structural embedder, so those
    resolve to the first match.  The ``semantic`` bucket can hold more than one
    name (e.g. a text-capable ``siglip`` plus a single-vector ``dinov2_single``);
    a text-capable embedder wins, matching :func:`score_marker_embedder`'s
    ``... ▸ text ▸ primary`` precedence.
    """
    if not target_type:
        return None
    semantic_text: str | None = None
    semantic_other: str | None = None
    for name in embedder_names:
        if not name or embedder_type(name) != target_type:
            continue
        if target_type != EMBEDDER_TYPE_SEMANTIC:
            return name
        supports_text, _, _ = _capabilities(name)
        if supports_text:
            if semantic_text is None:
                semantic_text = name
        elif semantic_other is None:
            semantic_other = name
    return semantic_text or semantic_other


def dataset_supplied_types(embedder_names: Iterable[str | None]) -> set[str]:
    """The set of embedder types a dataset's bound *embedder_names* cover.

    Used to gate detector/dataset compatibility and to drive the create-time
    type picker (one entry per type the dataset can score in).
    """
    return {t for name in embedder_names if (t := embedder_type(name))}


def detector_dataset_compatible(det_type: str, embedder_names: Iterable[str | None]) -> bool:
    """Whether a detector locked to *det_type* can run on a dataset binding *embedder_names*.

    An empty *det_type* (a legacy detector that has not yet chosen a type) is
    treated as compatible: it resolves at first train via the score precedence,
    preserving the pre-type migration path.  Otherwise the dataset must supply
    at least one embedder of the detector's type.
    """
    if not det_type:
        return True
    return embedder_of_type(embedder_names, det_type) is not None


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

    Under the *per-detector embedder type* model (see
    ``docs/plans/patch-embedder.md`` → "Per-detector embedder type"), a detector
    locks an embedder **type** (semantic / patch_semantic / structural) and
    scores in whatever concrete embedder of that type the active dataset
    supplies.  So the model/label caches stay valid as long as the active
    dataset keeps binding the *same concrete* embedder of the detector's type:

    * type set **and** the snap supplies an embedder of that type → return that
      concrete embedder name.  When it equals ``det_ctx.embedder`` (e.g. both
      datasets bind ``siglip``) nothing is invalidated; when it differs (e.g.
      switching a ``semantic`` detector from a ``siglip`` to a ``clip`` dataset)
      the compare invalidates and the cold path re-embeds the labelset in the
      new space - the intended same-type portability;
    * type set but the snap supplies **no** embedder of that type (the dataset
      can't score this detector) → return the snap's score precedence, which
      differs and invalidates the stale cache; the scoring route gates this pair
      out before it gets used;
    * no explicit type yet (legacy / pre-first-train ``det_ctx``) → the score
      precedence, the legacy-migration default.

    For a legacy detector with no chosen type (``embedder_type`` empty) this is
    exactly the dataset score precedence - byte-for-byte the pre-type behaviour -
    so a single-embedder dataset and every existing detector are unaffected.
    """
    det_type = (getattr(det_ctx, "embedder_type", "") or "") if det_ctx is not None else ""
    if det_type and snap:
        first = next(iter(snap.values()), {})
        concrete = embedder_of_type(media_embedder_names(first), det_type)
        if concrete:
            return concrete
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
