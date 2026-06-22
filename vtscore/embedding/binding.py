"""Role-typed embedder binding for a dataset.

A dataset binds **up to one text-capable embedder** and **up to one
patch-capable embedder** (the v3 "three-slot" model in
``docs/plans/patch-embedder.md``).  Text sort runs against the text
embedder; region similarity, region voting, and the detector MLP run
against the patch embedder; both can coexist on one dataset.

This module is the library-tier source of truth for two pure operations
on that binding:

* :func:`derive_binding` - map a single (legacy) embedder name onto the
  ``(text_embedder, patch_embedder)`` pair, by role-typing it against the
  embedder's declared capabilities.  This is how a pre-v3 dataset (one
  ``embedder`` name) resolves into the two-slot model on load.
* :func:`validate_binding` - reject an explicit binding whose slot points
  at an embedder lacking that role's capability.

Neither function holds state; the binding itself lives on
:class:`vtscore.state.core.DatasetContext`.  Capability lookups go
through the embedder registry, so an unknown name resolves to "no
capabilities" (and is therefore ineligible for either slot).
"""

from __future__ import annotations

from collections.abc import Iterable


def _capabilities(embedder_name: str) -> tuple[bool, bool]:
    """Return ``(supports_text, supports_patch_regions)`` for *embedder_name*.

    An unregistered name resolves to ``(False, False)`` so it can fill
    neither role - the caller treats that as "ineligible", not "raise".
    """
    from vtscore.media import get_embedder  # noqa: PLC0415 - avoid import cycle

    try:
        emb = get_embedder(embedder_name)
    except KeyError:
        return (False, False)
    return (bool(emb.supports_text), bool(emb.supports_patch_regions))


def derive_binding(embedder_name: str | None) -> tuple[str | None, str | None]:
    """Map a single (legacy) embedder name to ``(text_embedder, patch_embedder)``.

    A text-capable embedder fills the text slot; a patch-capable embedder
    fills the patch slot.  A single-vector, non-text embedder (e.g.
    ``dinov2_single``) fills **neither** slot (both ``None``): it still
    drives cosine sort and the detector MLP via each media's primary
    vector, which the routing layer reads directly rather than through a
    role slot.

    ``None`` / empty in → ``(None, None)``.
    """
    return derive_binding_from_names([embedder_name] if embedder_name else [])


def derive_binding_from_names(embedder_names: Iterable[str | None]) -> tuple[str | None, str | None]:
    """Role-type a *set* of embedder names into ``(text_embedder, patch_embedder)``.

    This is the multi-embedder generalisation of :func:`derive_binding`: a
    v3 dataset carries one vector per bound embedder under
    ``media["embeddings"]``, so its binding is recovered by role-typing the
    full set of embedder names present (the dict keys), not just one legacy
    name.  Each role slot takes the **first** name that advertises that
    capability; a dual-capability embedder (``supports_text`` *and*
    ``supports_patch_regions``) can fill both.  Single-vector, non-text
    names (e.g. ``dinov2_single``) fill neither and are skipped.

    Empty / all-unknown in → ``(None, None)``.
    """
    text_embedder: str | None = None
    patch_embedder: str | None = None
    for name in embedder_names:
        if not name:
            continue
        supports_text, supports_patch = _capabilities(name)
        if supports_text and text_embedder is None:
            text_embedder = name
        if supports_patch and patch_embedder is None:
            patch_embedder = name
    return (text_embedder, patch_embedder)


def validate_binding(text_embedder: str | None, patch_embedder: str | None) -> None:
    """Raise ``ValueError`` if a slot points at an embedder lacking its role.

    The slots are role-typed: ``text_embedder`` must name an embedder with
    ``supports_text``; ``patch_embedder`` must name one with
    ``supports_patch_regions``.  An unregistered name is rejected for any
    slot it is placed in (it advertises no capabilities).  ``None`` slots
    are always valid.

    Note this does **not** enforce "at least one slot is set" - a
    single-vector dataset legitimately binds neither role.  That
    higher-level rule (no usable sort/search) belongs to the
    dataset-create flow, not the per-slot type check.
    """
    if text_embedder is not None:
        supports_text, _ = _capabilities(text_embedder)
        if not supports_text:
            raise ValueError(
                f"text_embedder {text_embedder!r} does not support text queries (no supports_text capability)"
            )
    if patch_embedder is not None:
        _, supports_patch = _capabilities(patch_embedder)
        if not supports_patch:
            raise ValueError(
                f"patch_embedder {patch_embedder!r} does not produce patch regions "
                "(no supports_patch_regions capability)"
            )
