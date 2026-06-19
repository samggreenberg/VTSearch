"""Role-typed embedder slot binding for datasets (text / patch / structural).

A dataset's header binds up to one embedder per capability role:

- ``text_embedder``       — a ``supports_text`` embedder (text queries)
- ``patch_embedder``      — a ``supports_patch_regions`` embedder (region voting)
- ``structural_embedder`` — a ``supports_geometric_verification`` embedder
  (instance / geometric-verification matching)

Only a **capability-matching** embedder is eligible for a slot.  Plain
single-vector embedders that match none of the three capability flags
(``dinov2_single``, ``face``, ``whisper``, ``ast``, ``videomae``, …) are not
slottable and remain single-embedder datasets driven by the legacy
``embedder`` field, exactly as before.  The three slot fields are *additive*
to that legacy field, which stays as the always-present primary/content
embedder.

This module is library-tier (no Flask) and resolves embedder capabilities
lazily via :func:`vtscore.media.get_embedder`, so importing it never forces
the media registry to load.

See ``docs/plans/patch-embedder.md`` → "V3 — design" for the full spec.
"""

from __future__ import annotations

from dataclasses import dataclass

# Role identifiers.
TEXT = "text"
PATCH = "patch"
STRUCTURAL = "structural"

# meta.json keys for each slot.
TEXT_KEY = "text_embedder"
PATCH_KEY = "patch_embedder"
STRUCTURAL_KEY = "structural_embedder"

SLOT_KEYS = (TEXT_KEY, PATCH_KEY, STRUCTURAL_KEY)


def _slot_capability_ok(name: str, role: str) -> bool:
    """Whether *name* names a registered embedder eligible for *role*'s slot.

    Eligibility is the embedder's matching capability flag: ``supports_text``
    for the text slot, ``supports_patch_regions`` for the patch slot,
    ``supports_geometric_verification`` for the structural slot.  Unknown
    embedder names return ``False``.
    """
    from vtscore.media import get_embedder  # noqa: PLC0415

    try:
        emb = get_embedder(name)
    except KeyError:
        return False
    if role == TEXT:
        return bool(getattr(emb, "supports_text", False))
    if role == PATCH:
        return bool(getattr(emb, "supports_patch_regions", False))
    if role == STRUCTURAL:
        return bool(getattr(emb, "supports_geometric_verification", False))
    return False


def legacy_embedder_role(name: str) -> str | None:
    """Return the slot role a single legacy embedder migrates into, or ``None``.

    Capability flags are disjoint across the shipped embedders, but we resolve
    in a fixed priority (text → patch → structural) so the mapping is
    deterministic even if a future embedder reports more than one flag.  A
    plain single-vector embedder (no matching flag) and an unknown name both
    return ``None`` — they are not slottable.
    """
    if not name:
        return None
    for role in (TEXT, PATCH, STRUCTURAL):
        if _slot_capability_ok(name, role):
            return role
    return None


@dataclass(frozen=True)
class EmbedderSlots:
    """The three role-typed embedder bindings of a dataset.

    ``None`` means the slot is empty.  A multi-slot dataset has at least one
    slot set; a plain single-vector dataset has all three empty and is driven
    by the legacy ``embedder`` field instead.
    """

    text: str | None = None
    patch: str | None = None
    structural: str | None = None

    @property
    def supports_text(self) -> bool:
        return self.text is not None

    @property
    def supports_patch_regions(self) -> bool:
        return self.patch is not None

    @property
    def supports_geometric_verification(self) -> bool:
        return self.structural is not None

    @property
    def is_empty(self) -> bool:
        """True when no slot is bound (a plain-single / legacy dataset)."""
        return self.text is None and self.patch is None and self.structural is None

    def bound(self) -> list[str]:
        """Distinct bound embedder names, in role order (text, patch, structural)."""
        ordered: dict[str, None] = {}
        for name in (self.text, self.patch, self.structural):
            if name:
                ordered.setdefault(name, None)
        return list(ordered)

    def validate(self) -> None:
        """Raise ``ValueError`` if any slot holds a capability-mismatched embedder.

        A plain (all-empty) binding is valid here; the "must be non-empty"
        rule for *multi-slot* datasets is enforced at dataset-create time, not
        on every read (legacy plain-single datasets are legitimately empty).
        """
        for role, name in ((TEXT, self.text), (PATCH, self.patch), (STRUCTURAL, self.structural)):
            if name and not _slot_capability_ok(name, role):
                raise ValueError(f"Embedder {name!r} is not eligible for the {role!r} slot")

    def to_meta(self) -> dict[str, str]:
        """Slot fields for writing into a container ``meta.json`` (empties omitted)."""
        out: dict[str, str] = {}
        if self.text:
            out[TEXT_KEY] = self.text
        if self.patch:
            out[PATCH_KEY] = self.patch
        if self.structural:
            out[STRUCTURAL_KEY] = self.structural
        return out

    @classmethod
    def from_legacy(cls, embedder: str) -> EmbedderSlots:
        """Migrate a single legacy embedder name into its capability-matching slot.

        A plain single-vector embedder (or an empty/unknown name) maps to an
        all-empty binding — it is not slottable and stays on the legacy
        ``embedder`` field.
        """
        role = legacy_embedder_role(embedder)
        if role == TEXT:
            return cls(text=embedder)
        if role == PATCH:
            return cls(patch=embedder)
        if role == STRUCTURAL:
            return cls(structural=embedder)
        return cls()

    @classmethod
    def from_meta(cls, meta: dict) -> EmbedderSlots:
        """Build slots from a container ``meta.json`` dict.

        If any explicit slot key is present, use the slot keys verbatim.
        Otherwise fall back to migrating the legacy ``embedder`` field by
        capability (read-time migration of pre-slot containers).
        """
        if any(key in meta for key in SLOT_KEYS):
            return cls(
                text=meta.get(TEXT_KEY) or None,
                patch=meta.get(PATCH_KEY) or None,
                structural=meta.get(STRUCTURAL_KEY) or None,
            )
        return cls.from_legacy(meta.get("embedder") or "")
