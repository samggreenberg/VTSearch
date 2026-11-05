"""Detector sharing: the types, errors and helpers of a source's sharing capability.

Sharing a detector means keeping a copy of its labels in a *container* on an
external collaboration service, where other people can be given access to it.
It is an **optional capability of**
:class:`~vtscore.labels.sources.base.LabelsetSource`, not a separate plugin
kind: a source opts in by overriding the ``_do_share_*`` hooks on that class,
and :attr:`~vtscore.labels.sources.base.LabelsetSource.supports_sharing`
reports whether it did.  This module holds everything those hooks exchange
with the core:

* the value types - :class:`SharedRecord`, :class:`SharedContainer`,
  :class:`SharedContainerSummary`, :class:`RecipientSuggestion`,
  :class:`Manifest` and :class:`ShareMeta`;
* the typed errors a source raises - :class:`SharingError` and its
  subclasses - so the core can degrade instead of crashing;
* :func:`build_manifest` and :func:`labelset_digest`, the integrity record an
  owner's save writes beside the records;
* :func:`share_payload`, what the core hands a source for one detector file;
* :func:`active_sharing_source`, the one reader of the
  ``detector_sharing_source`` setting;
* the operations that create and keep sharing state on the registry -
  :func:`receive_container` (a container becomes a local, read-only *linked*
  detector), :func:`refresh_linked_detector` (bring that mirror up to date)
  and :func:`fork_detector` (an editable copy of any detector, shared
  anew when sharing is on) - and :func:`compute_integrity`, which says
  whether anyone but the owner has written to a container.

**Identities are the source's.**  A source resolves the caller itself, through
:func:`vtscore.state.current_user.get_current_user` (the core runs sharing work
as the right user with :func:`~vtscore.state.current_user.thread_user`), owns
whatever mapping from that VTSearch username to the service's accounts and
credentials it needs, and reports every identity - an owner, a record's
``created_by`` / ``last_modified_by``, a recipient - as an **opaque string**
the core stores and displays verbatim and only ever compares for equality.

**Why a manifest.**  The services sharing targets keep per-record
``created_by`` / ``last_modified_by`` identities that clients cannot forge, but
enforce no read-only for recipients.  A recipient who *changes* a record leaves
their identity on it; one who *deletes* a record leaves nothing to inspect.  So
an owner's save also writes a :class:`Manifest` (record count and digest), and
a refresh reports who last wrote it: a manifest the owner did not write, or one
that disagrees with the records, is a foreign modification.

The design narrative is ``docs/plans/detector-sharing.md``; the contract a
source author follows is ``vtscore/docs/extending/labelset-sources.md``
("The sharing contract").
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabeledElement, LabelSet
    from vtscore.labels.sources.base import LabelsetSource

logger = logging.getLogger(__name__)

__all__ = [
    "RECIPIENT_GROUP",
    "RECIPIENT_KINDS",
    "RECIPIENT_USER",
    "SHARED_VIA",
    "SHARED_VIA_DIRECT",
    "SHARED_VIA_GROUP",
    "SHARED_VIA_OWNER",
    "SHARED_VIA_PUBLIC",
    "FOREIGN_ABORT",
    "FOREIGN_IMPORT",
    "FOREIGN_POLICIES",
    "FOREIGN_SKIP",
    "INTEGRITY_CLEAN",
    "INTEGRITY_FOREIGN",
    "ActiveSharingSource",
    "ForeignModifications",
    "Manifest",
    "RecipientSuggestion",
    "RefreshResult",
    "RemoteNotFound",
    "RemotePermissionDenied",
    "ShareMeta",
    "SharedContainer",
    "SharedContainerSummary",
    "SharedRecord",
    "SharingError",
    "SharingUnavailable",
    "SharingUnsupported",
    "active_sharing_source",
    "build_manifest",
    "compute_integrity",
    "fork_detector",
    "labelset_digest",
    "receive_container",
    "refresh_linked_detector",
    "share_payload",
]

#: :attr:`RecipientSuggestion.kind` of one person.
RECIPIENT_USER = "user"
#: :attr:`RecipientSuggestion.kind` of a group of people the service knows.
RECIPIENT_GROUP = "group"
#: Every legal :attr:`RecipientSuggestion.kind`.
RECIPIENT_KINDS = frozenset({RECIPIENT_USER, RECIPIENT_GROUP})

#: :attr:`SharedContainerSummary.shared_via` of a container the caller owns.
SHARED_VIA_OWNER = "owner"
#: The caller is named as a recipient of the container.
SHARED_VIA_DIRECT = "direct"
#: The caller belongs to a group named as a recipient.
SHARED_VIA_GROUP = "group"
#: The container is shared with the service's well-known public group.
SHARED_VIA_PUBLIC = "public"
#: Every legal :attr:`SharedContainerSummary.shared_via`.
SHARED_VIA = frozenset({SHARED_VIA_OWNER, SHARED_VIA_DIRECT, SHARED_VIA_GROUP, SHARED_VIA_PUBLIC})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SharingError(Exception):
    """A sharing operation failed at the source.

    The base of every typed sharing error, and what a source raises for a
    failure none of the subclasses names (the service is unreachable, a
    response is malformed).  The core reports the message and leaves the
    detector as it was.
    """


class SharingUnsupported(SharingError, NotImplementedError):
    """The source does not implement this sharing operation.

    Every ``_do_share_*`` hook raises it by default.  It is also a
    :class:`NotImplementedError`, so code written against the family's older
    "an unimplemented hook raises ``NotImplementedError``" convention catches
    it too.  The core answers an unsupported optional operation (recipient
    search) with "not available here" rather than an error.
    """


class RemoteNotFound(SharingError):
    """The ``remote_ref`` names no container the caller can reach.

    Raised for a container that was never created, was archived, or was
    deleted at the service.  A service that hides a container's existence from
    someone it was not shared with may raise this rather than
    :class:`RemotePermissionDenied`; the core treats the two alike.
    """


class RemotePermissionDenied(SharingError):
    """The container exists, but the caller may not do this to it.

    For example, a recipient asking to change who else it is shared with.
    """


class SharingUnavailable(SharingError):
    """The operation needs a sharing source this deployment does not have active.

    Raised when sharing is off, and when a detector's sharing binding names a
    source other than the active one (or one not installed); the message is
    then the detector's :func:`~vtscore.labels.binding.sharing_note`.  Such a
    detector is never pointed at a different source.
    """


class ForeignModifications(SharingError):
    """Someone other than the owner wrote to the container, and the caller asked to abort.

    Raised by :func:`receive_container` under the ``"abort"`` policy, so
    nothing is created until the caller chooses ``"import"`` or ``"skip"``.

    Attributes:
        integrity: The :func:`compute_integrity` result.
        container: The container as the refresh read it.
    """

    def __init__(self, integrity: dict[str, Any], container: SharedContainer) -> None:
        self.integrity = integrity
        self.container = container
        super().__init__(
            f"'{container.name}' has {integrity['count']} change(s) not made by its owner, {container.owner}"
        )


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Manifest:
    """The integrity record an owner's save writes beside a container's records.

    Attributes:
        record_count: How many records the owner saved.
        digest: :func:`labelset_digest` of the labelset the owner saved.

    Built by :func:`build_manifest`, never by hand, so the core and every
    source agree on the digest.
    """

    record_count: int
    digest: str

    def to_dict(self) -> dict[str, Any]:
        """The JSON-ready form a source stores."""
        return {"record_count": self.record_count, "digest": self.digest}

    @classmethod
    def from_dict(cls, d: Any) -> Manifest | None:
        """Read back what :meth:`to_dict` wrote, or ``None`` when *d* is not that shape.

        A malformed manifest reads as no manifest, which a refresh reports as
        a foreign modification rather than failing on.
        """
        if not isinstance(d, Mapping):
            return None
        count, digest = d.get("record_count"), d.get("digest")
        if isinstance(count, bool) or not isinstance(count, int) or not isinstance(digest, str):
            return None
        return cls(record_count=count, digest=digest)


@dataclass(frozen=True)
class ShareMeta:
    """What a container carries beside its records, as the core hands it to a source.

    Attributes:
        name: The detector's name, which the container is shown under.
        detector_meta: The detector's ``detector_meta`` block
            (:func:`~vtscore.detectors.input_spec.build_detector_meta`), the
            same block the labelset passed alongside carries.  No threshold.
        manifest: The :class:`Manifest` of the labelset passed alongside.
            Store it, and report who last wrote it, so a refresh can tell an
            owner's save from someone else's edit.
    """

    name: str
    detector_meta: Mapping[str, Any]
    manifest: Manifest

    def __post_init__(self) -> None:
        object.__setattr__(self, "detector_meta", dict(self.detector_meta or {}))


@dataclass(frozen=True)
class SharedRecord:
    """One vote in a container, with the identities the service keeps on it.

    Attributes:
        element: The vote, as a :class:`~vtscore.datasets.labelset.LabeledElement`
            in its exported form (no score or rank).
        created_by: Who created the record, as the source reports identities.
        last_modified_by: Who last changed it.  A record either identity of
            which is not the owner's is a foreign modification.
        modified_at: When it last changed, as a Unix timestamp.
    """

    element: LabeledElement
    created_by: str
    last_modified_by: str
    modified_at: float


@dataclass(frozen=True)
class SharedContainer:
    """Everything a refresh reads from one container.

    Attributes:
        remote_ref: The opaque reference the container was created under.
        name: The name it is shown under.
        owner: The identity that owns it.
        detector_meta: The ``detector_meta`` block saved with it.
        records: One :class:`SharedRecord` per vote, in any order.
        recipients: Identities (people or groups) it is shared with, in any
            order.  The public group is :attr:`public`, never an entry here.
        public: Whether it is shared with the service's well-known public group.
        updated_at: When the container last changed, as a Unix timestamp.
        manifest: The last :class:`Manifest` written into it, or ``None``
            when there is none (or it no longer parses).
        manifest_modified_by: Who wrote that manifest, or ``None`` when the
            service cannot say.

    Lists passed for :attr:`records` and :attr:`recipients` are stored as
    tuples, and :attr:`detector_meta` as a copy.
    """

    remote_ref: str
    name: str
    owner: str
    detector_meta: Mapping[str, Any] = field(default_factory=dict)
    records: tuple[SharedRecord, ...] = ()
    recipients: tuple[str, ...] = ()
    public: bool = False
    updated_at: float = 0.0
    manifest: Manifest | None = None
    manifest_modified_by: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "detector_meta", dict(self.detector_meta or {}))
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "recipients", tuple(self.recipients))

    def as_labelset(self) -> LabelSet:
        """The container's records and ``detector_meta`` as a :class:`~vtscore.datasets.labelset.LabelSet`.

        ``build_manifest(container.as_labelset()) == container.manifest`` holds
        for a container nobody but its owner has written.
        """
        from vtscore.datasets.labelset import LabelSet as _LabelSet  # noqa: PLC0415

        return _LabelSet([r.element for r in self.records], detector_meta=dict(self.detector_meta) or None)


@dataclass(frozen=True)
class SharedContainerSummary:
    """One entry of a container listing, without its records.

    Attributes:
        remote_ref: The container's opaque reference.
        name: The name it is shown under.
        owner: The identity that owns it.
        media_type: The ``media_type`` of its ``detector_meta`` (``""`` when
            it has none), so a listing can say what kind of detector it is.
        record_count: How many records it holds.
        updated_at: When it last changed, as a Unix timestamp.
        shared_via: How the caller can see it: :data:`SHARED_VIA_OWNER` in
            ``list_mine``; :data:`SHARED_VIA_DIRECT`, :data:`SHARED_VIA_GROUP`
            or :data:`SHARED_VIA_PUBLIC` in ``list_received``, the most direct
            one that applies.
    """

    remote_ref: str
    name: str
    owner: str
    media_type: str
    record_count: int
    updated_at: float
    shared_via: str

    def __post_init__(self) -> None:
        if self.shared_via not in SHARED_VIA:
            raise ValueError(f"shared_via must be one of {sorted(SHARED_VIA)}, not {self.shared_via!r}")


@dataclass(frozen=True)
class RecipientSuggestion:
    """One match from a recipient search.

    Attributes:
        id: The identity to pass to ``share_add_recipients``.
        display_name: What to show a person choosing it.
        kind: :data:`RECIPIENT_USER` or :data:`RECIPIENT_GROUP`.
    """

    id: str
    display_name: str
    kind: str = RECIPIENT_USER

    def __post_init__(self) -> None:
        if self.kind not in RECIPIENT_KINDS:
            raise ValueError(f"kind must be one of {sorted(RECIPIENT_KINDS)}, not {self.kind!r}")


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def labelset_digest(labelset: LabelSet) -> str:
    """A sha256 content digest of *labelset*: its elements and its ``detector_meta``.

    The one digest of a labelset in this package: the sync engine's push
    deduplication and a sharing :class:`Manifest` both use it.  It does not
    depend on element order - a service is free to hand records back in any
    order - but it does count duplicates, and an absent ``detector_meta``
    digests the same as an empty one.
    """
    elements = sorted(json.dumps(el.to_dict(), sort_keys=True, default=str) for el in labelset.elements)
    canonical = {"labels": elements, "detector_meta": labelset.detector_meta or None}
    text = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(labelset: LabelSet) -> Manifest:
    """The :class:`Manifest` of *labelset*: its record count and :func:`labelset_digest`.

    *labelset* is the export payload an owner's save sends (records plus
    ``detector_meta``, as :func:`share_payload` builds it).
    """
    return Manifest(record_count=len(labelset.elements), digest=labelset_digest(labelset))


def share_payload(detector_data: Mapping[str, Any]) -> tuple[LabelSet, ShareMeta]:
    """What a sharing hook receives for the detector file *detector_data*.

    Returns ``(labelset, meta)``: *labelset* is the file's whole labelset in
    its exported form, exactly as a plain push sends it (every dataset's
    votes, ``detector_meta`` attached, no threshold, rank or score - #5131),
    and *meta* is the :class:`ShareMeta` that goes with it, its manifest drawn
    from that same labelset.
    """
    from vtscore.labels.sync import _push_payload  # noqa: PLC0415

    labelset = _push_payload(dict(detector_data))
    meta = ShareMeta(
        name=str(detector_data.get("name") or ""),
        detector_meta=labelset.detector_meta or {},
        manifest=build_manifest(labelset),
    )
    return labelset, meta


# ---------------------------------------------------------------------------
# The active sharing source
# ---------------------------------------------------------------------------

#: Setting names already warned about, so a bad setting is logged once per
#: name rather than on every read.  Cleared between tests by
#: :func:`reset_sharing_warnings_for_tests`.
_warned: set[str] = set()


@dataclass(frozen=True)
class ActiveSharingSource:
    """The sharing source the ``detector_sharing_source`` setting names, resolved.

    Attributes:
        source_name: The registered name of the source.
        source: The source plugin; its :attr:`supports_sharing` is true.
        field_values: The values the setting fixes: each non-empty entry of
            its ``field_values``, laid into every binding the source is given,
            the way ``default_labelset_source`` fixes its keys.
        missing_fields: Keys of the source's required fields the setting
            leaves without a value or a declared default.
    """

    source_name: str
    source: LabelsetSource
    field_values: Mapping[str, Any]
    missing_fields: tuple[str, ...] = ()


def _warn_once(name: str, message: str, *args: Any) -> None:
    if name in _warned:
        return
    _warned.add(name)
    logger.warning(message, *args)


def active_sharing_source() -> ActiveSharingSource | None:
    """The source detector sharing goes through, or ``None`` when sharing is off.

    Reads the ``detector_sharing_source`` setting
    (``{"source_name": ..., "field_values": {...}} | null``, default ``null``)
    with :func:`~vtscore.config.get_setting`.  Sharing is off - ``None`` - when
    the setting is null, malformed or the opt-out ``"none"``, and also, with a
    warning logged once per name, when it names a source that is not
    registered or one whose :attr:`supports_sharing` is false.  Discovery
    alone never switches sharing on: a source is active only when named here.
    """
    from vtscore.config import get_setting  # noqa: PLC0415
    from vtscore.labels.binding import OPT_OUT, _missing_required, _valid_config, fixed_field_values  # noqa: PLC0415
    from vtscore.labels.sources import get_labelset_source  # noqa: PLC0415

    try:
        cfg = _valid_config(get_setting("detector_sharing_source"))
    except Exception:  # noqa: BLE001 - unreadable settings mean no sharing
        return None
    if cfg is None:
        return None
    source_name = cfg["source_name"].strip()
    if source_name == OPT_OUT:
        return None
    source = get_labelset_source(source_name)
    if source is None:
        _warn_once(source_name, "detector_sharing_source names unknown labelset source %r; sharing is off", source_name)
        return None
    if not source.supports_sharing:
        _warn_once(
            source_name,
            "detector_sharing_source names labelset source %r, which does not implement sharing; sharing is off",
            source_name,
        )
        return None
    fixed = fixed_field_values(cfg)
    return ActiveSharingSource(
        source_name=source_name,
        source=source,
        field_values=dict(fixed),
        missing_fields=_missing_required(source, dict(fixed)),
    )


def reset_sharing_warnings_for_tests() -> None:
    """Forget which bad settings were already warned about (tests)."""
    _warned.clear()


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

#: :func:`compute_integrity` ``status`` of a container only its owner wrote.
INTEGRITY_CLEAN = "clean"
#: ``status`` of one someone else wrote to.
INTEGRITY_FOREIGN = "foreign_modifications"

#: Refresh policy: report foreign modifications and change nothing.
FOREIGN_ABORT = "abort"
#: Take every record, foreign ones included.
FOREIGN_IMPORT = "import"
#: Take only the records the owner both created and last modified.
FOREIGN_SKIP = "skip"
#: Every refresh policy, the default first.
FOREIGN_POLICIES = (FOREIGN_ABORT, FOREIGN_IMPORT, FOREIGN_SKIP)


def _owned(record: SharedRecord, owner: str) -> bool:
    return record.created_by == owner and record.last_modified_by == owner


def compute_integrity(container: SharedContainer) -> dict[str, Any]:
    """Whether anyone but the owner has written to *container*: ``{"status", "count"}``.

    A record is foreign when its ``created_by`` or ``last_modified_by`` is not
    the owner.  The manifest is foreign when it is missing, was not last
    written by the owner, or counts a different number of records than the
    container holds - which is how a deleted record shows, since a deletion
    leaves no identity behind.  ``count`` is the foreign records plus one for
    a foreign manifest; ``status`` is :data:`INTEGRITY_CLEAN` when it is 0,
    else :data:`INTEGRITY_FOREIGN`.
    """
    owner = container.owner
    count = sum(1 for r in container.records if not _owned(r, owner))
    manifest = container.manifest
    if manifest is None or container.manifest_modified_by != owner or manifest.record_count != len(container.records):
        count += 1
    return {"status": INTEGRITY_CLEAN if count == 0 else INTEGRITY_FOREIGN, "count": count}


def _check_policy(foreign: str) -> None:
    if foreign not in FOREIGN_POLICIES:
        raise ValueError(f"foreign must be one of {FOREIGN_POLICIES}, not {foreign!r}")


def _records_for(container: SharedContainer, foreign: str) -> list[LabeledElement]:
    """The elements a receive or refresh takes from *container* under the *foreign* policy."""
    if foreign == FOREIGN_SKIP:
        return [r.element for r in container.records if _owned(r, container.owner)]
    return [r.element for r in container.records]


# ---------------------------------------------------------------------------
# Receive, refresh, fork
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefreshResult:
    """What :func:`refresh_linked_detector` did.

    Attributes:
        applied: Whether the local copy was replaced; ``False`` when the
            ``"abort"`` policy met foreign modifications.
        integrity: The :func:`compute_integrity` result.
        added: Labels the local copy gained.
        changed: Labels whose value changed.
        removed: Labels the local copy lost.
    """

    applied: bool
    integrity: Mapping[str, Any]
    added: int = 0
    changed: int = 0
    removed: int = 0


def _require_active() -> ActiveSharingSource:
    active = active_sharing_source()
    if active is None:
        raise SharingUnavailable("Detector sharing is off on this server")
    return active


def _active_for(entry: dict[str, Any]) -> tuple[ActiveSharingSource, dict[str, Any], dict[str, Any]]:
    """``(active source, binding, field values)`` for *entry*'s container, or :class:`SharingUnavailable`."""
    from vtscore.labels.binding import binding_matches_active_source, sharing_binding, sharing_note  # noqa: PLC0415

    binding = sharing_binding(entry)
    if binding is None:
        raise ValueError(f"Detector '{entry.get('name', '')}' is not shared")
    if not binding_matches_active_source(entry):
        raise SharingUnavailable(sharing_note(entry) or "Detector sharing is off on this server")
    active = _require_active()
    return active, binding, _field_values(binding, active)


def _field_values(binding: Mapping[str, Any], active: ActiveSharingSource) -> dict[str, Any]:
    """The binding's own values with the setting's fixed ones laid over them.

    The fixed values (the service's connection values) are never stored per
    detector, as with ``default_labelset_source``: a changed setting reaches
    every binding, and a credential is not copied into the registry.
    """
    own = binding.get("field_values")
    return {**(own if isinstance(own, dict) else {}), **active.field_values}


def _label_map(labelset: LabelSet) -> dict[Any, tuple[str, Any]]:
    from vtscore.datasets.labelset import element_key  # noqa: PLC0415

    out: dict[Any, tuple[str, Any]] = {}
    for el in labelset.elements:
        key = element_key(el)
        out[key if key is not None else ("element", json.dumps(el.to_dict(), sort_keys=True))] = (
            el.label,
            el.region_box,
        )
    return out


def _diff(old: LabelSet, new: LabelSet) -> tuple[int, int, int]:
    """``(added, changed, removed)`` going from *old* to *new*, item by item."""
    before, after = _label_map(old), _label_map(new)
    added = sum(1 for k in after if k not in before)
    removed = sum(1 for k in before if k not in after)
    changed = sum(1 for k, v in after.items() if k in before and before[k] != v)
    return added, changed, removed


def _update_binding(detector_id: str, binding: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    from vtscore.detectors.registry import update_detector  # noqa: PLC0415

    updated = {**binding, **changes}
    update_detector(detector_id, labelset_source=updated)
    return updated


def receive_container(remote_ref: str, *, foreign: str = FOREIGN_ABORT) -> dict[str, Any]:
    """Make the container *remote_ref* a local, read-only detector of the caller's; return its registry entry.

    Refreshes it through the active sharing source as the caller
    (:func:`~vtscore.state.current_user.get_current_user`), applies the
    *foreign* policy, writes the detector file (its labelset the container's
    records, its ``detector_meta`` applied) and registers the entry:
    ``created_by`` the caller, ``sync_mode`` ``"linked"``, and a binding
    carrying ``remote_ref``, ``remote_owner``, ``last_synced`` and
    ``integrity``.  A name another detector holds is suffixed, never
    overwritten.

    Raises:
        ValueError: *foreign* is not a policy, or the caller already has a
            detector holding this container.
        SharingUnavailable: Sharing is off.
        ForeignModifications: The policy is ``"abort"`` and someone other
            than the owner wrote to the container; nothing was created.
        SharingError: The source failed (``RemoteNotFound``,
            ``RemotePermissionDenied``, ...).
    """
    from vtscore.datasets.labelset import LabelSet as _LabelSet  # noqa: PLC0415
    from vtscore.detectors.input_spec import apply_detector_meta  # noqa: PLC0415
    from vtscore.detectors.registry import (  # noqa: PLC0415
        SYNC_MODE_LINKED,
        free_detector_name,
        list_detectors,
        register_detector,
    )
    from vtscore.detectors.store import (  # noqa: PLC0415
        _detector_path,
        _write_detector,
        allow_linked_write,
        discard_pending_detector_write,
    )
    from vtscore.labels.binding import sharing_binding  # noqa: PLC0415
    from vtscore.state.current_user import get_current_user  # noqa: PLC0415

    _check_policy(foreign)
    active = _require_active()
    user = get_current_user()
    for held in list_detectors():
        binding = sharing_binding(held)
        if (
            binding is not None
            and held.get("created_by", "default") == user
            and binding.get("remote_ref") == remote_ref
            and binding["source_name"].strip() == active.source_name
        ):
            raise ValueError(f"You already have this shared detector, as '{held.get('name', '')}'")

    container = active.source.share_refresh(remote_ref, dict(active.field_values))
    integrity = compute_integrity(container)
    if foreign == FOREIGN_ABORT and integrity["status"] != INTEGRITY_CLEAN:
        raise ForeignModifications(integrity, container)

    elements = _records_for(container, foreign)
    now = time.time()
    name = free_detector_name(container.name or "Shared detector")
    data: dict[str, Any] = {
        "name": name,
        "text_query": "",
        "media_example": "",
        "media_type": "",
        "examples": [],
        "created_at": now,
        "embedder_type": "",
        "labelset": _LabelSet(elements).to_dict(),
    }
    apply_detector_meta(data, dict(container.detector_meta))
    path = _detector_path(name)
    with allow_linked_write():
        _write_detector(path, data)
    binding = {
        "source_name": active.source_name,
        "field_values": {},
        "remote_ref": remote_ref,
        "remote_owner": container.owner,
        "last_synced": now,
        "integrity": integrity,
        "sync_error": None,
    }
    try:
        return register_detector(
            name=name,
            media_type=data.get("media_type", "") or "",
            num_training=len(elements),
            created_by=user,
            sync_mode=SYNC_MODE_LINKED,
            labelset_source=binding,
        )
    except Exception:
        discard_pending_detector_write(path)
        path.unlink(missing_ok=True)
        raise


def refresh_linked_detector(detector_id: str, *, foreign: str = FOREIGN_ABORT) -> RefreshResult:
    """Bring the received detector *detector_id* level with its container.

    The local labelset is **replaced** by the container's records - a mirror,
    not a merge - under the *foreign* policy: ``"abort"`` (the default)
    changes nothing when anyone but the owner wrote to the container and
    reports it, ``"import"`` takes every record and ``"skip"`` only the
    owner's.  The container's ``detector_meta`` is applied, the binding's
    ``last_synced`` / ``integrity`` / ``remote_owner`` are updated (an
    aborted refresh records the integrity too, so the listing can warn), and a
    loaded context re-derives its votes from the new file.

    Raises:
        LookupError: No detector has that id.
        ValueError: *foreign* is not a policy, or the detector is not a
            received one.
        SharingUnavailable: Sharing is off, or the detector's container
            belongs to a source that is not the active one (the message is its
            :func:`~vtscore.labels.binding.sharing_note`).
        SharingError: The source failed.
    """
    from vtscore.datasets.labelset import LabelSet as _LabelSet  # noqa: PLC0415
    from vtscore.detectors.input_spec import apply_detector_meta  # noqa: PLC0415
    from vtscore.detectors.label_sync import label_sync_write_lock  # noqa: PLC0415
    from vtscore.detectors.registry import get_detector, is_linked, update_detector  # noqa: PLC0415
    from vtscore.detectors.store import (  # noqa: PLC0415
        _detector_path,
        _read_detector,
        _write_detector,
        allow_linked_write,
    )
    from vtscore.labels.sync import _rehydrate_loaded  # noqa: PLC0415

    _check_policy(foreign)
    entry = get_detector(detector_id)
    if entry is None:
        raise LookupError(f"No detector {detector_id!r}")
    if not is_linked(entry):
        raise ValueError(f"Detector '{entry.get('name', '')}' is not a received detector")
    active, binding, field_values = _active_for(entry)

    container = active.source.share_refresh(binding["remote_ref"], field_values)
    integrity = compute_integrity(container)
    if foreign == FOREIGN_ABORT and integrity["status"] != INTEGRITY_CLEAN:
        _update_binding(detector_id, binding, integrity=integrity, remote_owner=container.owner)
        return RefreshResult(applied=False, integrity=integrity)

    new = _LabelSet(_records_for(container, foreign))
    path = _detector_path(entry["name"])
    with label_sync_write_lock, allow_linked_write():
        data = _read_detector(path) or {"name": entry["name"], "media_type": entry.get("media_type", "")}
        added, changed, removed = _diff(_LabelSet.from_dict(data.get("labelset") or {}), new)
        data["labelset"] = new.to_dict()
        apply_detector_meta(data, dict(container.detector_meta))
        _write_detector(path, data)
    _update_binding(
        detector_id,
        binding,
        integrity=integrity,
        remote_owner=container.owner,
        last_synced=time.time(),
        sync_error=None,
    )
    update_detector(detector_id, num_training=len(new))
    _rehydrate_loaded(detector_id)
    return RefreshResult(applied=True, integrity=integrity, added=added, changed=changed, removed=removed)


def fork_detector(detector_id: str, *, name: str | None = None) -> dict[str, Any]:
    """Copy the detector *detector_id* into a new, editable detector owned by the caller; return its entry.

    Works on any detector - a received one (whose local copy is the one
    forked: after a refresh that skipped foreign records, that is the trusted
    labelset) or a local one (a copy with provenance).  The new detector is
    named *name*, or ``"<name> (fork)"`` suffixed until free; it has
    ``sync_mode`` ``"forked"``, ``forked_from`` provenance, ``created_by`` the
    caller, and no binding.  Then, when sharing is on, a container is created
    from its labelset (``share_create``) and bound to it.  A failure there
    leaves a working local fork: the returned entry's ``share_error`` (not
    persisted) says what went wrong, and is ``None`` otherwise.

    ``forked_from`` holds ``detector_id``, ``name`` and ``owner`` - the
    container's owner for a received detector, else the local creator - plus
    ``source_name`` and ``remote_ref`` when the original was shared.

    Raises:
        LookupError: No detector has that id, or its file is missing.
        ValueError: *name* is blank or another detector holds it.
    """
    from vtscore.datasets.labelset import LabelSet as _LabelSet  # noqa: PLC0415
    from vtscore.detectors.registry import (  # noqa: PLC0415
        SYNC_MODE_FORKED,
        detector_name_taken,
        free_detector_name,
        get_detector,
        register_detector,
    )
    from vtscore.detectors.store import _detector_path, _read_detector, _write_detector  # noqa: PLC0415
    from vtscore.labels.binding import sharing_binding  # noqa: PLC0415
    from vtscore.state.current_user import get_current_user  # noqa: PLC0415

    entry = get_detector(detector_id)
    if entry is None:
        raise LookupError(f"No detector {detector_id!r}")
    source_data = _read_detector(_detector_path(entry["name"]))
    if source_data is None:
        raise LookupError(f"Detector '{entry['name']}' has no labelset file")
    if name is None:
        new_name = free_detector_name(f"{entry['name']} (fork)")
    else:
        new_name = name.strip()
        if not new_name:
            raise ValueError("A fork needs a name")
        if detector_name_taken(new_name):
            raise ValueError(f"A detector named '{new_name}' already exists")

    data = copy.deepcopy(source_data)
    data["name"] = new_name
    data["created_at"] = time.time()
    _write_detector(_detector_path(new_name), data)

    shared = sharing_binding(entry)
    forked_from: dict[str, Any] = {
        "detector_id": entry["id"],
        "name": entry["name"],
        "owner": ((shared or {}).get("remote_owner") or entry.get("created_by", "default")),
    }
    if shared is not None:
        forked_from["source_name"] = shared["source_name"]
        forked_from["remote_ref"] = shared["remote_ref"]
    new_entry = register_detector(
        name=new_name,
        media_type=entry.get("media_type", "") or data.get("media_type", "") or "",
        num_training=len(_LabelSet.from_dict(data.get("labelset") or {})),
        text_query=entry.get("text_query", "") or "",
        media_example=entry.get("media_example", "") or "",
        examples=entry.get("examples") or data.get("examples") or [],
        created_by=get_current_user(),
        embedder=entry.get("embedder", "") or "",
        embedder_type=entry.get("embedder_type", "") or "",
        sync_mode=SYNC_MODE_FORKED,
        forked_from=forked_from,
    )
    return {**new_entry, **_share_fork(new_entry, data)}


def _share_fork(entry: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Create a container for the new fork *entry* when sharing is on: the keys to add to the returned entry."""
    active = active_sharing_source()
    if active is None:
        return {"share_error": None}
    try:
        labelset, meta = share_payload(data)
        remote_ref = active.source.share_create(labelset, meta, dict(active.field_values))
    except Exception as exc:  # noqa: BLE001 - a fork that cannot be shared is still a fork
        logger.warning("Fork %r was not shared through %s: %s", entry.get("name"), active.source_name, exc)
        return {"share_error": str(exc) or type(exc).__name__}
    binding = {
        "source_name": active.source_name,
        "field_values": {},
        "remote_ref": remote_ref,
        "last_synced": time.time(),
        "sync_error": None,
    }
    from vtscore.detectors.registry import update_detector  # noqa: PLC0415

    update_detector(entry["id"], labelset_source=binding)
    return {"labelset_source": binding, "share_error": None}
