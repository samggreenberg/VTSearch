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
  ``detector_sharing_source`` setting.

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

import hashlib
import json
import logging
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
    "ActiveSharingSource",
    "Manifest",
    "RecipientSuggestion",
    "RemoteNotFound",
    "RemotePermissionDenied",
    "ShareMeta",
    "SharedContainer",
    "SharedContainerSummary",
    "SharedRecord",
    "SharingError",
    "SharingUnsupported",
    "active_sharing_source",
    "build_manifest",
    "labelset_digest",
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
