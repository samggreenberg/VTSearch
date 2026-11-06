"""Base class for Labelset Sources.

A labelset source is a bidirectional sync target for detector labels:
it can both *load* labels (like a label importer) and *save* them back.
When a detector has a linked source, labels are auto-imported on load
and auto-exported whenever votes change.

A source may also implement **sharing** - keeping a detector's labels in a
container on an external collaboration service that other people can be
given access to - by overriding the ``_do_share_*`` hooks.  Sharing is an
optional capability of this one family, not a separate plugin kind, and the
types and errors the hooks exchange live in :mod:`vtscore.labels.sharing`.
A source may implement plain sync, sharing, or both; the derived flags
:attr:`LabelsetSource.supports_sync`, :attr:`~LabelsetSource.supports_sharing`
and :attr:`~LabelsetSource.supports_recipient_search` say which.

Standalone label importers and exporters remain fully functional
regardless of whether a source is active.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vtscore.labels.sharing import (
    RecipientSuggestion,
    SharedContainer,
    SharedContainerSummary,
    ShareMeta,
    SharingError,
    SharingUnsupported,
)
from vtscore.plugins import PluginField
from vtscore.sync import SyncSource

if TYPE_CHECKING:
    # Imported for the `"LabelSet"` forward reference in the generic
    # subscription below. Ruff can't see string-based references, so the
    # noqa suppresses an otherwise-correct F401.
    from vtscore.datasets.labelset import LabelSet  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = ["OPTIONAL_SHARING_HOOKS", "REQUIRED_SHARING_HOOKS", "LabelsetSource", "PluginField"]

#: The sharing hooks a source overrides, every one, for
#: :attr:`LabelsetSource.supports_sharing` to be true.
REQUIRED_SHARING_HOOKS: tuple[str, ...] = (
    "_do_share_create",
    "_do_share_save",
    "_do_share_refresh",
    "_do_share_list_mine",
    "_do_share_list_received",
    "_do_share_add_recipients",
    "_do_share_remove_recipients",
    "_do_share_set_public",
    "_do_share_archive",
)

#: The optional sharing hook behind :attr:`LabelsetSource.supports_recipient_search`.
OPTIONAL_SHARING_HOOKS: tuple[str, ...] = ("_do_share_search_recipients",)


def _overrides(cls: type, hook: str) -> bool:
    """Whether *cls* (or a base below :class:`LabelsetSource`) overrides *hook*."""
    return getattr(cls, hook, None) is not getattr(LabelsetSource, hook, None)


def _check_ref(remote_ref: Any) -> str:
    """*remote_ref*, when it is the non-empty string a container is named by."""
    if not isinstance(remote_ref, str) or not remote_ref:
        raise ValueError(f"remote_ref must be a non-empty string, not {remote_ref!r}")
    return remote_ref


def _identities(ids: Iterable[str]) -> list[str]:
    """*ids* as a list of distinct, non-empty identity strings, in order.

    Identities are opaque, so they are neither stripped nor case-folded.  A
    bare string is refused rather than iterated character by character.
    """
    if isinstance(ids, str):
        raise TypeError("ids must be an iterable of identity strings, not a single string")
    out: list[str] = []
    for ident in ids:
        if not isinstance(ident, str) or not ident:
            raise ValueError(f"Each recipient id must be a non-empty string, not {ident!r}")
        if ident not in out:
            out.append(ident)
    return out


def _listed(result: Any, item_type: type, hook: str) -> list[Any]:
    """*result* as a list of *item_type*, or :class:`SharingError` naming *hook*."""
    try:
        items = list(result)
    except TypeError:
        raise SharingError(f"{hook}() returned {type(result).__name__}, not a list of {item_type.__name__}") from None
    for item in items:
        if not isinstance(item, item_type):
            raise SharingError(f"{hook}() returned a {type(item).__name__}, not a {item_type.__name__}")
    return items


class LabelsetSource(SyncSource[list[dict[str, str]], "LabelSet"]):
    """Abstract base class for labelset sources.

    Subclass this, set the class-level attributes, implement
    :meth:`_do_load` and :meth:`_do_save`, and expose a module-level
    ``LABELSET_SOURCE = YourSource()`` - the registry picks it up
    automatically.  Public :meth:`load` / :meth:`save` are framework
    wrappers that normalize *field_values* before dispatching; see
    :mod:`vtscore.sync`.

    ``load(field_values)`` returns a list of label dicts
    (``{"md5": ..., "label": "good"|"bad"}``).
    ``save(labelset, field_values)`` persists a :class:`LabelSet`.

    A subclass MAY additionally override :meth:`_do_load_full` to
    surface any detector-level metadata the source carries alongside
    the labels (e.g. ``media_type``, ``input_spec``, ``threshold``).
    The default implementation wraps :meth:`_do_load` into a
    :class:`LabelSet` with no ``detector_meta``, so sources that only
    round-trip labels keep working unchanged.

    **Sharing** is optional.  A source that can keep a detector's labels in
    a container other people are given access to overrides every hook in
    :data:`REQUIRED_SHARING_HOOKS` (and optionally
    :meth:`_do_share_search_recipients`); the ``share_*`` methods are the
    framework wrappers, which normalize *field_values*, check arguments and
    return values, and raise
    :class:`~vtscore.labels.sharing.SharingUnsupported` for a hook the source
    does not implement.  A sharing-only source (no :meth:`_do_save`) is legal:
    :attr:`supports_sync` is false for it, so it is never offered as a plain
    sync target.
    """

    #: Abstract family base: no auto-derived metadata, and concrete
    #: subclasses strip ``LabelsetSource`` from their class names.
    _is_plugin_family_base = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        sharing = [h for h in (*REQUIRED_SHARING_HOOKS, *OPTIONAL_SHARING_HOOKS) if _overrides(cls, h)]
        missing = [h for h in REQUIRED_SHARING_HOOKS if not _overrides(cls, h)]
        if sharing and missing:
            # A half-implemented capability is a bug, and a derived flag would
            # otherwise hide it: the source would simply never be shared through.
            logger.warning(
                "%s overrides some sharing hooks but not %s, so supports_sharing is false and none of its "
                "sharing hooks will be used. See vtscore/docs/extending/labelset-sources.md (The sharing contract).",
                cls.__name__,
                ", ".join(missing),
            )

    def load_full(self, field_values: dict[str, str]) -> "LabelSet":
        """Return the full :class:`LabelSet` (labels + optional detector_meta).

        Normalizes *field_values* then delegates to
        :meth:`_do_load_full`.  Subclasses override
        :meth:`_do_load_full`, not this method.
        """
        return self._do_load_full(self._normalize(field_values))

    def _do_load_full(self, field_values: dict[str, str]) -> "LabelSet":
        """Subclass hook for :meth:`load_full`.

        Receives an already-normalized *field_values* dict.  Default
        implementation calls :meth:`_do_load` (also pre-normalized) so
        sources that only return raw label dicts still produce a valid
        :class:`LabelSet`.  Subclasses with access to richer metadata
        should override this to attach a ``detector_meta`` block.
        """
        from vtscore.datasets.labelset import LabeledElement, LabelSet

        entries = self._do_load(field_values)
        elements = [LabeledElement.from_dict(e) for e in entries if isinstance(e, dict)]
        return LabelSet(elements)

    # -- Derived capability flags ---------------------------------------

    @property
    def supports_sync(self) -> bool:
        """Whether this source can be a plain sync target: a detector's labelset pushed to and pulled from it.

        True when the source overrides :meth:`_do_save` and a load hook
        (:meth:`_do_load` or :meth:`_do_load_full`): a detector's first push
        in a process reads the source first, so a source that can only be
        written to would never be pushed to.  Derived from the overrides,
        never declared, the way
        :attr:`~vtscore.exporters.base.ResultsExporter.supported_payloads` is,
        so it cannot drift from the implementation.  The plain-binding
        pickers (New Detector, Auto-export) offer only sources that report it.
        """
        cls = type(self)
        saves = cls._do_save is not SyncSource._do_save
        loads = cls._do_load is not SyncSource._do_load or cls._do_load_full is not LabelsetSource._do_load_full
        return saves and loads

    @property
    def supports_sharing(self) -> bool:
        """Whether this source implements detector sharing.

        True when it overrides every hook in :data:`REQUIRED_SHARING_HOOKS`.
        Derived, never declared (see :attr:`supports_sync`).  A source that
        overrides only some of them reports false, and a warning naming the
        rest is logged when its class is defined.
        """
        return all(_overrides(type(self), h) for h in REQUIRED_SHARING_HOOKS)

    @property
    def supports_recipient_search(self) -> bool:
        """Whether :meth:`share_search_recipients` works: sharing plus the optional search hook."""
        return self.supports_sharing and _overrides(type(self), "_do_share_search_recipients")

    def to_dict(self) -> dict[str, Any]:
        """Serialise source metadata, adding the three derived capability flags.

        They ride along so ``GET /api/labelset-sources`` tells the frontend
        which sources can be a plain sync target and which can share, without
        a second request.
        """
        return {
            **super().to_dict(),
            "supports_sync": self.supports_sync,
            "supports_sharing": self.supports_sharing,
            "supports_recipient_search": self.supports_recipient_search,
        }

    # -- Sharing: public API (framework-owned; do not override) ---------

    def _require(self, hook: str) -> None:
        """Raise :class:`SharingUnsupported` when this source does not implement *hook*.

        Every sharing hook needs the whole required set first: a source that
        does not share answers each operation, the optional one included,
        with "does not implement sharing".
        """
        if not self.supports_sharing:
            raise SharingUnsupported(f"{type(self).__name__} does not implement sharing")
        if not _overrides(type(self), hook):
            raise SharingUnsupported(f"{type(self).__name__} does not implement {hook}()")

    def share_create(self, labelset: "LabelSet", meta: ShareMeta, field_values: dict[str, Any]) -> str:
        """Create a container holding *labelset*, owned by the caller; return its ``remote_ref``.

        Normalizes *field_values* then delegates to :meth:`_do_share_create`.
        *labelset* and *meta* are what
        :func:`~vtscore.labels.sharing.share_payload` builds for the detector.

        Raises:
            SharingError: The source failed, or its hook returned something
                other than a non-empty string.
        """
        self._require("_do_share_create")
        if not isinstance(meta, ShareMeta):
            raise TypeError(f"meta must be a ShareMeta, not {type(meta).__name__}")
        ref = self._do_share_create(labelset, meta, self._normalize(field_values))
        if not isinstance(ref, str) or not ref:
            raise SharingError(f"{type(self).__name__}._do_share_create() returned {ref!r}, not a remote_ref")
        return ref

    def share_save(self, remote_ref: str, labelset: "LabelSet", meta: ShareMeta, field_values: dict[str, Any]) -> None:
        """Rewrite the container *remote_ref* in place with *labelset* (an owner's save).

        Normalizes *field_values* then delegates to :meth:`_do_share_save`.
        """
        self._require("_do_share_save")
        if not isinstance(meta, ShareMeta):
            raise TypeError(f"meta must be a ShareMeta, not {type(meta).__name__}")
        self._do_share_save(_check_ref(remote_ref), labelset, meta, self._normalize(field_values))

    def share_refresh(self, remote_ref: str, field_values: dict[str, Any]) -> SharedContainer:
        """Read the container *remote_ref*: its records with their identities, recipients and manifest.

        Normalizes *field_values* then delegates to :meth:`_do_share_refresh`.
        """
        self._require("_do_share_refresh")
        container = self._do_share_refresh(_check_ref(remote_ref), self._normalize(field_values))
        if not isinstance(container, SharedContainer):
            raise SharingError(
                f"{type(self).__name__}._do_share_refresh() returned {type(container).__name__}, not a SharedContainer"
            )
        return container

    def share_list_mine(self, field_values: dict[str, Any]) -> list[SharedContainerSummary]:
        """The containers the caller owns, archived ones excluded.

        Normalizes *field_values* then delegates to :meth:`_do_share_list_mine`.
        """
        self._require("_do_share_list_mine")
        result = self._do_share_list_mine(self._normalize(field_values))
        return _listed(result, SharedContainerSummary, f"{type(self).__name__}._do_share_list_mine")

    def share_list_received(self, field_values: dict[str, Any]) -> list[SharedContainerSummary]:
        """The containers shared with the caller - directly, through a group or publicly - that they do not own.

        Archived containers are excluded.  Normalizes *field_values* then
        delegates to :meth:`_do_share_list_received`.
        """
        self._require("_do_share_list_received")
        result = self._do_share_list_received(self._normalize(field_values))
        return _listed(result, SharedContainerSummary, f"{type(self).__name__}._do_share_list_received")

    def share_add_recipients(self, remote_ref: str, ids: Iterable[str], field_values: dict[str, Any]) -> None:
        """Share the container *remote_ref* with each identity in *ids* (people or groups).

        *ids* is deduplicated, order kept.  Normalizes *field_values* then
        delegates to :meth:`_do_share_add_recipients`.
        """
        self._require("_do_share_add_recipients")
        self._do_share_add_recipients(_check_ref(remote_ref), _identities(ids), self._normalize(field_values))

    def share_remove_recipients(self, remote_ref: str, ids: Iterable[str], field_values: dict[str, Any]) -> None:
        """Stop sharing the container *remote_ref* with each identity in *ids*.

        Normalizes *field_values* then delegates to
        :meth:`_do_share_remove_recipients`.
        """
        self._require("_do_share_remove_recipients")
        self._do_share_remove_recipients(_check_ref(remote_ref), _identities(ids), self._normalize(field_values))

    def share_set_public(self, remote_ref: str, public: bool, field_values: dict[str, Any]) -> None:
        """Share the container *remote_ref* with the service's public group, or stop.

        Normalizes *field_values* then delegates to :meth:`_do_share_set_public`.
        """
        self._require("_do_share_set_public")
        self._do_share_set_public(_check_ref(remote_ref), bool(public), self._normalize(field_values))

    def share_archive(self, remote_ref: str, field_values: dict[str, Any]) -> None:
        """Stop sharing the container *remote_ref* altogether.

        Normalizes *field_values* then delegates to :meth:`_do_share_archive`.
        """
        self._require("_do_share_archive")
        self._do_share_archive(_check_ref(remote_ref), self._normalize(field_values))

    def share_search_recipients(self, query: str, field_values: dict[str, Any]) -> list[RecipientSuggestion]:
        """People and groups matching *query*, for a recipient picker.

        An empty (or whitespace) *query* returns ``[]`` without asking the
        source.  Normalizes *field_values* then delegates to
        :meth:`_do_share_search_recipients`.

        Raises:
            SharingUnsupported: The source does not implement search
                (:attr:`supports_recipient_search` is false).
        """
        self._require("_do_share_search_recipients")
        query = str(query or "").strip()
        if not query:
            return []
        result = self._do_share_search_recipients(query, self._normalize(field_values))
        return _listed(result, RecipientSuggestion, f"{type(self).__name__}._do_share_search_recipients")

    # -- Sharing: template methods (subclasses override) ---------------
    #
    # Each receives an already-normalized *field_values* dict, runs as the
    # caller (resolve them with ``vtscore.state.current_user.get_current_user``),
    # and reports identities as opaque strings.  Raise the typed errors of
    # ``vtscore.labels.sharing``: ``RemoteNotFound``, ``RemotePermissionDenied``,
    # or ``SharingError`` for anything else.

    def _do_share_create(self, labelset: "LabelSet", meta: ShareMeta, field_values: dict[str, Any]) -> str:
        """Create a container owned by the caller; return its opaque ``remote_ref``.

        Store *labelset*'s elements as one record per vote (stamping the
        caller as ``created_by`` / ``last_modified_by``), *meta*'s name and
        ``detector_meta``, and *meta*'s manifest with the caller as its writer.
        The container starts shared with nobody and not public.  The core
        also forks a detector by calling this, from the forker's local
        labelset.
        """
        raise SharingUnsupported(f"{type(self).__name__}._do_share_create() is not implemented")

    def _do_share_save(
        self, remote_ref: str, labelset: "LabelSet", meta: ShareMeta, field_values: dict[str, Any]
    ) -> None:
        """Rewrite the container *remote_ref* in place: an owner's save.

        Afterwards it holds exactly *labelset*'s records (new ones added,
        changed ones updated, missing ones deleted), *meta*'s name and
        ``detector_meta``, and *meta*'s manifest, the caller recorded as the
        writer of whatever changed.  Recipients and the public flag stay.
        """
        raise SharingUnsupported(f"{type(self).__name__}._do_share_save() is not implemented")

    def _do_share_refresh(self, remote_ref: str, field_values: dict[str, Any]) -> SharedContainer:
        """Read the container *remote_ref* as a :class:`~vtscore.labels.sharing.SharedContainer`.

        Report the service's own record identities, never the caller's
        assumptions, and the manifest's writer as ``manifest_modified_by``:
        that is what lets the core tell an owner's save from someone else's
        edit.  Raise ``RemoteNotFound`` for an unknown or archived container
        and ``RemotePermissionDenied`` when the caller may not read it.
        """
        raise SharingUnsupported(f"{type(self).__name__}._do_share_refresh() is not implemented")

    def _do_share_list_mine(self, field_values: dict[str, Any]) -> list[SharedContainerSummary]:
        """Summaries of the containers the caller owns, archived ones excluded."""
        raise SharingUnsupported(f"{type(self).__name__}._do_share_list_mine() is not implemented")

    def _do_share_list_received(self, field_values: dict[str, Any]) -> list[SharedContainerSummary]:
        """Summaries of the containers shared with the caller that they do not own, archived ones excluded."""
        raise SharingUnsupported(f"{type(self).__name__}._do_share_list_received() is not implemented")

    def _do_share_add_recipients(self, remote_ref: str, ids: list[str], field_values: dict[str, Any]) -> None:
        """Share *remote_ref* with each of *ids*; only its owner may.  Adding a present one is a no-op."""
        raise SharingUnsupported(f"{type(self).__name__}._do_share_add_recipients() is not implemented")

    def _do_share_remove_recipients(self, remote_ref: str, ids: list[str], field_values: dict[str, Any]) -> None:
        """Stop sharing *remote_ref* with each of *ids*; only its owner may.  Removing an absent one is a no-op."""
        raise SharingUnsupported(f"{type(self).__name__}._do_share_remove_recipients() is not implemented")

    def _do_share_set_public(self, remote_ref: str, public: bool, field_values: dict[str, Any]) -> None:
        """Share *remote_ref* with the service's well-known public group, or stop; only its owner may."""
        raise SharingUnsupported(f"{type(self).__name__}._do_share_set_public() is not implemented")

    def _do_share_archive(self, remote_ref: str, field_values: dict[str, Any]) -> None:
        """Stop sharing *remote_ref*: it leaves every listing and refreshes raise ``RemoteNotFound``; only its owner may."""
        raise SharingUnsupported(f"{type(self).__name__}._do_share_archive() is not implemented")

    def _do_share_search_recipients(self, query: str, field_values: dict[str, Any]) -> list[RecipientSuggestion]:
        """Optional: people and groups matching the non-empty, stripped *query*, best match first."""
        raise SharingUnsupported(f"{type(self).__name__}._do_share_search_recipients() is not implemented")
