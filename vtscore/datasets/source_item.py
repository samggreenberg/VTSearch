"""Find the item a converter output was cut from, in its sibling dataset (#4749).

A converter output knows its source only as a *file*: ``origin.params``
records the scan-relative ``source_file`` and the resolved ``source_path``
(see :func:`vtscore.converters.runner._build_converter_origin`).  That file
is often gone by the time anyone asks — an archive's extract dir, a browser
upload's staging dir — so "show me the photo this face came from" cannot open
it.  It has to find the photo as a **media item** in the dataset the same
multi-dataset import produced beside the faces (#4747).

The join is already in the data.  Both datasets of one run were built from
one scan of one folder, so a face's ``source_file`` is its photo's
``origin_name`` in the Image sibling, and ``source_path`` is that photo's
``media_path``.

Three layers, each usable alone:

* :func:`resolve_source_item` — pure: one output against one sibling's
  ``medias``.  :class:`SourceResolver` is the same, built once for many.
* :func:`find_source_sibling` — which registry entry is the sibling: the
  entry of the same ``import_group`` whose ``output_category`` is the
  converter's ``source_type``.
* :class:`SourceLocator` / :func:`locate_source` — both, against the live
  contexts: read the sibling through :func:`~vtscore.state.core.get_context`
  (never the active-context proxies, which are bound to the *requesting*
  dataset) and report a :class:`SourceLocation` saying where the source is,
  or why it cannot be said.

Generic over converters, not face-specific: a video frame's source is the
video in the Video sibling, a face's is the photo in the Image sibling.  A
rendered document page has none, because its source category (``document``)
*is* the dataset the page sits in, which is never its own sibling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from vtscore.media.provenance import SOURCE_BOX_FIELD

__all__ = [
    "SourceLocation",
    "SourceLocator",
    "SourceResolver",
    "converter_refs",
    "find_source_sibling",
    "locate_source",
    "resolve_source_item",
    "source_type_of",
    "RESOLVED",
    "NOT_DERIVED",
    "NO_SIBLING",
    "SIBLING_NOT_LOADED",
    "NOT_FOUND",
]

#: :attr:`SourceLocation.status` values.
RESOLVED = "resolved"
#: The media is no converter output, or its converter is not installed.
NOT_DERIVED = "not_derived"
#: Its dataset has no import group, or no readable dataset of the group holds
#: the converter's source type.
NO_SIBLING = "no_sibling"
#: The sibling is registered but not in memory; :attr:`SourceLocation.dataset_id`
#: names it so a caller can load it and ask again.
SIBLING_NOT_LOADED = "sibling_not_loaded"
#: The sibling is loaded but no item in it matches (or several do, and nothing
#: tells them apart).
NOT_FOUND = "not_found"


def converter_refs(media: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``origin.params`` of every converter origin *media* records.

    One entry for a plain converter output.  A ``dupe_set`` representative
    (exact-duplicate collapse runs on every import, so two byte-identical
    photos make two byte-identical faces that collapse into one) has no
    converter origin of its own; each member that was a converter output
    contributes its params, in member order.  Anything else returns ``[]``.
    """
    origin = media.get("origin")
    if not isinstance(origin, dict):
        return []
    importer = origin.get("importer")
    if importer == "converter":
        return [origin.get("params") or {}]
    if importer != "dupe_set":
        return []
    refs: list[dict[str, Any]] = []
    for member in origin.get("members") or []:
        member_origin = member.get("origin") if isinstance(member, dict) else None
        if isinstance(member_origin, dict) and member_origin.get("importer") == "converter":
            refs.append(member_origin.get("params") or {})
    return refs


def source_type_of(media: dict[str, Any]) -> str | None:
    """The media type id *media* was converted from, or ``None``.

    Read off the first of :func:`converter_refs` whose converter is
    registered; ``None`` when the media is not derived or its converter is
    not installed here (a plugin removed since the import).
    """
    from vtscore.converters import get_converter  # noqa: PLC0415 - converter discovery is heavy; defer it

    for params in converter_refs(media):
        name = params.get("converter")
        converter = get_converter(name) if name else None
        if converter is None:
            continue
        try:
            return converter.source_type
        except Exception:  # an entry-point tombstone re-raises its import error here
            continue
    return None


class SourceResolver:
    """Resolve converter outputs to items of one sibling dataset; build once, resolve many.

    *sibling_medias* is the sibling's medias dict (a snapshot is fine).
    *name_lookup* is its ``origin_name`` index, the third element of
    :func:`~vtscore.state.media_lookup.build_media_lookup`; pass the cached
    one (:func:`~vtscore.state.media_lookup.cached_media_lookups`) to skip
    the O(N) build, else it is built here.
    """

    def __init__(
        self,
        sibling_medias: dict[int, dict[str, Any]],
        *,
        name_lookup: dict[str, list[int]] | None = None,
    ) -> None:
        if name_lookup is None:
            from vtscore.state.media_lookup import build_media_lookup  # noqa: PLC0415

            name_lookup = build_media_lookup(sibling_medias)[2]
        self._medias = sibling_medias
        self._names = name_lookup
        # Built on the first miss: ``origin_name`` → dupe-set representative,
        # for a source that duplicate collapse folded under another name.
        self._member_names: dict[str, list[int]] | None = None

    def resolve(self, media: dict[str, Any]) -> int | None:
        """The sibling media id *media* was converted from, or ``None``.

        Tries each of :func:`converter_refs` in turn and returns the first
        that names exactly one sibling item.
        """
        for params in converter_refs(media):
            media_id = self._resolve_ref(params)
            if media_id is not None:
                return media_id
        return None

    def _resolve_ref(self, params: dict[str, Any]) -> int | None:
        name = params.get("source_file") or ""
        if not name:
            return None
        candidates = [cid for cid in self._names.get(name, ()) if cid in self._medias]
        if candidates:
            return self._pick(candidates, params.get("source_path") or "")
        # No item carries the name itself: the source may be a duplicate that
        # collapse folded into a representative named after its first member.
        reps = list(dict.fromkeys(cid for cid in self._member_index().get(name, ()) if cid in self._medias))
        return reps[0] if len(reps) == 1 else None

    def _pick(self, candidates: list[int], source_path: str) -> int | None:
        """One candidate wins outright; several are told apart by path, or not at all.

        An ambiguous name answers ``None`` rather than the first match: a
        plausible wrong photo is worse than no photo.  (No importer records a
        source's md5, so the path is the only tiebreak there is.)
        """
        if len(candidates) == 1:
            return candidates[0]
        if source_path:
            on_path = [cid for cid in candidates if self._medias[cid].get("media_path") == source_path]
            if len(on_path) == 1:
                return on_path[0]
        return None

    def _member_index(self) -> dict[str, list[int]]:
        if self._member_names is None:
            index: dict[str, list[int]] = {}
            for media in self._medias.values():
                origin = media.get("origin")
                if not (isinstance(origin, dict) and origin.get("importer") == "dupe_set"):
                    continue
                for member in origin.get("members") or []:
                    if not isinstance(member, dict):
                        continue
                    member_name = member.get("origin_name") or member.get("filename") or ""
                    if member_name:
                        index.setdefault(member_name, []).append(media["id"])
            self._member_names = index
        return self._member_names


def resolve_source_item(
    media: dict[str, Any],
    sibling_medias: dict[int, dict[str, Any]],
    *,
    name_lookup: dict[str, list[int]] | None = None,
) -> int | None:
    """The id of the item in *sibling_medias* that *media* was converted from, or ``None``.

    ``source_file`` → the sibling's ``origin_name``; a name several items
    share is settled by ``source_path`` against their ``media_path``, and is
    ``None`` when that does not single one out.  ``None`` too when *media* is
    not a converter output or nothing matches.  See :class:`SourceResolver`
    for *name_lookup*, and to resolve many outputs against one sibling.
    """
    return SourceResolver(sibling_medias, name_lookup=name_lookup).resolve(media)


def _entry_category(entry: dict[str, Any]) -> str:
    # A legacy or single-import entry has no category; its media type is the
    # closest thing.  Within a group the category is always set, and is what
    # tells the Image output from the Document output (both hold images).
    return entry.get("output_category") or entry.get("media_type") or ""


def find_source_sibling(
    dataset_id: str,
    source_type: str,
    *,
    readable_by: str | None = None,
    entries: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The registry entry holding the items *dataset_id*'s converter outputs came from.

    That is the other entry of *dataset_id*'s ``import_group`` whose
    ``output_category`` is *source_type*: faces (``image2face``, source
    ``image``) find the Image output, never the Document output beside it,
    though both are image datasets.  ``None`` when *dataset_id* has no group,
    or no other member of it stands for *source_type*.

    *readable_by* confines the answer to entries that user may read
    (:func:`~vtscore.datasets.registry.can_user_access`); ``None`` skips the
    check.  *entries* defaults to the live registry.
    """
    from vtscore.datasets import registry  # noqa: PLC0415

    rows = list(entries) if entries is not None else registry.list_datasets()
    own = next((row for row in rows if row.get("id") == dataset_id), None)
    group = own.get("import_group") if own else None
    if not group:
        return None
    for row in rows:
        if row.get("id") == dataset_id or row.get("import_group") != group:
            continue
        if _entry_category(row) != source_type:
            continue
        if readable_by is not None and not registry.can_user_access(row["id"], readable_by):
            continue
        return dict(row)
    return None


@dataclass(frozen=True)
class SourceLocation:
    """Where a converter output's source item lives, or why that cannot be said.

    ``status`` is one of :data:`RESOLVED`, :data:`NOT_DERIVED`,
    :data:`NO_SIBLING`, :data:`SIBLING_NOT_LOADED`, :data:`NOT_FOUND`.
    ``dataset_id`` names the sibling whenever one was found (resolved, not
    loaded, or loaded without a match); ``media_id`` is set only when
    resolved.  ``box`` is the output's ``source_box`` — where in the source
    it sat, normalised (#4748) — or ``None`` when the converter records none.
    """

    status: str
    dataset_id: str | None = None
    media_id: int | None = None
    box: list[float] | None = None


def _box_of(media: dict[str, Any]) -> list[float] | None:
    box = media.get(SOURCE_BOX_FIELD)
    return [float(v) for v in box] if box is not None else None


class SourceLocator:
    """Locate the sources of many outputs of one dataset against the live registry.

    Built per request (or per batch): it reads the registry once, finds each
    source type's sibling once, and builds one :class:`SourceResolver` per
    loaded sibling, reusing the sibling context's cached ``origin_name``
    index.  *user*, when given, confines every answer to siblings that user
    may read (an unreadable sibling is reported as :data:`NO_SIBLING`, the
    same as none, so the reply says nothing about datasets the user cannot
    see).
    """

    def __init__(self, dataset_id: str, *, user: str | None = None) -> None:
        from vtscore.datasets import registry  # noqa: PLC0415

        self._dataset_id = dataset_id
        self._user = user
        self._entries = registry.list_datasets()
        self._siblings: dict[str, dict[str, Any] | None] = {}
        self._resolvers: dict[str, SourceResolver | None] = {}

    def locate(self, media: dict[str, Any]) -> SourceLocation:
        """Where *media*'s source item is; see :class:`SourceLocation`."""
        source_type = source_type_of(media)
        if source_type is None:
            return SourceLocation(NOT_DERIVED)
        box = _box_of(media)
        sibling = self._sibling_for(source_type)
        if sibling is None:
            return SourceLocation(NO_SIBLING, box=box)
        sibling_id = sibling["id"]
        resolver = self._resolver_for(sibling_id)
        if resolver is None:
            return SourceLocation(SIBLING_NOT_LOADED, dataset_id=sibling_id, box=box)
        media_id = resolver.resolve(media)
        if media_id is None:
            return SourceLocation(NOT_FOUND, dataset_id=sibling_id, box=box)
        return SourceLocation(RESOLVED, dataset_id=sibling_id, media_id=media_id, box=box)

    def _sibling_for(self, source_type: str) -> dict[str, Any] | None:
        if source_type not in self._siblings:
            self._siblings[source_type] = find_source_sibling(
                self._dataset_id, source_type, readable_by=self._user, entries=self._entries
            )
        return self._siblings[source_type]

    def _resolver_for(self, sibling_id: str) -> SourceResolver | None:
        if sibling_id not in self._resolvers:
            from vtscore.state.core import _state_lock, get_context  # noqa: PLC0415
            from vtscore.state.media_lookup import cached_media_lookups  # noqa: PLC0415

            ctx = get_context(sibling_id)
            if ctx is None:
                self._resolvers[sibling_id] = None
            else:
                with _state_lock:
                    medias = dict(ctx.medias)
                self._resolvers[sibling_id] = SourceResolver(medias, name_lookup=cached_media_lookups(ctx)[2])
        return self._resolvers[sibling_id]


def locate_source(media: dict[str, Any], dataset_id: str, *, user: str | None = None) -> SourceLocation:
    """Where *media* (an item of dataset *dataset_id*) was converted from.

    One-shot :class:`SourceLocator`; build a locator directly to locate many.
    """
    return SourceLocator(dataset_id, user=user).locate(media)
