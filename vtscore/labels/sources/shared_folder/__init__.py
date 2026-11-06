"""Shared folder labelset source - detector sharing through a directory on the server.

The filesystem-backed implementation of the sharing contract
(:mod:`vtscore.labels.sharing`), for tests and local development: it stands in
for the external collaboration service a real sharing source talks to, and the
conformance suite holds it to the same contract any such source must meet.
It implements sharing only - it is not a plain sync target
(:attr:`~vtscore.labels.sources.base.LabelsetSource.supports_sync` is false) -
and, like every source, is active only when the ``detector_sharing_source``
setting names it.

**Layout.**  One field, ``folder``.  Each container is one JSON file,
``<folder>/<remote_ref>.json``, holding its owner, name, ``detector_meta``,
public flag, recipients, archived flag, the records - each a labelset element
with the ``created_by`` / ``last_modified_by`` / ``modified_at`` the folder
maintains - and the manifest with its own ``manifest_modified_by``.

**Identities** are VTSearch usernames, read with
:func:`~vtscore.state.current_user.get_current_user`.  The folder behaves like
the services it models: the owner alone changes who a container is shared
with, its recipients can still write its records (there is no server-side
read-only), and every write stamps the writer's identity on what it changed,
which no caller can forge.  :func:`simulate_foreign_modification` writes what
a misbehaving recipient would, for the tests of the code that detects it.

``folder`` is a ``server_path`` field, so under a multi-user login provider it
is confined to the caller's own data directory like every server path, and two
users cannot reach one folder; run it single-user.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vtscore.config import DATA_DIR
from vtscore.labels.sharing import (
    RECIPIENT_USER,
    SHARED_VIA_DIRECT,
    SHARED_VIA_OWNER,
    SHARED_VIA_PUBLIC,
    Manifest,
    RecipientSuggestion,
    RemoteNotFound,
    RemotePermissionDenied,
    SharedContainer,
    SharedContainerSummary,
    SharedRecord,
    ShareMeta,
    SharingError,
)
from vtscore.labels.sources.base import LabelsetSource, PluginField
from vtscore.utils.io import atomic_write_json, file_lock, read_server_json

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabeledElement, LabelSet

logger = logging.getLogger(__name__)

__all__ = ["FORMAT", "SharedFolderLabelsetSource", "simulate_foreign_modification"]

#: The ``format`` stamp of a container file; files without it are not ours.
FORMAT = "vtsearch.shared_folder/1"

#: A ``remote_ref`` this source could have issued.  Anything else names no
#: container, which also keeps a crafted ref from leaving the folder.
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _caller() -> str:
    from vtscore.state.current_user import get_current_user  # noqa: PLC0415

    return get_current_user()


def _container_path(folder: Path, remote_ref: str) -> Path:
    if not _REF.match(remote_ref):
        raise RemoteNotFound(f"No shared detector {remote_ref!r}")
    return folder / f"{remote_ref}.json"


def _read_doc(path: Path) -> dict[str, Any] | None:
    """The container file at *path*, ``None`` when there is none, or :class:`SharingError`."""
    try:
        doc = read_server_json(path, missing_ok=True)
    except ValueError as exc:
        raise SharingError(f"Cannot read shared detector {path.stem!r}: {exc}") from exc
    if doc is None:
        return None
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise SharingError(f"{path} is not a shared-folder container")
    return doc


def _iter_docs(folder: Path) -> Iterator[dict[str, Any]]:
    """Every container file in *folder*, skipping (and logging) any that do not parse."""
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("*.json")):
        try:
            doc = _read_doc(path)
        except SharingError as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue
        if doc is not None:
            yield doc


def _live_doc(path: Path) -> dict[str, Any]:
    """The container at *path*, or :class:`RemoteNotFound` when it is missing or archived."""
    doc = _read_doc(path)
    if doc is None or doc.get("archived"):
        raise RemoteNotFound(f"No shared detector {path.stem!r}")
    return doc


def _recipients(doc: Mapping[str, Any]) -> list[str]:
    return [r for r in doc.get("recipients") or [] if isinstance(r, str)]


def _can_read(doc: Mapping[str, Any], who: str) -> bool:
    return doc.get("owner") == who or who in _recipients(doc) or bool(doc.get("public"))


def _can_write(doc: Mapping[str, Any], who: str) -> bool:
    """Owner or named recipient: like the services this models, no read-only for recipients."""
    return doc.get("owner") == who or who in _recipients(doc)


def _require_owner(doc: Mapping[str, Any], who: str, action: str) -> None:
    if doc.get("owner") != who:
        raise RemotePermissionDenied(f"Only the owner of shared detector {doc.get('remote_ref')!r} can {action}")


def _element_text(el: LabeledElement) -> str:
    return json.dumps(el.to_dict(), sort_keys=True, default=str)


def _record_key(el: LabeledElement) -> Any:
    """Which stored record a new element continues: same media (and drawn box), whatever its label."""
    from vtscore.datasets.labelset import element_key  # noqa: PLC0415

    key = element_key(el)
    if key is None:
        key = ("element", _element_text(el))
    return (key, el.region_box)


def _record(element: dict[str, Any], *, created_by: str, last_modified_by: str, modified_at: float) -> dict[str, Any]:
    return {
        "element": element,
        "created_by": created_by,
        "last_modified_by": last_modified_by,
        "modified_at": modified_at,
    }


def _rewritten_records(prior: list[dict[str, Any]], labelset: LabelSet, who: str, now: float) -> list[dict[str, Any]]:
    """*labelset*'s elements as records, keeping the identities of every record they leave as it was.

    A record whose element is unchanged keeps its identities; one whose label
    (or anything else) changed keeps ``created_by`` and is stamped *who* as
    its last modifier; a new element is a record created by *who*; and prior
    records the labelset no longer holds are dropped.
    """
    from vtscore.datasets.labelset import LabeledElement as _LabeledElement  # noqa: PLC0415

    by_key: dict[Any, list[dict[str, Any]]] = {}
    for rec in prior:
        element = rec.get("element")
        if isinstance(element, dict):
            by_key.setdefault(_record_key(_LabeledElement.from_dict(element)), []).append(rec)

    out: list[dict[str, Any]] = []
    for el in labelset.elements:
        element = el.to_dict()
        candidates = by_key.get(_record_key(el)) or []
        same = next((r for r in candidates if r["element"] == element), None)
        if same is not None:
            candidates.remove(same)
            out.append(same)
        elif candidates:
            old = candidates.pop(0)
            out.append(_record(element, created_by=old.get("created_by") or who, last_modified_by=who, modified_at=now))
        else:
            out.append(_record(element, created_by=who, last_modified_by=who, modified_at=now))
    return out


def _container(doc: Mapping[str, Any]) -> SharedContainer:
    from vtscore.datasets.labelset import LabeledElement as _LabeledElement  # noqa: PLC0415

    records = [
        SharedRecord(
            element=_LabeledElement.from_dict(rec["element"]),
            created_by=str(rec.get("created_by") or ""),
            last_modified_by=str(rec.get("last_modified_by") or ""),
            modified_at=float(rec.get("modified_at") or 0.0),
        )
        for rec in doc.get("records") or []
        if isinstance(rec, dict) and isinstance(rec.get("element"), dict)
    ]
    meta = doc.get("detector_meta")
    return SharedContainer(
        remote_ref=str(doc.get("remote_ref") or ""),
        name=str(doc.get("name") or ""),
        owner=str(doc.get("owner") or ""),
        detector_meta=meta if isinstance(meta, dict) else {},
        records=tuple(records),
        recipients=tuple(_recipients(doc)),
        public=bool(doc.get("public")),
        updated_at=float(doc.get("updated_at") or 0.0),
        manifest=Manifest.from_dict(doc.get("manifest")),
        manifest_modified_by=doc.get("manifest_modified_by") or None,
    )


def _summary(doc: Mapping[str, Any], shared_via: str) -> SharedContainerSummary:
    meta = doc.get("detector_meta")
    media_type = meta.get("media_type", "") if isinstance(meta, dict) else ""
    return SharedContainerSummary(
        remote_ref=str(doc.get("remote_ref") or ""),
        name=str(doc.get("name") or ""),
        owner=str(doc.get("owner") or ""),
        media_type=str(media_type or ""),
        record_count=len(doc.get("records") or []),
        updated_at=float(doc.get("updated_at") or 0.0),
        shared_via=shared_via,
    )


def _sorted(summaries: list[SharedContainerSummary]) -> list[SharedContainerSummary]:
    """Newest first, then by name: a stable order for a listing."""
    return sorted(summaries, key=lambda s: (-s.updated_at, s.name, s.remote_ref))


class SharedFolderLabelsetSource(LabelsetSource):
    """Share detectors through a folder of JSON files on the server (tests and local development)."""

    name = "shared_folder"
    display_name = "Shared Folder"
    description = "Share detectors through a folder on the server, one JSON file per shared detector."
    icon = "\U0001f4c2"  # open file folder
    fields = [
        PluginField(
            key="folder",
            label="Shared folder (server path)",
            field_type="server_path",
            description="The directory every shared detector is kept in, one JSON file each.",
            hint="Absolute or relative server path. Fix it in the detector_sharing_source setting.",
            placeholder=f"{DATA_DIR}/shared-detectors",
        ),
    ]

    # -- writing ------------------------------------------------------

    def _do_share_create(self, labelset: LabelSet, meta: ShareMeta, field_values: dict[str, Any]) -> str:
        who, now = _caller(), time.time()
        remote_ref = uuid.uuid4().hex
        doc = {
            "format": FORMAT,
            "remote_ref": remote_ref,
            "name": meta.name,
            "owner": who,
            "detector_meta": dict(meta.detector_meta),
            "public": False,
            "recipients": [],
            "archived": False,
            "created_at": now,
            "updated_at": now,
            "records": _rewritten_records([], labelset, who, now),
            "manifest": meta.manifest.to_dict(),
            "manifest_modified_by": who,
        }
        atomic_write_json(_container_path(Path(field_values["folder"]), remote_ref), doc)
        return remote_ref

    def _do_share_save(
        self, remote_ref: str, labelset: LabelSet, meta: ShareMeta, field_values: dict[str, Any]
    ) -> None:
        path = _container_path(Path(field_values["folder"]), remote_ref)
        who, now = _caller(), time.time()
        with file_lock(path):
            doc = _live_doc(path)
            if not _can_write(doc, who):
                raise RemotePermissionDenied(f"You cannot write to shared detector {remote_ref!r}")
            doc["records"] = _rewritten_records(list(doc.get("records") or []), labelset, who, now)
            doc["name"] = meta.name
            doc["detector_meta"] = dict(meta.detector_meta)
            doc["manifest"] = meta.manifest.to_dict()
            doc["manifest_modified_by"] = who
            doc["updated_at"] = now
            atomic_write_json(path, doc)

    def _change(
        self, remote_ref: str, field_values: dict[str, Any], action: str, edit: Callable[[dict[str, Any]], None]
    ) -> None:
        """Apply *edit* to the owner's container *remote_ref* under its lock."""
        path = _container_path(Path(field_values["folder"]), remote_ref)
        who = _caller()
        with file_lock(path):
            doc = _live_doc(path)
            _require_owner(doc, who, action)
            edit(doc)
            doc["updated_at"] = time.time()
            atomic_write_json(path, doc)

    def _do_share_add_recipients(self, remote_ref: str, ids: list[str], field_values: dict[str, Any]) -> None:
        def edit(doc: dict[str, Any]) -> None:
            current = _recipients(doc)
            current += [i for i in ids if i not in current and i != doc.get("owner")]
            doc["recipients"] = current

        self._change(remote_ref, field_values, "change who it is shared with", edit)

    def _do_share_remove_recipients(self, remote_ref: str, ids: list[str], field_values: dict[str, Any]) -> None:
        def edit(doc: dict[str, Any]) -> None:
            doc["recipients"] = [r for r in _recipients(doc) if r not in ids]

        self._change(remote_ref, field_values, "change who it is shared with", edit)

    def _do_share_set_public(self, remote_ref: str, public: bool, field_values: dict[str, Any]) -> None:
        def edit(doc: dict[str, Any]) -> None:
            doc["public"] = bool(public)

        self._change(remote_ref, field_values, "make it public or private", edit)

    def _do_share_archive(self, remote_ref: str, field_values: dict[str, Any]) -> None:
        def edit(doc: dict[str, Any]) -> None:
            doc["archived"] = True

        self._change(remote_ref, field_values, "stop sharing it", edit)

    # -- reading ------------------------------------------------------

    def _do_share_refresh(self, remote_ref: str, field_values: dict[str, Any]) -> SharedContainer:
        doc = _live_doc(_container_path(Path(field_values["folder"]), remote_ref))
        if not _can_read(doc, _caller()):
            raise RemotePermissionDenied(f"Shared detector {remote_ref!r} is not shared with you")
        return _container(doc)

    def _do_share_list_mine(self, field_values: dict[str, Any]) -> list[SharedContainerSummary]:
        who = _caller()
        return _sorted(
            [
                _summary(doc, SHARED_VIA_OWNER)
                for doc in _iter_docs(Path(field_values["folder"]))
                if doc.get("owner") == who and not doc.get("archived")
            ]
        )

    def _do_share_list_received(self, field_values: dict[str, Any]) -> list[SharedContainerSummary]:
        who = _caller()
        out: list[SharedContainerSummary] = []
        for doc in _iter_docs(Path(field_values["folder"])):
            if doc.get("archived") or doc.get("owner") == who:
                continue
            if who in _recipients(doc):
                out.append(_summary(doc, SHARED_VIA_DIRECT))
            elif doc.get("public"):
                out.append(_summary(doc, SHARED_VIA_PUBLIC))
        return _sorted(out)

    def _do_share_search_recipients(self, query: str, field_values: dict[str, Any]) -> list[RecipientSuggestion]:
        """Every identity seen in the folder (owners, recipients, record writers) containing *query*.

        Case-insensitive; identities starting with the query come first, and
        the caller is never suggested.
        """
        who, needle = _caller(), query.casefold()
        seen: set[str] = set()
        for doc in _iter_docs(Path(field_values["folder"])):
            seen.add(str(doc.get("owner") or ""))
            seen.update(_recipients(doc))
            for rec in doc.get("records") or []:
                if isinstance(rec, dict):
                    seen.update(str(rec.get(k) or "") for k in ("created_by", "last_modified_by"))
        matches = [s for s in seen if s and s != who and needle in s.casefold()]
        matches.sort(key=lambda s: (not s.casefold().startswith(needle), s.casefold(), s))
        return [RecipientSuggestion(id=s, display_name=s, kind=RECIPIENT_USER) for s in matches]


def _flip(label: str) -> str:
    return "bad" if label == "good" else "good"


def simulate_foreign_modification(
    folder: str | Path,
    remote_ref: str,
    *,
    actor: str,
    relabel: Mapping[str, str] | Iterable[str] | None = None,
    add: Iterable[LabeledElement | Mapping[str, Any]] | None = None,
    delete: Iterable[str] | None = None,
    touch_manifest: bool = False,
) -> None:
    """Write to a container what a misbehaving recipient would (a test hook).

    Goes straight to the file, as a recipient of a service with no read-only
    for recipients could, and stamps *actor* as the writer of everything it
    changes:

    * *relabel* - md5s whose records change label: a mapping of md5 to the new
      label, or an iterable of md5s to flip between ``"good"`` and ``"bad"``;
      each changed record gets *actor* as ``last_modified_by``;
    * *add* - new elements (``LabeledElement`` or element dicts), recorded as
      created and last modified by *actor*;
    * *delete* - md5s whose records are removed, which leaves no identity
      behind (only the manifest can tell);
    * *touch_manifest* - also rewrite the manifest to match the modified
      records, as a cover-up would; its ``manifest_modified_by`` becomes
      *actor*, so it is still not the owner's.

    Raises :class:`RemoteNotFound` when there is no such container.
    """
    from vtscore.datasets.labelset import LabeledElement as _LabeledElement  # noqa: PLC0415
    from vtscore.labels.sharing import build_manifest  # noqa: PLC0415

    if isinstance(relabel, str):
        relabel = [relabel]
    if isinstance(delete, str):
        delete = [delete]
    path = _container_path(Path(folder), remote_ref)
    now = time.time()
    with file_lock(path):
        doc = _live_doc(path)
        records = [r for r in doc.get("records") or [] if isinstance(r, dict) and isinstance(r.get("element"), dict)]
        if relabel is not None:
            targets = dict(relabel) if isinstance(relabel, Mapping) else {md5: None for md5 in relabel}
            for rec in records:
                md5 = rec["element"].get("md5")
                if md5 in targets:
                    label = targets[md5] or _flip(rec["element"].get("label", ""))
                    rec["element"] = {**rec["element"], "label": label}
                    rec["last_modified_by"] = actor
                    rec["modified_at"] = now
        if delete is not None:
            gone = set(delete)
            records = [r for r in records if r["element"].get("md5") not in gone]
        for item in add or ():
            element = item.to_dict() if isinstance(item, _LabeledElement) else dict(item)
            records.append(_record(element, created_by=actor, last_modified_by=actor, modified_at=now))
        doc["records"] = records
        if touch_manifest:
            doc["manifest"] = build_manifest(_container(doc).as_labelset()).to_dict()
            doc["manifest_modified_by"] = actor
        doc["updated_at"] = now
        atomic_write_json(path, doc)


LABELSET_SOURCE = SharedFolderLabelsetSource()
