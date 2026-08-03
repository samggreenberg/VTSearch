"""Default dataset-name derivation shared by every importer.

:meth:`ImporterBase.default_display_name` is the single hook an importer
overrides to name the datasets it produces.  Its base implementation lives
here: a generic derivation over the importer's own declared fields (URL
tail, path leaf, uploaded filename), so a plugin that declares a ``url`` or
``server_path`` field gets a sensible human-readable default without
writing any naming code at all.

This is also what the Add-Dataset form shows: ``POST /api/dataset/import/
<name>/suggested-name`` calls ``default_display_name`` live as the user
edits the form and prefills the Dataset Name box with the result.  The
derivation therefore exists in exactly one place -- what the box shows is
what the import will use.
"""

from __future__ import annotations

from typing import Any, Iterable

from vtscore.plugins import PluginField

from .origin import DATASET_NAME_FIELD_KEY

#: Compound and plain archive extensions stripped from a derived name, so a
#: ``photos.tar.gz`` download becomes ``photos`` rather than ``photos.tar``.
#: Longest-first so the compound forms win over the bare ``.tar``.
_ARCHIVE_SUFFIXES = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tgz",
    ".tbz2",
    ".txz",
    ".tar",
    ".zip",
    ".rar",
)

#: Field types whose value names a filesystem location the derivation can
#: read a leaf out of.  ``file`` is handled separately (its value is an
#: upload object, and files always carry an extension worth stripping).
_PATH_FIELD_TYPES = frozenset({"server_path", "folder"})


def strip_archive_suffix(name: str) -> str:
    """Return *name* without a trailing archive extension.

    Falls back to *name* unchanged when stripping would leave nothing
    (e.g. a file literally called ``.zip``).
    """
    lowered = name.lower()
    for suffix in _ARCHIVE_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)] or name
    return name


def _basename(raw: str) -> str:
    """Return the last path segment of *raw*, tolerating either separator."""
    return next((part for part in reversed(raw.replace("\\", "/").split("/")) if part), "")


def _upload_filename(value: Any) -> str:
    """Return the client-supplied filename behind a ``file`` field value.

    Handles the three shapes a file field arrives as: a Werkzeug upload
    (``.filename``), a file-like object opened from disk (``.name``), or a
    plain path string (CLI runs).
    """
    if value is None:
        return ""
    for attr in ("filename", "name"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str) and candidate:
            return candidate
    return value if isinstance(value, str) else ""


def _candidate_from(field: PluginField, raw: Any) -> str:
    """Return the name *field* suggests given its current *raw* value, or ``""``."""
    if field.field_type == "file":
        basename = _basename(_upload_filename(raw))
        if not basename:
            return ""
        stripped = strip_archive_suffix(basename)
        if stripped != basename:
            return stripped
        # A file always has an extension worth dropping; a directory (below)
        # does not, which is why only this branch takes the stem.
        dot = stripped.rfind(".")
        return stripped[:dot] if dot > 0 else stripped

    if not isinstance(raw, str) or not raw.strip():
        return ""
    raw = raw.strip()

    if field.field_type == "url":
        tail = _basename(raw.split("?")[0].split("#")[0])
        return strip_archive_suffix(tail) if tail else ""

    # A path field may name a directory, so only archive suffixes come off:
    # a folder called ``2024.raw`` keeps its dotted name.
    if field.field_type in _PATH_FIELD_TYPES or field.key == "path":
        leaf = _basename(raw)
        return strip_archive_suffix(leaf) if leaf else ""

    return ""


def derive_display_name(fields: Iterable[PluginField], field_values: dict[str, Any]) -> str:
    """Return a dataset name derived from *field_values*, or ``""``.

    Walks *fields* in declaration order and returns the first name any of
    them yields, so an importer controls precedence by field ordering.  The
    synthetic ``dataset_name`` field is skipped: it holds the user's own
    answer, which :meth:`ImporterBase.resolve_display_name` already
    prefers over anything derived here.
    """
    for field in fields:
        if field.key == DATASET_NAME_FIELD_KEY:
            continue
        candidate = _candidate_from(field, field_values.get(field.key))
        if candidate:
            return candidate
    return ""
