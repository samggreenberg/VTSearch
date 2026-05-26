"""Normalized upload type shared by library-tier plugin bases.

Library-tier plugin bases (:class:`vtscore.datasets.importers.base.DatasetImporter`,
:class:`vtscore.labels.importers.base.LabelImporter`) declare
``field_type="file"`` form inputs that historically arrived as a
Werkzeug :class:`~werkzeug.datastructures.FileStorage` - a Flask-side
type that the library tier is not supposed to know about.

This module provides :class:`UploadedFile`, a structural protocol that
both Flask's ``FileStorage`` and the CLI-side :class:`CliUploadedFile`
adapter satisfy.  Plugin bodies can accept either source without
importing any Flask / Werkzeug symbols:

- The Flask request path passes the ``FileStorage`` straight through -
  it already exposes ``.filename``, ``.read()`` / ``.stream``, and
  ``.save(dst)``.
- The CLI path (:meth:`DatasetImporter.run_cli`) wraps the user's
  ``--file <path>`` argument in :class:`CliUploadedFile`, which exposes
  the same surface backed by a local filesystem path.
- The background-thread upload path
  (``file_mode="bytesio"`` in :func:`vtsearch.routes._shared.validate_plugin_args`)
  uses :class:`BytesIOUploadedFile`, which holds the upload bytes in
  memory so the thread can read them after the Flask request context
  has been torn down.

All three carry a ``.filename`` attr (matching Werkzeug's name); the
historical ``.name`` attr on the bytesio adapter remains as a back-compat
shim for plugin code that still reads it.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vtscore.plugins import PluginField


@runtime_checkable
class UploadedFile(Protocol):
    """Structural type for an uploaded file in a plugin field.

    Satisfied by Werkzeug's :class:`~werkzeug.datastructures.FileStorage`
    (Flask request uploads), :class:`CliUploadedFile` (CLI path
    arguments), and :class:`BytesIOUploadedFile` (in-memory bytes for
    background-thread consumption).  Plugin bodies that accept a
    ``field_type="file"`` value should rely only on the attributes
    declared here.
    """

    #: Original (user-visible) filename, never a server-side path.
    filename: str

    def read(self, size: int = -1) -> bytes:
        """Return up to *size* bytes (all bytes when *size* is ``-1``)."""
        ...

    def save(self, dst: str | Path) -> None:
        """Persist the upload to *dst* on the local filesystem."""
        ...


class CliUploadedFile:
    """Adapter that exposes a local filesystem path as an :class:`UploadedFile`.

    Used by :meth:`DatasetImporter.run_cli` (and the equivalent label /
    settings importer methods) so a plugin body written against the
    :class:`UploadedFile` surface can be reached from the CLI without
    the plugin author handling string-vs-FileStorage branching.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self.filename = self._path.name
        self._fh: BinaryIO | None = None

    def _open(self) -> BinaryIO:
        if self._fh is None:
            self._fh = self._path.open("rb")
        return self._fh

    def read(self, size: int = -1) -> bytes:
        fh = self._open()
        return fh.read() if size == -1 else fh.read(size)

    def save(self, dst: str | Path) -> None:
        shutil.copyfile(self._path, dst)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def stream(self) -> BinaryIO:
        return self._open()


class BytesIOUploadedFile:
    """In-memory :class:`UploadedFile` backed by a :class:`io.BytesIO` buffer.

    Used by the dataset-importer request path
    (``file_mode="bytesio"`` in :func:`validate_plugin_args`) so an
    upload's bytes survive past the Flask request lifetime - the import
    body runs in a background thread that may outlive the request, by
    which time the underlying ``FileStorage`` is no longer readable.

    The buffer exposes both ``.filename`` (UploadedFile-canonical) and
    ``.name`` (back-compat: pre-existing plugin code reads either).
    """

    def __init__(self, data: bytes, filename: str) -> None:
        self.filename = filename
        self._buf = io.BytesIO(data)
        # ``.name`` retained for plugins that read it (matches the
        # pre-Phase-C in-tree convention).  ``BytesIO`` doesn't claim
        # ``.name`` as part of its public surface, so setting it
        # post-construction is safe.
        self._buf.name = filename  # type: ignore[attr-defined]

    @property
    def name(self) -> str:
        return self.filename

    def read(self, size: int = -1) -> bytes:
        return self._buf.read() if size == -1 else self._buf.read(size)

    def save(self, dst: str | Path) -> None:
        Path(dst).write_bytes(self._buf.getvalue())

    @property
    def stream(self) -> BinaryIO:
        return self._buf


def wrap_cli_file_fields(fields: list[PluginField], field_values: dict[str, Any]) -> dict[str, Any]:
    """Return *field_values* with each ``file`` field's path string
    wrapped as a :class:`CliUploadedFile`.

    Used by the default ``run_cli`` implementations on every plugin
    family that accepts ``field_type="file"`` (dataset importers, label
    importers).  Values that are already an :class:`UploadedFile`, or
    ``None``, pass through untouched so CLI test harnesses constructing
    field_values by hand can still inject fakes.
    """
    wrapped = dict(field_values)
    for f in fields:
        if f.field_type != "file":
            continue
        value = wrapped.get(f.key)
        if value is None or isinstance(value, UploadedFile):
            continue
        if isinstance(value, (str, Path)):
            wrapped[f.key] = CliUploadedFile(value)
    return wrapped


__all__ = [
    "BytesIOUploadedFile",
    "CliUploadedFile",
    "UploadedFile",
    "wrap_cli_file_fields",
]
