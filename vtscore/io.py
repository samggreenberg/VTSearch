"""Shared file I/O helpers for plugins that read or write server-side files.

Two recurring patterns lived inline across importers / exporters / sources
before this module existed:

* Reading a JSON file from a server path — every label / settings importer
  and labelset / settings source repeated the same ``Path.exists()`` /
  ``is_file()`` / ``read_bytes()`` / ``json.loads()`` dance with slightly
  different error messages.
* Writing a file atomically — every server-side exporter and source
  re-implemented the tmp-file + ``fsync`` + ``os.replace`` ritual.  Forget
  one piece and a process crash mid-write leaves a half-written file behind.

The helpers here are deliberately small.  They standardise the
:class:`ValueError` text the framework surfaces to users, and they make
"future file-writing plugin that forgets `fsync`" impossible without
explicitly working around the helper.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = [
    "atomic_write_json",
    "atomic_write_text",
    "read_server_json",
]


def read_server_json(path: Path | str, *, missing_ok: bool = False) -> Any:
    """Read and parse a JSON file from the server filesystem.

    Args:
        path: The file to read.
        missing_ok: When ``True``, return ``None`` if *path* doesn't
            exist.  Otherwise raise :class:`ValueError`.  Sources whose
            "no file yet" state is normal (sync sources on first load,
            etc.) pass ``missing_ok=True``; importers that demand a
            specific user-supplied file leave the default.

    Raises:
        ValueError: If the path exists but is not a regular file, the
            file is not valid UTF-8 JSON, or the file doesn't exist
            (only when ``missing_ok=False``).
    """
    p = Path(path)
    if not p.exists():
        if missing_ok:
            return None
        raise ValueError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")
    raw = p.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write *text* to *path* atomically.

    The implementation writes to a sibling ``.tmp`` file, ``fsync``\\ s it,
    then renames into place via :func:`os.replace` — which is atomic on
    POSIX, so a concurrent reader always sees either the pre- or
    post-write content, never a partial write.  Parent directories are
    created if needed.

    The file is opened with ``newline=""`` so any ``\\r\\n`` or ``\\n``
    sequences already present in *text* are preserved exactly.  Callers
    that built their content with :mod:`csv` (which emits ``\\r\\n``)
    therefore don't end up with doubled ``\\r``\\ s on Windows.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        # Best-effort tmp cleanup so a failed write doesn't leak a
        # half-written tmp file next to the destination.
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path | str, obj: Any, *, indent: int = 2) -> None:
    """Serialise *obj* to JSON and write atomically.

    Thin wrapper around :func:`atomic_write_text`; the trailing newline
    keeps the file POSIX-friendly (text-mode tools that expect a final
    newline don't complain).
    """
    atomic_write_text(path, json.dumps(obj, indent=indent) + "\n")
