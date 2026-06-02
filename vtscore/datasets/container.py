"""ZIP-based dataset container format.

Replaces the raw-pickle-plus-sidecars layout with a single append-friendly
ZIP file.  The container stores:

- ``medias.pkl``     — the pickled ``{"medias": {...}, ...}`` dict
- ``meta.json``      — embedder, clipper, media_type, timestamps, age-off
- ``projection.npz`` — (optional, appended later) frozen 2-D layout + pyramid

Legacy raw-pickle files (no ZIP wrapper) are detected by magic bytes and
handled transparently — the reader returns the same dict regardless of
format.

See ``docs/design/vtsbrowse.md`` for the persistence carve-out.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from vtscore.security.pickle import safe_pickle_load

logger = logging.getLogger(__name__)

_ZIP_MAGIC = b"PK\x03\x04"


def is_container(path: str | Path) -> bool:
    """Return ``True`` if *path* is a ZIP container (vs. a raw pickle)."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == _ZIP_MAGIC
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_container(
    dest: str | Path | io.BytesIO,
    medias_pickle_bytes: bytes,
    meta: dict[str, Any],
    *,
    extra_pickle_keys: dict[str, Any] | None = None,
) -> None:
    """Write a dataset container (ZIP).

    *dest* may be a file path (written atomically) or a ``BytesIO``
    buffer (for in-memory use by ``export_dataset_to_file``).

    *medias_pickle_bytes* is the raw pickle blob (the ``{"medias": ...}``
    dict).  *meta* is written as ``meta.json`` inside the ZIP.

    *extra_pickle_keys* are additional top-level keys that need to live in
    the pickle payload (e.g. ``audio_dir``, ``name``).  They are merged
    into the existing pickle dict before writing.
    """
    if extra_pickle_keys:
        data = safe_pickle_load(io.BytesIO(medias_pickle_bytes))
        data.update(extra_pickle_keys)
        buf = io.BytesIO()
        pickle.dump(data, buf)
        medias_pickle_bytes = buf.getvalue()

    if isinstance(dest, io.BytesIO):
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("medias.pkl", medias_pickle_bytes)
            zf.writestr("meta.json", json.dumps(meta, indent=2))
        return

    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as raw:
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("medias.pkl", medias_pickle_bytes)
                zf.writestr("meta.json", json.dumps(meta, indent=2))
        os.replace(tmp, str(p))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_container(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a container, returning ``(data_dict, meta)``.

    Works with both ZIP containers and legacy raw pickles.  For legacy
    files, ``meta`` is synthesized from sidecars / the pickle itself.
    """
    p = Path(path)

    if not is_container(p):
        return _read_legacy(p)

    with zipfile.ZipFile(str(p), "r") as zf:
        pkl_bytes = zf.read("medias.pkl")
        data = safe_pickle_load(io.BytesIO(pkl_bytes))
        if not isinstance(data, dict) or "medias" not in data:
            raise ValueError(f"Invalid container {p.name}: pickle missing 'medias' key.")

        meta: dict[str, Any] = {}
        if "meta.json" in zf.namelist():
            meta = json.loads(zf.read("meta.json").decode("utf-8"))

    dir_keys = {k: data[k] for k in list(data) if k.endswith("_dir") and isinstance(data[k], str)}
    meta.setdefault("dir_keys", dir_keys)

    return data, meta


def read_meta(path: str | Path) -> dict[str, Any]:
    """Read just the metadata from a container without loading medias."""
    p = Path(path)
    if not is_container(p):
        return _read_legacy_meta(p)

    with zipfile.ZipFile(str(p), "r") as zf:
        if "meta.json" in zf.namelist():
            return json.loads(zf.read("meta.json").decode("utf-8"))
    return {}


def _read_legacy(p: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a legacy raw-pickle file and synthesize metadata from sidecars."""
    from vtscore.datasets.loader_pickle import (
        _read_pickle_dataset,
        read_pkl_clipper,
        read_pkl_embedder,
    )

    data = _read_pickle_dataset(p)
    meta: dict[str, Any] = {
        "embedder": read_pkl_embedder(p) or "",
        "clipper": read_pkl_clipper(p) or "",
        "name": data.get("name", ""),
    }
    return data, meta


def _read_legacy_meta(p: Path) -> dict[str, Any]:
    """Synthesize metadata from sidecar files without loading the pickle."""
    from vtscore.datasets.loader_pickle import read_pkl_clipper, read_pkl_embedder

    return {
        "embedder": read_pkl_embedder(p) or "",
        "clipper": read_pkl_clipper(p) or "",
    }


# ---------------------------------------------------------------------------
# Projection append / read (inside the container)
# ---------------------------------------------------------------------------


def append_projection(
    path: str | Path,
    projection: Any,
    pyramid: Any,
) -> None:
    """Append (or replace) the projection inside an existing container.

    For legacy raw-pickle files, falls back to a ``.projection`` sidecar.
    """
    from vtscore.projection.persistence import save_projection as _save_sidecar

    p = Path(path)
    if not is_container(p):
        _save_sidecar(p, projection, pyramid)
        return

    npz_bytes = _serialize_projection(projection, pyramid)

    with zipfile.ZipFile(str(p), "a") as zf:
        if "projection.npz" in zf.namelist():
            _rewrite_without(p, "projection.npz")
            with zipfile.ZipFile(str(p), "a") as zf2:
                zf2.writestr("projection.npz", npz_bytes)
        else:
            zf.writestr("projection.npz", npz_bytes)

    logger.info("Appended projection to container: %s", p)


def read_projection(path: str | Path) -> tuple[Any, Any] | None:
    """Read the projection + pyramid from a container, or ``None``.

    Falls back to the ``.projection`` sidecar for legacy files.
    """
    from vtscore.projection.persistence import load_projection as _load_sidecar

    p = Path(path)
    if not is_container(p):
        return _load_sidecar(p)

    try:
        with zipfile.ZipFile(str(p), "r") as zf:
            if "projection.npz" not in zf.namelist():
                return None
            npz_bytes = zf.read("projection.npz")
    except Exception:
        logger.warning("Failed to read projection from container %s", p, exc_info=True)
        return None

    return _deserialize_projection(npz_bytes)


def _serialize_projection(projection: Any, pyramid: Any) -> bytes:
    """Serialize a Projection + Pyramid to ``.npz`` bytes."""
    from vtscore.projection.persistence import _pyramid_to_meta

    meta = _pyramid_to_meta(projection, pyramid)
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        coords=projection.coords,
        ids=np.asarray(projection.ids, dtype=np.int64),
        meta=np.frombuffer(meta_bytes, dtype=np.uint8),
    )
    return buf.getvalue()


def _deserialize_projection(npz_bytes: bytes) -> tuple[Any, Any] | None:
    """Deserialize a Projection + Pyramid from ``.npz`` bytes."""
    from vtscore.projection.persistence import _rebuild_from_npz_arrays

    try:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz:
            coords = np.ascontiguousarray(npz["coords"], dtype=np.float32)
            ids = npz["ids"].tolist()
            meta_bytes = npz["meta"].tobytes()
        return _rebuild_from_npz_arrays(coords, ids, meta_bytes)
    except Exception:
        logger.warning("Failed to deserialize projection from container", exc_info=True)
        return None


def _rewrite_without(path: Path, entry_name: str) -> None:
    """Rewrite a ZIP, removing a single entry."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as raw:
            with zipfile.ZipFile(str(path), "r") as src, zipfile.ZipFile(raw, "w") as dst:
                for item in src.infolist():
                    if item.filename != entry_name:
                        dst.writestr(item, src.read(item.filename))
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
