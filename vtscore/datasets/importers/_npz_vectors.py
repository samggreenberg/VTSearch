"""Helpers for reading pre-computed embedding vectors from ``.npz`` files.

The ``server_files`` and ``local_files`` importers accept an ``.npz`` archive
of pre-computed embeddings instead of (or alongside) raw media files, so
users who have already embedded their data don't have to re-embed it.

Two NPZ layouts are supported:

1. **filenames + vectors** (preferred) - Two top-level arrays,
   ``filenames`` (1-D string-like) and ``vectors`` (2-D float).  The
   i-th filename maps to the i-th row of ``vectors``.  Produced e.g. by
   ``np.savez(path, filenames=names, vectors=vecs)``.  This is the
   memory-efficient form: the vectors live in one contiguous array, and
   per-row ``np.asarray`` calls return cheap views rather than copies.
2. **per-key** - Each archive key is a filename and the corresponding
   value is its vector.  Produced e.g. by
   ``np.savez(path, **{name: vec for name, vec in zip(names, vecs)})``.
   Convenient but materialises every row into its own ndarray in a Python
   ``dict``, so it carries a noticeable memory overhead at scale (roughly
   ~450 MB resident for 100k rows of 1152-dim ``float32``).  Fine for
   ~10k-row archives; prefer layout 1 above that.

The standard layout is tried first; if neither expected key is present
the per-key layout is assumed.  ``allow_pickle`` is disabled to avoid
loading arbitrary Python objects from untrusted archives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


_FILENAMES_KEYS = ("filenames", "names", "paths", "filename")
_VECTORS_KEYS = ("vectors", "embeddings", "vecs", "embedding")
_EMBEDDER_NAME_KEYS = ("embedder_name", "embedder")
_MEMBERS_KEYS = ("members", "member")
_ARCHIVES_KEYS = ("archives", "archive", "shards", "shard")


def read_npz_embedder_name(npz_path: Path) -> str:
    """Return the embedder name stored in *npz_path*, or ``""`` if absent.

    Checks for a scalar or 1-element string array under the keys
    ``"embedder_name"`` or ``"embedder"``.  Returns ``""`` for any archive
    that does not contain such a key or where the stored value is blank.
    Does not raise; returns ``""`` on any read error.
    """
    p = Path(npz_path)
    if not p.is_file():
        return ""
    try:
        with np.load(p, allow_pickle=False) as data:
            for key in _EMBEDDER_NAME_KEYS:
                if key in data.files:
                    val = data[key]
                    raw = str(val) if val.ndim == 0 else str(val.flat[0])
                    return raw.strip()
    except Exception:
        pass
    return ""


def read_npz_filenames_and_vectors(npz_path: Path) -> dict[str, np.ndarray]:
    """Return a ``{filename: vector}`` mapping read from *npz_path*.

    Preserves insertion order (NumPy ``.npz`` preserves key order).
    Raises ``FileNotFoundError`` if the file does not exist and
    ``ValueError`` for malformed archives.
    """
    p = Path(npz_path)
    if not p.is_file():
        raise FileNotFoundError(f"NPZ file not found: {p}")

    with np.load(p, allow_pickle=False) as data:
        keys = list(data.files)
        key_set = set(keys)

        filenames_key = next((k for k in _FILENAMES_KEYS if k in key_set), None)
        vectors_key = next((k for k in _VECTORS_KEYS if k in key_set), None)

        if filenames_key and vectors_key:
            names_arr = data[filenames_key]
            vecs_arr = data[vectors_key]
            if names_arr.ndim != 1:
                raise ValueError(f"NPZ '{filenames_key}' array must be 1-D, got shape {names_arr.shape}")
            if len(vecs_arr) != len(names_arr):
                raise ValueError(
                    f"NPZ '{filenames_key}' and '{vectors_key}' have mismatched lengths "
                    f"({len(names_arr)} vs {len(vecs_arr)})"
                )
            mapping: dict[str, np.ndarray] = {}
            for i, raw_name in enumerate(names_arr):
                name = str(raw_name).strip()
                if not name:
                    continue
                mapping[name] = np.asarray(vecs_arr[i])
            if not mapping:
                raise ValueError(f"NPZ '{filenames_key}' array is empty")
            return mapping

        # Per-key layout: every archive key is a filename.
        if not keys:
            raise ValueError(f"NPZ archive {p} is empty")
        mapping = {}
        for k in keys:
            mapping[k] = np.asarray(data[k])
        return mapping


def read_npz_archive_member_rows(npz_path: Path) -> list[dict]:
    """Read archive-member rows from an ``.npz`` manifest.

    This is the no-extraction counterpart of
    :func:`read_npz_filenames_and_vectors`: instead of mapping a filesystem
    path to a vector, each row references a **member inside a tar/zip shard**
    plus its pre-computed embedding, so a filtered subset of a WebDataset-style
    corpus imports without ever extracting the shards.

    Expected arrays (NumPy ``.npz``, ``allow_pickle=False``):

    * ``vectors`` / ``embeddings`` - ``(N, D)`` float embeddings, one per row.
    * ``members`` - ``(N,)`` member names within their archive.
    * ``archives`` - ``(N,)`` archive paths, **or** a single scalar/1-element
      value applied to every row (one-shard manifests).  Relative paths are
      resolved against the manifest's directory.
    * ``filenames`` *(optional)* - ``(N,)`` display names; defaults to the
      member's basename.
    * ``embedder_name`` *(optional)* - scalar embedder name (see
      :func:`read_npz_embedder_name`).

    Returns a list of ``{"archive", "member", "filename", "vector"}`` dicts
    (archive paths resolved to absolute strings).  Raises ``ValueError`` for a
    malformed manifest and ``FileNotFoundError`` if the file is missing.
    """
    p = Path(npz_path)
    if not p.is_file():
        raise FileNotFoundError(f"NPZ file not found: {p}")

    base_dir = p.resolve().parent
    with np.load(p, allow_pickle=False) as data:
        vecs_arr, members_arr, archives_arr, filenames_arr = _manifest_columns(data, p)
        rows = [
            row
            for i in range(len(members_arr))
            if (row := _manifest_row(i, members_arr, archives_arr, filenames_arr, vecs_arr, base_dir)) is not None
        ]
        if not rows:
            raise ValueError(f"NPZ manifest {p} produced no rows")
        return rows


def _manifest_columns(data, p: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Resolve and validate the ``(vectors, members, archives, filenames?)`` columns."""
    key_set = set(data.files)
    vectors_key = next((k for k in _VECTORS_KEYS if k in key_set), None)
    members_key = next((k for k in _MEMBERS_KEYS if k in key_set), None)
    archives_key = next((k for k in _ARCHIVES_KEYS if k in key_set), None)
    filenames_key = next((k for k in _FILENAMES_KEYS if k in key_set), None)

    if vectors_key is None or members_key is None:
        raise ValueError(
            f"NPZ manifest {p} must contain a vectors array (one of {_VECTORS_KEYS}) "
            f"and a members array (one of {_MEMBERS_KEYS})"
        )

    vecs_arr = data[vectors_key]
    members_arr = np.atleast_1d(data[members_key])
    if members_arr.ndim != 1:
        raise ValueError(f"NPZ '{members_key}' array must be 1-D, got shape {members_arr.shape}")
    n = len(members_arr)
    if len(vecs_arr) != n:
        raise ValueError(f"NPZ '{members_key}' and '{vectors_key}' have mismatched lengths ({n} vs {len(vecs_arr)})")
    if archives_key is None:
        raise ValueError(f"NPZ manifest {p} must contain an archives array (one of {_ARCHIVES_KEYS})")
    archives_arr = _broadcast_column(data[archives_key], n, "archives")
    filenames_arr = _broadcast_column(data[filenames_key], n, "filenames") if filenames_key is not None else None
    return vecs_arr, members_arr, archives_arr, filenames_arr


def _manifest_row(i, members_arr, archives_arr, filenames_arr, vecs_arr, base_dir: Path) -> dict | None:
    """Build one ``{archive, member, filename, vector}`` row, or ``None`` to skip."""
    member = str(members_arr[i]).strip()
    if not member:
        return None
    archive = str(archives_arr[i]).strip()
    if not archive:
        raise ValueError(f"manifest row {i} has an empty archive path")
    archive_path = Path(archive)
    if not archive_path.is_absolute():
        archive_path = (base_dir / archive_path).resolve()
    display = str(filenames_arr[i]).strip() if filenames_arr is not None else ""
    return {
        "archive": str(archive_path),
        "member": member,
        "filename": display or Path(member).name,
        "vector": np.asarray(vecs_arr[i]),
    }


def _broadcast_column(arr: np.ndarray, n: int, label: str) -> np.ndarray:
    """Return a length-*n* 1-D view of *arr*, broadcasting a scalar/1-element.

    A manifest whose every row shares one archive may store ``archives`` as a
    single scalar (or 1-element array); this expands it to one entry per row.
    Raises ``ValueError`` for any other length mismatch.
    """
    flat = np.atleast_1d(arr)
    if flat.ndim != 1:
        raise ValueError(f"NPZ '{label}' array must be 1-D, got shape {flat.shape}")
    if len(flat) == n:
        return flat
    if len(flat) == 1:
        return np.repeat(flat, n)
    raise ValueError(f"NPZ '{label}' has length {len(flat)}, expected {n} or 1")
