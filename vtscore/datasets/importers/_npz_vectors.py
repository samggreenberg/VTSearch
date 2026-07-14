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

import math
from pathlib import Path
from typing import Any

import numpy as np


_FILENAMES_KEYS = ("filenames", "names", "paths", "filename")
_VECTORS_KEYS = ("vectors", "embeddings", "vecs", "embedding")
_EMBEDDER_NAME_KEYS = ("embedder_name", "embedder")
_MEMBERS_KEYS = ("members", "member")
_ARCHIVES_KEYS = ("archives", "archive", "shards", "shard")
_CLIP_START_KEYS = ("clip_start", "clip_starts", "starts", "start")
_CLIP_END_KEYS = ("clip_end", "clip_ends", "ends", "end")
_WINDOW_ID_KEYS = ("window_id", "window_ids", "windows", "window")


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


def validate_manifest_embedder_name(
    embedder_name: str, media_type_id: str, *, source_label: str = "NPZ manifest"
) -> None:
    """Reject a manifest *embedder_name* that VTSearch can't route to a real embedder.

    An NPZ manifest may name the embedder that produced its vectors.  VTSearch
    binds a dataset's text / region search slots by looking that name up in the
    embedder registry (see :func:`vtscore.embedding.binding.derive_binding`), so a
    name that is unregistered - or registered for a *different* media type - binds
    no slot, and text queries later fail with a confusing 400 ("does not support
    text queries").  Catch it here, at import, and point the user at the embedders
    that are actually valid for this media type.

    An empty / whitespace name is allowed: it simply means the manifest declares no
    embedder (the importer falls back to the media type's default).  Raises
    :class:`ValueError` for a non-empty name that is not a registered
    *media_type_id* embedder.
    """
    name = (embedder_name or "").strip()
    if not name:
        return

    from vtscore.media import embedders_for_type  # noqa: PLC0415 - avoid import cycle

    valid = [e.name for e in embedders_for_type(media_type_id)]
    if name in valid:
        return
    options = ", ".join(valid) if valid else "(none registered for this media type)"
    raise ValueError(
        f"{source_label} names embedder {name!r}, which is not a registered VTSearch "
        f"{media_type_id} embedder. Text and region search bind by embedder name, so "
        f"an unregistered name silently disables them. Set embedder_name to one of the "
        f"registered {media_type_id} embedders: {options}."
    )


def is_archive_member_manifest(npz_path: Path) -> bool:
    """Return ``True`` if *npz_path* is a no-extraction **archive-member** manifest.

    The ``Manifest`` importer accepts two shapes of ``.npz`` under one field: a
    plain *path → vector* manifest (files that live on disk) and an
    *archive-member* manifest (members packed inside tar/zip shards, streamed
    without extraction).  They are told apart by a single distinguishing array:
    an archive-member manifest carries a ``members`` (or ``member``) array,
    which a plain path manifest never has.  This lets the importer auto-detect
    the shape and dispatch to the archive-member path, so users never pick
    between two tabs for what is, to them, "a manifest of pre-computed vectors".

    Only ``.npz`` files can be archive manifests; a ``.txt`` / ``.list`` paths
    file never is.  Never raises -- returns ``False`` on a missing file, a
    non-``.npz`` suffix, or any read error.
    """
    p = Path(npz_path)
    if p.suffix.lower() != ".npz" or not p.is_file():
        return False
    try:
        with np.load(p, allow_pickle=False) as data:
            return any(k in data.files for k in _MEMBERS_KEYS)
    except Exception:
        return False


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
    * ``clip_start`` / ``clip_end`` *(optional)* - ``(N,)`` window extents in
      seconds.  When present, one member can appear in several rows, each a
      distinct **sub-file clip window** (e.g. ≈14 × 10 s CLAP windows per
      chunk); the importer fans them out into separate windowed media that the
      player seeks/loops within (display-only, no byte slicing).
    * ``window_id`` *(optional)* - ``(N,)`` per-window identifiers used to keep
      each window's synthesized content-id unique; defaults to the clip start.
    * ``embedder_name`` *(optional)* - scalar embedder name (see
      :func:`read_npz_embedder_name`).

    Returns a list of ``{"archive", "member", "filename", "vector",
    "clip_start", "clip_end", "window_id"}`` dicts (the three clip fields are
    ``None`` for an un-windowed manifest; archive paths resolved to absolute
    strings).  Raises ``ValueError`` for a malformed manifest and
    ``FileNotFoundError`` if the file is missing.
    """
    p = Path(npz_path)
    if not p.is_file():
        raise FileNotFoundError(f"NPZ file not found: {p}")

    base_dir = p.resolve().parent
    with np.load(p, allow_pickle=False) as data:
        cols = _manifest_columns(data, p)
        n = len(cols["members"])  # always present (validated in _manifest_columns)
        rows = [row for i in range(n) if (row := _manifest_row(i, cols, base_dir)) is not None]
        if not rows:
            raise ValueError(f"NPZ manifest {p} produced no rows")
        return rows


def _manifest_columns(data, p: Path) -> dict[str, Any]:
    """Resolve and validate every manifest column.

    Returns a dict with the required ``vectors`` / ``members`` / ``archives``
    columns and the optional ``filenames`` / ``clip_start`` / ``clip_end`` /
    ``window_id`` columns (``None`` when the manifest omits them).  Every
    optional column is broadcast to length *N* so a scalar can stand in for a
    whole-manifest constant.
    """
    key_set = set(data.files)
    vectors_key = next((k for k in _VECTORS_KEYS if k in key_set), None)
    members_key = next((k for k in _MEMBERS_KEYS if k in key_set), None)
    archives_key = next((k for k in _ARCHIVES_KEYS if k in key_set), None)

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

    return {
        "vectors": vecs_arr,
        "members": members_arr,
        "archives": _broadcast_column(data[archives_key], n, "archives"),
        "filenames": _optional_column(data, key_set, _FILENAMES_KEYS, n, "filenames"),
        "clip_start": _optional_column(data, key_set, _CLIP_START_KEYS, n, "clip_start"),
        "clip_end": _optional_column(data, key_set, _CLIP_END_KEYS, n, "clip_end"),
        "window_id": _optional_column(data, key_set, _WINDOW_ID_KEYS, n, "window_id"),
    }


def _optional_column(data, key_set: set, keys: tuple, n: int, label: str) -> np.ndarray | None:
    """Return a length-*n* broadcast of the first present *keys* column, else ``None``."""
    key = next((k for k in keys if k in key_set), None)
    if key is None:
        return None
    return _broadcast_column(data[key], n, label)


def _manifest_row(i: int, cols: dict, base_dir: Path) -> dict | None:
    """Build one window/member row, or ``None`` to skip an empty member."""
    member = str(cols["members"][i]).strip()
    if not member:
        return None
    archive = str(cols["archives"][i]).strip()
    if not archive:
        raise ValueError(f"manifest row {i} has an empty archive path")
    archive_path = Path(archive)
    if not archive_path.is_absolute():
        archive_path = (base_dir / archive_path).resolve()
    filenames_arr = cols["filenames"]
    display = str(filenames_arr[i]).strip() if filenames_arr is not None else ""
    return {
        "archive": str(archive_path),
        "member": member,
        "filename": display or Path(member).name,
        "vector": np.asarray(cols["vectors"][i]),
        "clip_start": _row_float(cols["clip_start"], i),
        "clip_end": _row_float(cols["clip_end"], i),
        "window_id": _row_window_id(cols["window_id"], i),
    }


def _row_float(arr: np.ndarray | None, i: int) -> float | None:
    """Return ``arr[i]`` as a float, or ``None`` when the column is absent/blank.

    ``None`` covers three "no window for this row" encodings so a manifest can
    mix windowed and whole-member rows in one float column: an absent column, a
    blank string entry, and a ``NaN`` entry.
    """
    if arr is None:
        return None
    raw = arr[i]
    if isinstance(raw, (bytes, np.bytes_, str, np.str_)) and not str(raw).strip():
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(val) else val


def _row_window_id(arr: np.ndarray | None, i: int) -> str | None:
    """Return ``arr[i]`` as a trimmed string id, or ``None`` when absent/blank."""
    if arr is None:
        return None
    val = str(arr[i]).strip()
    return val or None


def window_suffix(window_id: str | None, clip_start: float | None) -> str:
    """Return the per-window discriminator appended to a member's identity.

    A windowed import fans one member into several media, so the member name
    alone is no longer unique.  Prefer an explicit ``window_id``; otherwise key
    on the window's start time; an un-windowed (whole-member) row contributes
    the empty string.  Shared by the importer (synthesized md5 + ``origin_name``)
    and the manifest-backed media source (re-supplying a specific window's
    vector) so the two agree on a window's identity.
    """
    if window_id:
        return f"#{window_id}"
    if clip_start is not None:
        return f"@{clip_start:g}"
    return ""


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
