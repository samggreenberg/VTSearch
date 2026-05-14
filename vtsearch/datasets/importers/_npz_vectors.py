"""Helpers for reading pre-computed embedding vectors from ``.npz`` files.

The ``server_files`` and ``local_files`` importers accept an ``.npz`` archive
of pre-computed embeddings instead of (or alongside) raw media files, so
users who have already embedded their data don't have to re-embed it.

Two NPZ layouts are supported:

1. **filenames + vectors** (preferred) — Two top-level arrays,
   ``filenames`` (1-D string-like) and ``vectors`` (2-D float).  The
   i-th filename maps to the i-th row of ``vectors``.  Produced e.g. by
   ``np.savez(path, filenames=names, vectors=vecs)``.  This is the
   memory-efficient form: the vectors live in one contiguous array, and
   per-row ``np.asarray`` calls return cheap views rather than copies.
2. **per-key** — Each archive key is a filename and the corresponding
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
                raise ValueError(
                    f"NPZ '{filenames_key}' array must be 1-D, got shape {names_arr.shape}"
                )
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
