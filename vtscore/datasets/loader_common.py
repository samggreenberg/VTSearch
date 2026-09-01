"""Shared helpers for the dataset loaders.

A dependency-free leaf that :mod:`vtscore.datasets.loader` and each of its
implementation siblings (``loader_folder`` / ``loader_pickle`` /
``loader_demo``) import.  It exists so the façade can import its siblings at
the *top* of the file: when these helpers lived in ``loader.py`` itself, every
sibling imported back up into the partially-initialised façade, and the only
way to make that work was to push the façade's own imports to the bottom of
the file behind ``# noqa: E402``.

Nothing here performs I/O or touches the media registry; the module-level
imports are deliberately limited to numpy so importing it can never pull in a
cycle.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

#: Signature of the progress reporter every public loader accepts:
#: ``(status, message, current, total) -> None``.
ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    """Lazily resolve the progress callback for the current thread.

    Checks for a per-thread callback first (set during parallel dataset
    loading) and falls back to the global singleton.
    """
    from vtscore.concurrency.progress import get_thread_progress

    cb = get_thread_progress()
    if cb is not None:
        return cb
    from vtscore.concurrency.progress import update_progress

    return update_progress


def _pop_md5_key(d: dict[str, Any]) -> str:
    """Pop and return the MD5 value from *d*, trying both ``"md5"`` and ``"MD5"`` keys.

    Returns the value (or ``""`` if neither key is present) and removes the
    matched key from *d* so it doesn't leak into downstream metadata.
    """
    for key in ("md5", "MD5"):
        val = d.get(key)
        if val:
            del d[key]
            return val
    return ""


def _get_md5_value(d: dict[str, Any]) -> str:
    """Return the MD5 value from *d*, trying both ``"md5"`` and ``"MD5"`` keys.

    Unlike :func:`_pop_md5_key` this does **not** mutate *d*.
    """
    return d.get("md5") or d.get("MD5") or ""


def _get_embedding_value(d: dict[str, Any]) -> Any:
    """Return the embedding value from *d* without mutating it.

    Returns ``None`` when the key is absent.
    """
    return d.get("embedding")


def _embedding_for_pickle(embedding: Any) -> np.ndarray | None:
    """Coerce an embedding to a compact ``float32`` ndarray for serialisation.

    Storing the vector as a contiguous ``float32`` array (rather than the old
    Python ``list`` of boxed floats) roughly halves its pickled footprint and
    avoids reconstructing hundreds of ``PyFloat`` objects per media on load —
    the load side already runs every embedding through ``l2_normalize`` /
    ``np.asarray``, so it accepts arrays and legacy lists alike.
    """
    if embedding is None:
        return None
    return np.ascontiguousarray(embedding, dtype=np.float32)


def _embeddings_dict_for_pickle(embeddings: Any) -> dict[str, np.ndarray] | None:
    """Coerce a per-embedder ``{name: vector}`` map to compact ``float32`` arrays.

    Returns ``None`` when there is nothing to store (no dict / empty), so a
    legacy single-vector media adds no key to the pickle and a v3 media carries
    one entry per bound embedder.  Entries whose vector coerces to ``None`` are
    dropped.
    """
    if not isinstance(embeddings, dict) or not embeddings:
        return None
    out: dict[str, np.ndarray] = {}
    for name, vec in embeddings.items():
        coerced = _embedding_for_pickle(vec)
        if name and coerced is not None:
            out[name] = coerced
    return out or None
