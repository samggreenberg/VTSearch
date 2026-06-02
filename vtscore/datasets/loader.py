"""Dataset loading and management utilities.

This module is the public façade for dataset loading.  Implementations
live in dedicated sibling modules; this file holds the shared helpers,
``export_dataset_to_file``, and re-exports so existing imports
(``from vtscore.datasets.loader import ...``) continue to work.

* :mod:`vtscore.datasets.loader_folder` - folder loaders
* :mod:`vtscore.datasets.loader_pickle` - pickle/container loaders, image embed
* :mod:`vtscore.datasets.loader_demo` - demo dataset loader

All public functions that perform I/O accept an optional ``on_progress``
callback with the signature
``(status: str, message: str, current: int, total: int) -> None``.
When omitted the functions fall back to the application-wide
:func:`~vtscore.concurrency.progress.update_progress` reporter; pass an explicit callback
to use these functions outside the Flask app.
"""

from __future__ import annotations

import hashlib
import io
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vtscore.config import EMBEDDINGS_DIR  # noqa: F401  - re-exported & patched in tests
from vtscore.datasets.metadata import (  # noqa: F401  - re-exported for consumers
    load_audio_metadata_from_folders,
    load_cifar10_batch,
    load_esc50_metadata,
    load_image_metadata_from_folders,
    load_oxford_flowers_metadata,
    load_paragraph_metadata_from_folders,
    load_places365_metadata,
    load_urbansound8k_metadata,
    load_video_metadata_from_folders,
)
from vtscore.security.pickle import (  # noqa: F401  - re-exported for consumers
    RestrictedUnpickler,
    _PICKLE_SAFE_CLASSES,
    safe_pickle_load,
)

ProgressCallback = Callable[[str, str, int, int], None]


# ---------------------------------------------------------------------------
# Shared helpers (used by loader_folder / loader_pickle / loader_demo)
# ---------------------------------------------------------------------------


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


def _streaming_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file using constant memory."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public loaders - re-exported from sibling modules
# ---------------------------------------------------------------------------

from vtscore.datasets.loader_folder import (  # noqa: E402, F401
    apply_custom_metadata_md5,
    load_dataset_from_folder,
    load_dataset_from_folder_chunked,
)
from vtscore.datasets.loader_pickle import (  # noqa: E402, F401
    load_dataset_from_pickle,
    load_dataset_from_pickle_chunked,
    read_pkl_clipper,
    read_pkl_embedder,
)
from vtscore.datasets.loader_demo import (  # noqa: E402, F401
    _stamp_demo_origin,
    load_demo_dataset,
)

# Backward-compat alias - canonical location is vtscore.converters.runner
from vtscore.converters.runner import apply_converter_to_demo as _apply_converter_to_demo  # noqa: E402, F401


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_dataset_to_file(
    medias: dict[int, dict[str, Any]],
    *,
    embedder: str = "",
    clipper: str = "",
    media_type: str = "",
    name: str = "",
    created_at: float | None = None,
    expires_at: float | None = None,
    extra_pickle_keys: dict[str, Any] | None = None,
) -> bytes:
    """Serialise the current media dataset to a ZIP container byte string.

    The container holds ``medias.pkl`` (the pickled media dict) and
    ``meta.json`` (embedder, clipper, timestamps, age-off).  Reloadable
    with :func:`load_dataset_from_pickle` which auto-detects both this
    format and legacy raw pickles.

    Args:
        medias: Mapping of media ID to media data dict.
        embedder: Name of the embedder used to produce the embeddings.
        clipper: Name of the clipper used (audio datasets).
        media_type: Media type identifier.
        name: Dataset display name.
        created_at: Unix timestamp of creation (defaults to now).
        expires_at: Unix timestamp when the dataset expires (``None`` = never).
        extra_pickle_keys: Additional top-level keys for the pickle dict
            (e.g. ``audio_dir``, ``video_dir``).

    Returns:
        Raw bytes of the ZIP container.
    """
    import time

    from vtscore.datasets.container import write_container

    data: dict[str, Any] = {
        "medias": {
            cid: {
                "id": media["id"],
                "media_type": media.get("media_type", "audio"),
                "duration": media["duration"],
                "file_size": media["file_size"],
                "md5": media["md5"],
                "embedder": media.get("embedder", ""),
                "embedding": media["embedding"].tolist()
                if isinstance(media["embedding"], np.ndarray)
                else media["embedding"],
                "filename": media.get("filename", f"media_{cid}.wav"),
                "category": media.get("category", "unknown"),
                "origin": media.get("origin"),
                "origin_name": media.get("origin_name", media.get("filename", "")),
                "media_bytes": media.get("media_bytes"),
                "media_string": media.get("media_string"),
                "media_path": media.get("media_path"),
                "word_count": media.get("word_count"),
                "character_count": media.get("character_count"),
                "width": media.get("width"),
                "height": media.get("height"),
                "custom_metadata": media.get("custom_metadata"),
            }
            for cid, media in medias.items()
        }
    }

    pkl_buf = io.BytesIO()
    pickle.dump(data, pkl_buf)
    medias_pkl_bytes = pkl_buf.getvalue()

    meta = {
        "format_version": 1,
        "embedder": embedder,
        "clipper": clipper,
        "media_type": media_type,
        "name": name,
        "created_at": created_at or time.time(),
        "expires_at": expires_at,
    }

    out_buf = io.BytesIO()
    write_container(
        out_buf,
        medias_pkl_bytes,
        meta,
        extra_pickle_keys=extra_pickle_keys,
    )
    return out_buf.getvalue()
