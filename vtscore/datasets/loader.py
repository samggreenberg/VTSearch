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

import io
import pickle
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

# Pickle protocol 5 (PEP 574) serialises numpy arrays via out-of-band buffers,
# making it both smaller and faster than the interpreter default (4 on 3.11).
# Available on every Python we support (>=3.10), so pin it explicitly.
_PICKLE_PROTOCOL = 5


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
    on_stage: Callable[[str], None] | None = None,
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
        on_stage: Optional callback fired with a human-readable message
            before each internal stage (build dict / pickle / package), so
            a caller can keep a progress bar moving during serialisation.

    Returns:
        Raw bytes of the ZIP container.
    """
    import time

    from vtscore.datasets.container import write_container

    if on_stage:
        on_stage("Serializing dataset…")

    # Per-type extra fields (e.g. the image/audio/video ``thumbnail_bytes``)
    # are declared by each media type's ``pickle_extra_fields`` and copied back
    # on load; the export side must write them too or they silently drop out of
    # the round-trip.  Build the type→fields map once and merge each media's
    # extra fields into its serialized entry below.
    from vtscore.media import all_types  # noqa: PLC0415

    extra_fields_by_type: dict[str, list[str]] = {mt.type_id: mt.pickle_extra_fields for mt in all_types()}

    def _serialize_media(cid: int, media: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "id": media["id"],
            "media_type": media.get("media_type", "audio"),
            "duration": media["duration"],
            "file_size": media["file_size"],
            "md5": media["md5"],
            "embedder": media.get("embedder", ""),
            # Per-embedder vectors (v3 three-slot model) are the sole vector
            # store: a single-embedder dataset writes a one-entry dict that the
            # load side re-keys back; a text+patch dataset writes both.  There
            # is no singular ``embedding`` key (Phase 2c dropped the mirror).
            "embeddings": _embeddings_dict_for_pickle(media.get("embeddings")),
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
        for field in extra_fields_by_type.get(media.get("media_type", ""), ()):
            entry[field] = media.get(field)
        # Persist a precomputed grid/list thumbnail so reloads stream the bytes
        # instead of decoding the full-resolution original on every cold tile
        # fetch (the browse first-paint delay).  Image *demos* never generate
        # ``thumbnail_bytes`` at build time and ``_write_demo_cache`` strips
        # them, so any image media still missing one at save gets it generated
        # here from the in-memory source bytes -- a one-time, offline cost.
        if media.get("media_type") == "image" and not entry.get("thumbnail_bytes") and entry.get("media_bytes"):
            from vtscore.media.image.thumbnail import make_image_thumbnail  # noqa: PLC0415

            thumb = make_image_thumbnail(entry["media_bytes"])
            if thumb is not None:
                entry["thumbnail_bytes"] = thumb[0]
        return entry

    data: dict[str, Any] = {"medias": {cid: _serialize_media(cid, media) for cid, media in medias.items()}}
    # Merge the extra top-level keys (e.g. ``diversity_tree``, ``audio_dir``)
    # into the dict *before* the single pickle.dump, so ``write_container``
    # doesn't have to unpickle+update+re-pickle the whole (embedding-heavy)
    # blob a second time — the container save then pickles exactly once.
    data.update(extra_pickle_keys or {})

    pkl_buf = io.BytesIO()
    pickle.dump(data, pkl_buf, protocol=_PICKLE_PROTOCOL)
    medias_pkl_bytes = pkl_buf.getvalue()

    # Role-typed binding slots (v3): derived from the embedders actually
    # present on the medias, so a text+patch dataset records both.  The legacy
    # singular ``embedder`` is kept for older readers and the dashboard.  On
    # reload the binding is re-derived from each media's ``embeddings`` keys, so
    # these meta fields are informational (read_meta / external tooling), not
    # the load-time source of truth.
    from vtscore.embedding.binding import derive_binding_from_names  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedder_names  # noqa: PLC0415

    text_embedder = patch_embedder = structural_embedder = None
    if medias:
        first = next(iter(medias.values()))
        text_embedder, patch_embedder, structural_embedder = derive_binding_from_names(media_embedder_names(first))

    meta = {
        "format_version": 1,
        "embedder": embedder,
        "text_embedder": text_embedder,
        "patch_embedder": patch_embedder,
        "structural_embedder": structural_embedder,
        "clipper": clipper,
        "media_type": media_type,
        "name": name,
        "created_at": created_at or time.time(),
        "expires_at": expires_at,
    }

    if on_stage:
        on_stage("Packaging dataset…")
    out_buf = io.BytesIO()
    write_container(
        out_buf,
        medias_pkl_bytes,
        meta,
        extra_pickle_keys=None,
    )
    return out_buf.getvalue()
