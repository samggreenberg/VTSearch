"""Embedding generation — delegates to the embedder registry.

The actual embedding logic lives inside each
:class:`~vtsearch.media.base.MediaEmbedder` implementation.  This module keeps
its original public API as thin wrappers so that existing callers
(``datasets/loader.py``, ``routes/sorting.py``, etc.) continue to work
without modification.
"""

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from vtsearch.media.embedder import media_from_path


def _get_embedder_for_media_type(media_type: str):
    """Return the first registered embedder for *media_type*, or None."""
    from vtsearch.media import embedders_for_type

    avail = embedders_for_type(media_type)
    return avail[0] if avail else None


def embed_audio_file(audio_path: Path) -> Optional[np.ndarray]:
    """Generate a CLAP audio embedding for *audio_path*."""
    emb = _get_embedder_for_media_type("audio")
    return emb.embed_media(media_from_path(audio_path)) if emb else None


def embed_video_file(video_path: Path) -> Optional[np.ndarray]:
    """Generate an X-CLIP video embedding for *video_path*."""
    emb = _get_embedder_for_media_type("video")
    return emb.embed_media(media_from_path(video_path)) if emb else None


def embed_image_file(image_path: Path) -> Optional[np.ndarray]:
    """Generate a CLIP image embedding for *image_path*."""
    emb = _get_embedder_for_media_type("image")
    return emb.embed_media(media_from_path(image_path)) if emb else None


def embed_paragraph_file(text_path: Path) -> Optional[np.ndarray]:
    """Generate an E5-base-v2 embedding for *text_path*."""
    emb = _get_embedder_for_media_type("text")
    return emb.embed_media(media_from_path(text_path)) if emb else None


# In-memory LRU cache of recently-embedded text queries. Keyed by
# (embedder_name, media_type, enrich, text); the embedder name is included
# because the same string embedded by CLIP vs. SigLIP lands in different
# vector spaces. Caching avoids re-running the text encoder when the user
# toggles sort modes or re-submits the same query on a different dataset
# that shares the embedder. Vectors are never persisted (see CLAUDE.md
# "No Persisted Vectors or MLPs") — this lives for the process lifetime.
_QUERY_CACHE_MAXSIZE = 32
_query_cache: "OrderedDict[tuple[str, str, bool, str], np.ndarray]" = OrderedDict()
_query_cache_lock = Lock()


def _query_cache_get(key: tuple[str, str, bool, str]) -> Optional[np.ndarray]:
    with _query_cache_lock:
        vec = _query_cache.get(key)
        if vec is not None:
            _query_cache.move_to_end(key)
        return vec


def _query_cache_put(key: tuple[str, str, bool, str], vec: np.ndarray) -> None:
    with _query_cache_lock:
        _query_cache[key] = vec
        _query_cache.move_to_end(key)
        while len(_query_cache) > _QUERY_CACHE_MAXSIZE:
            _query_cache.popitem(last=False)


def clear_text_query_cache() -> None:
    """Drop all cached query embeddings (test helper / manual reset)."""
    with _query_cache_lock:
        _query_cache.clear()


def embed_text_query(text: str, media_type: str, enrich: bool = False, embedder_name: str = "") -> Optional[np.ndarray]:
    """Embed *text* in the vector space of the given *media_type* (or specific *embedder_name*).

    When *embedder_name* is provided, uses that specific embedder.  Otherwise
    falls back to the first registered embedder for the media type.

    Results are cached in a small in-memory LRU keyed by
    ``(embedder_name, media_type, enrich, text)`` so repeated queries
    (e.g. switching sort modes and back, or re-submitting the same search)
    skip the text encoder.
    """
    cache_key = (embedder_name, media_type, bool(enrich), text)
    cached = _query_cache_get(cache_key)
    if cached is not None:
        return cached

    if embedder_name:
        from vtsearch.media import get_embedder

        try:
            emb = get_embedder(embedder_name)
        except KeyError:
            return None
    else:
        emb = _get_embedder_for_media_type(media_type)

    if emb is None:
        return None

    vec = emb.embed_text_enriched(text) if enrich else emb.embed_text(text)
    if vec is None:
        return None

    _query_cache_put(cache_key, vec)
    return vec
