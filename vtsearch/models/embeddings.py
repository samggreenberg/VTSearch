"""Embedding generation — delegates to the embedder registry.

The actual embedding logic lives inside each
:class:`~vtsearch.media.base.MediaEmbedder` implementation.  This module keeps
its original public API as thin wrappers so that existing callers
(``datasets/loader.py``, ``routes/sorting.py``, etc.) continue to work
without modification.
"""

from pathlib import Path
from typing import Optional

import numpy as np


def _get_embedder_for_media_type(media_type: str):
    """Return the first registered embedder for *media_type*, or None."""
    from vtsearch.media import embedders_for_type

    avail = embedders_for_type(media_type)
    return avail[0] if avail else None


def embed_audio_file(audio_path: Path) -> Optional[np.ndarray]:
    """Generate a CLAP audio embedding for *audio_path*."""
    emb = _get_embedder_for_media_type("audio")
    return emb.embed_media(audio_path) if emb else None


def embed_video_file(video_path: Path) -> Optional[np.ndarray]:
    """Generate an X-CLIP video embedding for *video_path*."""
    emb = _get_embedder_for_media_type("video")
    return emb.embed_media(video_path) if emb else None


def embed_image_file(image_path: Path) -> Optional[np.ndarray]:
    """Generate a CLIP image embedding for *image_path*."""
    emb = _get_embedder_for_media_type("image")
    return emb.embed_media(image_path) if emb else None


def embed_paragraph_file(text_path: Path) -> Optional[np.ndarray]:
    """Generate an E5-base-v2 embedding for *text_path*."""
    emb = _get_embedder_for_media_type("text")
    return emb.embed_media(text_path) if emb else None


def embed_text_query(text: str, media_type: str, enrich: bool = False, embedder_name: str = "") -> Optional[np.ndarray]:
    """Embed *text* in the vector space of the given *media_type* (or specific *embedder_name*).

    When *embedder_name* is provided, uses that specific embedder.  Otherwise
    falls back to the first registered embedder for the media type.
    """
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

    if enrich:
        return emb.embed_text_enriched(text)
    return emb.embed_text(text)
