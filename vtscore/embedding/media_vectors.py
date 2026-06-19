"""Per-media embedding vector access (dict-keyed, with legacy fallback).

Phase 2 of the three-slot embedder work (``docs/plans/patch-embedder.md`` →
"V3 — design") moves a media's embedding from a single ``media["embedding"]``
ndarray to ``media["embeddings"]`` — a dict keyed by embedder name — so the
vectors of multiple bound embedders can coexist on one media.

During the migration the singular ``media["embedding"]`` is kept as the
**primary mirror / legacy fallback**: it always reflects the dataset's single
bound embedder while only one is bound, so the many read sites not yet
converted to this accessor keep working unchanged.  This module is the one
place that knows how to resolve a vector from either shape; new and converted
read sites go through :func:`media_embedding`.

Library-tier: imports only numpy, no Flask, no media registry.
"""

from __future__ import annotations

from typing import Any

EMBEDDINGS_KEY = "embeddings"


def primary_embedder_name(media: dict[str, Any]) -> str | None:
    """Return the media's primary embedder name, or ``None`` if unknown.

    Prefers the recorded ``media["embedder"]``; falls back to the first key of
    the ``embeddings`` dict when the field is absent.
    """
    name = media.get("embedder")
    if name:
        return name
    embs = media.get(EMBEDDINGS_KEY)
    if embs:
        return next(iter(embs))
    return None


def media_embedding(media: dict[str, Any], embedder_name: str | None = None) -> Any:
    """Return the embedding vector for *embedder_name*, or the primary one.

    Resolution order:

    * ``embedder_name`` given → ``media["embeddings"][embedder_name]`` when a
      dict is present; otherwise the legacy singular vector, but only if it
      belongs to that embedder (the media's recorded ``embedder`` matches or is
      unset).  Returns ``None`` when no matching vector exists.
    * ``embedder_name`` omitted → the primary vector: the recorded embedder's
      entry in the dict, else the sole dict entry, else the legacy singular
      ``media["embedding"]``.
    """
    embs = media.get(EMBEDDINGS_KEY)
    if embedder_name is not None:
        if isinstance(embs, dict) and embedder_name in embs:
            return embs[embedder_name]
        # No dict entry: the legacy singular vector only answers for the
        # media's own embedder.
        if media.get("embedder") in (None, "", embedder_name):
            return media.get("embedding")
        return None

    if isinstance(embs, dict) and embs:
        name = media.get("embedder")
        if name and name in embs:
            return embs[name]
        if len(embs) == 1:
            return next(iter(embs.values()))
        # Ambiguous (multiple entries, no recorded primary) → fall through.
    return media.get("embedding")


def ensure_embeddings_dict(media: dict[str, Any]) -> None:
    """Materialize ``media["embeddings"]`` from the legacy singular vector.

    Idempotent.  A media that already carries a non-empty ``embeddings`` dict
    is left untouched.  One with only a singular ``embedding`` *and* a known
    embedder name gets a one-entry dict; a media whose embedder is unknown
    stays singular-only (the accessor falls back to ``embedding``), so no
    empty-string keys are minted.
    """
    if media.get(EMBEDDINGS_KEY):
        return
    vec = media.get("embedding")
    name = media.get("embedder")
    if vec is None or not name:
        return
    media[EMBEDDINGS_KEY] = {name: vec}


def set_media_embedding(media: dict[str, Any], embedder_name: str, vec: Any) -> None:
    """Store *vec* for *embedder_name* on *media*, maintaining the primary mirror.

    Writes ``media["embeddings"][embedder_name]`` (creating the dict) and keeps
    the legacy singular ``media["embedding"]`` pointing at the primary embedder's
    vector so unconverted readers stay correct while only one embedder is bound.
    """
    embs = media.get(EMBEDDINGS_KEY)
    if not isinstance(embs, dict):
        embs = {}
        media[EMBEDDINGS_KEY] = embs
    embs[embedder_name] = vec
    if media.get("embedder") in (None, "", embedder_name) or media.get("embedding") is None:
        media["embedding"] = vec
        if not media.get("embedder"):
            media["embedder"] = embedder_name
