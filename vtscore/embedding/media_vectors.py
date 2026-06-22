"""Per-media embedding vector access (dict-keyed).

Phase 2 of the three-slot embedder work (``docs/plans/patch-embedder.md`` →
"V3 — design") moved a media's embedding from a single ``media["embedding"]``
ndarray to ``media["embeddings"]`` — a dict keyed by embedder name — so the
vectors of multiple bound embedders can coexist on one media.  Phase 2c
**dropped the singular mirror entirely**: ``media["embeddings"]`` is now the
*only* per-media vector store, and there is no ``media["embedding"]`` on a
live media.

This module is the one place that resolves a vector for a media; every read
site goes through :func:`media_embedding` and every write through
:func:`set_media_embedding`.  Legacy single-vector pickles (which stored only
``media["embedding"]``) are re-keyed into the dict by
:func:`ensure_embeddings_dict` on load, which then deletes the singular key.

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


def media_embedder_names(media: dict[str, Any]) -> list[str]:
    """Return every embedder name that has a vector on *media*, primary first.

    Reads the keys of ``media["embeddings"]`` (the dict-keyed form) and falls
    back to the recorded ``media["embedder"]`` when no dict is present (e.g. an
    un-embedded media whose vector hasn't been written yet).  The recorded
    primary embedder is ordered first so role derivation and primary-vector
    reads agree on which embedder leads.
    """
    embs = media.get(EMBEDDINGS_KEY)
    if isinstance(embs, dict) and embs:
        names = list(embs.keys())
        primary = media.get("embedder")
        if primary and primary in names:
            names.remove(primary)
            names.insert(0, primary)
        return names
    name = primary_embedder_name(media)
    return [name] if name else []


def init_embeddings(embedder_name: str | None, vec: Any) -> dict[str, Any]:
    """Build a fresh ``embeddings`` dict for a media being created.

    Returns ``{embedder_name: vec}`` when both are present, else an empty dict
    (the deferred-embed placeholder a creation site uses when the vector is
    filled later by the embed stage via :func:`set_media_embedding`).  This is
    the construction-time counterpart of :func:`set_media_embedding`; use it in
    media-dict literals in place of the old ``"embedding": vec`` field.
    """
    if vec is None or not embedder_name:
        return {}
    return {embedder_name: vec}


def media_embedding(media: dict[str, Any], embedder_name: str | None = None) -> Any:
    """Return the embedding vector for *embedder_name*, or the primary one.

    Resolution order:

    * ``embedder_name`` given → ``media["embeddings"][embedder_name]``, or
      ``None`` when that embedder has no vector on this media.
    * ``embedder_name`` omitted → the primary vector: the recorded embedder's
      entry in the dict, else the sole dict entry, else ``None``.

    There is no singular ``media["embedding"]`` fallback: the dict is the only
    per-media vector store (legacy pickles are re-keyed on load by
    :func:`ensure_embeddings_dict`).
    """
    embs = media.get(EMBEDDINGS_KEY)
    if not isinstance(embs, dict) or not embs:
        return None
    if embedder_name is not None:
        return embs.get(embedder_name)
    name = media.get("embedder")
    if name and name in embs:
        return embs[name]
    if len(embs) == 1:
        return next(iter(embs.values()))
    # Ambiguous (multiple entries, no recorded primary) → no primary vector.
    return None


def ensure_embeddings_dict(media: dict[str, Any]) -> None:
    """Re-key a legacy singular ``media["embedding"]`` into the dict, then drop it.

    Idempotent.  A media that already carries a non-empty ``embeddings`` dict
    has its stale singular ``embedding`` (if any) dropped.  A legacy media with
    only a singular ``embedding`` *and* a known embedder name gets a one-entry
    dict; a media whose embedder is unknown keeps no vector (no empty-string
    keys are minted).  Either way ``media["embedding"]`` is removed so the dict
    is the sole vector store afterward.
    """
    if not media.get(EMBEDDINGS_KEY):
        vec = media.get("embedding")
        name = media.get("embedder")
        if vec is not None and name:
            media[EMBEDDINGS_KEY] = {name: vec}
    media.pop("embedding", None)


def set_media_embedding(media: dict[str, Any], embedder_name: str, vec: Any) -> None:
    """Store *vec* for *embedder_name* on *media* in the per-embedder dict.

    Writes ``media["embeddings"][embedder_name]`` (creating the dict) and
    records ``media["embedder"]`` as the primary embedder name when none is set
    yet.  No singular ``media["embedding"]`` is written — the dict is the only
    vector store.
    """
    embs = media.get(EMBEDDINGS_KEY)
    if not isinstance(embs, dict):
        embs = {}
        media[EMBEDDINGS_KEY] = embs
    embs[embedder_name] = vec
    if not media.get("embedder"):
        media["embedder"] = embedder_name
