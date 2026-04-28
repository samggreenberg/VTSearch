"""ReCaller dataset importer — import media from a ReCaller query.

Given a ReCaller *queryID* and a *mediaType*, this importer:

1. Calls ReCaller to fetch all results for the query.
2. Filters results to the requested media type.
3. Uses PullWrest (PW) to download media files (skipped in thin mode).
4. Uses DataWrest (DW) to obtain pre-computed embeddings and embedder name.
5. Uses the MD5 hashes supplied by ReCaller (no local recalculation).

Each media item receives a **per-media origin** keyed by ``contentID``
(not the ephemeral ``queryID``), so labels can round-trip through Holder
and back::

    {
        "importer": "recaller",
        "params": {
            "contentID": "...",
            "mediaID": "...",
            "media_url": "...",
            "media_type": "audio",
        }
    }

The importer also stores ``{contentID, mediaID, media_url}`` in each
media's ``custom_metadata`` so they are visible in enriched label exports
and available to the Holder exporter.

Thin mode
---------
When ``thin=True``, PullWrest downloads are skipped entirely.  Each media
stores ``media_url`` instead of ``media_bytes``/``media_path``, relying on
VTSearch's URL-based lazy-fetch in :meth:`MediaType._resolve_media_bytes`.
Embeddings still come from DataWrest, so sorting and scoring work without
downloading the actual media.
"""

from __future__ import annotations

from typing import Any

from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.media import all_folder_names


# ---------------------------------------------------------------------------
# TODO(dev): Implement these client functions to talk to the real services.
# Each should be a thin wrapper around an HTTP call (requests, httpx, etc.).
# ---------------------------------------------------------------------------


def _rc_fetch_results(query_id: str) -> list[dict[str, Any]]:
    """Call ReCaller and return the result list for *query_id*.

    Each result dict should have at least::

        {
            "contentID": str,
            "mediaID": str,
            "media_type": str,   # e.g. "audio", "image"
            "media_url": str,    # PullWrest-resolvable URL
            "md5": str,          # hex digest from ReCaller
            ...                  # any other RC fields
        }
    """
    raise NotImplementedError("TODO: implement ReCaller API client")


def _dw_get_embedding(media_id: str) -> dict[str, Any]:
    """Call DataWrest and return embedding info for *media_id*.

    Expected return::

        {
            "embedding": numpy.ndarray,   # 1-D float32 vector
            "embedder": str,              # VTSearch-registered embedder name
        }
    """
    raise NotImplementedError("TODO: implement DataWrest API client")


def _pw_fetch_media(media_url: str) -> bytes:
    """Call PullWrest to download the raw media bytes for *media_url*."""
    raise NotImplementedError("TODO: implement PullWrest API client")


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------


class ReCallerDatasetImporter(DatasetImporter):
    """Import a dataset from a ReCaller query."""

    name = "recaller"
    display_name = "ReCaller Query"
    description = "Import media from a ReCaller browsing session."
    icon = "\U0001f50d"  # magnifying glass
    hidden_from_picker = True  # flip to False once API clients are implemented
    fields = [
        ImporterField(
            key="query_id",
            label="Query ID",
            field_type="text",
            description="The ReCaller query ID referencing a recent browsing session.",
        ),
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            options=all_folder_names(),
            default="audio",
            description="Only results matching this media type will be imported.",
        ),
    ]

    # The dataset-level origin (queryID) is NOT useful for provenance —
    # each media gets its own origin keyed by contentID.  Return an empty
    # origin here; run() sets per-media origins.
    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        return {"importer": self.name, "params": {}}

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        query_id = (field_values.get("query_id") or "").strip()
        if not query_id:
            raise ValueError("A query ID is required.")
        media_type = field_values.get("media_type", "audio")

        # 1. Fetch and filter RC results
        all_results = _rc_fetch_results(query_id)
        results = [r for r in all_results if r.get("media_type") == media_type]
        if not results:
            raise ValueError(f"No results of type '{media_type}' found for query '{query_id}'.")

        # 2. Build media dicts
        for i, rc in enumerate(results, start=1):
            content_id = rc["contentID"]
            media_id = rc["mediaID"]
            media_url = rc["media_url"]
            md5 = rc["md5"]

            # --- Embedding from DataWrest ---
            dw = _dw_get_embedding(media_id)
            embedding = dw["embedding"]
            embedder = dw["embedder"]

            # --- Media bytes (skip in thin mode) ---
            media_bytes = None
            media_path = None
            if not thin:
                media_bytes = _pw_fetch_media(media_url)
                # Optionally write to a temp folder and set media_path:
                # media_path = _save_to_temp(media_bytes, content_id, media_type)

            # --- Per-media origin (NOT queryID) ---
            origin = {
                "importer": self.name,
                "params": {
                    "contentID": content_id,
                    "mediaID": media_id,
                    "media_url": media_url,
                    "media_type": media_type,
                },
            }

            # --- Custom metadata (visible in enriched label exports) ---
            custom_metadata = {
                "contentID": content_id,
                "mediaID": media_id,
                "media_url": media_url,
            }

            medias[i] = {
                "id": i,
                "type": media_type,
                "filename": content_id,  # use contentID as the filename key
                "md5": md5,
                "embedding": embedding,
                "embedder": embedder,
                "media_bytes": media_bytes,
                "media_path": media_path,
                "media_url": media_url,  # URL-based lazy-fetch fallback
                "media_string": None,
                "file_size": len(media_bytes) if media_bytes else 0,
                "duration": 0,
                "category": "",
                "origin": origin,
                "origin_name": content_id,
                "custom_metadata": custom_metadata,
            }

    def origin_display(self, origin: dict[str, Any]) -> str:
        content_id = origin.get("params", {}).get("contentID", "")
        return f"recaller:{content_id}" if content_id else "recaller"

    def resolve_file(self, origin: dict[str, Any], origin_name: str = "", filename: str = "") -> None:
        # File resolution for RC media goes through the PullWrest MediaSource.
        # See vtsearch/datasets/sources/pullwrest.py
        return None


IMPORTER = ReCallerDatasetImporter()
