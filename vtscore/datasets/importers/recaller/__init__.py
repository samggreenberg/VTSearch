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

Bulk fetch
----------
This importer overrides
:meth:`~vtscore.datasets.importers.base.DatasetImporter._fetch_records_bulk_impl`
to issue DataWrest embedding lookups and PullWrest media-byte downloads
concurrently via a thread pool.  The per-record :meth:`fetch_record` is
kept as a serial fallback (and as the simplest possible reference
implementation).  Importers built on top of this base get the same
per-record / bulk-record split: implement :meth:`fetch_record` for a
working baseline, override :meth:`_fetch_records_bulk_impl` when the
backing service can be batched.
"""

from __future__ import annotations

from typing import Any

from vtscore.datasets.importers.base import DatasetImporter, ImporterField
from vtscore.media import all_folder_names


# ---------------------------------------------------------------------------
# TODO(dev): Implement these client functions to talk to the real services.
# Each should be a thin wrapper around an HTTP call (requests, httpx, etc.).
# ---------------------------------------------------------------------------


def _rc_list_queries(media_type: str) -> list[str]:
    """Call ReCaller and return the list of query IDs for *media_type*.

    Used to populate the ``query_id`` dropdown dynamically once the user
    picks a media type.  Each entry is a ReCaller queryID string.
    """
    raise NotImplementedError("TODO: implement ReCaller list-queries API client")


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
# Media-dict assembly (shared by per-item and bulk paths)
# ---------------------------------------------------------------------------


def _build_media(
    record: dict[str, Any],
    media_type: str,
    embedding_info: dict[str, Any],
    media_bytes: bytes | None,
    importer_name: str,
) -> dict[str, Any]:
    content_id = record["contentID"]
    media_id = record["mediaID"]
    media_url = record["media_url"]
    origin = {
        "importer": importer_name,
        "params": {
            "contentID": content_id,
            "mediaID": media_id,
            "media_url": media_url,
            "media_type": media_type,
        },
    }
    return {
        "type": media_type,
        "filename": content_id,
        "md5": record["md5"],
        "embedding": embedding_info["embedding"],
        "embedder": embedding_info["embedder"],
        "media_bytes": media_bytes,
        "media_path": None,
        "media_url": media_url,
        "media_string": None,
        "file_size": len(media_bytes) if media_bytes else 0,
        "duration": 0,
        "category": "",
        "origin": origin,
        "origin_name": content_id,
        "custom_metadata": {
            "contentID": content_id,
            "mediaID": media_id,
            "media_url": media_url,
        },
    }


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
    category = "services"
    # ReCaller queries return a single ``media_type`` selected by the
    # user — the import flow does not pull in additional source types
    # via converters.  Flag set to keep the in-tree importer set
    # uniformly off the legacy shim; the existing ``list_records``
    # implementation reads ``field_values["media_type"]`` directly.
    multi_media = True
    fields = [
        ImporterField(
            key="media_type",
            label="Dataset MediaType",
            field_type="select",
            options=all_folder_names(),
            default="audio",
            description="Only results matching this media type will be imported.",
        ),
        ImporterField(
            key="query_id",
            label="Query ID",
            field_type="select",
            description="The ReCaller query referencing a recent browsing session.",
            dynamic_options=True,
            depends_on=["media_type"],
        ),
    ]

    def get_field_options(self, field_key: str, current_values: dict[str, Any]) -> list[str]:
        """Populate ``query_id`` from ReCaller, scoped by the picked media type."""
        if field_key == "query_id":
            media_type = current_values.get("media_type") or "audio"
            return list(_rc_list_queries(media_type))
        return super().get_field_options(field_key, current_values)

    # The dataset-level origin (queryID) is NOT useful for provenance —
    # each media gets its own origin keyed by contentID via fetch_record.
    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        return {"importer": self.name, "params": {}}

    # ------------------------------------------------------------------
    # Bulk import hooks
    # ------------------------------------------------------------------

    def list_records(self, field_values: dict[str, Any]) -> list[dict[str, Any]]:
        query_id = (field_values.get("query_id") or "").strip()
        if not query_id:
            raise ValueError("A query ID is required.")
        media_type = field_values.get("media_type", "audio")

        all_results = _rc_fetch_results(query_id)
        results = [r for r in all_results if r.get("media_type") == media_type]
        if not results:
            raise ValueError(f"No results of type '{media_type}' found for query '{query_id}'.")
        return results

    def fetch_record(
        self,
        record: dict[str, Any],
        field_values: dict[str, Any],
        thin: bool = False,
    ) -> dict[str, Any]:
        """Fetch a single ReCaller record into a media dict.

        Per-item path: one DataWrest call for the embedding plus (in
        non-thin mode) one PullWrest call for the bytes.  Override
        :meth:`_fetch_records_bulk_impl` to batch these across many
        records concurrently.
        """
        media_type = field_values.get("media_type", "audio")
        embedding_info = _dw_get_embedding(record["mediaID"])
        media_bytes = None if thin else _pw_fetch_media(record["media_url"])
        return _build_media(record, media_type, embedding_info, media_bytes, self.name)

    def _fetch_records_bulk_impl(
        self,
        records: list[dict[str, Any]],
        field_values: dict[str, Any],
        thin: bool = False,
    ) -> list[dict[str, Any] | None]:
        """Concurrent bulk fetch.

        Issues all DataWrest embedding lookups and PullWrest media-byte
        downloads in parallel via a thread pool, then assembles the media
        dicts in the original order.  Each record still goes through
        :func:`_build_media`, so the per-item shape stays identical to
        :meth:`fetch_record`.
        """
        from concurrent.futures import ThreadPoolExecutor

        from vtscore.concurrency.progress import update_progress

        media_type = field_values.get("media_type", "audio")
        total = len(records)
        update_progress("loading", f"Fetching {total} ReCaller records…", 0, total)

        max_workers = min(16, max(1, total))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            embed_infos = list(pool.map(lambda r: _dw_get_embedding(r["mediaID"]), records))
            if thin:
                byte_blobs: list[bytes | None] = [None] * total
            else:
                byte_blobs = list(pool.map(lambda r: _pw_fetch_media(r["media_url"]), records))

        results: list[dict[str, Any] | None] = []
        for i, (rec, ei, mb) in enumerate(zip(records, embed_infos, byte_blobs)):
            update_progress("loading", f"Importing {i + 1} of {total}…", i + 1, total)
            results.append(_build_media(rec, media_type, ei, mb, self.name))
        return results

    def origin_display(self, origin: dict[str, Any]) -> str:
        content_id = origin.get("params", {}).get("contentID", "")
        return f"recaller:{content_id}" if content_id else "recaller"

    def resolve_file(self, origin: dict[str, Any], origin_name: str = "", filename: str = "") -> None:
        # File resolution for RC media goes through the PullWrest MediaSource.
        # See vtsearch/datasets/sources/pullwrest.py
        return None


IMPORTER = ReCallerDatasetImporter()
