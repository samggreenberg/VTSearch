"""ReCaller dataset importer — import media from a ReCaller query.

Given a ReCaller *queryID* and an output media type, this importer pulls
the matching results from ReCaller, downloads their bytes via PullWrest
(PW), and fetches their pre-computed embeddings via DataWrest (DW).  MD5
hashes come straight from ReCaller — no local recalculation.

Multi-media imports
-------------------
ReCaller is a :attr:`~DatasetImporter.multi_media` importer: a single
import can pull in **multiple source types** in one go and let the
framework convert each to the dataset's output type.  The user picks
the output media type plus a list of :class:`SourceSpec` rows in the
modal, e.g.::

    output_type = "image"
    source_specs = [
        {"source_type": "image",    "converter": None,           "params": {}},
        {"source_type": "video",    "converter": "video2image",  "params": {"n_clips": "30"}},
        {"source_type": "document", "converter": "document2image", "params": {}},
    ]

The importer's job is to yield raw source-type media — one dict per
ReCaller record matching ``spec.source_type`` — from
:meth:`~ReCallerDatasetImporter.fetch_source_media`.  The framework
loops the spec list and, when the spec has a converter, runs
``converter.convert(raw, spec.params)`` on every yielded media before
storing it.  This importer **never** invokes a converter itself.

Each ingested media receives a **per-media origin** keyed by
``contentID`` (not the ephemeral ``queryID``), so labels can round-trip
through Holder and back::

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
:meth:`~ReCallerDatasetImporter.fetch_source_media` issues all DataWrest
embedding lookups and PullWrest media-byte downloads for a spec
concurrently via a thread pool, then yields the assembled media dicts in
the original ReCaller order.
"""

from __future__ import annotations

from typing import Any, Iterator

from vtscore.datasets.importers.base import DatasetImporter, ImporterField, SourceSpec
from vtscore.media import all_folder_names


# ---------------------------------------------------------------------------
# TODO(dev): Implement these client functions to talk to the real services.
# Each should be a thin wrapper around an HTTP call (requests, httpx, etc.).
# ---------------------------------------------------------------------------


def _rc_list_queries(output_type: str) -> list[str]:
    """Call ReCaller and return the list of query IDs scoped by *output_type*.

    Used to populate the ``query_id`` dropdown dynamically once the user
    picks an output media type.  Each entry is a ReCaller queryID string.
    A query may contain records of any media type; *output_type* is used
    to narrow the listing to queries the user is likely to import for.
    """
    raise NotImplementedError("TODO: implement ReCaller list-queries API client")


def _rc_fetch_results(query_id: str) -> list[dict[str, Any]]:
    """Call ReCaller and return every result for *query_id*.

    Each result dict should have at least::

        {
            "contentID": str,
            "mediaID": str,
            "media_type": str,   # e.g. "audio", "image", "video", "document"
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
# Media-dict assembly
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
    """Import a dataset from a ReCaller query, with multi-source-type support."""

    name = "recaller"
    display_name = "ReCaller Query"
    description = "Import media from a ReCaller browsing session."
    icon = "\U0001f50d"  # magnifying glass
    hidden_from_picker = True  # flip to False once API clients are implemented
    category = "services"
    multi_media = True
    fields = [
        ImporterField(
            key="media_type",
            label="Output Media Type",
            field_type="select",
            options=all_folder_names(),
            default="audio",
            description="The media type this dataset will hold.  Source-type rows below specify which ReCaller record types to pull in and how to convert them to this output type.",
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
        """Populate ``query_id`` from ReCaller, scoped by the output type."""
        if field_key == "query_id":
            output_type = current_values.get("media_type") or "audio"
            return list(_rc_list_queries(output_type))
        return super().get_field_options(field_key, current_values)

    # The dataset-level origin (queryID) is NOT useful for provenance —
    # each media gets its own origin keyed by contentID via _build_media.
    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        return {"importer": self.name, "params": {}}

    # ------------------------------------------------------------------
    # Multi-media source hook
    # ------------------------------------------------------------------

    def fetch_source_media(
        self,
        spec: SourceSpec,
        field_values: dict[str, Any],
        thin: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield raw ReCaller records matching ``spec.source_type``.

        Filters :func:`_rc_fetch_results` by ``media_type ==
        spec.source_type`` and batches DataWrest + PullWrest calls
        concurrently via a thread pool.  Yields :func:`_build_media`
        dicts in ReCaller's result order.

        The framework owns conversion: when the caller's :class:`SourceSpec`
        carries a converter, :meth:`DatasetImporter.run` passes each
        yielded media through ``converter.convert(raw, spec.params)``
        before assigning an ID and storing it.
        """
        query_id = (field_values.get("query_id") or "").strip()
        if not query_id:
            raise ValueError("A query ID is required.")

        all_results = _rc_fetch_results(query_id)
        results = [r for r in all_results if r.get("media_type") == spec.source_type]
        if not results:
            return

        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        from vtscore.concurrency.progress import update_progress  # noqa: PLC0415

        total = len(results)
        update_progress("loading", f"Fetching {total} {spec.source_type} records…", 0, total)

        max_workers = min(16, max(1, total))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            embed_infos = list(pool.map(lambda r: _dw_get_embedding(r["mediaID"]), results))
            if thin:
                byte_blobs: list[bytes | None] = [None] * total
            else:
                byte_blobs = list(pool.map(lambda r: _pw_fetch_media(r["media_url"]), results))

        for i, (rec, ei, mb) in enumerate(zip(results, embed_infos, byte_blobs)):
            update_progress("loading", f"Importing {i + 1} of {total}…", i + 1, total)
            yield _build_media(rec, spec.source_type, ei, mb, self.name)

    def origin_display(self, origin: dict[str, Any]) -> str:
        content_id = origin.get("params", {}).get("contentID", "")
        return f"recaller:{content_id}" if content_id else "recaller"

    def resolve_file(self, origin: dict[str, Any], origin_name: str = "", filename: str = "") -> None:
        # File resolution for RC media goes through the PullWrest MediaSource.
        # See vtsearch/datasets/sources/pullwrest.py
        return None


IMPORTER = ReCallerDatasetImporter()
