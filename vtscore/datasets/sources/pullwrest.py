"""PullWrest media source - resolve media files via the PullWrest service.

This source handles media items whose origin is ``"recaller"``.  When
VTSearch needs to resolve a file (e.g. for cross-dataset label matching
or re-embedding), this source calls PullWrest to download the media to a
temporary directory, then returns the local path.

Registration
------------
The ``SOURCE`` sentinel is auto-discovered by the media source registry.
The factory matches origins with ``importer == "recaller"`` and reads the
``media_url`` from origin params.

Caching
-------
Downloaded files are cached in a temporary directory keyed by
``contentID``.  Call :meth:`PullWrestSource.cleanup` to remove the
temporary directory when done.

Bulk fetch
----------
:meth:`PullWrestSource.fetch_items` overrides the default loop-per-item
implementation to issue a single batch request to the PullWrest service.
The batch response carries not just the file bytes but also pre-computed
embeddings and file-level metadata (size, duration, created_at, etc.) so
VTSearch can skip local re-embedding and avoid redundant stat calls.

The single-item :meth:`fetch_item` path remains as a fallback and for
callers that only need one file.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Iterator

from vtscore.datasets.sources.base import FetchedItem, MediaItem, MediaSource

logger = logging.getLogger(__name__)

__all__ = ["PullWrestSource"]


# ---------------------------------------------------------------------------
# TODO(dev): Implement the PullWrest client functions below.
# ---------------------------------------------------------------------------


def _pw_fetch_media(media_url: str) -> bytes:
    """Call PullWrest to download the raw media bytes for *media_url*.

    This is the same function as in the ReCaller importer - consider
    extracting a shared ``pullwrest_client`` module that both use.
    """
    raise NotImplementedError("TODO: implement PullWrest API client")


def _pw_fetch_media_batch(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Call PullWrest to fetch a batch of media items in one request.

    Args:
        items: List of dicts, each with at least ``"media_url"`` and
            ``"content_id"`` keys identifying the items to fetch.

    Returns:
        List of result dicts aligned with *items*.  Each result has:

        - ``"data"`` (``bytes``): raw media bytes.
        - ``"embedding"`` (``list[float] | None``): pre-computed embedding
          vector from PullWrest, or ``None`` if the service did not return
          one.
        - ``"embedder_name"`` (``str``): name of the embedder used to
          produce ``"embedding"`` (e.g. ``"laion-clap"``).
        - ``"file_size"`` (``int``): byte count of the media file.
        - ``"duration"`` (``float | None``): duration in seconds (audio /
          video), or ``None`` for non-temporal media.
        - ``"created_at"`` (``str | None``): ISO 8601 creation timestamp
          from the PullWrest catalogue, or ``None``.

    Raise:
        NotImplementedError: Until the PullWrest batch endpoint is wired up.
    """
    raise NotImplementedError("TODO: implement PullWrest batch API client")


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class PullWrestSource(MediaSource):
    """A media source that fetches files on demand via PullWrest.

    Args:
        media_url: The PullWrest-resolvable URL for the media.
        content_id: The contentID (used as the filename/key).
        media_type: The media type string (e.g. ``"audio"``).
    """

    name = "pullwrest"

    def __init__(
        self,
        media_url: str,
        content_id: str = "",
        media_type: str = "",
    ) -> None:
        self._media_url = media_url
        self._content_id = content_id
        self._media_type = media_type
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    def _ensure_tmpdir(self) -> Path:
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="pullwrest_")
        return Path(self._tmpdir.name)

    def _download_to_cache(self) -> Path | None:
        """Download the media file and return the local path."""
        if not self._media_url:
            return None
        tmpdir = self._ensure_tmpdir()
        filename = self._content_id or "media"
        cached = tmpdir / filename
        if cached.exists():
            return cached
        try:
            data = _pw_fetch_media(self._media_url)
            cached.write_bytes(data)
            return cached
        except Exception:
            logger.warning("PullWrest download failed for %s", self._media_url, exc_info=True)
            return None

    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        """Yield the single media item this source represents."""
        if self._content_id:
            yield MediaItem(
                key=self._content_id,
                filename=self._content_id,
                source_name=self.name,
            )

    def fetch_item(self, key: str) -> Path | None:
        """Download and return a local path for the media."""
        if key != self._content_id:
            return None
        return self._download_to_cache()

    def resolve_path(self, origin_name: str = "", filename: str = "") -> Path | None:
        """Resolve by downloading if origin_name or filename matches."""
        if origin_name == self._content_id or filename == self._content_id:
            return self._download_to_cache()
        return None

    # ------------------------------------------------------------------
    # Bulk fetch - override to use the PullWrest batch endpoint
    # ------------------------------------------------------------------

    def fetch_items(self, keys: list[str]) -> dict[str, FetchedItem]:
        """Fetch multiple items via the PullWrest batch API.

        Issues a single batch request that returns file bytes, pre-computed
        embeddings, and file-level metadata for all requested keys.  The
        returned :class:`~vtscore.datasets.sources.base.FetchedItem` objects
        carry the embedding and ``extra`` metadata (file_size, duration,
        created_at) so the ingest path can skip local re-embedding and
        redundant stat calls.

        Falls back to the single-item path for keys that are not
        ``self._content_id``.
        """
        import numpy as np

        # Collect items this source actually owns.
        owned = [k for k in keys if k == self._content_id]
        result: dict[str, FetchedItem] = {}

        if owned:
            tmpdir = self._ensure_tmpdir()
            batch_requests = [{"media_url": self._media_url, "content_id": self._content_id}]
            try:
                responses = _pw_fetch_media_batch(batch_requests)
                for req, resp in zip(batch_requests, responses):
                    cid = req["content_id"]
                    cached = tmpdir / cid
                    cached.write_bytes(resp["data"])

                    raw_emb = resp.get("embedding")
                    embedding = np.array(raw_emb, dtype=np.float32) if raw_emb else None

                    extra: dict[str, Any] = {}
                    for field in ("file_size", "duration", "created_at"):
                        if resp.get(field) is not None:
                            extra[field] = resp[field]

                    result[cid] = FetchedItem(
                        path=cached,
                        embedding=embedding,
                        embedder_name=resp.get("embedder_name", ""),
                        extra=extra,
                    )
            except Exception:
                logger.warning(
                    "PullWrest batch fetch failed; falling back to single-item path",
                    exc_info=True,
                )
                # Fall through to single-item path below.

        # Single-item fallback for anything not resolved by the batch call.
        for key in keys:
            if key not in result:
                result[key] = FetchedItem(path=self.fetch_item(key))

        return result

    def cleanup(self) -> None:
        """Remove the temporary download directory."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


# ---------------------------------------------------------------------------
# Factory for auto-discovery
# ---------------------------------------------------------------------------


class _PullWrestSourceFactory:
    """Factory discovered by :class:`~vtscore.plugins.PluginRegistry`."""

    name = "recaller"  # matches origin["importer"] from ReCaller imports

    def create_from_origin(self, origin: dict[str, Any]) -> PullWrestSource | None:
        params = origin.get("params", {})
        media_url = params.get("media_url", "")
        if not media_url:
            return None
        return PullWrestSource(
            media_url=media_url,
            content_id=params.get("contentID", ""),
            media_type=params.get("media_type", ""),
        )


SOURCE = _PullWrestSourceFactory()
