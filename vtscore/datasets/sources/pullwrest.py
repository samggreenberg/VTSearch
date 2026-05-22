"""PullWrest media source — resolve media files via the PullWrest service.

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
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Iterator

from vtscore.datasets.sources.base import MediaItem, MediaSource

logger = logging.getLogger(__name__)

__all__ = ["PullWrestSource"]


# ---------------------------------------------------------------------------
# TODO(dev): Implement the PullWrest client function below.
# ---------------------------------------------------------------------------


def _pw_fetch_media(media_url: str) -> bytes:
    """Call PullWrest to download the raw media bytes for *media_url*.

    This is the same function as in the ReCaller importer — consider
    extracting a shared ``pullwrest_client`` module that both use.
    """
    raise NotImplementedError("TODO: implement PullWrest API client")


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
