"""URL datasource importer: download one media file from a public URL."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from vtscore.datasource_importers.base import DataSourceImporter, FetchedMediaItem
from vtscore.plugins import PluginField
from vtscore.security.url_validation import validate_url


def _filename_from_url(url: str) -> str:
    """Derive a filename (with its real extension) from *url*'s path."""
    name = Path(unquote(urlparse(url).path)).name
    return name or "download.bin"


class UrlDownloadDataSourceImporter(DataSourceImporter):
    """Download a media file from a public http(s) URL."""

    name = "url_download"
    display_name = "URL"
    icon = "\U0001f310"
    category = "services"
    fields = [
        PluginField(
            key="url",
            label="Media URL",
            field_type="url",
            description="Public http(s) URL of a single media file",
            placeholder="https://example.com/sound.wav",
            required=True,
        ),
    ]

    def fetch(self, field_values: dict[str, Any]) -> FetchedMediaItem:
        raw = str(field_values.get("url") or "").strip()
        if not raw:
            raise ValueError("Media URL is required.")
        # Re-validate for direct library callers (idempotent after the
        # ingress normalize pass); the downloader re-checks every redirect
        # hop against the same SSRF guard.
        url = validate_url(raw)

        from vtscore.datasets.downloader import download_file_with_progress

        with tempfile.TemporaryDirectory(prefix="vts_datasource_url_") as tmp_dir:
            dest = Path(tmp_dir) / "download"
            download_file_with_progress(url, dest, on_progress=lambda *a, **k: None)
            data = dest.read_bytes()
        if not data:
            raise ValueError(f"The URL returned no data: {url}")
        return FetchedMediaItem(data=data, filename=_filename_from_url(url))


DATASOURCE_IMPORTER = UrlDownloadDataSourceImporter()
