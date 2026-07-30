"""Server-file datasource importer: pick one media file from the server's filesystem."""

from __future__ import annotations

from typing import Any

from vtscore.datasource_importers.base import DataSourceImporter, FetchedMediaItem
from vtscore.plugins import PluginField


class ServerFileDataSourceImporter(DataSourceImporter):
    """Use a media file already on the server's filesystem."""

    name = "server_file"
    display_name = "Server File"
    icon = "\U0001f4c4"
    category = "server"
    fields = [
        PluginField(
            key="path",
            label="Media file",
            field_type="server_path",
            description="Path to a media file on the server",
            placeholder="/absolute/server/path/to/file",
            required=True,
        ),
    ]

    def fetch(self, field_values: dict[str, Any]) -> FetchedMediaItem:
        # Call-time import so monkeypatched validators (e.g. the tests_lib
        # tmp-path widening) are honoured, mirroring plugins/normalize.py.
        from vtscore.security.path_validation import get_file_access_base_dir, validate_server_filepath

        raw = str(field_values.get("path") or "").strip()
        if not raw:
            raise ValueError("Media file path is required.")
        # The HTTP/CLI ingress already confined the path via
        # normalize_field_values; re-validate so direct library callers get
        # the same per-user confinement (idempotent on validated paths).
        path = validate_server_filepath(raw, base_dir=get_file_access_base_dir())
        if not path.is_file():
            raise ValueError(f"File not found: {raw}")
        return FetchedMediaItem(data=path.read_bytes(), filename=path.name)


DATASOURCE_IMPORTER = ServerFileDataSourceImporter()
