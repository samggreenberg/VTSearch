"""Server-folder importer – wraps :class:`FolderDatasetImporter` for the
"Import from Server Folder" picker entry.

The dataset-importer modal renders this importer with a built-in directory
browser rooted at ``saved_datasets_dir``, so the user can pick a folder
visually instead of typing an absolute path.  All import logic is reused
from :class:`FolderDatasetImporter`.
"""

from __future__ import annotations

from vtsearch.datasets.importers.base import ImporterField
from vtsearch.datasets.importers.folder import FolderDatasetImporter


class ServerFolderDatasetImporter(FolderDatasetImporter):
    name = "server_folder"
    display_name = "Import from Server Folder"
    description = "Browse directories on the server and import media files"
    icon = "\U0001f5a5"  # 🖥 desktop computer

    fields = [
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            description="Type of media files to scan for in the folder.",
            default="audio",
        ),
        ImporterField(
            key="path",
            label="Folder",
            field_type="server_path",
            description="Path to a folder on the server containing media files.",
        ),
    ]


IMPORTER = ServerFolderDatasetImporter()
