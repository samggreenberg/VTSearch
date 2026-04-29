"""Local-folder importer placeholder — drives the browser-side upload card.

The actual upload flow does **not** invoke this importer's ``run()``.
Browsers POST a multipart body to ``/api/dataset/import-local-folder``,
which streams the files to a server-side temp directory and then
delegates to the regular :mod:`server_folder` importer for scanning
and embedding.

This importer exists so that the dataset-importer modal can be fully
data-driven — the picker reads :attr:`display_name`, :attr:`description`,
:attr:`icon`, and :attr:`picker_view` from the registry instead of
hard-coding "Local Folder" markup in HTML.
"""

from __future__ import annotations

from typing import Any

from vtsearch.datasets.importers.base import DatasetImporter, ImporterField


class LocalFolderDatasetImporter(DatasetImporter):
    """Placeholder for the browser-side folder upload card.

    The card opens the modal's ``"local_folder"`` view (see
    :attr:`picker_view`), which uses
    ``<input type="file" webkitdirectory>`` to let the user pick a folder
    on the **browser machine** and POSTs the files to
    ``/api/dataset/import-local-folder``.
    """

    name = "local_folder"
    display_name = "Local Folder"
    description = "Upload a folder of media files from this computer (your browser machine) to the server"
    icon = "\U0001f4c1"  # 📁 — frontend renders as a folder icon
    ui_mode = "custom"
    picker_view = "local_folder"
    fields = [
        ImporterField(
            key="recursive",
            label="Include subfolders",
            field_type="checkbox",
            description=(
                "When enabled, subfolders inside the chosen folder are also "
                "uploaded and imported.  When disabled, only files directly "
                "inside the chosen folder are imported."
            ),
            default="true",
            required=False,
        ),
    ]

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        raise NotImplementedError(
            "LocalFolderDatasetImporter is browser-only. "
            "POST to /api/dataset/import-local-folder for the upload flow, "
            "or use the regular `server_folder` importer for server-side directories."
        )


IMPORTER = LocalFolderDatasetImporter()
