"""Local upload importer placeholder — drives the browser-side upload card.

The user picks either individual files or a folder from their browser
machine and the contents stream up to the server.  The actual upload
flow does **not** invoke this importer's ``run()``: the browser POSTs
a multipart body to ``/api/dataset/import-local-folder``, which writes
each file to a server-side temp directory (preserving any
``webkitRelativePath`` sub-structure for folder picks) and then
delegates to the regular :mod:`server_folder` importer for scanning
and embedding.

The card accepts an optional ``vectors_file`` form input — a ``.npz``
archive of pre-computed embedding vectors keyed by uploaded-file name
(basename or relative path).  Files whose name matches an NPZ key
reuse the supplied vector instead of running the embedding model,
which lets users import media they have already embedded offline
without paying for embedding twice.

This importer exists so that the dataset-importer modal can be fully
data-driven — the picker reads :attr:`display_name`,
:attr:`description`, :attr:`icon`, and :attr:`picker_view` from the
registry instead of hard-coding "Local" markup in HTML.
"""

from __future__ import annotations

from typing import Any

from vtscore.datasets.importers.base import DatasetImporter, ImporterField


class LocalDatasetImporter(DatasetImporter):
    """Placeholder for the browser-side files / folder upload card.

    The card opens the modal's ``"local"`` view (see :attr:`picker_view`),
    which renders two browse buttons — one using ``<input type="file"
    webkitdirectory>`` to pick a folder, the other using ``<input
    type="file" multiple>`` to pick one or more individual files — plus
    a drop zone that accepts either.  Whatever the user picks is POSTed
    to ``/api/dataset/import-local-folder`` and re-imported through
    :mod:`server_folder` on a server-side temp directory.
    """

    name = "local"
    display_name = "Local"
    description = (
        "Upload files or a folder of media from this computer (your browser machine) to the server.  "
        "Optionally include a .npz archive of pre-computed embedding vectors to skip re-embedding "
        "for media you have already embedded offline."
    )
    icon = "\U0001f4c1"  # 📁 — frontend renders as a folder icon
    ui_mode = "custom"
    picker_view = "local"
    category = "local"
    # The upload flow delegates to server_folder, which already
    # participates in the multi-media flow.  Surfacing the flag here
    # lets the frontend render the "Include media" repeater for this
    # picker too.
    multi_media = True
    fields = [
        ImporterField(
            key="recursive",
            label="Include subfolders",
            field_type="checkbox",
            description=(
                "Only applies when picking a folder.  When enabled, subfolders "
                "inside the chosen folder are also uploaded and imported.  When "
                "disabled, only files directly inside the chosen folder are imported."
            ),
            default="true",
            required=False,
        ),
    ]

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        raise NotImplementedError(
            "LocalDatasetImporter is browser-only. "
            "POST to /api/dataset/import-local-folder for the upload flow, "
            "or use the regular `server_folder` importer for server-side directories."
        )


IMPORTER = LocalDatasetImporter()
