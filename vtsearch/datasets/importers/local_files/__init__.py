"""Local-files importer placeholder — drives the browser-side multi-file upload card.

Like :mod:`local_folder`, this importer's ``run()`` is **never invoked**:
the browser POSTs a multipart body to ``/api/dataset/import-local-folder``
which streams the uploaded media files into a server-side temp directory
and then delegates to the regular :mod:`server_folder` importer for
scanning and embedding.

The only difference between the Local Folder and Local Files cards is
the browser-side input: Local Folder uses ``<input type="file"
webkitdirectory>`` (a single directory pick), while Local Files uses
``<input type="file" multiple>`` (one or more individual file picks).
Both flows deliver the same multipart shape to the same upload
endpoint.

The Local Files card additionally accepts an optional ``vectors_file``
form input — a ``.npz`` archive containing pre-computed embedding
vectors keyed by uploaded-file name.  Files whose name matches an NPZ
key reuse the supplied vector and skip the embedding model, which lets
users import media they have already embedded offline without
re-embedding it on the server.
"""

from __future__ import annotations

from typing import Any

from vtsearch.datasets.importers.base import DatasetImporter


class LocalFilesDatasetImporter(DatasetImporter):
    """Placeholder for the browser-side multi-file upload card.

    The card opens the modal's ``"local_files"`` view (see
    :attr:`picker_view`), which uses
    ``<input type="file" multiple>`` to let the user pick one or more
    individual files on the **browser machine** and POSTs them to
    ``/api/dataset/import-local-folder`` — the server then runs the
    regular :mod:`server_folder` importer over the upload temp dir.
    """

    name = "local_files"
    display_name = "Files"
    description = (
        "Upload one or more individual media files from this computer (your browser machine) "
        "to the server.  Optionally include a .npz archive of pre-computed embedding vectors "
        "to skip re-embedding for media you have already embedded offline."
    )
    icon = "\U0001f4c4"  # 📄 — falls back to a generic file icon
    ui_mode = "custom"
    picker_view = "local_files"
    category = "local"
    fields = []

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        raise NotImplementedError(
            "LocalFilesDatasetImporter is browser-only. "
            "POST to /api/dataset/import-local-folder for the upload flow, "
            "or use the regular `server_folder` importer for server-side directories."
        )


IMPORTER = LocalFilesDatasetImporter()
