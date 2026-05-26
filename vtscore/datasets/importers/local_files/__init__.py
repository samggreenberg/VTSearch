"""Local-files importer placeholder - drives the browser-side paths-file upload card.

The actual upload flow does **not** invoke this importer's ``run()``.
The browser POSTs a multipart body to ``/api/dataset/import-local-files``
carrying a single ``paths_file`` (a UTF-8 text file listing one
server-side media path per line, or a ``.npz`` archive that also
supplies pre-computed embedding vectors).  The endpoint saves the
uploaded file to a server-side temp location and then delegates to the
regular :mod:`server_files` importer for resolution, symlinking, and
embedding.

The card is the browser-upload equivalent of Server Files.  Use it when
you have a list of media paths (or a NumPy archive of pre-computed
vectors) on your laptop that you'd like to feed to the server without
typing the file's absolute server path by hand.
"""

from __future__ import annotations

from typing import Any

from vtscore.datasets.importers.base import DatasetImporter


class LocalFilesDatasetImporter(DatasetImporter):
    """Placeholder for the browser-side single-file paths upload card.

    The card opens the modal's ``"local_files"`` view (see
    :attr:`picker_view`), which uses a single-file ``<input type="file">``
    accepting ``.txt`` / ``.list`` / ``.npz`` and POSTs it to
    ``/api/dataset/import-local-files`` - the server then runs the
    regular :mod:`server_files` importer over the uploaded paths file.
    """

    name = "local_files"
    display_name = "Files"
    description = (
        "Upload a single file from this computer (your browser machine) that lists the media "
        "to import: a UTF-8 text file with one server-side path per line, or a .npz archive "
        "that also supplies pre-computed embedding vectors."
    )
    icon = "\U0001f4c4"  # 📄 - falls back to a generic file icon
    ui_mode = "custom"
    picker_view = "local_files"
    category = "local"
    # The upload flow delegates to server_files, which already
    # participates in the multi-media flow.  Surfacing the flag here
    # lets the frontend render the "Include media" repeater for this
    # picker too.
    multi_media = True
    fields = []

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        raise NotImplementedError(
            "LocalFilesDatasetImporter is browser-only. "
            "POST to /api/dataset/import-local-files for the upload flow, "
            "or use the regular `server_files` importer for server-side paths files."
        )


IMPORTER = LocalFilesDatasetImporter()
