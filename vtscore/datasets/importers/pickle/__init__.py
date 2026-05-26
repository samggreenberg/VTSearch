"""Pickle-file importer \u2013 loads a previously exported ``.pkl`` dataset.

No additional pip packages are required; everything needed is already in
the core requirements.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterator

from vtscore.config import DATA_DIR
from vtscore.datasets.importers.base import DatasetImporter, ImporterField
from vtscore.datasets.loader import load_dataset_from_pickle, load_dataset_from_pickle_chunked


def _get_progress():
    from vtscore.concurrency.progress import update_progress

    return update_progress


class PickleDatasetImporter(DatasetImporter):
    """Load a dataset from a ``.pkl`` file exported by VTSearch.

    The user picks the file via the browser's file-upload input.  The file
    is streamed to a temporary path on the server, deserialized, and then
    the temporary file is deleted.
    """

    name = "pickle"
    display_name = "Upload Saved Dataset"
    description = "Load a .pkl dataset file that was previously exported from VTSearch"
    icon = "\U0001f4e4"
    ui_mode = "file_upload"
    hidden_from_picker = True
    # Pickle imports carry a single embedded media type from the .pkl
    # itself - there is no user-chosen output type and no converter rows.
    # The flag is set so the in-tree importer set is uniformly off the
    # legacy ``effective_source_specs()`` shim, not because the upload
    # flow has anything multi-media about it.
    multi_media = True
    fields = [
        ImporterField(
            key="file",
            label="Upload a file",
            field_type="file",
            description="A .pkl file that was exported from VTSearch.",
            accept=".pkl",
        ),
    ]

    def run(self, field_values: dict, medias: dict, thin: bool = False) -> None:
        file_obj = field_values["file"]  # UploadedFile (FileStorage / CliUploadedFile / BytesIOUploadedFile)
        progress = _get_progress()
        progress("loading", "Loading dataset from file...", 0, 0)
        DATA_DIR.mkdir(exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(suffix=".pkl", dir=DATA_DIR)
        temp_path = Path(tmp_name)
        try:
            import os

            os.close(fd)
            # UploadedFile.save() persists to disk; FileStorage, CliUploadedFile,
            # and BytesIOUploadedFile all implement it.
            file_obj.save(temp_path)
            load_dataset_from_pickle(temp_path, medias, thin=thin)
        finally:
            temp_path.unlink(missing_ok=True)
        progress("idle", f"Loaded {len(medias)} medias from file")

    @property
    def supports_chunked(self) -> bool:
        return True

    def run_chunked_cli(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        """Yield chunks from a pickle file path (string)."""
        file_path = Path(field_values["file"])
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        yield from load_dataset_from_pickle_chunked(file_path, chunk_size, thin=thin)

    def default_display_name(self, field_values: dict[str, Any]) -> str:
        file_obj = field_values.get("file")
        candidate = ""
        if file_obj is None:
            pass
        elif hasattr(file_obj, "filename") and file_obj.filename:
            candidate = file_obj.filename
        elif hasattr(file_obj, "name") and file_obj.name:
            candidate = file_obj.name
        elif isinstance(file_obj, str):
            candidate = file_obj
        if candidate:
            stem = Path(candidate).stem
            if stem:
                return stem
        return self.display_name

    def origin_display(self, origin: dict[str, Any]) -> str:
        params = origin.get("params", {})
        filename = params.get("filename", params.get("path", ""))
        return f"file:{filename}" if filename else "pickle"

    def can_reload_from_origin(self, origin: dict[str, Any]) -> bool:
        params = origin.get("params", {})
        pkl_path = params.get("path", "")
        return bool(pkl_path) and Path(pkl_path).is_file()

    def reload_from_origin(self, origin: dict[str, Any]) -> dict[str, Any] | None:
        params = origin.get("params", {})
        pkl_path = params.get("path", "")
        if not pkl_path or not Path(pkl_path).is_file():
            return None
        return {"file": pkl_path}


IMPORTER = PickleDatasetImporter()
