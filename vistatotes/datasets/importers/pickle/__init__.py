"""Pickle-file importer – loads a previously exported ``.pkl`` dataset.

No additional pip packages are required; everything needed is already in
the core requirements.
"""

from __future__ import annotations

from config import DATA_DIR
from vistatotes.datasets.importers.base import DatasetImporter, ImporterField
from vistatotes.datasets.loader import load_dataset_from_pickle
from vistatotes.utils import update_progress


class PickleImporter(DatasetImporter):
    """Load a dataset from a ``.pkl`` file exported by VistaTotes.

    The user picks the file via the browser's file-upload input.  The file
    is streamed to a temporary path on the server, deserialized, and then
    the temporary file is deleted.
    """

    name = "pickle"
    display_name = "Pickle File"
    description = "Load a previously exported .pkl dataset file."
    fields = [
        ImporterField(
            key="file",
            label="Dataset File",
            field_type="file",
            description="A .pkl file that was exported from VistaTotes.",
            accept=".pkl",
        ),
    ]

    def run(self, field_values: dict, clips: dict) -> None:
        file_obj = field_values["file"]  # werkzeug FileStorage
        update_progress("loading", "Loading dataset from file...", 0, 0)
        temp_path = DATA_DIR / "temp_upload.pkl"
        DATA_DIR.mkdir(exist_ok=True)
        file_obj.save(temp_path)
        try:
            load_dataset_from_pickle(temp_path, clips)
        finally:
            temp_path.unlink(missing_ok=True)
        update_progress("idle", f"Loaded {len(clips)} clips from file")


IMPORTER = PickleImporter()
