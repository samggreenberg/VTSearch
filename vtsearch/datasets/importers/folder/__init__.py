"""Local-folder importer \u2013 scans a directory of media files and embeds them.

No additional pip packages are required; librosa, opencv, and Pillow are
already in the core requirements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.datasets.loader import load_dataset_from_folder, load_dataset_from_folder_chunked


class FolderDatasetImporter(DatasetImporter):
    """Embed all media files found in a local directory into a dataset.

    The user supplies an absolute filesystem path and selects the media type
    so that the correct file extensions are matched during the scan.
    """

    name = "folder"
    display_name = "Generate from Folder"
    description = "Import media files from a folder."
    icon = "\U0001f4c2"
    fields = [
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            description="Type of media files to scan for in the folder.",
            options=["sounds", "videos", "images", "paragraphs"],
            default="sounds",
        ),
        ImporterField(
            key="path",
            label="Folder",
            field_type="folder",
            description="Absolute path to the directory containing media files.",
        ),
    ]

    def run(self, field_values: dict, medias: dict, thin: bool = False) -> None:
        folder = Path(field_values["path"])
        media_type = field_values.get("media_type", "sounds")
        load_dataset_from_folder(folder, media_type, medias, thin=thin)

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        folder = Path(field_values["path"])
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
        self.run(field_values, medias, thin=thin)

    @property
    def supports_chunked(self) -> bool:
        return True

    def run_chunked(
        self, field_values: dict[str, Any], chunk_size: int, thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        folder = Path(field_values["path"])
        media_type = field_values.get("media_type", "sounds")
        yield from load_dataset_from_folder_chunked(folder, media_type, chunk_size, thin=thin)

    def run_chunked_cli(
        self, field_values: dict[str, Any], chunk_size: int, thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        folder = Path(field_values["path"])
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
        yield from self.run_chunked(field_values, chunk_size, thin=thin)


IMPORTER = FolderDatasetImporter()
