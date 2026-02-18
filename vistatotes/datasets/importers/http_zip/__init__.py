"""HTTP-ZIP importer – downloads a public ZIP of media files and loads them.

Requires only ``requests``, which is already a core dependency.
"""

from __future__ import annotations

import zipfile

from config import DATA_DIR
from vistatotes.datasets.downloader import download_file_with_progress
from vistatotes.datasets.importers.base import DatasetImporter, ImporterField
from vistatotes.datasets.loader import load_dataset_from_folder
from vistatotes.utils import update_progress


class HttpZipImporter(DatasetImporter):
    """Download a publicly-accessible ZIP archive and load its media files.

    The archive is streamed to ``DATA_DIR/http_zip_download.zip``, extracted
    to ``DATA_DIR/http_zip_extract/``, then scanned with the standard
    :func:`~vistatotes.datasets.loader.load_dataset_from_folder` pipeline.
    Both temporary paths are cleaned up after a successful run.
    """

    name = "http_zip"
    display_name = "HTTP ZIP Download"
    description = "Download a .zip archive from a URL and load the media files inside."
    fields = [
        ImporterField(
            key="url",
            label="ZIP URL",
            field_type="url",
            description="URL to a publicly accessible .zip file of media files.",
        ),
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            description="Type of media files contained in the ZIP.",
            options=["sounds", "videos", "images", "paragraphs"],
            default="sounds",
        ),
    ]

    def run(self, field_values: dict, clips: dict) -> None:
        url = field_values["url"]
        media_type = field_values.get("media_type", "sounds")

        DATA_DIR.mkdir(exist_ok=True)
        zip_path = DATA_DIR / "http_zip_download.zip"
        extract_dir = DATA_DIR / "http_zip_extract"

        update_progress("downloading", "Downloading ZIP...", 0, 0)
        download_file_with_progress(url, zip_path)

        update_progress("loading", "Extracting ZIP...", 0, 0)
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                update_progress(
                    "loading",
                    f"Extracting {member.split('/')[-1]}...",
                    i,
                    total,
                )
                zf.extract(member, extract_dir)
        zip_path.unlink(missing_ok=True)

        load_dataset_from_folder(extract_dir, media_type, clips)


IMPORTER = HttpZipImporter()
