"""HTTP-Archive importer – downloads a public archive of media files and loads them.

Supports .zip, .tar, .tar.gz, .tar.bz2, .tar.xz archives.
RAR support requires the optional ``rarfile`` package.

Requires only ``requests``, which is already a core dependency.

Converter support
-----------------
When the ``converters`` field value is set (a comma-separated list of
converter names), the importer also scans the extracted archive for source
files matching each converter's input type, converts them to the target
media type, and appends them to the dataset.
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from uuid import uuid4

from vtsearch.config import DATA_DIR
from vtsearch.datasets.downloader import download_file_with_progress
from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.datasets.loader import load_dataset_from_folder
from vtsearch.utils.url_validation import validate_url

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    from vtsearch.utils.progress import get_thread_progress

    cb = get_thread_progress()
    if cb is not None:
        return cb
    from vtsearch.utils import update_progress

    return update_progress


def _extract_archive(
    archive_path: Path,
    extract_dir: Path,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Extract *archive_path* into *extract_dir*, supporting zip/tar/rar."""
    if on_progress is None:
        on_progress = _default_progress()

    name = archive_path.name.lower()

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                on_progress(
                    "loading",
                    f"Extracting {member.split('/')[-1]}...",
                    i,
                    total,
                )
                target = (extract_dir / member).resolve()
                if not str(target).startswith(str(extract_dir.resolve())):
                    raise ValueError(f"Path traversal detected in archive: {member}")
                zf.extract(member, extract_dir)

    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            members = tf.getmembers()
            total = len(members)
            for i, member in enumerate(members, 1):
                on_progress(
                    "loading",
                    f"Extracting {member.name.split('/')[-1]}...",
                    i,
                    total,
                )
                tf.extract(member, extract_dir, filter="data")

    elif name.endswith(".rar"):
        try:
            import rarfile  # optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "RAR extraction requires the 'rarfile' package. Install it with: pip install rarfile"
            ) from exc
        with rarfile.RarFile(archive_path, "r") as rf:
            members = rf.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                on_progress(
                    "loading",
                    f"Extracting {member.split('/')[-1]}...",
                    i,
                    total,
                )
                target = (extract_dir / member).resolve()
                if not str(target).startswith(str(extract_dir.resolve())):
                    raise ValueError(f"Path traversal detected in archive: {member}")
                rf.extract(member, extract_dir)

    else:
        raise ValueError(
            f"Unsupported archive format: {archive_path.name}. "
            "Supported formats: .zip, .tar, .tar.gz, .tar.bz2, .tar.xz, .rar"
        )


def _run_selected_converters(
    folder: Path,
    media_type: str,
    field_values: dict,
    medias: dict,
    thin: bool = False,
) -> None:
    """Run any user-selected converters from *field_values*."""
    converters_str = field_values.get("converters", "")
    if not converters_str:
        return
    converter_names = [c.strip() for c in converters_str.split(",") if c.strip()]
    if not converter_names:
        return

    from vtsearch.converters.runner import run_converters_on_folder  # noqa: PLC0415

    base_origin = {"importer": "http_archive", "params": {
        "url": field_values.get("url", ""),
        "media_type": media_type,
    }}
    run_converters_on_folder(
        folder_path=folder,
        converter_names=converter_names,
        target_media_type=media_type,
        medias=medias,
        thin=thin,
        base_origin=base_origin,
    )


class HttpArchiveDatasetImporter(DatasetImporter):
    """Download a publicly-accessible archive and load its media files.

    The archive is streamed to a temporary file in ``DATA_DIR``, extracted
    to a unique ``DATA_DIR/http_archive_extract_<id>/`` directory, then
    scanned with the standard
    :func:`~vtsearch.datasets.loader.load_dataset_from_folder` pipeline.
    Both temporary paths are cleaned up after the run completes.

    Supported archive formats: ``.zip``, ``.tar``, ``.tar.gz``,
    ``.tar.bz2``, ``.tar.xz``, ``.rar`` (requires ``rarfile`` package).

    When converters are selected (via the ``converters`` field value), files
    matching each converter's source type are also scanned, converted, and
    added to the dataset.
    """

    name = "http_archive"
    display_name = "Import from URL"
    description = "Download an archive (.zip, .tar, .rar) from a web URL and embed the media files inside"
    icon = "\U0001f310"
    hidden_from_picker = True
    fields = [
        ImporterField(
            key="url",
            label="Archive URL",
            field_type="url",
            description="URL to a publicly accessible archive (.zip, .tar.gz, .rar, \u2026) of media files.",
        ),
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            description="Type of media files contained in the archive.",
            default="sounds",
        ),
    ]

    def __init__(self) -> None:
        super().__init__()
        from vtsearch.media import all_folder_names

        for f in self.fields:
            if f.key == "media_type":
                f.options = all_folder_names()
                break

    def to_dict(self) -> dict:
        d = super().to_dict()
        from vtsearch.converters import list_converters_for_target
        from vtsearch.media import all_types_dict

        converters_by_target: dict[str, list[dict]] = {}
        for mt_info in all_types_dict():
            type_id = mt_info["type_id"]
            convs = list_converters_for_target(type_id)
            if convs:
                converters_by_target[type_id] = [c.to_dict() for c in convs]
        d["available_converters_by_media_type"] = converters_by_target
        return d

    def run(self, field_values: dict, medias: dict, thin: bool = False) -> None:
        url = field_values["url"]
        validate_url(url)
        media_type = field_values.get("media_type", "sounds")

        DATA_DIR.mkdir(exist_ok=True)

        progress = _default_progress()

        # Derive a local filename from the URL so we preserve the extension
        url_path = url.split("?")[0].rstrip("/")
        url_filename = url_path.split("/")[-1] or "archive"
        run_id = uuid4().hex[:12]
        archive_path = DATA_DIR / f"http_archive_download_{run_id}_{url_filename}"
        extract_dir = DATA_DIR / f"http_archive_extract_{run_id}"

        progress("downloading", "Downloading archive...", 0, 0)
        download_file_with_progress(url, archive_path, on_progress=progress)

        progress("loading", "Extracting archive...", 0, 0)
        extract_dir.mkdir(exist_ok=True)
        _extract_archive(archive_path, extract_dir, on_progress=progress)
        archive_path.unlink(missing_ok=True)

        emb_name = field_values.get("embedder", "")
        skip_emb = bool(field_values.get("skip_embedding"))
        try:
            load_dataset_from_folder(
                extract_dir, media_type, medias, on_progress=progress, thin=thin, embedder_name=emb_name,
                content_vectors=self.content_vectors or None,
                content_md5s=self.content_md5s or None,
                custom_metadata_map=self.custom_metadata_map or None,
                skip_embedding=skip_emb,
            )
            # Run any user-selected converters on the extracted folder.
            _run_selected_converters(extract_dir, media_type, field_values, medias, thin=thin)
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        url = field_values.get("url", "")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL (must start with http:// or https://): {url}")
        self.run(field_values, medias, thin=thin)

    @property
    def supports_chunked(self) -> bool:
        return True

    def _download_and_extract(self, field_values: dict[str, Any]) -> Path:
        """Download and extract the archive, returning the extraction dir.

        Each call creates a unique extraction directory so concurrent imports
        do not interfere with each other.  Callers are responsible for cleaning
        up the returned directory when they are done with it.
        """
        url = field_values["url"]
        validate_url(url)

        DATA_DIR.mkdir(exist_ok=True)
        progress = _default_progress()

        url_path = url.split("?")[0].rstrip("/")
        url_filename = url_path.split("/")[-1] or "archive"
        run_id = uuid4().hex[:12]
        archive_path = DATA_DIR / f"http_archive_download_{run_id}_{url_filename}"
        extract_dir = DATA_DIR / f"http_archive_extract_{run_id}"

        progress("downloading", "Downloading archive...", 0, 0)
        download_file_with_progress(url, archive_path, on_progress=progress)

        progress("loading", "Extracting archive...", 0, 0)
        extract_dir.mkdir(exist_ok=True)
        _extract_archive(archive_path, extract_dir, on_progress=progress)
        archive_path.unlink(missing_ok=True)

        return extract_dir

    def run_chunked(
        self, field_values: dict[str, Any], chunk_size: int, thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        from vtsearch.datasets.loader import load_dataset_from_folder_chunked

        extract_dir = self._download_and_extract(field_values)
        media_type = field_values.get("media_type", "sounds")
        emb_name = field_values.get("embedder", "")
        skip_emb = bool(field_values.get("skip_embedding"))
        try:
            yield from load_dataset_from_folder_chunked(
                extract_dir, media_type, chunk_size, thin=thin, embedder_name=emb_name,
                content_vectors=self.content_vectors or None,
                content_md5s=self.content_md5s or None,
                custom_metadata_map=self.custom_metadata_map or None,
                skip_embedding=skip_emb,
            )
            # Run converters on the extracted folder and yield as a chunk.
            converters_str = field_values.get("converters", "")
            if converters_str:
                converter_chunk: dict[int, dict[str, Any]] = {}
                _run_selected_converters(extract_dir, media_type, field_values, converter_chunk, thin=thin)
                if converter_chunk:
                    yield converter_chunk
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

    def run_chunked_cli(
        self, field_values: dict[str, Any], chunk_size: int, thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        url = field_values.get("url", "")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL (must start with http:// or https://): {url}")
        yield from self.run_chunked(field_values, chunk_size, thin=thin)

    def build_cli_args(self, field_values: dict[str, Any]) -> str:
        base = super().build_cli_args(field_values)
        converters = field_values.get("converters", "")
        if converters:
            base += f" --converters {converters}"
        return base

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        origin = super().build_origin(field_values)
        converters = field_values.get("converters", "")
        if converters:
            origin["params"]["converters"] = converters
        return origin


    def resolve_file(
        self,
        origin: dict[str, Any],
        origin_name: str = "",
        filename: str = "",
    ) -> Path | None:
        url = origin.get("params", {}).get("url", "")
        if not url:
            return None

        from vtsearch.datasets.sources.http_archive import HttpArchiveSource

        source = HttpArchiveSource(url)
        # Note: we intentionally do NOT call source.cleanup() here because
        # the extracted archive should remain cached for future resolve_file
        # calls (matching the previous behaviour of keeping resolve dirs).
        return source.resolve_path(origin_name, filename)


def _search_dir_for_file(
    directory: Path, origin_name: str, filename: str
) -> Path | None:
    """Search a directory for a file matching origin_name or filename."""
    for name in [origin_name, filename]:
        if not name:
            continue
        candidate = directory / name
        if candidate.is_file():
            return candidate
        matches = list(directory.rglob(Path(name).name))
        if matches:
            return matches[0]
    return None


IMPORTER = HttpArchiveDatasetImporter()
