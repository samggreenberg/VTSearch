"""HTTP-Archive importer – downloads a public archive of media files and loads them.

Supports .zip, .tar, .tar.gz, .tar.bz2, .tar.xz archives.
RAR support requires the optional ``rarfile`` package.

Requires only ``requests``, which is already a core dependency.

Converter support
-----------------
The importer participates in the multi-media import flow.  Each
:class:`~vtscore.datasets.importers.base.SourceSpec` row in
``source_specs`` is applied to the extracted archive:

* a direct row (``converter is None``) embeds files of the spec's
  source type with the target embedder, and
* a converter row scans the extracted archive for files of the
  converter's source type and runs the converter (with per-row params)
  to produce media of the chosen output type.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

from vtscore.config import DATA_DIR
from vtscore.datasets.archive import extract_archive, is_archive_path, load_archive_into
from vtscore.datasets.downloader import download_file_with_progress
from vtscore.datasets.importers.base import DatasetImporter, ImporterField, SourceSpec
from vtscore.datasets.loader import load_dataset_from_folder

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    from vtscore.concurrency.progress import get_thread_progress

    cb = get_thread_progress()
    if cb is not None:
        return cb
    from vtscore.concurrency.progress import update_progress

    return update_progress


def _is_url(value: str) -> bool:
    """Return ``True`` when *value* looks like an http(s) URL (vs a local path)."""
    return value.startswith(("http://", "https://"))


def _run_converter_specs(
    folder: Path,
    media_type: str,
    field_values: dict,
    converter_specs: list[SourceSpec],
    medias: dict,
    thin: bool = False,
) -> None:
    """Hand the converter rows of a multi-media spec to the runner."""
    runnable = [s for s in converter_specs if s.converter is not None]
    if not runnable:
        return

    from vtscore.converters.runner import run_converters_on_folder  # noqa: PLC0415

    base_origin = {
        "importer": "http_archive",
        "params": {
            "url": field_values.get("url", ""),
            "media_type": media_type,
        },
    }
    run_converters_on_folder(
        folder_path=folder,
        converter_specs=runnable,
        target_media_type=media_type,
        medias=medias,
        thin=thin,
        base_origin=base_origin,
    )


class HttpArchiveDatasetImporter(DatasetImporter):
    """Load media files from an archive given by a web URL **or** a local path.

    For an ``http://`` / ``https://`` URL the archive is streamed to a
    temporary file in ``DATA_DIR``, extracted to a unique
    ``DATA_DIR/http_archive_extract_<id>/`` directory, then scanned with the
    standard :func:`~vtscore.datasets.loader.load_dataset_from_folder`
    pipeline (both temporary paths are cleaned up afterwards).

    For a local **server path** to an archive file, the download step is
    skipped and the archive is extracted (and cached) via
    :func:`~vtscore.datasets.archive.load_archive_into`; the resulting media
    carry a ``local_archive`` origin so they re-derive on demand.

    Supported archive formats: ``.zip``, ``.tar``, ``.tar.gz``,
    ``.tar.bz2``, ``.tar.xz``, ``.rar`` (requires ``rarfile`` package).

    Multi-media imports work the same as in
    :class:`~vtscore.datasets.importers.server_folder.ServerFolderDatasetImporter`:
    each :class:`SourceSpec` row either embeds files of its source type
    directly or runs a converter to produce media of the chosen output
    media type.
    """

    name = "http_archive"
    display_name = "Import from URL"
    description = "Download an archive (.zip, .tar, .rar) from a web URL or local path and embed the media inside"
    icon = "\U0001f310"
    hidden_from_picker = True
    fields = [
        ImporterField(
            key="url",
            label="Path or URL",
            field_type="url",
            description="A web URL or local server path to an archive of media files to unpack.",
            hint=(
                "An http(s):// URL or an absolute server path. "
                "Supported archive formats: .zip, .tar, .tar.gz / .tgz, .tar.bz2, .rar."
            ),
        ),
        ImporterField(
            key="media_type",
            label="Dataset MediaType",
            field_type="select",
            description="Type of media files contained in the archive.",
            default="audio",
            required=False,
        ),
    ]

    def __init__(self) -> None:
        super().__init__()
        from vtscore.media import all_folder_names

        for f in self.fields:
            if f.key == "media_type":
                f.options = all_folder_names()
                break

    def _output_type_id(self, media_type: str) -> str:
        """Resolve a folder-name media type (e.g. ``"audio"``) to its type id."""
        from vtscore.media import get_by_folder_name  # noqa: PLC0415

        try:
            return get_by_folder_name(media_type).type_id
        except (KeyError, AttributeError):
            return media_type

    def run(self, field_values: dict, medias: dict, thin: bool = False) -> None:
        url = field_values["url"]
        media_type = field_values.get("media_type", "audio")
        specs = self.effective_source_specs(field_values)

        if not _is_url(url):
            # Local server path to an archive: extract (cached) and load,
            # stamping local_archive origins for on-demand re-derivation.
            medias.clear()
            load_archive_into(
                url,
                self._output_type_id(media_type),
                specs,
                medias,
                thin=thin,
                content_vectors=self.content_vectors,
                content_md5s=self.content_md5s,
                custom_metadata_map=self.custom_metadata_map,
            )
            return

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
        extract_archive(archive_path, extract_dir, on_progress=progress)
        archive_path.unlink(missing_ok=True)

        try:
            load_dataset_from_folder(
                extract_dir,
                media_type,
                medias,
                on_progress=progress,
                thin=thin,
                content_vectors=self.content_vectors or None,
                content_md5s=self.content_md5s or None,
                custom_metadata_map=self.custom_metadata_map or None,
            )
            _run_converter_specs(extract_dir, media_type, field_values, specs, medias, thin=thin)
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        url = field_values.get("url", "")
        if not _is_url(url):
            path = Path(url)
            if not path.exists():
                raise FileNotFoundError(f"Archive not found: {path}")
            if not is_archive_path(path):
                raise ValueError(f"Not a supported archive: {path}")
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
        extract_archive(archive_path, extract_dir, on_progress=progress)
        archive_path.unlink(missing_ok=True)

        return extract_dir

    def run_chunked(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        from vtscore.datasets.loader import load_dataset_from_folder_chunked

        if not _is_url(field_values.get("url", "")):
            # Local archive path: load everything as a single chunk via run().
            local_medias: dict[int, dict[str, Any]] = {}
            self.run(field_values, local_medias, thin=thin)
            if local_medias:
                yield local_medias
            return

        extract_dir = self._download_and_extract(field_values)
        media_type = field_values.get("media_type", "audio")
        specs = self.effective_source_specs(field_values)
        converter_specs = [s for s in specs if s.converter is not None]
        try:
            yield from load_dataset_from_folder_chunked(
                extract_dir,
                media_type,
                chunk_size,
                thin=thin,
                content_vectors=self.content_vectors or None,
                content_md5s=self.content_md5s or None,
                custom_metadata_map=self.custom_metadata_map or None,
            )
            if converter_specs:
                converter_chunk: dict[int, dict[str, Any]] = {}
                _run_converter_specs(
                    extract_dir,
                    media_type,
                    field_values,
                    converter_specs,
                    converter_chunk,
                    thin=thin,
                )
                if converter_chunk:
                    yield converter_chunk
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

    def run_chunked_cli(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        url = field_values.get("url", "")
        if not _is_url(url):
            path = Path(url)
            if not path.exists():
                raise FileNotFoundError(f"Archive not found: {path}")
            if not is_archive_path(path):
                raise ValueError(f"Not a supported archive: {path}")
        yield from self.run_chunked(field_values, chunk_size, thin=thin)

    def build_cli_args(self, field_values: dict[str, Any]) -> str:
        base = super().build_cli_args(field_values)
        specs = field_values.get("source_specs")
        if specs:
            if not isinstance(specs, str):
                specs = json.dumps(specs)
            base += f" --source-specs '{specs}'"
        return base

    def default_display_name(self, field_values: dict[str, Any]) -> str:
        url = (field_values.get("url") or "").strip()
        if url:
            url_path = url.split("?")[0].rstrip("/")
            tail = url_path.split("/")[-1]
            if tail:
                # Strip a few common archive extensions for a tidier label.
                for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar", ".zip", ".rar"):
                    if tail.lower().endswith(suffix):
                        return tail[: -len(suffix)] or self.display_name
                return tail
        return self.display_name

    def resolve_file(
        self,
        origin: dict[str, Any],
        origin_name: str = "",
        filename: str = "",
    ) -> Path | None:
        url = origin.get("params", {}).get("url", "")
        if not url:
            return None

        from vtscore.datasets.sources.http_archive import HttpArchiveSource

        source = HttpArchiveSource(url)
        # Note: we intentionally do NOT call source.cleanup() here because
        # the extracted archive should remain cached for future resolve_file
        # calls (matching the previous behaviour of keeping resolve dirs).
        return source.resolve_path(origin_name, filename).path


IMPORTER = HttpArchiveDatasetImporter()
