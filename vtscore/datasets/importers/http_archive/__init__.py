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
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from uuid import uuid4

from vtscore.config import DATA_DIR
from vtscore.datasets.downloader import download_file_with_progress
from vtscore.datasets.importers.base import DatasetImporter, ImporterField, SourceSpec
from vtscore.datasets.loader import load_dataset_from_folder
from vtscore.security.url_validation import validate_url

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    from vtscore.concurrency.progress import get_thread_progress

    cb = get_thread_progress()
    if cb is not None:
        return cb
    from vtscore.concurrency.progress import update_progress

    return update_progress


def _reject_traversal(extract_dir_resolved: Path, member_name: str) -> None:
    """Raise ValueError if *member_name* would extract outside extract_dir.

    Validates before extraction so a malicious member is never written to disk.
    Rejects absolute paths, ``..`` traversal, and any name that — once joined
    and normalised — escapes the extraction root.
    """
    # Reject absolute member names outright; on Windows they'd also drop the
    # root prefix when joined, but we want to fail loudly either way.
    if member_name.startswith(("/", "\\")) or (len(member_name) > 1 and member_name[1] == ":"):
        raise ValueError(f"Path traversal detected in archive: {member_name}")

    # Use os.path.normpath-style joining without resolving symlinks: the
    # extract_dir is freshly created and contains no symlinks yet, and we
    # don't want a symlink planted by an earlier member in the same archive
    # to mask a later traversal.
    target = Path(os.path.normpath(extract_dir_resolved / member_name))
    if target != extract_dir_resolved and not target.is_relative_to(extract_dir_resolved):
        raise ValueError(f"Path traversal detected in archive: {member_name}")


def _extract_archive(  # noqa: C901
    archive_path: Path,
    extract_dir: Path,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Extract *archive_path* into *extract_dir*, supporting zip/tar/rar."""
    if on_progress is None:
        on_progress = _default_progress()

    name = archive_path.name.lower()
    extract_dir_resolved = extract_dir.resolve()

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
                _reject_traversal(extract_dir_resolved, member)
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
                _reject_traversal(extract_dir_resolved, member.name)
                tf.extract(member, extract_dir, filter="data")

    elif name.endswith(".rar"):
        try:
            import rarfile  # optional dependency  # pyright: ignore[reportMissingImports]
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
                _reject_traversal(extract_dir_resolved, member)
                rf.extract(member, extract_dir)

    else:
        raise ValueError(
            f"Unsupported archive format: {archive_path.name}. "
            "Supported formats: .zip, .tar, .tar.gz, .tar.bz2, .tar.xz, .rar"
        )


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
    """Download a publicly-accessible archive and load its media files.

    The archive is streamed to a temporary file in ``DATA_DIR``, extracted
    to a unique ``DATA_DIR/http_archive_extract_<id>/`` directory, then
    scanned with the standard
    :func:`~vtscore.datasets.loader.load_dataset_from_folder` pipeline.
    Both temporary paths are cleaned up after the run completes.

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
    description = "Download an archive (.zip, .tar, .rar) from a web URL and embed the media files inside"
    icon = "\U0001f310"
    hidden_from_picker = True
    multi_media = True
    fields = [
        ImporterField(
            key="url",
            label="Path or URL",
            field_type="url",
            description="A publicly accessible archive of media files to download and unpack.",
            hint="Supported archive formats: .zip, .tar, .tar.gz / .tgz, .tar.bz2, .rar.",
        ),
        ImporterField(
            key="media_type",
            label="Dataset MediaType",
            field_type="select",
            description="Type of media files contained in the archive.",
            default="audio",
        ),
    ]

    def __init__(self) -> None:
        super().__init__()
        from vtscore.media import all_folder_names

        for f in self.fields:
            if f.key == "media_type":
                f.options = all_folder_names()
                break

    def run(self, field_values: dict, medias: dict, thin: bool = False) -> None:
        url = field_values["url"]
        validate_url(url)
        media_type = field_values.get("media_type", "audio")
        specs = self.effective_source_specs(field_values)

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
                extract_dir,
                media_type,
                medias,
                on_progress=progress,
                thin=thin,
                embedder_name=emb_name,
                content_vectors=self.content_vectors or None,
                content_md5s=self.content_md5s or None,
                custom_metadata_map=self.custom_metadata_map or None,
                skip_embedding=skip_emb,
            )
            _run_converter_specs(extract_dir, media_type, field_values, specs, medias, thin=thin)
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
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        from vtscore.datasets.loader import load_dataset_from_folder_chunked

        extract_dir = self._download_and_extract(field_values)
        media_type = field_values.get("media_type", "audio")
        specs = self.effective_source_specs(field_values)
        converter_specs = [s for s in specs if s.converter is not None]
        emb_name = field_values.get("embedder", "")
        skip_emb = bool(field_values.get("skip_embedding"))
        try:
            yield from load_dataset_from_folder_chunked(
                extract_dir,
                media_type,
                chunk_size,
                thin=thin,
                embedder_name=emb_name,
                content_vectors=self.content_vectors or None,
                content_md5s=self.content_md5s or None,
                custom_metadata_map=self.custom_metadata_map or None,
                skip_embedding=skip_emb,
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
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL (must start with http:// or https://): {url}")
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
                for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tar", ".zip", ".rar"):
                    if tail.lower().endswith(suffix):
                        return tail[: -len(suffix)] or self.display_name
                return tail
        return self.display_name

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        origin = super().build_origin(field_values)
        specs = field_values.get("source_specs")
        if specs:
            if not isinstance(specs, str):
                specs = json.dumps(specs)
            origin["params"]["source_specs"] = specs
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

        from vtscore.datasets.sources.http_archive import HttpArchiveSource

        source = HttpArchiveSource(url)
        # Note: we intentionally do NOT call source.cleanup() here because
        # the extracted archive should remain cached for future resolve_file
        # calls (matching the previous behaviour of keeping resolve dirs).
        return source.resolve_path(origin_name, filename)


IMPORTER = HttpArchiveDatasetImporter()
