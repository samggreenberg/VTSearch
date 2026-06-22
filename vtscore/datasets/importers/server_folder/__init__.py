"""Server-folder importer – scans a directory of media files and embeds them.

The importer participates in the multi-media flow (see
:doc:`/docs/EXTENDING-media`).  The user picks an output media
type and an ordered list of :class:`~vtscore.datasets.importers.base.SourceSpec`
rows that declare which media types to scan for.  Each row is either:

* a "direct" row (``converter`` ``= None``) whose ``source_type`` equals
  the chosen output media type - the scanner reads matching files and
  embeds them with the target embedder, or
* a converter row, where the scanner reads files of the converter's
  ``source_type`` and feeds them through the converter (with the
  per-row params) to produce media of the output type.

When the output type is ``"image"``, ``*.pdf`` files in the folder are
also expanded as per-page images.  (PDFs participate independently of
the explicit converter rows - they are tied to the "image" output type
rather than to a converter.)

Archives
--------
The folder field accepts either a directory **or** a single archive file
(``.zip`` / ``.tar`` / ``.rar`` …): an archive path is extracted and its
contents loaded directly.  Additionally, when the ``dig_archives`` checkbox
is enabled, any archives found *inside* the scanned directory are extracted
and their media folded into the dataset.  Archive-derived media carry a
``local_archive`` origin so they re-derive by re-extracting on demand (see
:mod:`vtscore.datasets.archive`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from vtscore.datasets.archive import (
    append_medias,
    extract_archive_cached,
    find_archives,
    is_archive_path,
    iter_archive_chunks,
    load_archive_into,
)
from vtscore.datasets.importers.base import DatasetImporter, ImporterField, SourceSpec
from vtscore.datasets.loader import load_dataset_from_folder, load_dataset_from_folder_chunked
from vtscore.datasets.pdf import load_pdf_images_into


def _coerce_recursive(field_values: dict[str, Any]) -> bool:
    """Parse the ``recursive`` field value as a bool; default ``True``."""
    val = field_values.get("recursive", True)
    if isinstance(val, bool):
        return val
    return str(val).lower() != "false"


def _coerce_dig_archives(field_values: dict[str, Any]) -> bool:
    """Parse the ``dig_archives`` field value as a bool; default ``False``."""
    val = field_values.get("dig_archives", False)
    if isinstance(val, bool):
        return val
    return str(val).lower() == "true"


def _run_converter_specs(
    folder: Path,
    output_type: str,
    converter_specs: list[SourceSpec],
    medias: dict,
    thin: bool = False,
    recursive: bool = True,
    folder_path_for_origin: str = "",
) -> None:
    """Hand the non-direct rows of a multi-media spec to the runner."""
    runnable = [s for s in converter_specs if s.converter is not None]
    if not runnable:
        return

    from vtscore.converters.runner import run_converters_on_folder  # noqa: PLC0415

    base_origin = {
        "importer": "server_folder",
        "params": {"path": folder_path_for_origin or str(folder), "media_type": output_type},
    }
    run_converters_on_folder(
        folder_path=folder,
        converter_specs=runnable,
        target_media_type=output_type,
        medias=medias,
        thin=thin,
        base_origin=base_origin,
        recursive=recursive,
    )


class ServerFolderDatasetImporter(DatasetImporter):
    """Embed all media files found in a directory on the server's filesystem.

    The user supplies an absolute filesystem path (on the **server**) plus
    an output media type and a list of :class:`SourceSpec` rows describing
    which source types to scan for (and which converters to apply).  See
    :doc:`/docs/EXTENDING-media` for the full design.

    The path may also point at a single **archive file** (``.zip`` / ``.tar``
    / ``.rar`` …), in which case the archive is extracted and its contents
    imported.  With ``dig_archives`` enabled, archives found inside the
    scanned directory are extracted and folded in too.

    When the output media type is ``"image"``, any ``*.pdf`` files in the
    folder are also processed: each page is rendered as a separate image.

    .. note::
       This importer reads files from the server's filesystem.  In the web
       UI it powers the dedicated "Server Folder" flow via
       :attr:`picker_view` ``= "server_folder"``, which opens a server-side
       directory browser instead of the generic form.  For importing files
       from the **browser machine** (which may be different from the
       server), there is a separate :class:`LocalFolderDatasetImporter`
       whose card delegates to ``/api/dataset/import-local-folder``; that
       endpoint streams the upload to a temp directory and then re-enters
       this importer to do the actual scanning and embedding.
    """

    name = "server_folder"
    display_name = "Folder"
    description = "Browse the server's filesystem and import media files from a directory or archive"
    icon = "\U0001f4c1"  # 📁 - frontend renders as a folder icon
    picker_view = "server_folder"
    category = "server"
    fields = [
        ImporterField(
            key="media_type",
            label="Dataset MediaType",
            field_type="select",
            description="Type of media files the dataset ends up holding.",
            default="audio",
            required=False,
        ),
        ImporterField(
            key="path",
            label="Folder or archive",
            field_type="folder",
            description=(
                "Absolute path to a directory containing media files, or to a single "
                "archive file (.zip, .tar, .tar.gz, .tar.bz2, .tar.xz, .rar)."
            ),
        ),
        ImporterField(
            key="recursive",
            label="Include subfolders",
            field_type="checkbox",
            description=(
                "When enabled, scan subdirectories recursively.  When disabled, "
                "only files directly inside the chosen folder are imported."
            ),
            default="true",
            required=False,
        ),
        ImporterField(
            key="dig_archives",
            label="Dig into archives",
            field_type="checkbox",
            description=(
                "When enabled, archives (.zip, .tar, .rar …) found inside the chosen "
                "folder are extracted and their media imported too.  Has no effect when "
                "the path itself is an archive (that is always extracted)."
            ),
            default="false",
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

    def _resolve_output_type(self, field_values: dict[str, Any], specs: list[SourceSpec]) -> str:
        """Resolve the output media type from the direct spec or ``media_type``."""
        for spec in specs:
            if spec.converter is None:
                return spec.source_type
        # Multi-media imports without a direct row still have an output type
        # from ``media_type``; resolve it explicitly so PDF expansion and
        # origin recording behave correctly.
        from vtscore.media import get_by_folder_name  # noqa: PLC0415

        return get_by_folder_name(field_values.get("media_type", "")).type_id

    def _accumulate_direct(
        self,
        folder: Path,
        spec: SourceSpec,
        medias: dict,
        thin: bool,
        recursive: bool,
    ) -> bool:
        """Load files of ``spec.source_type`` and append them to *medias*.

        Loads into a temporary dict (so the per-call ``medias.clear()`` of
        :func:`load_dataset_from_folder` doesn't wipe earlier rows / archives)
        and merges the result in.  Returns ``True`` if any files were found.
        """
        from vtscore.media import get  # noqa: PLC0415

        mt = get(spec.source_type)
        temp: dict[int, dict[str, Any]] = {}
        try:
            load_dataset_from_folder(
                folder,
                mt.folder_import_name,
                temp,
                thin=thin,
                content_vectors=self.content_vectors or None,
                content_md5s=self.content_md5s or None,
                custom_metadata_map=self.custom_metadata_map or None,
                recursive=recursive,
            )
        except ValueError:
            return False
        append_medias(medias, temp)
        return True

    def _load_archive(self, archive: Path, output_type: str, specs: list[SourceSpec], medias: dict, thin: bool) -> None:
        load_archive_into(
            archive,
            output_type,
            specs,
            medias,
            thin=thin,
            content_vectors=self.content_vectors,
            content_md5s=self.content_md5s,
            custom_metadata_map=self.custom_metadata_map,
        )

    def run(self, field_values: dict, medias: dict, thin: bool = False) -> None:
        path = Path(field_values["path"])
        recursive = _coerce_recursive(field_values)
        dig_archives = _coerce_dig_archives(field_values)
        specs = self.effective_source_specs(field_values)
        output_type = self._resolve_output_type(field_values, specs)

        # Single clear up front: every loader below appends from here.
        medias.clear()

        if is_archive_path(path):
            self._load_archive(path, output_type, specs, medias, thin)
            if not medias:
                raise ValueError(f"No {output_type} files found in archive")
            return

        has_direct_files = False
        for spec in specs:
            if spec.converter is None:
                if self._accumulate_direct(path, spec, medias, thin, recursive):
                    has_direct_files = True

        if output_type == "image":
            load_pdf_images_into(path, medias, thin=thin, recursive=recursive)

        _run_converter_specs(
            path,
            output_type,
            [s for s in specs if s.converter is not None],
            medias,
            thin=thin,
            recursive=recursive,
            folder_path_for_origin=str(path),
        )

        if dig_archives:
            for archive in find_archives(path, recursive):
                self._load_archive(archive, output_type, specs, medias, thin)

        if not has_direct_files and not medias:
            raise ValueError(f"No {output_type} files found in folder")

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        path = Path(field_values["path"])
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if not is_archive_path(path) and not path.is_dir():
            raise NotADirectoryError(f"Not a directory or supported archive: {path}")
        self.run(field_values, medias, thin=thin)

    @property
    def supports_chunked(self) -> bool:
        return True

    def run_chunked(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        path = Path(field_values["path"])
        recursive = _coerce_recursive(field_values)
        dig_archives = _coerce_dig_archives(field_values)

        specs = self.effective_source_specs(field_values)
        direct_specs = [s for s in specs if s.converter is None]
        converter_specs = [s for s in specs if s.converter is not None]
        output_type = self._resolve_output_type(field_values, specs)

        if is_archive_path(path):
            yield from self._chunk_archive(path, output_type, specs, chunk_size, thin)
            return

        # Chunked load only fires for the direct rows.  Converter rows produce
        # a separate chunk afterwards (matching legacy behaviour).
        for spec in direct_specs:
            from vtscore.media import get  # noqa: PLC0415

            mt = get(spec.source_type)
            try:
                yield from load_dataset_from_folder_chunked(
                    path,
                    mt.folder_import_name,
                    chunk_size,
                    thin=thin,
                    content_vectors=self.content_vectors or None,
                    content_md5s=self.content_md5s or None,
                    custom_metadata_map=self.custom_metadata_map or None,
                    recursive=recursive,
                )
            except ValueError:
                # Empty folder for this source type - keep going; later
                # PDF / converter rows may still produce output.
                pass

        if output_type == "image":
            chunk: dict[int, dict[str, Any]] = {}
            load_pdf_images_into(path, chunk, thin=thin, recursive=recursive)
            if chunk:
                yield chunk

        if converter_specs:
            converter_chunk: dict[int, dict[str, Any]] = {}
            _run_converter_specs(
                path,
                output_type,
                converter_specs,
                converter_chunk,
                thin=thin,
                recursive=recursive,
                folder_path_for_origin=str(path),
            )
            if converter_chunk:
                yield converter_chunk

        if dig_archives:
            for archive in find_archives(path, recursive):
                yield from self._chunk_archive(archive, output_type, specs, chunk_size, thin)

    def _chunk_archive(
        self,
        archive: Path,
        output_type: str,
        specs: list[SourceSpec],
        chunk_size: int,
        thin: bool,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        yield from iter_archive_chunks(
            archive,
            output_type,
            specs,
            chunk_size,
            thin=thin,
            content_vectors=self.content_vectors,
            content_md5s=self.content_md5s,
            custom_metadata_map=self.custom_metadata_map,
        )

    def run_chunked_cli(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        path = Path(field_values["path"])
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if not is_archive_path(path) and not path.is_dir():
            raise NotADirectoryError(f"Not a directory or supported archive: {path}")
        yield from self.run_chunked(field_values, chunk_size, thin=thin)

    def build_cli_args(self, field_values: dict[str, Any]) -> str:
        base = super().build_cli_args(field_values)
        specs = field_values.get("source_specs")
        if specs:
            import json as _json  # noqa: PLC0415

            if not isinstance(specs, str):
                specs = _json.dumps(specs)
            base += f" --source-specs '{specs}'"
        return base

    def default_display_name(self, field_values: dict[str, Any]) -> str:
        path_str = (field_values.get("path") or "").strip()
        if path_str:
            leaf = Path(path_str).name
            if leaf and is_archive_path(leaf):
                # Strip the archive extension for a tidier dataset name.
                for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar", ".zip", ".rar"):
                    if leaf.lower().endswith(suffix):
                        return leaf[: -len(suffix)] or self.display_name
            if leaf:
                return leaf
        return self.display_name

    def origin_display(self, origin: dict[str, Any]) -> str:
        params = origin.get("params", {})
        return f"server_folder:{params.get('path', '')}"

    def can_reload_from_origin(self, origin: dict[str, Any]) -> bool:
        params = origin.get("params", {})
        folder_path = params.get("path", "")
        if not folder_path:
            return False
        p = Path(folder_path)
        return p.is_dir() or (is_archive_path(p) and p.is_file())

    def resolve_file(
        self,
        origin: dict[str, Any],
        origin_name: str = "",
        filename: str = "",
    ) -> Path | None:
        folder = origin.get("params", {}).get("path", "")
        if not folder:
            return None
        if is_archive_path(folder):
            # A server_folder dataset whose path is a single archive: resolve
            # via the extracted (cached) directory.
            from vtscore.datasets.sources.local_folder import LocalFolderSource  # noqa: PLC0415

            extract_dir = extract_archive_cached(folder)
            return LocalFolderSource(extract_dir).resolve_path(origin_name, filename).path
        from vtscore.datasets.sources.local_folder import LocalFolderSource  # noqa: PLC0415

        source = LocalFolderSource(folder)
        return source.resolve_path(origin_name, filename).path


IMPORTER = ServerFolderDatasetImporter()
