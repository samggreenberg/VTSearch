"""Server-files importer - embed media files listed in a server-side file.

The user supplies the absolute path of a file on the server that
identifies the media files to import.  Two file formats are accepted:

- **Text** (``.txt`` / ``.list``) - a UTF-8 text file with one media
  path per non-empty, non-comment line.  Lines beginning with ``#`` are
  comments.  The listed files are embedded by the server.
- **NumPy archive** (``.npz``) - a ``.npz`` file holding both the media
  paths and their pre-computed embedding vectors.  See
  :mod:`vtscore.datasets.importers._npz_vectors` for the supported
  array layouts.  When supplied, the importer **skips re-embedding** for
  every listed file and uses the vector from the archive instead.  This
  lets users import media that they have already embedded offline
  without paying for embedding twice.

Each listed entry may be the path of a file or a directory.  Symlinks
(to either files or directories) are followed, and directory entries
are walked recursively for media files.  The importer symlinks every
resulting file into a temporary directory and then delegates to
:mod:`server_folder` for the actual scanning/embedding.  After the
import each media's origin is rewritten to point at this importer so
that :meth:`resolve_file` can find the original file on disk.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vtscore.datasets.importers._npz_vectors import read_npz_embedder_name, read_npz_filenames_and_vectors
from vtscore.datasets.importers.base import DatasetImporter, ImporterField, SourceSpec
from vtscore.datasets.loader import load_dataset_from_folder, load_dataset_from_folder_chunked

logger = logging.getLogger(__name__)


def _cleanup_staging(staging: Path) -> None:
    """Remove the staging symlink farm, logging (not raising) failures.

    Cleanup runs in ``finally`` blocks where re-raising would mask the
    real error; but silently swallowing leaves a symlink farm behind on
    e.g. a permission error, so log every per-path failure.
    """

    def _onerror(func, path, exc_info):
        logger.warning("Failed to remove staging entry %s: %s", path, exc_info[1])

    shutil.rmtree(staging, onerror=_onerror)


def _read_text_paths_file(paths_file: Path) -> list[Path]:
    """Read media file paths from a UTF-8 text file, one per line."""
    base_dir = paths_file.resolve().parent
    paths: list[Path] = []
    for raw in paths_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        paths.append(candidate)
    return paths


def _read_npz_paths_file(paths_file: Path) -> tuple[list[Path], dict[str, Any], str]:
    """Read media paths + pre-computed vectors + embedder name from a ``.npz``.

    Returns ``(paths, path_to_vector, embedder_name)`` where keys of
    *path_to_vector* are the absolute path strings of the resolved entries
    (the same strings that appear in *paths*), and *embedder_name* is the
    value stored under ``"embedder_name"`` or ``"embedder"`` in the archive
    (``""`` when absent).
    """
    name_to_vector = read_npz_filenames_and_vectors(paths_file)
    embedder_name = read_npz_embedder_name(paths_file)
    base_dir = paths_file.resolve().parent
    paths: list[Path] = []
    path_to_vector: dict[str, Any] = {}
    for raw_name, vec in name_to_vector.items():
        line = raw_name.strip()
        if not line:
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        paths.append(candidate)
        path_to_vector[str(candidate)] = vec
    return paths, path_to_vector, embedder_name


def _read_paths_file(paths_file: Path) -> list[Path]:
    """Read media file paths from *paths_file*.

    Both ``.txt``/``.list`` (one path per line) and ``.npz`` (paths +
    pre-computed vectors) formats are supported.  This helper returns
    only the path list; callers that also want the NPZ vectors should
    use :func:`_read_paths_and_vectors`.
    """
    if not paths_file.is_file():
        raise FileNotFoundError(f"Paths file not found: {paths_file}")
    if paths_file.suffix.lower() == ".npz":
        paths, _, _embedder = _read_npz_paths_file(paths_file)
        return paths
    return _read_text_paths_file(paths_file)


def _read_paths_and_vectors(paths_file: Path) -> tuple[list[Path], dict[str, Any], str]:
    """Like :func:`_read_paths_file` but also returns pre-computed vectors and embedder name.

    For ``.txt`` / ``.list`` inputs the second and third tuple elements
    are an empty dict and ``""`` respectively.  For ``.npz`` inputs the
    second element maps each resolved path string to its pre-computed
    embedding vector, and the third element is the embedder name (``""``
    if the archive doesn't record one).
    """
    if not paths_file.is_file():
        raise FileNotFoundError(f"Paths file not found: {paths_file}")
    if paths_file.suffix.lower() == ".npz":
        return _read_npz_paths_file(paths_file)
    return _read_text_paths_file(paths_file), {}, ""


def _expand_paths(paths: list[Path]) -> list[Path]:
    """Expand any directory entries in *paths* into the files they contain.

    Both ``Path.is_file`` and ``Path.is_dir`` follow symlinks by default, so
    a list entry that's a symlink to a file or to a directory is treated
    just like the underlying target.  Directory entries are walked
    recursively with ``followlinks=True`` so symlinked sub-directories are
    also descended.
    """
    expanded: list[Path] = []
    for src in paths:
        if src.is_file():
            expanded.append(src)
        elif src.is_dir():
            for dirpath, _dirnames, filenames in os.walk(src, followlinks=True):
                for name in filenames:
                    expanded.append(Path(dirpath) / name)
    return expanded


def _symlink_paths(paths: list[Path], target_dir: Path) -> dict[str, Path]:
    """Symlink each *paths* entry into *target_dir*; return name→source map.

    Entries that are directories (or symlinks to directories) are walked
    recursively for files; every regular file found inside is symlinked
    into the staging directory with a disambiguated basename.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    name_to_source: dict[str, Path] = {}
    for src in _expand_paths(paths):
        if not src.is_file():
            continue
        # Disambiguate identical basenames by appending an index.
        base = src.name
        candidate_name = base
        idx = 1
        while candidate_name in name_to_source:
            candidate_name = f"{src.stem}__{idx}{src.suffix}"
            idx += 1
        link_path = target_dir / candidate_name
        link_path.symlink_to(src.resolve())
        name_to_source[candidate_name] = src.resolve()
    return name_to_source


class ServerFilesDatasetImporter(DatasetImporter):
    """Embed a list of media files identified by a server-side text file.

    Workflow:

    1. The user supplies the path of a text file on the server (one
       media-file path per line; ``#`` comments allowed).
    2. The importer reads the listed paths, symlinks each into a fresh
       temp directory, and runs the :mod:`server_folder` importer over
       that temp dir.
    3. Each loaded media's ``origin`` is rewritten to point back at this
       importer; ``origin_name`` is set to the original absolute path so
       that :meth:`resolve_file` can locate the file later.

    Cross-dataset features (e.g. applying a saved Detector to a different
    dataset via "Find") therefore continue to work, since each media
    carries enough provenance to re-fetch its bytes.
    """

    name = "server_files"
    display_name = "Files Manifest"
    description = (
        "Read a server-side file listing media paths (text file, one per line, "
        "OR a .npz archive that also supplies pre-computed embedding vectors) "
        "and embed every listed file"
    )
    icon = "\U0001f5c2"  # 🗂 - falls back to a generic file icon
    picker_view = "form"
    category = "server"
    fields = [
        ImporterField(
            key="media_type",
            label="Dataset MediaType",
            field_type="select",
            description=(
                "Type of media files the dataset ends up holding.  When source "
                "files of other types are listed in the paths file, an Include "
                "row with a matching converter pulls them in too."
            ),
            default="audio",
            required=False,
        ),
        ImporterField(
            key="paths_file",
            label="Paths file",
            field_type="server_path",
            description="A file that lists the media files (or folders) to pull into the dataset.",
            hint=(
                "Accepted formats:\n"
                " • .txt / .list - UTF-8 text, one path per line (lines starting with # are comments).\n"
                " • .npz - NumPy archive of paths plus pre-computed embedding vectors;\n"
                "   listed files skip re-embedding and use the supplied vectors.\n"
                "Symlinks are followed; directory entries are scanned recursively for media files."
            ),
            accept=".txt,.list,.npz",
        ),
    ]

    def __init__(self) -> None:
        super().__init__()
        from vtscore.media import all_folder_names

        for f in self.fields:
            if f.key == "media_type":
                f.options = all_folder_names()
                break

    def _stage_paths(self, field_values: dict[str, Any]) -> tuple[Path, dict[str, Path], dict[str, Any], str]:
        """Read the paths file and symlink each entry into a fresh temp dir.

        Returns ``(staging_dir, name_to_source, content_vectors, embedder_name)`` where:

        - ``staging_dir`` is the freshly created temp directory holding
          one symlink per imported file (caller must ``rmtree`` it).
        - ``name_to_source`` maps each staged symlink basename to the
          original absolute path it points at.
        - ``content_vectors`` maps each staged symlink basename to a
          pre-computed embedding vector.  Populated when the paths file
          is itself a ``.npz`` archive that holds vectors alongside the
          paths.  Returned as a local dict (not threaded through
          :meth:`~DatasetImporter.yield_precomputed`) so the singleton
          importer instance does not accumulate per-import state.
        - ``embedder_name`` is the name of the embedder that produced the
          pre-computed vectors (``""`` for plain text paths files or NPZ
          archives that don't record an embedder name).
        """
        paths_file = Path(field_values["paths_file"])
        paths, path_to_vector, embedder_name = _read_paths_and_vectors(paths_file)
        if not paths:
            raise ValueError(f"No paths found in {paths_file}")

        staging = Path(tempfile.mkdtemp(prefix="server_files_"))
        name_to_source = _symlink_paths(paths, staging)
        if not name_to_source:
            _cleanup_staging(staging)
            raise ValueError(f"None of the paths in {paths_file} resolved to existing files")

        # Rekey npz vectors from the original absolute path to the
        # staged symlink basename, which is what ``server_folder``'s
        # loader uses as the ``content_vectors`` key.
        content_vectors: dict[str, Any] = {}
        if path_to_vector:
            for name, source in name_to_source.items():
                vec = path_to_vector.get(str(source))
                if vec is not None:
                    content_vectors[name] = vec
        return staging, name_to_source, content_vectors, embedder_name

    def _rewrite_origins(
        self,
        medias: dict[int, dict[str, Any]],
        name_to_source: dict[str, Path],
        origin: dict[str, Any],
        embedder_name: str = "",
    ) -> None:
        """Point each media at its real source path instead of the symlink.

        Each media gets a fresh copy of *origin* so a later mutation of one
        media's ``origin.params`` cannot leak across siblings.  When
        *embedder_name* is non-empty it is stored in the origin params so
        the ``server_files`` :class:`~vtscore.datasets.sources.MediaSource`
        can surface the embedder on re-ingestion.
        """
        for media in medias.values():
            src = name_to_source.get(media.get("origin_name", "")) or name_to_source.get(media.get("filename", ""))
            if src is None:
                continue
            params = dict(origin.get("params", {}))
            if embedder_name:
                params["embedder_name"] = embedder_name
            media["origin"] = {
                "importer": origin.get("importer", ""),
                "params": params,
            }
            media["origin_name"] = str(src)
            media["media_path"] = str(src)

    def _load_direct_into(
        self,
        staging: Path,
        spec: SourceSpec,
        field_values: dict[str, Any],
        medias: dict,
        thin: bool,
        merged_vectors: dict[str, Any],
        content_embedder_name: str = "",
    ) -> bool:
        """Load files of ``spec.source_type`` from the staging dir.

        Returns ``True`` when the loader produced any output (i.e. did
        not raise ``ValueError`` for an empty folder).
        """
        from vtscore.media import get  # noqa: PLC0415

        mt = get(spec.source_type)
        try:
            load_dataset_from_folder(
                staging,
                mt.folder_import_name,
                medias,
                thin=thin,
                content_vectors=merged_vectors or None,
                content_embedder_name=content_embedder_name,
                content_md5s=self.content_md5s or None,
                custom_metadata_map=self.custom_metadata_map or None,
            )
        except ValueError:
            return False
        return True

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        from vtscore.media import get_by_folder_name  # noqa: PLC0415

        specs = self.effective_source_specs(field_values)
        output_type = get_by_folder_name(field_values.get("media_type", "")).type_id

        staging, name_to_source, npz_vectors, embedder_name = self._stage_paths(field_values)
        # Merge npz-supplied vectors with any vectors set externally on
        # ``self.content_vectors``.  NPZ vectors take priority for keys
        # that overlap.
        merged_vectors: dict[str, Any] = dict(self.content_vectors or {})
        merged_vectors.update(npz_vectors)
        try:
            had_direct = False
            for spec in specs:
                if spec.converter is None:
                    if self._load_direct_into(staging, spec, field_values, medias, thin, merged_vectors, embedder_name):
                        had_direct = True

            converter_rows = [s for s in specs if s.converter is not None]
            if converter_rows:
                from vtscore.converters.runner import run_converters_on_folder  # noqa: PLC0415

                run_converters_on_folder(
                    folder_path=staging,
                    converter_specs=converter_rows,
                    target_media_type=output_type,
                    medias=medias,
                    thin=thin,
                    base_origin={
                        "importer": self.name,
                        "params": {"paths_file": str(field_values.get("paths_file", ""))},
                    },
                )

            self._rewrite_origins(medias, name_to_source, self.build_origin(field_values), embedder_name)

            if not had_direct and not medias:
                raise ValueError(f"No {output_type} files found in listed paths")
        finally:
            _cleanup_staging(staging)

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        paths_file = Path(field_values["paths_file"])
        if not paths_file.exists():
            raise FileNotFoundError(f"Paths file not found: {paths_file}")
        if not paths_file.is_file():
            raise IsADirectoryError(f"Paths file must be a file: {paths_file}")
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
        from vtscore.media import get as media_get, get_by_folder_name  # noqa: PLC0415

        specs = self.effective_source_specs(field_values)
        output_type = get_by_folder_name(field_values.get("media_type", "")).type_id

        staging, name_to_source, npz_vectors, embedder_name = self._stage_paths(field_values)
        origin = self.build_origin(field_values)
        merged_vectors: dict[str, Any] = dict(self.content_vectors or {})
        merged_vectors.update(npz_vectors)
        try:
            for spec in specs:
                if spec.converter is not None:
                    continue
                mt = media_get(spec.source_type)
                try:
                    for chunk in load_dataset_from_folder_chunked(
                        staging,
                        mt.folder_import_name,
                        chunk_size,
                        thin=thin,
                        content_vectors=merged_vectors or None,
                        content_embedder_name=embedder_name,
                        content_md5s=self.content_md5s or None,
                        custom_metadata_map=self.custom_metadata_map or None,
                    ):
                        self._rewrite_origins(chunk, name_to_source, origin, embedder_name)
                        yield chunk
                except ValueError:
                    # Empty for this source type - keep going; converter
                    # rows may still produce output.
                    pass

            converter_rows = [s for s in specs if s.converter is not None]
            if converter_rows:
                from vtscore.converters.runner import run_converters_on_folder  # noqa: PLC0415

                converter_chunk: dict[int, dict[str, Any]] = {}
                run_converters_on_folder(
                    folder_path=staging,
                    converter_specs=converter_rows,
                    target_media_type=output_type,
                    medias=converter_chunk,
                    thin=thin,
                    base_origin={
                        "importer": self.name,
                        "params": {"paths_file": str(field_values.get("paths_file", ""))},
                    },
                )
                if converter_chunk:
                    # Converter-origin medias keep their converter origin
                    # (don't rewrite to the server_files origin - the
                    # converted output isn't a directly-listed source
                    # file).
                    yield converter_chunk
        finally:
            _cleanup_staging(staging)

    def run_chunked_cli(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        paths_file = Path(field_values["paths_file"])
        if not paths_file.exists():
            raise FileNotFoundError(f"Paths file not found: {paths_file}")
        if not paths_file.is_file():
            raise IsADirectoryError(f"Paths file must be a file: {paths_file}")
        yield from self.run_chunked(field_values, chunk_size, thin=thin)

    def default_display_name(self, field_values: dict[str, Any]) -> str:
        paths_file = (field_values.get("paths_file") or "").strip()
        if paths_file:
            stem = Path(paths_file).stem
            if stem:
                return stem
        return self.display_name

    def origin_display(self, origin: dict[str, Any]) -> str:
        params = origin.get("params", {})
        return f"server_files:{params.get('paths_file', '')}"

    def can_reload_from_origin(self, origin: dict[str, Any]) -> bool:
        params = origin.get("params", {})
        paths_file = params.get("paths_file", "")
        return bool(paths_file) and Path(paths_file).is_file()

    def resolve_file(
        self,
        origin: dict[str, Any],
        origin_name: str = "",
        filename: str = "",
    ) -> Path | None:
        # Each media's origin_name is the original absolute path
        # (set during :meth:`run`), so we can return it directly.
        for candidate in (origin_name, filename):
            if candidate:
                p = Path(candidate)
                if p.is_file():
                    return p
        return None


IMPORTER = ServerFilesDatasetImporter()
