"""Server-files importer — embed media files listed in a server-side text file.

The user supplies the absolute path of a UTF-8 text file on the server.
Each non-empty, non-comment line of that file is treated as the
absolute path (or path relative to the text file's directory) of a
media file to embed.  The importer symlinks each listed file into a
temporary directory and then delegates to :mod:`server_folder` for the
actual scanning/embedding.  After the import each media's origin is
rewritten to point at this importer so that
:meth:`resolve_file` can find the original file on disk.

Lines beginning with ``#`` are treated as comments and skipped.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.datasets.loader import load_dataset_from_folder, load_dataset_from_folder_chunked


def _read_paths_file(paths_file: Path) -> list[Path]:
    """Read media file paths from *paths_file*, one per line."""
    if not paths_file.is_file():
        raise FileNotFoundError(f"Paths file not found: {paths_file}")

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


def _symlink_paths(paths: list[Path], target_dir: Path) -> dict[str, Path]:
    """Symlink each *paths* entry into *target_dir*; return name→source map."""
    target_dir.mkdir(parents=True, exist_ok=True)
    name_to_source: dict[str, Path] = {}
    for src in paths:
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
    display_name = "Files"
    description = (
        "Read a text file on the server containing media-file paths (one per line) and embed every listed file"
    )
    icon = "\U0001f5c2"  # 🗂 — falls back to a generic file icon
    picker_view = "form"
    category = "server"
    fields = [
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            description="Type of media files listed in the paths file.",
            default="audio",
        ),
        ImporterField(
            key="paths_file",
            label="Paths File",
            field_type="server_path",
            description="Absolute server path to a text file containing one media-file path per line.",
            accept=".txt,.list",
        ),
    ]

    def __init__(self) -> None:
        super().__init__()
        from vtsearch.media import all_folder_names

        for f in self.fields:
            if f.key == "media_type":
                f.options = all_folder_names()
                break

    def _stage_paths(self, field_values: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
        """Read the paths file and symlink each entry into a fresh temp dir.

        Returns the staging directory and a name→source-path mapping.  The
        caller is responsible for ``rmtree``-ing the staging directory.
        """
        paths_file = Path(field_values["paths_file"])
        paths = _read_paths_file(paths_file)
        if not paths:
            raise ValueError(f"No paths found in {paths_file}")

        staging = Path(tempfile.mkdtemp(prefix="server_files_"))
        name_to_source = _symlink_paths(paths, staging)
        if not name_to_source:
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError(f"None of the paths in {paths_file} resolved to existing files")
        return staging, name_to_source

    def _rewrite_origins(
        self,
        medias: dict[int, dict[str, Any]],
        name_to_source: dict[str, Path],
        origin: dict[str, Any],
    ) -> None:
        """Point each media at its real source path instead of the symlink."""
        for media in medias.values():
            src = name_to_source.get(media.get("origin_name", "")) or name_to_source.get(media.get("filename", ""))
            if src is None:
                continue
            media["origin"] = origin
            media["origin_name"] = str(src)
            media["media_path"] = str(src)

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        media_type = field_values.get("media_type", "audio")
        embedder = field_values.get("embedder", "")
        skip_emb = bool(field_values.get("skip_embedding"))

        staging, name_to_source = self._stage_paths(field_values)
        try:
            load_dataset_from_folder(
                staging,
                media_type,
                medias,
                thin=thin,
                embedder_name=embedder,
                content_vectors=self.content_vectors or None,
                content_md5s=self.content_md5s or None,
                custom_metadata_map=self.custom_metadata_map or None,
                skip_embedding=skip_emb,
            )

            self._rewrite_origins(medias, name_to_source, self.build_origin(field_values))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

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
        media_type = field_values.get("media_type", "audio")
        embedder = field_values.get("embedder", "")
        skip_emb = bool(field_values.get("skip_embedding"))

        staging, name_to_source = self._stage_paths(field_values)
        origin = self.build_origin(field_values)
        try:
            for chunk in load_dataset_from_folder_chunked(
                staging,
                media_type,
                chunk_size,
                thin=thin,
                embedder_name=embedder,
                content_vectors=self.content_vectors or None,
                content_md5s=self.content_md5s or None,
                custom_metadata_map=self.custom_metadata_map or None,
                skip_embedding=skip_emb,
            ):
                self._rewrite_origins(chunk, name_to_source, origin)
                yield chunk
        finally:
            shutil.rmtree(staging, ignore_errors=True)

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

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, str] = {}
        paths_file = field_values.get("paths_file", "")
        if paths_file:
            params["paths_file"] = str(paths_file)
        media_type = field_values.get("media_type", "")
        if media_type:
            params["media_type"] = media_type
        return {"importer": self.name, "params": params}

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
