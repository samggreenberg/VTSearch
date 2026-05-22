"""Server-folder importer – scans a directory of media files and embeds them.

The importer participates in the new multi-media flow (see
:doc:`/docs/plans/multi-media-import`).  The user picks an output media
type and an ordered list of :class:`~vtscore.datasets.importers.base.SourceSpec`
rows that declare which media types to scan for.  Each row is either:

* a "direct" row (``converter`` ``= None``) whose ``source_type`` equals
  the chosen output media type — the scanner reads matching files and
  embeds them with the target embedder, or
* a converter row, where the scanner reads files of the converter's
  ``source_type`` and feeds them through the converter (with the
  per-row params) to produce media of the output type.

When the output type is ``"image"``, ``*.pdf`` files in the folder are
also expanded as per-page images.  (PDFs participate independently of
the explicit converter rows — they are tied to the "image" output type
rather than to a converter.)
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Iterator, cast

from vtscore.datasets.importers.base import DatasetImporter, ImporterField, SourceSpec
from vtscore.datasets.loader import load_dataset_from_folder, load_dataset_from_folder_chunked
from vtscore.security.path_validation import glob_top_level, rglob_follow_symlinks


def _coerce_recursive(field_values: dict[str, Any]) -> bool:
    """Parse the ``recursive`` field value as a bool; default ``True``."""
    val = field_values.get("recursive", True)
    if isinstance(val, bool):
        return val
    return str(val).lower() != "false"


def _load_pdf_images(  # noqa: C901
    folder: Path,
    medias: dict[int, dict[str, Any]],
    thin: bool = False,
    embedder_name: str = "",
    recursive: bool = True,
) -> None:
    """Expand all PDFs in *folder* into per-page image medias.

    Each page is rendered at 150 DPI, embedded with the image embedder, and
    appended to *medias* with sequential IDs continuing from the current
    maximum.  The ``origin`` is set to ``{"importer": "pdf", "params":
    {"path": ...}}`` so the provenance points back to the source document.
    """
    pdf_files = sorted(rglob_follow_symlinks(folder, "*.pdf") if recursive else glob_top_level(folder, "*.pdf"))
    if not pdf_files:
        return

    from vtscore.datasets.pdf import render_pdf_pages  # noqa: PLC0415
    from vtscore.media import embedders_for_type, get_by_folder_name, get_embedder  # noqa: PLC0415

    mt = get_by_folder_name("image")

    # Resolve the embedder
    emb = None
    if embedder_name:
        try:
            emb = get_embedder(embedder_name)
        except KeyError:
            pass
    if emb is None:
        avail = embedders_for_type(mt.type_id)
        if avail:
            emb = avail[0]
    if emb is None:
        return

    if getattr(emb, "_model", None) is None:
        emb.load_models()

    embedder_id = emb.name
    media_id = max(medias.keys(), default=0) + 1

    for pdf_path in pdf_files:
        origin = {"importer": "pdf", "params": {"path": str(pdf_path)}}
        pages = render_pdf_pages(pdf_path)
        # Relative path prefix so that PDFs in different subdirectories
        # produce distinct page names.
        pdf_rel = pdf_path.relative_to(folder).as_posix()

        for page_name, pil_image in pages:
            # Image-type embedders all implement embed_pil_image, but
            # it's not declared on the MediaEmbedder ABC.
            embedding = cast(Any, emb).embed_pil_image(pil_image)
            if embedding is None:
                continue

            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            # page_name is e.g. "doc.pdf-1"; prefix with relative dir
            # so that identically-named PDFs in different folders stay
            # distinct (e.g. "subdir/doc.pdf-1").
            rel_dir = str(Path(pdf_rel).parent)
            if rel_dir and rel_dir != ".":
                full_page_name = f"{rel_dir}/{page_name}"
            else:
                full_page_name = page_name

            media_data: dict[str, Any] = {
                "id": media_id,
                "media_type": mt.type_id,
                "embedder": embedder_id,
                "file_size": len(image_bytes),
                "md5": hashlib.md5(image_bytes).hexdigest(),
                "embedding": embedding,
                "filename": full_page_name,
                "category": "custom",
                "origin": origin,
                "origin_name": full_page_name,
                "media_bytes": None if thin else image_bytes,
                "media_string": None,
                "media_path": str(pdf_path.resolve()),
                "duration": 0,
                "width": pil_image.width,
                "height": pil_image.height,
            }
            medias[media_id] = media_data
            media_id += 1


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
    :doc:`/docs/plans/multi-media-import` for the full design.

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
    description = "Browse the server's filesystem and import media files from a directory"
    icon = "\U0001f4c1"  # 📁 — frontend renders as a folder icon
    picker_view = "server_folder"
    category = "server"
    multi_media = True
    fields = [
        ImporterField(
            key="media_type",
            label="Dataset MediaType",
            field_type="select",
            description="Type of media files the dataset ends up holding.",
            default="audio",
        ),
        ImporterField(
            key="path",
            label="Folder",
            field_type="folder",
            description="Absolute path to the directory containing media files.",
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
    ]

    def __init__(self) -> None:
        super().__init__()
        from vtscore.media import all_folder_names

        for f in self.fields:
            if f.key == "media_type":
                f.options = all_folder_names()
                break

    def _load_direct(
        self,
        folder: Path,
        spec: SourceSpec,
        field_values: dict[str, Any],
        medias: dict,
        thin: bool,
        recursive: bool,
    ) -> bool:
        """Load files of ``spec.source_type`` directly into *medias*.

        Returns ``True`` if any files were found (i.e. the loader did not
        raise ``ValueError`` for an empty folder), ``False`` otherwise.
        """
        from vtscore.media import get  # noqa: PLC0415

        mt = get(spec.source_type)
        try:
            load_dataset_from_folder(
                folder,
                mt.folder_import_name,
                medias,
                thin=thin,
                embedder_name=field_values.get("embedder", ""),
                content_vectors=self.content_vectors or None,
                content_md5s=self.content_md5s or None,
                custom_metadata_map=self.custom_metadata_map or None,
                skip_embedding=bool(field_values.get("skip_embedding")),
                recursive=recursive,
            )
        except ValueError:
            return False
        return True

    def run(self, field_values: dict, medias: dict, thin: bool = False) -> None:  # noqa: C901
        folder = Path(field_values["path"])
        recursive = _coerce_recursive(field_values)

        specs = self.effective_source_specs(field_values)
        output_type = ""
        for spec in specs:
            if spec.converter is None:
                output_type = spec.source_type
                break
        if not output_type:
            # Multi-media imports without a direct row still have an
            # output type from ``media_type``; resolve it explicitly so
            # PDF expansion and origin recording behave correctly.
            from vtscore.media import get_by_folder_name  # noqa: PLC0415

            output_type = get_by_folder_name(field_values.get("media_type", "")).type_id

        has_direct_files = False
        for spec in specs:
            if spec.converter is None:
                if self._load_direct(folder, spec, field_values, medias, thin, recursive):
                    has_direct_files = True

        if output_type == "image":
            _load_pdf_images(
                folder,
                medias,
                thin=thin,
                embedder_name=field_values.get("embedder", ""),
                recursive=recursive,
            )

        _run_converter_specs(
            folder,
            output_type,
            [s for s in specs if s.converter is not None],
            medias,
            thin=thin,
            recursive=recursive,
            folder_path_for_origin=str(folder),
        )

        if not has_direct_files and not medias:
            raise ValueError(f"No {output_type} files found in folder")

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
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        folder = Path(field_values["path"])
        recursive = _coerce_recursive(field_values)

        specs = self.effective_source_specs(field_values)
        direct_specs = [s for s in specs if s.converter is None]
        converter_specs = [s for s in specs if s.converter is not None]

        output_type = direct_specs[0].source_type if direct_specs else ""
        if not output_type:
            from vtscore.media import get_by_folder_name  # noqa: PLC0415

            output_type = get_by_folder_name(field_values.get("media_type", "")).type_id

        # Chunked load only fires for the direct row.  Converter rows
        # produce a separate chunk afterwards (matching legacy
        # behaviour).
        for spec in direct_specs:
            from vtscore.media import get  # noqa: PLC0415

            mt = get(spec.source_type)
            try:
                yield from load_dataset_from_folder_chunked(
                    folder,
                    mt.folder_import_name,
                    chunk_size,
                    thin=thin,
                    embedder_name=field_values.get("embedder", ""),
                    content_vectors=self.content_vectors or None,
                    content_md5s=self.content_md5s or None,
                    custom_metadata_map=self.custom_metadata_map or None,
                    skip_embedding=bool(field_values.get("skip_embedding")),
                    recursive=recursive,
                )
            except ValueError:
                # Empty folder for this source type — keep going; later
                # PDF / converter rows may still produce output.
                pass

        if output_type == "image":
            chunk: dict[int, dict[str, Any]] = {}
            _load_pdf_images(folder, chunk, thin=thin, recursive=recursive)
            if chunk:
                yield chunk

        if converter_specs:
            converter_chunk: dict[int, dict[str, Any]] = {}
            _run_converter_specs(
                folder,
                output_type,
                converter_specs,
                converter_chunk,
                thin=thin,
                recursive=recursive,
                folder_path_for_origin=str(folder),
            )
            if converter_chunk:
                yield converter_chunk

    def run_chunked_cli(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        folder = Path(field_values["path"])
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
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
            if leaf:
                return leaf
        return self.display_name

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        origin = super().build_origin(field_values)
        specs = field_values.get("source_specs")
        if specs:
            import json as _json  # noqa: PLC0415

            if not isinstance(specs, str):
                specs = _json.dumps(specs)
            origin["params"]["source_specs"] = specs
        return origin

    def origin_display(self, origin: dict[str, Any]) -> str:
        params = origin.get("params", {})
        return f"server_folder:{params.get('path', '')}"

    def can_reload_from_origin(self, origin: dict[str, Any]) -> bool:
        params = origin.get("params", {})
        folder_path = params.get("path", "")
        return bool(folder_path) and Path(folder_path).is_dir()

    def resolve_file(
        self,
        origin: dict[str, Any],
        origin_name: str = "",
        filename: str = "",
    ) -> Path | None:
        folder = origin.get("params", {}).get("path", "")
        if not folder:
            return None
        from vtscore.datasets.sources.local_folder import LocalFolderSource

        source = LocalFolderSource(folder)
        return source.resolve_path(origin_name, filename)


IMPORTER = ServerFolderDatasetImporter()
