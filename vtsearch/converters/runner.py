"""Run selected converters on a folder of media files.

This module provides :func:`run_converters_on_folder`, a reusable utility
that any dataset importer can call to scan a directory for source media
files, convert them via one or more :class:`MediaConverter` instances, embed
the results with the target media type's embedder, and append them to an
existing medias dict.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from vtsearch.converters import get_converter
from vtsearch.utils.paths import rglob_follow_symlinks

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    from vtsearch.utils.progress import get_thread_progress

    cb = get_thread_progress()
    if cb is not None:
        return cb
    from vtsearch.utils import update_progress

    return update_progress


def run_converters_on_folder(
    folder_path: Path,
    converter_names: list[str],
    target_media_type: str,
    medias: dict[int, dict[str, Any]],
    thin: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    base_origin: dict[str, Any] | None = None,
) -> None:
    """Scan *folder_path* for source files, convert them, and add to *medias*.

    For each converter named in *converter_names*:

    1. Look up the converter's source media type to get file extensions.
    2. Scan *folder_path* recursively for files matching those extensions.
    3. For each source file, run the converter to produce output media dicts.
    4. Embed each output with the target media type's embedder.
    5. Assign sequential IDs continuing from the current max in *medias*.
    6. Set origin to ``{"importer": "converter", "params": {...}}``.

    Args:
        folder_path: Root directory to scan.
        converter_names: List of converter names (e.g. ``["video2image"]``).
        target_media_type: The target media type identifier
            (e.g. ``"image"``).
        medias: The medias dict to append to (not cleared).
        thin: When ``True``, store converter output in a temp directory
            instead of holding bytes in memory.  Currently all converted
            media is kept in-memory regardless, since converters produce
            bytes directly.
        on_progress: Optional progress callback.
        base_origin: The origin dict of the parent import (e.g.
            ``{"importer": "folder", "params": {"path": "..."}}``) used
            to record provenance.
    """
    if not converter_names:
        return

    # Validate converter names up front — skip model loading if none are valid.
    valid_converters = []
    for cname in converter_names:
        c = get_converter(cname)
        if c is not None:
            valid_converters.append(c)
    if not valid_converters:
        return

    if on_progress is None:
        on_progress = _default_progress()

    from vtsearch.media import embedders_for_type, get as media_get, get_by_folder_name  # noqa: PLC0415

    # Resolve target media type and its embedder.
    target_mt = get_by_folder_name(target_media_type)

    # Filter to converters that actually target this media type.
    valid_converters = [c for c in valid_converters if c.target_type == target_mt.type_id]
    if not valid_converters:
        return

    # Resolve the embedder for the target media type
    avail = embedders_for_type(target_mt.type_id)
    target_emb = avail[0] if avail else None

    if target_emb is not None and getattr(target_emb, "_model", None) is None:
        target_emb.load_models()

    media_id = max(medias.keys(), default=0) + 1

    for converter in valid_converters:

        # Get the source media type to know which file extensions to scan.
        try:
            source_mt = media_get(converter.source_type)
        except KeyError:
            continue

        # Scan folder for source files.
        source_files: list[Path] = []
        for ext in source_mt.file_extensions:
            source_files.extend(rglob_follow_symlinks(folder_path, ext))
        source_files.sort()

        if not source_files:
            continue

        on_progress(
            "converting",
            f"Converting {source_mt.name} files via {converter.display_name}...",
            0,
            len(source_files),
        )

        for file_idx, source_path in enumerate(source_files):
            source_rel = source_path.relative_to(folder_path).as_posix()
            on_progress(
                "converting",
                f"Converting {source_rel} via {converter.display_name}...",
                file_idx + 1,
                len(source_files),
            )

            # Build a minimal source media dict for the converter.
            source_media: dict[str, Any] = {
                "filename": source_rel,
                "media_path": str(source_path.resolve()),
            }
            # Only load bytes if the file is small enough or thin is False.
            if not thin:
                try:
                    source_media["media_bytes"] = source_path.read_bytes()
                except Exception:
                    continue

            try:
                outputs = converter.convert(source_media)
            except Exception as exc:
                print(f"Converter {converter.name} failed on {source_rel}: {exc}")
                continue

            if not outputs:
                continue

            # Build origin for converted media.
            origin = {
                "importer": "converter",
                "params": {
                    "converter": converter.name,
                    "source_file": source_rel,
                },
            }
            if base_origin:
                origin["params"]["parent_importer"] = base_origin.get("importer", "")
                parent_params = base_origin.get("params", {})
                if "path" in parent_params:
                    origin["params"]["parent_path"] = parent_params["path"]
                if "url" in parent_params:
                    origin["params"]["parent_url"] = parent_params["url"]

            for output in outputs:
                output_filename = output.get("filename", f"converted_{media_id}")
                # Derive an origin_name that shows provenance:
                # "source_file → output_filename"
                origin_name = f"{source_rel}\u2192{output_filename}"

                # Embed the converted output.
                embedding = _embed_converted_output(target_emb, output)
                if embedding is None:
                    continue

                # Compute MD5 from the output bytes or string.
                md5 = _compute_md5(output)

                media_data: dict[str, Any] = {
                    "id": media_id,
                    "type": target_mt.type_id,
                    "embedder": target_emb.name if target_emb else "",
                    "file_size": len(output.get("media_bytes", b"") or output.get("media_string", "").encode()),
                    "md5": md5,
                    "embedding": embedding,
                    "filename": origin_name,
                    "category": "custom",
                    "origin": origin,
                    "origin_name": origin_name,
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": str(source_path.resolve()),
                    "duration": output.get("duration", 0),
                }

                # Merge in converter output fields.
                if "media_bytes" in output:
                    media_data["media_bytes"] = output["media_bytes"]
                if "media_string" in output:
                    media_data["media_string"] = output["media_string"]
                if "width" in output:
                    media_data["width"] = output["width"]
                if "height" in output:
                    media_data["height"] = output["height"]
                if "word_count" in output:
                    media_data["word_count"] = output["word_count"]
                if "character_count" in output:
                    media_data["character_count"] = output["character_count"]

                medias[media_id] = media_data
                media_id += 1


def _embed_converted_output(target_emb, output: dict[str, Any]):
    """Embed converter output using the target embedder.

    Writes the output to a temporary file and calls ``embed_media()``,
    which is the most general approach across all media types.
    """

    if target_emb is None:
        return None

    media_bytes = output.get("media_bytes")
    media_string = output.get("media_string")

    from vtsearch.media.embedder import media_from_path  # noqa: PLC0415

    if media_bytes:
        # Binary media (image, audio, video) — write to temp file.
        suffix = Path(output.get("filename", "output")).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(media_bytes)
            tmp_path = Path(tmp.name)
        try:
            embedding = target_emb.embed_media(media_from_path(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
        return embedding

    if media_string:
        # Text media — write to temp .txt file.
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as tmp:
            tmp.write(media_string)
            tmp_path = Path(tmp.name)
        try:
            embedding = target_emb.embed_media(media_from_path(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
        return embedding

    return None


def _compute_md5(output: dict[str, Any]) -> str:
    """Compute MD5 from converter output bytes or string."""
    data = output.get("media_bytes")
    if data:
        return hashlib.md5(data).hexdigest()
    text = output.get("media_string", "")
    if text:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    return hashlib.md5(b"").hexdigest()


def apply_converter_to_demo(
    converter_name: str,
    dataset_name: str,
    medias: dict[int, dict[str, Any]],
    embedder_name: str = "",
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Convert all medias in-place using the named converter.

    After conversion, *medias* contains the converted outputs (target type)
    instead of the original source-type medias.  Each converted media's
    origin records the demo dataset and the converter used.
    """
    converter = get_converter(converter_name)
    if converter is None:
        raise ValueError(f"Unknown converter: {converter_name}")

    if on_progress is None:
        on_progress = _default_progress()

    from vtsearch.media import embedders_for_type, get_embedder  # noqa: PLC0415

    # Resolve embedder for the *target* type.
    target_emb = None
    if embedder_name:
        try:
            target_emb = get_embedder(embedder_name)
        except KeyError:
            pass
    if target_emb is None:
        avail = embedders_for_type(converter.target_type)
        if avail:
            target_emb = avail[0]
    if target_emb is not None and getattr(target_emb, "_model", None) is None:
        target_emb.load_models()

    # Convert each original media and collect the outputs.
    source_items = list(medias.items())
    converted: dict[int, dict[str, Any]] = {}
    new_id = 1

    on_progress(
        "converting",
        f"Converting {len(source_items)} items via {converter.display_name or converter.name}...",
        0,
        len(source_items),
    )

    for idx, (_, src) in enumerate(source_items):
        on_progress(
            "converting",
            f"Converting {src.get('filename', '')} via {converter.display_name or converter.name}...",
            idx + 1,
            len(source_items),
        )

        try:
            outputs = converter.convert(src)
        except Exception:
            continue
        if not outputs:
            continue

        origin = {
            "importer": "converter",
            "params": {
                "converter": converter_name,
                "source_file": src.get("filename", ""),
                "parent_importer": "demo",
                "parent_demo": dataset_name,
            },
        }

        for output in outputs:
            output_filename = output.get("filename", f"converted_{new_id}")
            source_name = src.get("filename", str(src.get("id", "")))
            origin_name = f"{source_name}\u2192{output_filename}"

            embedding = _embed_converted_output(target_emb, output)
            if embedding is None:
                continue

            md5 = _compute_md5(output)

            media_data: dict[str, Any] = {
                "id": new_id,
                "type": converter.target_type,
                "embedder": target_emb.name if target_emb else "",
                "file_size": len(output.get("media_bytes", b"") or output.get("media_string", "").encode()),
                "md5": md5,
                "embedding": embedding,
                "filename": origin_name,
                "category": src.get("category", "custom"),
                "origin": origin,
                "origin_name": origin_name,
                "media_bytes": None,
                "media_string": None,
                "media_path": src.get("media_path", ""),
                "duration": output.get("duration", 0),
            }

            if "media_bytes" in output:
                media_data["media_bytes"] = output["media_bytes"]
            if "media_string" in output:
                media_data["media_string"] = output["media_string"]
            if "width" in output:
                media_data["width"] = output["width"]
            if "height" in output:
                media_data["height"] = output["height"]
            if "word_count" in output:
                media_data["word_count"] = output["word_count"]
            if "character_count" in output:
                media_data["character_count"] = output["character_count"]

            converted[new_id] = media_data
            new_id += 1

    medias.clear()
    medias.update(converted)
