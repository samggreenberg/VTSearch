"""Run selected converters on a folder of media files.

This module provides :func:`run_converters_on_folder`, a reusable utility
that any dataset importer can call to scan a directory for source media
files, convert them via one or more :class:`MediaConverter` instances,
and append them to an existing medias dict.  The converter outputs are
left with ``embedding=None``; the framework
:func:`vtscore.datasets.load_pipeline.embed_missing` stage fills them
in after the importer returns.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Optional

from vtscore.security.path_validation import glob_top_level, rglob_follow_symlinks

ProgressCallback = Callable[[str, str, int, int], None]


def _normalise_converter_specs(
    converter_specs: "list | None",
) -> list[tuple[Any, dict[str, Any]]]:
    """Build a ``list[(converter_instance, params)]`` from *converter_specs*.

    Accepts a list of :class:`~vtscore.datasets.importers.base.SourceSpec`
    objects or equivalent dicts with ``converter`` + ``params`` keys.
    Specs whose ``converter`` is ``None`` are skipped (those are the
    "include directly" rows, handled by the importer's own loader, not
    the converter runner).  Unknown converter names are silently dropped
    to match the runner's prior behaviour.
    """
    from vtscore.converters import get_converter  # noqa: PLC0415 — deferred to avoid circular import during eager registry discovery

    result: list[tuple[Any, dict[str, Any]]] = []
    if not converter_specs:
        return result
    for spec in converter_specs:
        if hasattr(spec, "converter"):
            name = spec.converter
            params = dict(spec.params or {})
        else:
            name = spec.get("converter") if isinstance(spec, dict) else None
            params = dict((spec.get("params") if isinstance(spec, dict) else None) or {})
        if not name:
            continue
        c = get_converter(name)
        if c is None:
            continue
        result.append((c, params))
    return result


def _default_progress() -> ProgressCallback:
    from vtscore.concurrency.progress import get_thread_progress

    cb = get_thread_progress()
    if cb is not None:
        return cb
    from vtscore.concurrency.progress import update_progress

    return update_progress


_OPTIONAL_OUTPUT_FIELDS = ("media_bytes", "media_string", "width", "height", "word_count", "character_count")


def _scan_source_files(folder_path: Path, source_mt, recursive: bool) -> list[Path]:
    source_files: list[Path] = []
    for ext in source_mt.file_extensions:
        if recursive:
            source_files.extend(rglob_follow_symlinks(folder_path, ext))
        else:
            source_files.extend(glob_top_level(folder_path, ext))
    source_files.sort()
    return source_files


def _build_converter_origin(
    converter_name: str,
    source_rel: str,
    conv_params: dict[str, Any],
    base_origin: dict[str, Any] | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"converter": converter_name, "source_file": source_rel}
    for pk, pv in conv_params.items():
        params[f"converter_param_{pk}"] = str(pv)
    if base_origin:
        params["parent_importer"] = base_origin.get("importer", "")
        parent_params = base_origin.get("params", {})
        if "path" in parent_params:
            params["parent_path"] = parent_params["path"]
        if "url" in parent_params:
            params["parent_url"] = parent_params["url"]
    return {"importer": "converter", "params": params}


def _build_converted_media_dict(
    media_id: int,
    output: dict[str, Any],
    target_type: str,
    origin: dict[str, Any],
    origin_name: str,
    media_path: str,
    category: str,
) -> dict[str, Any]:
    """Build the media dict for one converter output.

    ``embedding`` is left at ``None`` — the framework
    :func:`~vtscore.datasets.load_pipeline.embed_missing` stage embeds
    converter outputs via ``media_bytes`` / ``media_string`` after the
    importer returns.
    """
    media_data: dict[str, Any] = {
        "id": media_id,
        "media_type": target_type,
        "embedder": "",
        "file_size": len(output.get("media_bytes", b"") or output.get("media_string", "").encode()),
        "md5": _compute_md5(output),
        "embedding": None,
        "filename": origin_name,
        "category": category,
        "origin": origin,
        "origin_name": origin_name,
        "media_bytes": None,
        "media_string": None,
        "media_path": media_path,
        "duration": output.get("duration", 0),
    }
    for key in _OPTIONAL_OUTPUT_FIELDS:
        if key in output:
            media_data[key] = output[key]
    return media_data


def _run_converter_on_source(
    converter, source_path: Path, source_rel: str, conv_params: dict[str, Any], thin: bool
) -> list[dict[str, Any]] | None:
    """Build the source-media dict and invoke the converter; ``None`` on failure."""
    source_media: dict[str, Any] = {
        "filename": source_rel,
        "media_path": str(source_path.resolve()),
    }
    if not thin:
        try:
            source_media["media_bytes"] = source_path.read_bytes()
        except Exception:
            return None
    try:
        return converter.convert_normalized(source_media, conv_params)
    except Exception as exc:
        print(f"Converter {converter.name} failed on {source_rel}: {exc}")
        return None


def _emit_converted_outputs(
    *,
    outputs: list[dict[str, Any]],
    source_rel: str,
    source_path: Path,
    target_type: str,
    origin: dict[str, Any],
    medias: dict[int, dict[str, Any]],
    start_id: int,
    category: str,
) -> int:
    """Append each converter output to *medias*; return next media_id.

    Outputs leave with ``embedding=None``; the framework embed stage
    embeds them from ``media_bytes`` / ``media_string``.
    """
    media_id = start_id
    for output in outputs:
        output_filename = output.get("filename", f"converted_{media_id}")
        origin_name = f"{source_rel}\u2192{output_filename}"

        media_data = _build_converted_media_dict(
            media_id,
            output,
            target_type,
            origin,
            origin_name,
            str(source_path.resolve()),
            category,
        )
        medias[media_id] = media_data
        media_id += 1
    return media_id


def run_converters_on_folder(
    folder_path: Path,
    target_media_type: str = "",
    medias: dict[int, dict[str, Any]] | None = None,
    thin: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    base_origin: dict[str, Any] | None = None,
    recursive: bool = True,
    converter_specs: list | None = None,
) -> None:
    """Scan *folder_path* for source files, convert them, and add to *medias*.

    For each converter spec in *converter_specs*:

    1. Look up the converter's source media type to get file extensions.
    2. Scan *folder_path* recursively for files matching those extensions.
    3. For each source file, run the converter to produce output media dicts.
    4. Embed each output with the target media type's embedder.
    5. Assign sequential IDs continuing from the current max in *medias*.
    6. Set origin to ``{"importer": "converter", "params": {...}}``.

    Args:
        folder_path: Root directory to scan.
        target_media_type: The target media type identifier
            (e.g. ``"image"``).
        medias: The medias dict to append to (not cleared).
        thin: When ``True``, store converter output in a temp directory
            instead of holding bytes in memory.  Currently all converted
            media is kept in-memory regardless, since converters produce
            bytes directly.
        on_progress: Optional progress callback.
        base_origin: The origin dict of the parent import (e.g.
            ``{"importer": "server_folder", "params": {"path": "..."}}``)
            used to record provenance.
        converter_specs: A list of
            :class:`~vtscore.datasets.importers.base.SourceSpec`
            instances (or equivalent dicts) carrying both the converter
            name and the user-supplied params for that converter.  Specs
            whose ``converter`` is ``None`` are ignored here (those are
            "include directly" rows, handled by the importer's own
            file loader).
    """
    if medias is None:
        return
    converters_with_params = _normalise_converter_specs(converter_specs)
    if not converters_with_params:
        return

    if on_progress is None:
        on_progress = _default_progress()

    from vtscore.media import get as media_get, get_by_folder_name  # noqa: PLC0415

    target_mt = get_by_folder_name(target_media_type)
    converters_with_params = [(c, p) for c, p in converters_with_params if c.target_type == target_mt.type_id]
    if not converters_with_params:
        return

    media_id = max(medias.keys(), default=0) + 1

    for converter, conv_params in converters_with_params:
        try:
            source_mt = media_get(converter.source_type)
        except KeyError:
            continue

        source_files = _scan_source_files(folder_path, source_mt, recursive)
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

            outputs = _run_converter_on_source(converter, source_path, source_rel, conv_params, thin)
            if not outputs:
                continue

            origin = _build_converter_origin(converter.name, source_rel, conv_params, base_origin)
            media_id = _emit_converted_outputs(
                outputs=outputs,
                source_rel=source_rel,
                source_path=source_path,
                target_type=target_mt.type_id,
                origin=origin,
                medias=medias,
                start_id=media_id,
                category="custom",
            )


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
    embedder_name: str = "",  # noqa: ARG001 — kept for call-site compatibility; framework picks the embedder now
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Convert all medias in-place using the named converter.

    After conversion, *medias* contains the converted outputs (target type)
    instead of the original source-type medias.  Each converted media's
    origin records the demo dataset and the converter used.  Outputs
    leave with ``embedding=None``; the framework embed stage fills them
    in.
    """
    from vtscore.converters import get_converter  # noqa: PLC0415 — deferred to avoid circular import during eager registry discovery

    converter = get_converter(converter_name)
    if converter is None:
        raise ValueError(f"Unknown converter: {converter_name}")

    if on_progress is None:
        on_progress = _default_progress()

    source_items = list(medias.items())
    converted: dict[int, dict[str, Any]] = {}
    new_id = 1

    label = converter.display_name or converter.name
    on_progress("converting", f"Converting {len(source_items)} items via {label}...", 0, len(source_items))

    for idx, (_, src_media) in enumerate(source_items):
        on_progress(
            "converting",
            f"Converting {src_media.get('filename', '')} via {label}...",
            idx + 1,
            len(source_items),
        )

        try:
            outputs = converter.convert_normalized(src_media, {})
        except Exception:
            continue
        if not outputs:
            continue

        origin = {
            "importer": "converter",
            "params": {
                "converter": converter_name,
                "source_file": src_media.get("filename", ""),
                "parent_importer": "demo",
                "parent_demo": dataset_name,
            },
        }
        source_name = src_media.get("filename", str(src_media.get("id", "")))
        new_id = _emit_converted_demo_outputs(
            outputs=outputs,
            source_name=source_name,
            source_media=src_media,
            target_type=converter.target_type,
            origin=origin,
            converted=converted,
            start_id=new_id,
        )

    medias.clear()
    medias.update(converted)


def _emit_converted_demo_outputs(
    *,
    outputs: list[dict[str, Any]],
    source_name: str,
    source_media: dict[str, Any],
    target_type: str,
    origin: dict[str, Any],
    converted: dict[int, dict[str, Any]],
    start_id: int,
) -> int:
    """Append each converter output to *converted*, return next id.

    Outputs leave with ``embedding=None`` for the framework embed stage.
    """
    new_id = start_id
    for output in outputs:
        output_filename = output.get("filename", f"converted_{new_id}")
        origin_name = f"{source_name}\u2192{output_filename}"

        media_data = _build_converted_media_dict(
            new_id,
            output,
            target_type,
            origin,
            origin_name,
            source_media.get("media_path", ""),
            source_media.get("category", "custom"),
        )
        converted[new_id] = media_data
        new_id += 1
    return new_id
