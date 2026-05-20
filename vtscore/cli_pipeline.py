"""YAML pipeline-file runner for ``python app.py --pipeline pipeline.yaml``.

A pipeline file declares the same options as the ``--autodetect`` flag set
(importer + fields, settings file, detector list, chunk size, optional
one-shot label import, exporter + fields).  This module loads the YAML,
validates the shape against the active plugin registries, and dispatches
to the shared ``_run_pipeline`` in :mod:`vtscore.cli`.

See ``docs/CLI.md`` for the user-facing schema and examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


_TOP_LEVEL_KEYS = {
    "dataset",
    "importer",
    "settings",
    "detectors",
    "chunk_size",
    "import_labels",
    "exporter",
}

_IMPORT_LABELS_KEYS = {"detector", "importer", "file"}


def load_pipeline_file(path: str | Path) -> dict[str, Any]:  # noqa: C901
    """Read *path*, parse it as YAML, and return a normalised config dict.

    Raises :class:`FileNotFoundError` if the file is missing, and
    :class:`ValueError` for any shape problem (unknown key, wrong type,
    mutually-exclusive options both set, etc.).  The plugin names inside
    ``importer:`` / ``exporter:`` / ``import_labels.importer`` are looked
    up against the registries at parse time so a typo fails fast — before
    the pipeline starts loading any media.
    """
    import yaml  # noqa: PLC0415

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Pipeline file not found: {file_path}")

    raw = yaml.safe_load(file_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Pipeline file must contain a YAML mapping, got {type(raw).__name__}.")

    unknown = set(raw.keys()) - _TOP_LEVEL_KEYS
    if unknown:
        allowed = ", ".join(sorted(_TOP_LEVEL_KEYS))
        raise ValueError(f"Unknown pipeline key(s): {', '.join(sorted(unknown))}. Allowed: {allowed}.")

    dataset = raw.get("dataset")
    importer = raw.get("importer")
    if dataset is None and importer is None:
        raise ValueError("Pipeline file must set either 'dataset:' or 'importer:'.")
    if dataset is not None and importer is not None:
        raise ValueError("Pipeline file sets both 'dataset:' and 'importer:'; pick one.")

    if dataset is not None and not isinstance(dataset, str):
        raise ValueError("'dataset:' must be a string path.")

    importer_name: str | None = None
    importer_fields: dict[str, Any] = {}
    if importer is not None:
        importer_name, importer_fields = _parse_plugin_section(importer, "importer")
        _validate_importer_name(importer_name)
        _validate_field_keys(importer_name, importer_fields, "importer", _list_importer_field_keys)

    settings = raw.get("settings")
    if settings is not None and not isinstance(settings, str):
        raise ValueError("'settings:' must be a string path.")

    detectors = raw.get("detectors")
    if detectors is not None:
        if not isinstance(detectors, list) or not all(isinstance(d, str) for d in detectors):
            raise ValueError("'detectors:' must be a list of detector names (strings).")
        if not detectors:
            raise ValueError("'detectors:' must contain at least one name when present.")

    chunk_size = raw.get("chunk_size")
    if chunk_size is not None:
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError("'chunk_size:' must be a positive integer.")

    import_labels = raw.get("import_labels")
    parsed_import_labels: dict[str, str] | None = None
    if import_labels is not None:
        parsed_import_labels = _parse_import_labels(import_labels)

    exporter = raw.get("exporter")
    exporter_name: str | None = None
    exporter_fields: dict[str, Any] = {}
    if exporter is not None:
        exporter_name, exporter_fields = _parse_plugin_section(exporter, "exporter")
        _validate_exporter_name(exporter_name)
        _validate_field_keys(exporter_name, exporter_fields, "exporter", _list_exporter_field_keys)

    return {
        "dataset": dataset,
        "importer": importer_name,
        "importer_fields": importer_fields,
        "settings": settings,
        "detectors": list(detectors) if detectors else None,
        "chunk_size": chunk_size,
        "import_labels": parsed_import_labels,
        "exporter": exporter_name,
        "exporter_fields": exporter_fields,
    }


def _parse_plugin_section(value: Any, section: str) -> tuple[str, dict[str, Any]]:
    """Parse an ``importer:`` or ``exporter:`` block into ``(name, fields)``."""
    if not isinstance(value, dict):
        raise ValueError(f"'{section}:' must be a mapping with a 'name:' key.")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"'{section}.name' is required and must be a string.")

    unknown = set(value.keys()) - {"name", "fields"}
    if unknown:
        raise ValueError(f"'{section}' has unknown key(s): {', '.join(sorted(unknown))}. Allowed: name, fields.")

    fields = value.get("fields") or {}
    if not isinstance(fields, dict):
        raise ValueError(f"'{section}.fields' must be a mapping of field key → value.")
    return name, dict(fields)


def _parse_import_labels(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("'import_labels:' must be a mapping.")
    unknown = set(value.keys()) - _IMPORT_LABELS_KEYS
    if unknown:
        allowed = ", ".join(sorted(_IMPORT_LABELS_KEYS))
        raise ValueError(f"'import_labels' has unknown key(s): {', '.join(sorted(unknown))}. Allowed: {allowed}.")

    detector = value.get("detector")
    file = value.get("file")
    if not isinstance(detector, str) or not detector:
        raise ValueError("'import_labels.detector' is required and must be a string.")
    if not isinstance(file, str) or not file:
        raise ValueError("'import_labels.file' is required and must be a string path.")

    importer_name = value.get("importer", "server_json_file")
    if not isinstance(importer_name, str) or not importer_name:
        raise ValueError("'import_labels.importer' must be a string when set.")

    from vtscore.labels.importers import get_label_importer, list_label_importers  # noqa: PLC0415

    if get_label_importer(importer_name) is None:
        available = ", ".join(li.name for li in list_label_importers())
        raise ValueError(f"Unknown label importer: {importer_name!r}. Available: {available}.")

    return {"detector": detector, "importer": importer_name, "file": file}


def _validate_importer_name(name: str) -> None:
    from vtscore.datasets.importers import get_importer, list_importers  # noqa: PLC0415

    if get_importer(name) is None:
        available = ", ".join(imp.name for imp in list_importers())
        raise ValueError(f"Unknown importer: {name!r}. Available: {available}.")


def _validate_exporter_name(name: str) -> None:
    from vtscore.exporters import get_exporter, list_exporters  # noqa: PLC0415

    if get_exporter(name) is None:
        available = ", ".join(exp.name for exp in list_exporters())
        raise ValueError(f"Unknown exporter: {name!r}. Available: {available}.")


def _list_importer_field_keys(name: str) -> set[str]:
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415

    plugin = get_importer(name)
    return {f.key for f in (plugin.fields if plugin else [])}


def _list_exporter_field_keys(name: str) -> set[str]:
    from vtscore.exporters import get_exporter  # noqa: PLC0415

    plugin = get_exporter(name)
    return {f.key for f in (plugin.fields if plugin else [])}


def _validate_field_keys(
    plugin_name: str,
    fields: dict[str, Any],
    section: str,
    keys_for: Any,
) -> None:
    """Reject fields that the named plugin doesn't declare — matches argparse,
    which rejects unknown flags."""
    allowed = keys_for(plugin_name)
    unknown = set(fields.keys()) - allowed
    if unknown:
        allowed_str = ", ".join(sorted(allowed)) or "(no fields)"
        raise ValueError(
            f"'{section}.fields' has unknown key(s) for {plugin_name!r}: "
            f"{', '.join(sorted(unknown))}. Allowed: {allowed_str}."
        )


def run_pipeline_file(path: str | Path) -> None:
    """Load *path* and execute the pipeline it declares.

    Errors are converted to a one-line ``Error: ...`` on stderr followed by
    ``sys.exit(1)`` — same convention as the other CLI entry points in
    :mod:`vtscore.cli`.
    """
    import sys  # noqa: PLC0415

    try:
        config = load_pipeline_file(path)
        _dispatch(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _dispatch(config: dict[str, Any]) -> None:
    """Run the parsed *config* against the existing autodetect pipeline."""
    from vtscore.cli import (  # noqa: PLC0415
        _load_importer_chunked,
        _load_importer_whole,
        _load_pickle_chunked,
        _load_pickle_whole,
        _run_pipeline,
        import_labels_into_detector_from_file,
    )

    settings_path = config["settings"]
    if config["import_labels"] is not None:
        if settings_path:
            # Apply the settings path first so detector-dir lookups resolve
            # to the same place the pipeline run will use.  Routed through
            # CoreConfig so this module stays settings-import-free
            # (see Phase 2 of docs/architecture.md).
            from vtscore.config import CoreConfig  # noqa: PLC0415

            CoreConfig.from_settings(settings_path=settings_path)
        il = config["import_labels"]
        applied, skipped = import_labels_into_detector_from_file(il["detector"], il["importer"], il["file"])
        print(
            f"Imported {applied} label(s) into detector '{il['detector']}' (skipped {skipped} duplicate/invalid).",
            flush=True,
        )

    chunk_size = config["chunk_size"]
    if config["importer"]:
        if chunk_size:
            source = _load_importer_chunked(config["importer"], config["importer_fields"], chunk_size)
            empty_error = f"No medias loaded by importer '{config['importer']}'"
        else:
            source = _load_importer_whole(config["importer"], config["importer_fields"])
            empty_error = f"No medias loaded by importer '{config['importer']}'"
    else:
        dataset_path = config["dataset"]
        if chunk_size:
            source = _load_pickle_chunked(dataset_path, chunk_size)
        else:
            source = _load_pickle_whole(dataset_path)
        empty_error = f"No medias loaded from dataset: {dataset_path}"

    _run_pipeline(
        source,
        settings_path=settings_path,
        exporter_name=config["exporter"],
        exporter_field_values=config["exporter_fields"],
        override_detectors=config["detectors"],
        empty_error=empty_error,
    )
