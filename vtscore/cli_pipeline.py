"""YAML pipeline-file runner for ``python app.py --pipeline pipeline.yaml``.

A pipeline file declares the same options as the ``--autodetect`` flag set
(importer + fields, settings file, detector list, chunk size, optional
one-shot label import, exporter + fields, and ``tempimport`` - whether the
imported dataset is discarded rather than saved to the dashboard).  This module loads the YAML,
validates the shape against the active plugin registries, and dispatches
to the shared ``_run_pipeline`` in :mod:`vtscore.cli`.

An ``importer:`` block may carry an ``outputs:`` list (#4707): one mapping
per dataset the single importer run should produce, in the shape the web
API's ``outputs`` entries take.  It is parsed with
:func:`~vtscore.datasets.importers.base.parse_output_specs` and dispatched
as the source's ``outputs``.

See ``docs/CLI.md`` for the user-facing schema and examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vtscore.datasets.importers.base import OutputSpec


_TOP_LEVEL_KEYS = {
    "dataset",
    "importer",
    "settings",
    "detectors",
    "chunk_size",
    "stream_results",
    "keep_negatives",
    "import_labels",
    "exporter",
    "tempimport",
}

_IMPORT_LABELS_KEYS = {"detector", "importer", "file", "create", "media_type"}

#: Keys an entry of ``importer.outputs`` may carry: the two the importer reads
#: (``media_type``, ``source_specs``), the category that names the dataset,
#: and the per-dataset load options - the same set the web ``outputs`` take.
_OUTPUT_KEYS = {
    "media_type",
    "source_specs",
    "category",
    "dataset_name",
    "embedder",
    "embedders",
    "clipper",
    "clipper_params",
    "clipper_chain",
    "cleaners",
}


def load_pipeline_file(path: str | Path) -> dict[str, Any]:  # noqa: C901
    """Read *path*, parse it as YAML, and return a normalised config dict.

    Raises :class:`FileNotFoundError` if the file is missing, and
    :class:`ValueError` for any shape problem (unknown key, wrong type,
    mutually-exclusive options both set, etc.).  The plugin names inside
    ``importer:`` / ``exporter:`` / ``import_labels.importer`` are looked
    up against the registries at parse time so a typo fails fast - before
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
    importer_outputs: list[OutputSpec] = []
    if importer is not None:
        outputs_raw = None
        if isinstance(importer, dict) and "outputs" in importer:
            importer = dict(importer)
            outputs_raw = importer.pop("outputs")
        importer_name, importer_fields = _parse_plugin_section(importer, "importer")
        _validate_importer_name(importer_name)
        _validate_field_keys(importer_name, importer_fields, "importer", _list_importer_field_keys)
        if outputs_raw is not None:
            importer_outputs = _parse_importer_outputs(importer_name, importer_fields, outputs_raw)

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

    stream_results = raw.get("stream_results", False)
    if not isinstance(stream_results, bool):
        raise ValueError("'stream_results:' must be a boolean.")
    keep_negatives = raw.get("keep_negatives", False)
    if not isinstance(keep_negatives, bool):
        raise ValueError("'keep_negatives:' must be a boolean.")
    if stream_results and not chunk_size:
        raise ValueError("'stream_results:' requires 'chunk_size:' (it streams chunk by chunk).")
    if keep_negatives and not stream_results:
        raise ValueError("'keep_negatives:' only applies with 'stream_results:'.")
    tempimport = raw.get("tempimport", False)
    if not isinstance(tempimport, bool):
        raise ValueError("'tempimport:' must be a boolean.")
    if stream_results and not tempimport:
        from vtscore.cli import _STREAM_CANNOT_SAVE  # noqa: PLC0415

        raise ValueError(f"'stream_results:' requires 'tempimport: true': {_STREAM_CANNOT_SAVE}.")

    import_labels = raw.get("import_labels")
    parsed_import_labels: dict[str, Any] | None = None
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
        "importer_outputs": importer_outputs,
        "settings": settings,
        "detectors": list(detectors) if detectors else None,
        "chunk_size": chunk_size,
        "stream_results": stream_results,
        "keep_negatives": keep_negatives,
        "tempimport": tempimport,
        "import_labels": parsed_import_labels,
        "exporter": exporter_name,
        "exporter_fields": exporter_fields,
    }


def _parse_plugin_section(value: Any, section: str) -> tuple[str, dict[str, Any]]:
    """Parse an ``importer:`` or ``exporter:`` block into ``(name, fields)``.

    The ``importer:`` block's ``outputs`` key is taken off before this runs
    (see :func:`_parse_importer_outputs`), so it is named in the error only
    for the block that accepts it.
    """
    if not isinstance(value, dict):
        raise ValueError(f"'{section}:' must be a mapping with a 'name:' key.")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"'{section}.name' is required and must be a string.")

    unknown = set(value.keys()) - {"name", "fields"}
    if unknown:
        allowed = "name, fields, outputs" if section == "importer" else "name, fields"
        raise ValueError(f"'{section}' has unknown key(s): {', '.join(sorted(unknown))}. Allowed: {allowed}.")

    fields = value.get("fields") or {}
    if not isinstance(fields, dict):
        raise ValueError(f"'{section}.fields' must be a mapping of field key → value.")
    return name, dict(fields)


def _parse_importer_outputs(importer_name: str, fields: dict[str, Any], raw: Any) -> list[OutputSpec]:
    """Parse ``importer.outputs`` into the datasets the one importer run should produce.

    A list of mappings, one per dataset, each carrying the keys a web
    ``outputs`` entry takes (:data:`_OUTPUT_KEYS`); every mapping needs a
    ``media_type``.  The list and ``importer.fields.media_type`` /
    ``importer.fields.source_specs`` are mutually exclusive - each output names
    its own - and the importer has to make several datasets per run (the demo
    importer does not).  The entries themselves go through the web API's own
    validator, :func:`~vtscore.datasets.importers.base.parse_output_specs`,
    so an unknown media type or a converter that does not produce the output's
    type fails here with that validator's message.
    """
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415
    from vtscore.datasets.importers.base import parse_output_specs  # noqa: PLC0415

    if not isinstance(raw, list) or not raw:
        raise ValueError("'importer.outputs' must be a non-empty list, one mapping per dataset to make.")
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"'importer.outputs[{index}]' must be a mapping, got {type(item).__name__}.")
        unknown = set(item.keys()) - _OUTPUT_KEYS
        if unknown:
            raise ValueError(
                f"'importer.outputs[{index}]' has unknown key(s): {', '.join(sorted(unknown))}. "
                f"Allowed: {', '.join(sorted(_OUTPUT_KEYS))}."
            )
        if not item.get("media_type"):
            raise ValueError(f"'importer.outputs[{index}]' needs a 'media_type' (the dataset's type).")
    for key in ("media_type", "source_specs"):
        if key in fields:
            raise ValueError(
                f"'importer.outputs' and 'importer.fields.{key}' both set; with 'outputs:' each entry names its own."
            )

    importer = get_importer(importer_name)
    if importer is not None and (
        not getattr(importer, "multi_output", True) or not any(f.key == "media_type" for f in importer.fields)
    ):
        raise ValueError(
            f"'importer.outputs': importer {importer_name!r} produces one dataset per run; use 'fields.media_type'."
        )
    try:
        return parse_output_specs(raw)
    except ValueError as exc:
        raise ValueError(f"'importer.outputs': {exc}") from exc


def _parse_import_labels(value: Any) -> dict[str, Any]:
    """Parse the ``import_labels:`` block into ``{detector, importer, fields, create, media_type}``.

    ``importer`` takes the same ``{name, fields}`` mapping as the top-level
    ``importer:`` / ``exporter:`` blocks, so any label importer - including
    one that reads no file - can be driven from YAML.  The older flat form
    (``importer: <name>`` plus ``file: <path>``) is still accepted: ``file``
    is shorthand for the importer's ``filepath`` field.  ``create: true`` and
    ``media_type:`` mirror ``--create-detector`` / ``--detector-media-type``.
    """
    if not isinstance(value, dict):
        raise ValueError("'import_labels:' must be a mapping.")
    unknown = set(value.keys()) - _IMPORT_LABELS_KEYS
    if unknown:
        allowed = ", ".join(sorted(_IMPORT_LABELS_KEYS))
        raise ValueError(f"'import_labels' has unknown key(s): {', '.join(sorted(unknown))}. Allowed: {allowed}.")

    detector = value.get("detector")
    if not isinstance(detector, str) or not detector:
        raise ValueError("'import_labels.detector' is required and must be a string.")

    importer = value.get("importer", "server_json_file")
    if isinstance(importer, dict):
        importer_name, fields = _parse_plugin_section(importer, "import_labels.importer")
    elif isinstance(importer, str) and importer:
        importer_name, fields = importer, {}
    else:
        raise ValueError("'import_labels.importer' must be a label importer name or a mapping with a 'name:' key.")

    from vtscore.labels.importers import get_label_importer, list_label_importers  # noqa: PLC0415

    plugin = get_label_importer(importer_name)
    if plugin is None:
        available = ", ".join(li.name for li in list_label_importers())
        raise ValueError(f"Unknown label importer: {importer_name!r}. Available: {available}.")

    file = value.get("file")
    if file is not None:
        if not isinstance(file, str) or not file:
            raise ValueError("'import_labels.file' must be a string path when set.")
        if "filepath" in fields:
            raise ValueError("'import_labels' sets both 'file:' and 'importer.fields.filepath'; pick one.")
        fields["filepath"] = file

    _validate_field_keys(importer_name, fields, "import_labels.importer", lambda _name: {f.key for f in plugin.fields})
    _require_fields(plugin, importer_name, fields)

    create, media_type = _parse_import_labels_create(value)
    return {
        "detector": detector,
        "importer": importer_name,
        "fields": fields,
        "create": create,
        "media_type": media_type,
    }


def _parse_import_labels_create(value: dict[str, Any]) -> tuple[bool, str]:
    """Parse ``import_labels.create`` / ``import_labels.media_type`` into ``(create, media_type)``."""
    create = value.get("create", False)
    if not isinstance(create, bool):
        raise ValueError("'import_labels.create' must be a boolean.")
    media_type = value.get("media_type")
    if media_type is None:
        return create, ""
    if not isinstance(media_type, str) or not media_type:
        raise ValueError("'import_labels.media_type' must be a media type name when set.")
    if not create:
        raise ValueError("'import_labels.media_type' only applies with 'import_labels.create: true'.")
    return create, media_type


def _require_fields(plugin: Any, plugin_name: str, fields: dict[str, Any]) -> None:
    """Fail fast on a required label-importer field left out of the YAML."""
    missing = [
        f.key
        for f in plugin.fields
        if f.required and not f.default and f.field_type != "checkbox" and _is_blank(fields.get(f.key))
    ]
    if missing:
        hint = " (or 'import_labels.file')" if missing == ["filepath"] else ""
        raise ValueError(
            f"'import_labels.importer.fields' is missing required field(s) for {plugin_name!r}: "
            f"{', '.join(missing)}{hint}."
        )


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


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
    """Reject fields that the named plugin doesn't declare - matches argparse,
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
    ``sys.exit(1)`` - same convention as the other CLI entry points in
    :mod:`vtscore.cli`.
    """
    import sys  # noqa: PLC0415

    try:
        config = load_pipeline_file(path)
        _dispatch(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _import_labels(il: dict[str, Any], spec: Any) -> None:
    """Run the ``import_labels:`` block, creating the detector when ``create:`` asks."""
    from vtscore.cli import (  # noqa: PLC0415
        DetectorNotFoundError,
        _label_import_media_type,
        import_labels_into_detector,
    )

    create_media_type = _label_import_media_type(
        il["detector"],
        spec,
        create=il.get("create", False),
        media_type=il.get("media_type", ""),
        media_type_option="'import_labels.media_type'",
    )
    try:
        applied, skipped = import_labels_into_detector(
            il["detector"], il["importer"], il["fields"], create_media_type=create_media_type
        )
    except DetectorNotFoundError as exc:
        raise ValueError(f"{exc} Set 'import_labels.create: true' to create it from the imported labels.") from exc
    if create_media_type:
        done = f"Created detector '{il['detector']}' (media_type={create_media_type}) with {applied} label(s)"
    else:
        done = f"Imported {applied} label(s) into detector '{il['detector']}'"
    print(f"{done} (skipped {skipped} duplicate/invalid).", flush=True)


def _dispatch(config: dict[str, Any]) -> None:
    """Run the parsed *config* against the existing autodetect pipeline."""
    from vtscore.cli import (  # noqa: PLC0415
        _run_source,
        _SourceSpec,
    )

    if config["importer"]:
        spec = _SourceSpec(
            kind="importer",
            importer_name=config["importer"],
            field_values=config["importer_fields"],
            chunk_size=config["chunk_size"],
            outputs=tuple(config.get("importer_outputs") or ()),
        )
    else:
        spec = _SourceSpec(kind="pickle", dataset_path=config["dataset"], chunk_size=config["chunk_size"])

    settings_path = config["settings"]
    if config["import_labels"] is not None:
        if settings_path:
            # Apply the settings path first so detector-dir lookups resolve
            # to the same place the pipeline run will use.  Routed through
            # CoreConfig so this module stays settings-import-free
            # (see Phase 2 of docs/architecture.md).
            from vtscore.config import CoreConfig  # noqa: PLC0415

            CoreConfig.from_settings(settings_path=settings_path)
        _import_labels(config["import_labels"], spec)

    # As with ``--import-labels-into``, a label import names the detector the
    # run scores with, in place of the settings' AutoFind list - unless the
    # file lists ``detectors:`` explicitly, which always wins (#4235).
    detectors = config["detectors"]
    if detectors is None and config["import_labels"] is not None:
        detectors = [config["import_labels"]["detector"]]

    _run_source(
        spec,
        # Same default as the flags: the dataset is kept unless the file says
        # ``tempimport: true`` (#4226).
        save_dataset=not config.get("tempimport", False),
        settings_path=settings_path,
        exporter_name=config["exporter"],
        exporter_field_values=config["exporter_fields"],
        override_detectors=detectors,
        stream_results=config.get("stream_results", False),
        keep_negatives=config.get("keep_negatives", False),
    )
