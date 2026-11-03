"""The exporter end of a run: which exporter, with what field values, and running it.

Resolves the AutoFind exporter the settings name when no ``--exporter`` is
given, gives each dataset of a multi-dataset run its own destination, and
refuses an exporter that cannot stream (``--stream-results``) or resume
(``--resume``) before anything loads - the run and its ``--dry-run`` share
these, so the plan fails where the run would.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vtscore.cli import progress as cli_progress

if TYPE_CHECKING:
    from vtscore.cli.datasets import _RunDataset

logger = logging.getLogger(__name__)


def _list_exporter_names() -> list[str]:
    """Return the names of all registered exporters."""
    from vtscore.exporters import list_exporters

    return [exp.name for exp in list_exporters()]


def _run_exporter(
    exporter_name: str,
    field_values: dict[str, Any],
    results: dict[str, Any],
    detector_mlps: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Validate and run a named exporter, printing its confirmation message.

    Most exporters consume the scored *results*.  An exporter that instead
    exports the trained classifiers themselves (``needs_trained_detectors``,
    the portable-detector bundle) is handed the *detector_mlps* the pipeline
    trained, via :meth:`ResultsExporter.export_cli_detectors`.

    An exporter that returns an ``open_url`` gets it surfaced here rather than
    dropped: there is no browser to open it on the command line, so the URL is
    printed under the confirmation message (text mode) and carried as a field
    on the ``export_complete`` event (JSON mode), which is what lets a wrapping
    script open it itself.
    """
    from vtscore.exporters import get_exporter

    exporter = get_exporter(exporter_name)
    if exporter is None:
        available = _list_exporter_names()
        raise ValueError(f"Unknown exporter: {exporter_name}. Available: {', '.join(available)}")

    exporter.validate_cli_field_values(field_values)
    if "detector_bundles" in exporter.supported_payloads:
        descriptors = _portable_detector_descriptors(detector_mlps or {})
        result = exporter.export_cli_detectors(descriptors, field_values)
    else:
        result = exporter.export_cli(results, field_values)
    message = result.get("message", "Export complete.")

    open_url = _validated_open_url(result, exporter_name)
    if open_url is None:
        cli_progress.emit("export_complete", text=message, message=message)
    else:
        cli_progress.emit(
            "export_complete",
            text=f"{message}\n\n  {open_url}",
            message=message,
            open_url=open_url,
        )


def _validated_open_url(result: dict[str, Any], exporter_name: str) -> str | None:
    """Return *result*'s ``open_url`` if it is one, else ``None``.

    The same scheme allowlist the HTTP route applies, for the same reason: a
    plugin should never be able to put a ``javascript:`` URL in front of the
    user.  A bad one is dropped with a warning rather than raised, because the
    export itself already succeeded — sinking a completed delivery over a
    cosmetic field would lose the run.
    """
    raw = result.get("open_url")
    if raw is None:
        return None
    from vtscore.security.url_validation import validate_browser_url

    try:
        return validate_browser_url(str(raw))
    except ValueError as exc:
        logger.warning("Exporter %r returned an unusable open_url, ignoring it: %s", exporter_name, exc)
        return None


def _portable_detector_descriptors(detector_mlps: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialise each trained detector into a portable-export descriptor.

    Turns the pipeline's ``{name: {"mlp", "threshold", ...}}`` map into the
    list of plain-data dicts :meth:`ResultsExporter.export_cli_detectors`
    consumes - the live torch model is reduced to nested-list weights here so
    the exporter itself stays torch-free.
    """
    from vtscore.detectors.training import serialize_weights

    descriptors: list[dict[str, Any]] = []
    for name, info in detector_mlps.items():
        descriptors.append(
            {
                "detector_name": name,
                "media_type": info.get("media_type", "") or "",
                "weights": serialize_weights(info["mlp"]),
                "threshold": info["threshold"],
                "embedder": info.get("embedder", "") or "",
                "embedder_type": info.get("embedder_type", "") or "",
                "good_count": int(info.get("good_count", 0)),
                "bad_count": int(info.get("bad_count", 0)),
            }
        )
    return descriptors


def _resolve_exporter(
    exporter_name: str | None, exporter_field_values: dict[str, Any] | None
) -> tuple[str | None, dict[str, Any] | None]:
    """Fall back to the settings' AutoFind results exporter when no ``--exporter`` was given.

    Its per-exporter field values come along too.  An explicit ``--exporter``
    always wins; if neither is set the downstream default (``gui``) applies.
    """
    from vtscore.config import get_setting  # noqa: PLC0415

    if exporter_name is not None:
        return exporter_name, exporter_field_values
    configured = get_setting("autofind_exporter")
    if configured:
        exporter_name = configured
        if exporter_field_values is None:
            exporter_field_values = dict(get_setting("autofind_exporter_field_values").get(configured, {}))
    return exporter_name, exporter_field_values


def _exporter_fields_for_dataset(
    exporter_name: str, field_values: dict[str, Any] | None, dataset: _RunDataset
) -> dict[str, Any]:
    """The exporter's field values for one dataset of a multi-dataset run.

    The exporter runs once per dataset, so its destination has to tell them
    apart.  ``{dataset_name}`` in any of its fields is replaced by the
    dataset's name (made path-safe, as every template value is); when no field
    mentions it and the run produces more than one dataset, an exporter with a
    ``filepath`` field gets the name inserted before the extension, as the
    portable-detector exporter does with ``{detector_name}``, so the files do
    not overwrite each other.  A run of one dataset leaves the path alone.
    The framework's own template variables (``{YYYYMMDD}``, ``{username}``,
    ...) are substituted afterwards, by the exporter's usual validation.
    """
    from vtscore.exporters import get_exporter  # noqa: PLC0415
    from vtscore.security.path_validation import sanitize_template_value  # noqa: PLC0415

    values = dict(field_values or {})
    exporter = get_exporter(exporter_name)
    if exporter is None:
        # Reported, with the available names, when the exporter is resolved.
        return values
    slug = sanitize_template_value(dataset.name)
    mentioned = False
    for f in exporter.fields:
        value = values.get(f.key)
        if value is None or value == "":
            value = f.default
        if isinstance(value, str) and "{dataset_name}" in value:
            values[f.key] = value.replace("{dataset_name}", slug)
            mentioned = True
    if mentioned or dataset.count < 2:
        return values
    path_field = next((f for f in exporter.fields if f.key == "filepath"), None)
    if path_field is not None:
        raw = values.get("filepath") or path_field.default
        if isinstance(raw, str) and raw:
            path = Path(raw)
            values["filepath"] = str(path.with_name(f"{path.stem}-{slug}{path.suffix}"))
    return values


def _list_streaming_exporter_names(*, as_labelset: bool = False) -> list[str]:
    """Return the names of exporters that support ``--stream-results`` (``--as-labelset`` when asked)."""
    from vtscore.exporters import list_exporters

    flag = "supports_labelset_streaming" if as_labelset else "supports_streaming"
    return [exp.name for exp in list_exporters() if getattr(exp, flag, False)]


def _streaming_exporter(exporter_name: str | None, *, as_labelset: bool) -> Any:
    """Resolve the exporter a streaming run writes to, refusing one that cannot stream.

    *exporter_name* ``None`` is the ``gui`` default.  The error names the
    exporters that can, which is what a user who picked the wrong one needs.
    Shared by the run and its ``--dry-run``, so the plan fails where the run
    would.
    """
    from vtscore.exporters import get_exporter

    exporter = get_exporter(exporter_name or "gui")
    if exporter is None:
        available = _list_exporter_names()
        raise ValueError(f"Unknown exporter: {exporter_name}. Available: {', '.join(available)}")
    if as_labelset:
        if not getattr(exporter, "supports_labelset_streaming", False):
            streaming = ", ".join(_list_streaming_exporter_names(as_labelset=True))
            raise ValueError(
                f"Exporter '{exporter.name}' cannot stream a labelset (--stream-results --as-labelset). "
                f"Exporters that can: {streaming}."
            )
    elif not getattr(exporter, "supports_streaming", False):
        streaming = ", ".join(_list_streaming_exporter_names())
        raise ValueError(
            f"Exporter '{exporter.name}' does not support --stream-results. Streaming-capable exporters: {streaming}."
        )
    return exporter


def _resumable_exporter(exporter_name: str | None) -> Any:
    """Resolve the exporter a ``--resume`` run writes to, refusing one that cannot resume.

    Like :func:`_streaming_exporter`, and shared with ``--dry-run`` for the same
    reason: the plan fails where the run would.
    """
    exporter = _streaming_exporter(exporter_name, as_labelset=False)
    if not exporter.supports_resumable_streaming:
        from vtscore.exporters import list_exporters  # noqa: PLC0415

        resumable = ", ".join(e.name for e in list_exporters() if e.supports_resumable_streaming)
        raise ValueError(
            f"Exporter '{exporter.name}' cannot resume a streamed run (--resume). Exporters that can: {resumable}."
        )
    return exporter


def _refuse_dated_destination(exporter: Any, field_values: dict[str, Any]) -> None:
    """Refuse a ``--resume`` output path that names a different file on every run.

    The checkpoint sits beside the output, so a re-run finds it only if the
    path resolves the same way: a date or time template (``{YYYYMMDD-HHMMSS}``,
    the exporters' default) would point each attempt at a fresh file and
    silently start over.
    """
    from vtscore.plugins.normalize import _DATETIME_TEMPLATE_VARS  # noqa: PLC0415

    for f in exporter.fields:
        value = field_values.get(f.key)
        if not isinstance(value, str):
            continue
        dated = [var for var in f.template_vars if var in _DATETIME_TEMPLATE_VARS and "{" + var + "}" in value]
        if dated:
            flag = f"--{f.key.replace('_', '-')}"
            raise ValueError(
                f"--resume needs an output that names the same file on every run, but {flag} {value!r} "
                f"uses {', '.join('{' + var + '}' for var in dated)}, which changes from one run to the next."
            )
