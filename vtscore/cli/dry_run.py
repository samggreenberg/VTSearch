"""``--dry-run``: validate the planned run and print (or emit) its plan, loading nothing.

The source, the exporter and - for a streaming or resumable run - the
exporter's streaming support are checked with the same helpers the run uses,
so a plan fails where the run would.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtscore.cli import progress as cli_progress
from vtscore.cli.detectors import _describe_clipper_spec
from vtscore.cli.export import (
    _list_exporter_names,
    _refuse_dated_destination,
    _resumable_exporter,
    _streaming_exporter,
)
from vtscore.cli.sources import _check_multi_output, _list_importer_names


def _summarize_autofind_detectors(detector_names: list[str]) -> list[dict[str, Any]]:
    """Read each named detector's on-disk JSON and return a small summary.

    Used by ``--dry-run`` to describe which detectors would be trained and
    scored without actually loading models or embedding any media.
    """
    from vtscore.detectors.store import _detector_path, _read_detector

    summaries: list[dict[str, Any]] = []
    for name in detector_names:
        path = _detector_path(name)
        data = _read_detector(path)
        if data is None:
            summaries.append({"name": name, "path": str(path), "missing": True})
            continue
        labelset = data.get("labelset") or {}
        labels = labelset.get("labels") if isinstance(labelset, dict) else None
        n_labels = len(labels) if isinstance(labels, list) else 0
        summaries.append(
            {
                "name": name,
                "path": str(path),
                "media_type": data.get("media_type", "") or "",
                "labels": n_labels,
                "missing": False,
            }
        )
    return summaries


def _print_dry_run_source(source_description: dict[str, Any]) -> None:
    """Print the ``Source:`` block of the dry-run plan (kind, params, chunking)."""
    print("Source:", flush=True)
    kind = source_description.get("kind", "")
    if kind == "pickle":
        print(f"  Dataset pickle: {source_description.get('dataset', '')}", flush=True)
    elif kind == "importer":
        print(f"  Importer: {source_description.get('importer', '')}", flush=True)
        params = source_description.get("params") or {}
        if params:
            print("  Params:", flush=True)
            for k, v in params.items():
                print(f"    {k}: {v if v != '' else '(empty)'}", flush=True)
        else:
            print("  Params: (none)", flush=True)
        outputs = source_description.get("outputs") or []
        if outputs:
            print(f"  Datasets ({len(outputs)}; one importer run, one dataset per entry):", flush=True)
            for output in outputs:
                print(f"    - {_describe_output(output)}", flush=True)
    chunk_size = source_description.get("chunk_size")
    print(f"  Chunk size: {chunk_size if chunk_size else 'whole dataset'}", flush=True)
    _print_dry_run_streaming(source_description)
    if source_description.get("delete_after_detection"):
        print(
            "  Save to dashboard: until detection has run (the autofind_cli_delete_dataset setting then deletes it;"
            " a run that detects nothing keeps it)",
            flush=True,
        )
    elif source_description.get("save_dataset"):
        print("  Save to dashboard: yes (the imported dataset is kept; --tempimport discards it)", flush=True)
    else:
        print("  Save to dashboard: no (--tempimport: the dataset is discarded after detection)", flush=True)


def _print_dry_run_streaming(source_description: dict[str, Any]) -> None:
    """Print the dry-run plan's ``Streaming:`` and ``Resume:`` lines, when they apply."""
    if source_description.get("stream_results"):
        neg = "included" if source_description.get("keep_negatives") else "dropped"
        mode = "yes, as a labelset (label entries" if source_description.get("as_labelset") else "yes (hits"
        max_results = source_description.get("max_results")
        if max_results:
            written = f"each detector's top {max_results} by score written to the exporter after the last chunk"
        else:
            written = "written to the exporter per chunk"
        print(f"  Streaming: {mode} {written}; negatives {neg})", flush=True)
    if source_description.get("resume"):
        checkpoint = source_description.get("resume_checkpoint", "")
        done = source_description.get("resume_chunks_done")
        state = f"{done} chunk(s) already written" if done is not None else "none yet, so the run starts fresh"
        print(f"  Resume: yes (checkpoint {checkpoint}: {state})", flush=True)


def _describe_output(output: dict[str, Any]) -> str:
    """One line of the dry-run plan for an entry of a multi-dataset import's ``outputs``.

    Reads the :meth:`~vtscore.datasets.importers.base.OutputSpec.to_dict` shape
    the source description carries: the category label leads (it is what names
    the dataset), then the dataset's media type, its source rows as
    ``source[→converter]``, and whichever per-dataset options were set.
    """
    from vtscore.datasets.importers.base.outputs import _type_label  # noqa: PLC0415

    media_type = str(output.get("media_type") or "")
    category = str(output.get("category") or media_type)
    sources = ", ".join(
        f"{s.get('source_type', '')}→{s['converter']}" if s.get("converter") else str(s.get("source_type", ""))
        for s in output.get("source_specs") or []
    )
    parts = [f"media_type={media_type}"]
    if category != media_type:
        parts.append(f"category={category}")
    parts.append(f"sources={sources or media_type}")
    for key in ("dataset_name", "embedder", "clipper"):
        if output.get(key):
            parts.append(f"{key}={output[key]}")
    if output.get("embedders"):
        parts.append(f"embedders={','.join(output['embedders'])}")
    return f"{_type_label(category)}  [{', '.join(parts)}]"


def _print_dry_run_plan(
    *,
    source_description: dict[str, Any],
    settings_path: str | None,
    autofind_detectors: list[str],
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
    override_detectors: list[str] | None = None,
    override_clipper: dict[str, Any] | None = None,
) -> None:
    """Print the autodetect plan that ``--dry-run`` would otherwise execute.

    With *override_detectors* the plan lists those instead of the settings
    file's *autofind_detectors*, and says so - the run will not read the
    AutoFind list at all.  An *override_clipper* gets a line of its own.
    """
    print("DRY RUN - no media will be loaded, embedded, scored, or exported.", flush=True)
    print("", flush=True)

    _print_dry_run_source(source_description)
    print("", flush=True)

    print(f"Settings: {settings_path or '(default: data/settings.json)'}", flush=True)
    detector_names, heading, count_note = autofind_detectors, "AutoFind detectors", ""
    if override_detectors is not None:
        detector_names = override_detectors
        heading, count_note = "Detectors", "; overrides the settings' AutoFind list"
    if not detector_names:
        if source_description.get("save_dataset"):
            print(f"{heading}: (none - the dataset would be saved and detection skipped)", flush=True)
        else:
            print(f"{heading}: (none - pipeline would abort with an error)", flush=True)
    else:
        summaries = _summarize_autofind_detectors(detector_names)
        print(f"{heading} ({len(summaries)}{count_note}):", flush=True)
        for s in summaries:
            if s.get("missing"):
                print(f"  - {s['name']}  [MISSING - {s['path']}]", flush=True)
            else:
                print(
                    f"  - {s['name']}  [media_type={s['media_type'] or '?'}, labels={s['labels']}, file={s['path']}]",
                    flush=True,
                )
    if override_clipper:
        print(
            f"Clipper override: {_describe_clipper_spec(override_clipper)} (in place of each detector's input_spec)",
            flush=True,
        )
    print("", flush=True)

    print(f"Exporter: {exporter_name or 'gui (default - print to console)'}", flush=True)
    if exporter_field_values:
        for k, v in exporter_field_values.items():
            print(f"  {k}: {v if v != '' else '(empty)'}", flush=True)


def _validate_dry_run_source(sd: dict[str, Any]) -> None:
    """Validate the source description block passed to a dry run."""
    kind = sd.get("kind")
    if kind == "pickle":
        dataset = sd.get("dataset", "")
        if dataset and not Path(dataset).exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset}")
    elif kind == "importer":
        from vtscore.datasets.importers import get_importer

        importer_name = sd.get("importer", "")
        importer = get_importer(importer_name)
        if importer is None:
            available = _list_importer_names()
            raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")
        if sd.get("outputs"):
            _check_multi_output(importer)
        importer.validate_cli_field_values(sd.get("params") or {})


def _validate_dry_run_exporter(exporter_name: str, exporter_field_values: dict[str, Any] | None) -> None:
    """Resolve *exporter_name* in the registry and validate its CLI field values."""
    from vtscore.exporters import get_exporter

    exporter = get_exporter(exporter_name)
    if exporter is None:
        available = _list_exporter_names()
        raise ValueError(f"Unknown exporter: {exporter_name}. Available: {', '.join(available)}")
    exporter.validate_cli_field_values(exporter_field_values or {})


def _emit_dry_run_plan(
    source_description: dict[str, Any],
    settings_path: str | None,
    autofind_detectors: list[str],
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
    override_detectors: list[str] | None = None,
    override_clipper: dict[str, Any] | None = None,
) -> None:
    """Emit the JSON ``dry_run_plan`` event or the human-readable plan text.

    The JSON event's ``autofind_detectors`` always lists the detectors the run
    would train - *override_detectors* when given - and ``detectors_source``
    (``"autofind"`` / ``"override"``) says which list that was.  Its
    ``override_clipper`` is the clipper override's ``input_spec``, or ``None``.
    """
    if cli_progress.get_format() == "json":
        detector_names = autofind_detectors if override_detectors is None else override_detectors
        cli_progress.emit(
            "dry_run_plan",
            source=source_description,
            settings_path=settings_path,
            autofind_detectors=_summarize_autofind_detectors(detector_names),
            detectors_source="autofind" if override_detectors is None else "override",
            override_clipper=override_clipper,
            exporter=exporter_name,
            exporter_field_values=exporter_field_values or {},
        )
    else:
        _print_dry_run_plan(
            source_description=source_description,
            settings_path=settings_path,
            autofind_detectors=autofind_detectors,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            override_detectors=override_detectors,
            override_clipper=override_clipper,
        )


def _run_dry_run(
    source_description: dict[str, Any] | None,
    settings_path: str | None,
    autofind_detectors: list[str],
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
    override_detectors: list[str] | None = None,
    override_clipper: dict[str, Any] | None = None,
) -> None:
    """Validate the planned pipeline + exporter and emit the dry-run plan.

    A saving run whose user turned on ``autofind_cli_delete_dataset`` is planned
    as one that deletes its dataset again once detection has run (#4674).
    """
    from vtscore.config import get_setting  # noqa: PLC0415

    sd = source_description or {}
    if sd.get("save_dataset") and get_setting("autofind_cli_delete_dataset"):
        sd = {**sd, "delete_after_detection": True}
    _validate_dry_run_source(sd)
    if exporter_name:
        _validate_dry_run_exporter(exporter_name, exporter_field_values)
    if sd.get("stream_results"):
        _streaming_exporter(exporter_name, as_labelset=bool(sd.get("as_labelset")))
    if sd.get("resume"):
        sd = {**sd, **_dry_run_resume_state(exporter_name, exporter_field_values)}
    _emit_dry_run_plan(
        sd,
        settings_path,
        autofind_detectors,
        exporter_name,
        exporter_field_values,
        override_detectors,
        override_clipper,
    )


def _dry_run_resume_state(exporter_name: str | None, exporter_field_values: dict[str, Any] | None) -> dict[str, Any]:
    """Where a ``--resume`` run would keep its checkpoint, and how far an existing one got.

    Validates what the run would refuse - an exporter that cannot resume, an
    output path that changes from run to run, an unreadable checkpoint - so
    the plan fails where the run would.  It does not check the checkpoint
    against this run's identity: that reads every detector and the settings,
    which the run itself does before loading anything.
    """
    from vtscore.cli.resume import StreamCheckpoint, checkpoint_path_for  # noqa: PLC0415

    exporter = _resumable_exporter(exporter_name)
    _refuse_dated_destination(exporter, exporter_field_values or {})
    field_values = dict(exporter_field_values or {})
    exporter.validate_cli_field_values(field_values)
    checkpoint_path = checkpoint_path_for(exporter.resumable_stream_path(field_values))
    checkpoint = StreamCheckpoint.load(checkpoint_path)
    return {
        "resume_checkpoint": str(checkpoint_path),
        "resume_chunks_done": checkpoint.chunks_done if checkpoint is not None else None,
    }
