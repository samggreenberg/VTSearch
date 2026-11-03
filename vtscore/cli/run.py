"""Run one autodetect source end to end: optionally save it, then detect and export.

:func:`_run_source` is what the entry points and the YAML pipeline runner both
call; :func:`_run_pipeline` reads the settings and picks the pass - the dry
run, the in-memory pass (:func:`_run_live_pipeline`), a streamed one
(:mod:`vtscore.cli.streaming`) or a resumable one
(:mod:`vtscore.cli.resumable`).  The checks that refuse a flag combination
before anything loads live here too.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from vtscore.cli import progress as cli_progress
from vtscore.cli.datasets import (
    _delete_after_detection,
    _save_multi_source_datasets,
    _save_source_dataset,
    _temporary_output_runs,
)
from vtscore.cli.detectors import (
    _NoApplicableDetectorsError,
    _resolve_override_clipper,
    _train_detectors_for_first_chunk,
)
from vtscore.cli.dry_run import _run_dry_run
from vtscore.cli.export import _exporter_fields_for_dataset, _resolve_exporter, _run_exporter
from vtscore.cli.resumable import _run_resumable_streaming_pipeline
from vtscore.cli.scoring import (
    _build_multi_results_dict,
    _detect_media_type,
    _merge_detector_results,
    _score_medias_with_detectors,
    _sort_detector_results,
)
from vtscore.cli.sources import _SourceSpec
from vtscore.cli.streaming import _run_streaming_pipeline
from vtscore.datasets.loader import apply_custom_metadata_md5

if TYPE_CHECKING:
    from vtscore.cli.datasets import _OutputRun, _RunDataset
    from vtscore.cli.detectors import _RoutedSnapshots


def _score_chunk(
    chunk_medias: dict[int, dict[str, Any]],
    chunk_num: int,
    total_medias: int,
    detector_mlps: dict[str, dict[str, Any]],
    merged_results: dict[str, dict[str, Any]],
    routed: "_RoutedSnapshots | None" = None,
) -> None:
    """Emit chunk-start progress when relevant, score this chunk, and merge in place."""
    if chunk_num > 1 or total_medias != len(chunk_medias):
        cli_progress.emit(
            "chunk_start",
            text=f"Processing chunk {chunk_num} ({len(chunk_medias)} medias)...",
            chunk_num=chunk_num,
            chunk_size=len(chunk_medias),
        )
    chunk_results = _score_medias_with_detectors(chunk_medias, detector_mlps, routed)
    _merge_detector_results(merged_results, chunk_results)


def _run_live_pipeline(
    media_source: Iterator[dict[int, dict[str, Any]]],
    *,
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
    override_detectors: list[str] | None,
    autofind_detectors: list[str],
    empty_error: str,
    dataset: "_RunDataset | None" = None,
    override_clipper: dict[str, Any] | None = None,
) -> None:
    """Iterate *media_source*, score each chunk, and run the exporter on the merged results.

    *dataset* names the dataset this pass scores when the run produces several
    (#4707); it rides into the results the exporter receives.
    """
    merged_results: dict[str, dict[str, Any]] = {}
    media_type: str | None = None
    detector_mlps: dict[str, dict[str, Any]] | None = None
    # The first chunk's prepared snapshots, held only until that chunk is
    # scored against them; every later chunk routes itself.
    routed: _RoutedSnapshots | None = None
    total_medias = 0
    chunk_num = 0

    for chunk_num, chunk_medias in enumerate(media_source, 1):
        if not chunk_medias:
            continue

        apply_custom_metadata_md5(chunk_medias)
        total_medias += len(chunk_medias)

        if media_type is None:
            media_type = _detect_media_type(chunk_medias)
            detector_mlps, routed = _train_detectors_for_first_chunk(
                chunk_medias, media_type, override_detectors, autofind_detectors, override_clipper
            )

        if detector_mlps:
            _score_chunk(chunk_medias, chunk_num, total_medias, detector_mlps, merged_results, routed)
        routed = None

    if not merged_results:
        raise ValueError(empty_error)

    _sort_detector_results(merged_results)

    if chunk_num > 1:
        cli_progress.emit(
            "chunks_done",
            text=f"Finished processing {total_medias} medias across {chunk_num} chunk(s).",
            total_medias=total_medias,
            chunks=chunk_num,
        )

    results = _build_multi_results_dict(merged_results, media_type or "unknown", dataset)
    _run_exporter(exporter_name or "gui", exporter_field_values or {}, results, detector_mlps)


def _run_pipeline(
    media_source: Iterator[dict[int, dict[str, Any]]],
    *,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    override_detectors: list[str] | None = None,
    empty_error: str = "No medias loaded",
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    as_labelset: bool = False,
    source_description: dict[str, Any] | None = None,
    skip_without_detectors: bool = False,
    dataset: "_RunDataset | None" = None,
    override_clipper: dict[str, Any] | None = None,
    resumable_source: _SourceSpec | None = None,
    max_results: int | None = None,
) -> bool:
    """Shared pipeline: read settings, iterate media chunks, score, export.

    All four CLI entry points (pickle / importer, whole / chunked) delegate
    to this single function, differing only in the *media_source* iterator
    they supply.

    When *dry_run* is True the function prints the plan derived from
    *source_description* + the settings file and returns without consuming
    the iterator, so no importer runs and no embedding or scoring occurs.

    When *override_detectors* is supplied, that list of detector names is
    used in place of the settings file's ``autofind_detectors``.  The pipeline
    YAML loader uses this to declare detectors inline, and
    ``--import-labels-into`` to run just the detector it imported into, both
    without mutating the settings file on disk.

    *override_clipper*, an ``input_spec`` already through
    :func:`_resolve_override_clipper`, re-clips the dataset for every detector
    at that granularity in place of the detector's own ``input_spec``
    (``--override-clipper``, #5282).

    *skip_without_detectors* is set by a run that saved its dataset first:
    there the import is the point and detection is the extra, so having no
    detector to run - none configured, or none for this media type - ends the
    run with a note instead of an error.  A multi-dataset run sets it for every
    dataset, saved or not: one dataset no detector applies to must not stop
    the others being scored (the caller errors only when none was).

    *dataset* is the dataset this pass scores when the run produces several
    (#4707).  It names the pass in the progress output, rides into the results
    the exporter gets, and gives that exporter its own destination:
    ``{dataset_name}`` in any exporter field is replaced by the dataset's name,
    and a ``filepath`` that does not mention it gains the name before its
    extension, so two datasets' files do not overwrite each other (see
    :func:`_exporter_fields_for_dataset`).

    *resumable_source* makes a streaming run resumable (``--resume``): its
    media are read from that source instead of *media_source*, raw, so the
    chunks an earlier attempt completed can be skipped before they are
    embedded (see :func:`_run_resumable_streaming_pipeline`).

    *max_results*, with *stream_results*, exports only each detector's best
    *max_results* records, sorted by score (``--max-results``; see
    :func:`_top_records`).

    Returns whether detection ran: ``True`` once the detectors have scored the
    source and the exporter has run, ``False`` for a dry run and for a saving
    run that skipped detection.
    """
    from vtscore.config import get_setting, redirect_settings_path

    # Settings are read through ``get_setting`` so this function - and the
    # helpers below - never import ``vtsearch.settings`` directly.
    if settings_path:
        redirect_settings_path(settings_path)
    autofind_detectors = list(get_setting("autofind_detectors"))
    detector_names = list(override_detectors) if override_detectors is not None else autofind_detectors
    exporter_name, exporter_field_values = _resolve_exporter(exporter_name, exporter_field_values)
    if dataset is not None and not dry_run:
        exporter_field_values = _exporter_fields_for_dataset(exporter_name or "gui", exporter_field_values, dataset)

    if dry_run:
        _run_dry_run(
            source_description,
            settings_path,
            autofind_detectors,
            exporter_name,
            exporter_field_values,
            override_detectors,
            override_clipper,
        )
        return False

    if dataset is not None:
        dataset.announce()

    if skip_without_detectors and not detector_names:
        # Checked before the source is opened: with nothing to score, reading
        # the whole dataset back in would be wasted work.
        _emit_detection_skipped("no AutoFind detectors are configured", dataset)
        return False

    try:
        if stream_results and resumable_source is not None:
            _run_resumable_streaming_pipeline(
                resumable_source,
                exporter_name=exporter_name,
                exporter_field_values=exporter_field_values,
                override_detectors=override_detectors,
                autofind_detectors=autofind_detectors,
                keep_negatives=keep_negatives,
                empty_error=empty_error,
                override_clipper=override_clipper,
            )
        elif stream_results:
            _run_streaming_pipeline(
                media_source,
                exporter_name=exporter_name,
                exporter_field_values=exporter_field_values,
                override_detectors=override_detectors,
                autofind_detectors=autofind_detectors,
                keep_negatives=keep_negatives,
                as_labelset=as_labelset,
                empty_error=empty_error,
                dataset=dataset,
                override_clipper=override_clipper,
                max_results=max_results,
            )
        else:
            _run_live_pipeline(
                media_source,
                exporter_name=exporter_name,
                exporter_field_values=exporter_field_values,
                override_detectors=override_detectors,
                autofind_detectors=autofind_detectors,
                empty_error=empty_error,
                dataset=dataset,
                override_clipper=override_clipper,
            )
    except _NoApplicableDetectorsError as exc:
        if not skip_without_detectors:
            raise
        _emit_detection_skipped(str(exc), dataset)
        return False
    return True


def _emit_detection_skipped(reason: str, dataset: "_RunDataset | None" = None) -> None:
    """Report that a run had nothing to detect its dataset with.

    A saved dataset stays on the dashboard, and the note says so; one dataset
    of a multi-dataset run is named, since the others may still be scored.
    """
    saved = dataset is None or bool(dataset.dataset_id)
    subject = f"Detection skipped for dataset {dataset.name!r}" if dataset is not None else "Detection skipped"
    kept = " The dataset was still saved to the dashboard." if saved else ""
    fields: dict[str, Any] = {"reason": reason}
    if dataset is not None:
        fields["dataset"] = dataset.name
    cli_progress.emit("detection_skipped", text=f"{subject}: {reason}.{kept}", **fields)


def _run_multi_source(
    spec: _SourceSpec,
    *,
    save_dataset: bool,
    settings_path: str | None,
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
    override_detectors: list[str] | None,
    dry_run: bool,
    stream_results: bool,
    keep_negatives: bool,
    as_labelset: bool,
    override_clipper: dict[str, Any] | None = None,
    max_results: int | None = None,
) -> None:
    """Run a multi-dataset source (#4707): one importer run, then detect and export per dataset.

    The multi-dataset form of :func:`_run_source`.  A saving run imports every
    output through the dashboard's own multi-dataset pipeline first and scores
    each saved copy; a temporary run scores each output straight off the
    importer.  Either way each dataset is then its own pass of
    :func:`_run_pipeline`: its own detectors (the AutoFind ones that apply to
    its media type), its own results, its own exporter run with its own
    destination (:func:`_exporter_fields_for_dataset`).

    What is per dataset and what is the run's: a dataset no detector applies
    to is skipped with a note, and only a run in which *no* dataset could be
    scored fails for it (when temporary; a saving run's datasets are the point
    and stay saved).  A dataset whose import failed is reported and skipped,
    and the run fails once every other dataset has been scored and exported,
    so one bad output costs nothing but its own results.
    """
    source_description = spec.describe(
        stream_results=stream_results,
        keep_negatives=keep_negatives,
        save_dataset=save_dataset,
        as_labelset=as_labelset,
        max_results=max_results,
    )
    if dry_run:
        _run_pipeline(
            iter(()),
            settings_path=settings_path,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            override_detectors=override_detectors,
            empty_error=spec.empty_error,
            dry_run=True,
            stream_results=stream_results,
            keep_negatives=keep_negatives,
            as_labelset=as_labelset,
            source_description=source_description,
            skip_without_detectors=save_dataset,
            override_clipper=override_clipper,
            max_results=max_results,
        )
        return

    runs: Iterator[_OutputRun]
    if save_dataset:
        if stream_results:
            raise ValueError(f"--stream-results: {_STREAM_CANNOT_SAVE}. Run it as a temporary import.")
        runs = iter(_save_multi_source_datasets(spec))
    else:
        runs = _temporary_output_runs(spec)

    failures: list[str] = []
    detected_any = False
    for run in runs:
        if run.medias is None:
            failures.append(f"{run.dataset.name}: {run.error}")
            continue
        detected = _run_pipeline(
            run.medias,
            settings_path=settings_path,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            override_detectors=override_detectors,
            empty_error=f"No medias loaded for dataset {run.dataset.name!r} by importer '{spec.importer_name}'",
            stream_results=stream_results,
            keep_negatives=keep_negatives,
            as_labelset=as_labelset,
            source_description=source_description,
            skip_without_detectors=True,
            dataset=run.dataset,
            override_clipper=override_clipper,
            max_results=max_results,
        )
        detected_any = detected_any or detected
        if detected and run.entry is not None:
            _delete_after_detection(run.entry, settings_path)

    if failures:
        raise ValueError(f"{len(failures)} of {len(spec.outputs)} dataset(s) failed to import: " + "; ".join(failures))
    if not detected_any and not save_dataset:
        raise _NoApplicableDetectorsError(
            f"No AutoFind detector applies to any of the {len(spec.outputs)} dataset(s) this run produced. "
            "Add detectors for their media types to the settings file's autofind_detectors list."
        )


#: Why a streaming run cannot save its dataset, shared by every entry point that refuses one.
_STREAM_CANNOT_SAVE = "streaming never holds the whole dataset in memory, so it cannot save it to the dashboard"


def _run_source(
    spec: _SourceSpec,
    *,
    save_dataset: bool,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    override_detectors: list[str] | None = None,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    as_labelset: bool = False,
    override_clipper: dict[str, Any] | None = None,
    resume: bool = False,
    max_results: int | None = None,
) -> None:
    """Optionally save *spec*'s dataset to the dashboard, then detect and export.

    Shared by the flag-driven entry points and the YAML pipeline runner, so the
    two cannot disagree about what a saving run does.  When *save_dataset* is
    set the source is imported and registered first, and detection then runs
    over the saved pickle - so its hits are the ones the user will find on that
    dashboard row.  A temporary run (the pre-#4226 behaviour) scores the source
    straight from the importer and keeps nothing.  A saving run whose user
    turned on ``autofind_cli_delete_dataset`` deletes what it imported once
    detection has run (:func:`_delete_after_detection`).  A source with
    ``outputs`` is several datasets from one importer run, each saved, scored
    and exported on its own (:func:`_run_multi_source`).

    *settings_path* is applied first, before the save: the import reads the
    server-tier settings too (the saved-datasets directory, the retention
    stamp, the download and embedding gates), so a run whose settings switched
    over only at the scoring step saved its dataset under the default file's.

    A run that will load media then gets the same runtime preparation the
    server's startup does (:func:`~vtscore.embedding.initialize_models`), before
    the first worker thread exists: it warms sklearn's threadpool controller
    while the process is still single-threaded (the loader-lock/GIL deadlock
    guard) and seeds the stat-free ``packages_distributions`` before anything
    imports transformers (#3715).  A dry run loads nothing and skips it.

    *override_clipper* is validated here, before any of that, so an unknown
    clipper or parameter fails before a dataset is saved
    (:func:`_resolve_override_clipper`).

    *as_labelset* is a form of streaming, so it is refused without
    *stream_results* rather than ignored.

    *resume* makes a chunked, single-dataset hit stream resumable
    (``--resume``; :func:`_run_resumable_streaming_pipeline`), and is refused
    for any other run (:func:`_check_resumable`).

    *max_results*, with *stream_results*, exports only each detector's best
    *max_results* records, sorted by score (``--max-results``); it is refused
    without streaming and with *resume* (:func:`_check_max_results`).
    """
    if as_labelset and not stream_results:
        raise ValueError("as_labelset streams the hits as a labelset, so it requires stream_results.")
    _check_max_results(max_results, stream_results=stream_results, resume=resume)
    if resume:
        _check_resumable(spec, stream_results=stream_results, as_labelset=as_labelset)
    override_clipper = _resolve_override_clipper(override_clipper)
    if settings_path:
        from vtscore.config import redirect_settings_path  # noqa: PLC0415

        redirect_settings_path(settings_path)
    if not dry_run:
        from vtscore.embedding import initialize_models  # noqa: PLC0415

        initialize_models()
    if spec.outputs:
        _run_multi_source(
            spec,
            save_dataset=save_dataset,
            settings_path=settings_path,
            exporter_name=exporter_name,
            exporter_field_values=exporter_field_values,
            override_detectors=override_detectors,
            dry_run=dry_run,
            stream_results=stream_results,
            keep_negatives=keep_negatives,
            as_labelset=as_labelset,
            override_clipper=override_clipper,
            max_results=max_results,
        )
        return
    source_description = spec.describe(
        stream_results=stream_results,
        keep_negatives=keep_negatives,
        save_dataset=save_dataset,
        as_labelset=as_labelset,
        resume=resume,
        max_results=max_results,
    )
    imported: dict[str, Any] | None = None
    if save_dataset and not dry_run:
        if stream_results:
            raise ValueError(f"--stream-results: {_STREAM_CANNOT_SAVE}. Run it as a temporary import.")
        entry, is_new = _save_source_dataset(spec)
        if is_new:
            imported = entry
        spec = _SourceSpec(kind="pickle", dataset_path=entry["pkl_path"], chunk_size=spec.chunk_size)
    detected = _run_pipeline(
        # A resumable run reads its source itself (resumable_source), raw.
        spec.load() if not (dry_run or resume) else iter(()),
        settings_path=settings_path,
        exporter_name=exporter_name,
        exporter_field_values=exporter_field_values,
        override_detectors=override_detectors,
        empty_error=spec.empty_error,
        dry_run=dry_run,
        stream_results=stream_results,
        keep_negatives=keep_negatives,
        as_labelset=as_labelset,
        source_description=source_description,
        skip_without_detectors=save_dataset,
        override_clipper=override_clipper,
        resumable_source=spec if resume else None,
        max_results=max_results,
    )
    if detected and imported is not None:
        _delete_after_detection(imported, settings_path)


def _check_resumable(spec: _SourceSpec, *, stream_results: bool, as_labelset: bool) -> None:
    """Refuse ``--resume`` on a run it cannot make resumable.

    A resumable run streams hits a chunk at a time and checkpoints after each
    one, so it needs ``stream_results`` and a chunk size.  It writes one
    output, so a multi-dataset import is out; and a streamed labelset's file
    closes with a trailer that appending cannot extend.  Raised before anything
    loads.
    """
    if not stream_results:
        raise ValueError("--resume checkpoints a streamed run, so it requires --stream-results.")
    if not spec.chunk_size:
        raise ValueError("--resume checkpoints after each chunk, so it requires --chunk-size.")
    if as_labelset:
        raise ValueError("--resume cannot resume a labelset stream (--as-labelset); it resumes hit streams only.")
    if spec.outputs:
        raise ValueError("--resume cannot resume a multi-dataset run (--outputs); run each dataset on its own.")


def _check_max_results(max_results: int | None, *, stream_results: bool, resume: bool) -> None:
    """Refuse a ``--max-results`` the run cannot honour.

    It keeps the best of a streamed run's records, so it needs
    ``stream_results``.  It writes nothing until the last chunk is scored, so
    a ``--resume`` run would have no written chunk to checkpoint, and the best
    records so far would die with an interrupted process.  Raised before
    anything loads.
    """
    if max_results is None:
        return
    if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
        raise ValueError(f"--max-results must be a positive integer, not {max_results!r}.")
    if not stream_results:
        raise ValueError("--max-results keeps the best of a streamed run's results, so it requires --stream-results.")
    if resume:
        raise ValueError(
            "--max-results cannot be combined with --resume: it writes nothing until the last chunk is scored, "
            "so there is no chunk to checkpoint."
        )
