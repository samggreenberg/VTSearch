"""The datasets a run scores: saving them to the dashboard first, and multi-dataset runs.

A saving run (the default; ``--tempimport`` turns it off) imports its source
through the GUI's own load pipeline and registers it before scoring
(:func:`_save_source_dataset`), and may delete it again once detection has run
(:func:`_delete_after_detection`, #4674).  A multi-dataset import (#4707) is
one :class:`_OutputRun` per output, each a :class:`_RunDataset` scored and
exported on its own.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vtscore.cli import progress as cli_progress
from vtscore.cli.sources import (
    _check_multi_output,
    _list_importer_names,
    _output_dataset_names,
    _output_index,
    _SourceSpec,
)


@dataclass
class _RunDataset:
    """One dataset of a multi-dataset run (#4707), as the scoring pass sees it.

    *name* is the dataset's name (what a saving run registers it under, and
    what a temporary run would have); *dataset_id* its registry id when it was
    saved, empty otherwise.  *index* / *count* place it among the run's
    datasets, for the progress output and for telling whether an exporter's
    file needs the name to stay distinct.
    """

    name: str
    media_type: str
    category: str
    dataset_id: str = ""
    index: int = 0
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        """The ``dataset`` block carried by the results the exporter receives."""
        return {
            "name": self.name,
            "media_type": self.media_type,
            "category": self.category,
            "id": self.dataset_id or None,
        }

    def announce(self) -> None:
        """Say which of the run's datasets is about to be scored."""
        cli_progress.emit(
            "dataset_start",
            text=f"=== Dataset {self.index + 1} of {self.count}: {self.name!r} ({self.media_type}) ===",
            name=self.name,
            media_type=self.media_type,
            category=self.category,
            dataset_id=self.dataset_id or None,
            index=self.index,
            count=self.count,
        )


def _registered_entry_for_pickle(dataset_path: str) -> dict[str, Any] | None:
    """Return the registry entry whose saved pickle *is* *dataset_path*, if any.

    Pointing a saving run at a dataset the dashboard already holds (one of the
    ``ds_<uuid>.pkl`` files under the saved-datasets directory) must not import
    a second copy of it.
    """
    from vtscore.datasets.registry import list_datasets  # noqa: PLC0415

    target = Path(dataset_path).resolve()
    for entry in list_datasets():
        pkl_path = entry.get("pkl_path")
        if pkl_path and Path(pkl_path).resolve() == target:
            return entry
    return None


def _relay_import_progress(label: str = "") -> Callable[[dict[str, Any]], None]:
    """Build a load-tracker subscriber that narrates a saving import on the CLI.

    JSON mode forwards every tick as a ``progress`` event, the same stream the
    embedding stack feeds.  Text mode prints one line per phase rather than per
    tick: the tracker updates once per embedded item, and a line each would
    bury the run's real output.  *label* names the dataset when the run
    imports several at once (#4707), so the lines of one row can be told from
    another's; the ``progress`` events then carry it as ``dataset``.
    """
    last_phase: list[Any] = [None]

    def relay(snapshot: dict[str, Any]) -> None:
        status = str(snapshot.get("status") or "")
        message = str(snapshot.get("message") or "")
        if cli_progress.get_format() == "json":
            if not label:
                cli_progress.progress_callback(
                    status, message, int(snapshot.get("current") or 0), int(snapshot.get("total") or 0)
                )
            elif message or snapshot.get("total"):
                cli_progress.emit(
                    "progress",
                    status=status,
                    message=message,
                    current=int(snapshot.get("current") or 0),
                    total=int(snapshot.get("total") or 0),
                    dataset=label,
                )
            return
        phase = (status, snapshot.get("step"))
        if phase == last_phase[0] or status == "idle" or not message:
            return
        last_phase[0] = phase
        subject = f"Importing {label!r}" if label else "Importing"
        cli_progress.emit("import_progress", text=f"{subject}: {message}")

    return relay


def _await_import_tasks(task_ids: list[str], labels: list[str] | None = None) -> list[tuple[str, str]]:
    """Block until every background load in *task_ids* finishes; return ``(dataset_id, error)`` per task.

    The GUI's load pipeline runs on a worker thread and reports through the
    shared ``loading_tasks`` tracker, which is also where the registry id of
    the saved dataset is posted.  A Ctrl-C here cancels every load
    cooperatively - the same stop the dashboard's cancel button sends - so an
    interrupted run does not leave a half-built dataset registered.  Each
    task's outcome is reported on its own: its dataset id when it saved, else
    why it did not (its tracker's error, or a finish that registered nothing).

    *labels* name the datasets for the progress lines when there are several
    (a multi-dataset import, #4707); a single task is narrated unlabelled.
    """
    from vtscore.concurrency.progress import loading_tasks  # noqa: PLC0415

    trackers = [_import_tracker(task_id) for task_id in task_ids]
    done = threading.Event()
    registered: dict[str, str] = {}

    def on_tasks(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("task_id") in task_ids and row.get("dataset_id"):
                registered[row["task_id"]] = row["dataset_id"]
        if all(loading_tasks.is_finished(task_id) for task_id in task_ids):
            done.set()

    relays = [
        _relay_import_progress(labels[index] if labels and len(task_ids) > 1 else "") for index in range(len(task_ids))
    ]
    for tracker, relay in zip(trackers, relays, strict=True):
        tracker.subscribe(relay)
    loading_tasks.subscribe(on_tasks)
    try:
        # The worker may have got some way (or all the way) before the
        # subscription existed; read the current state once so neither the
        # dataset id nor the finish is missed.
        on_tasks(loading_tasks.list_tasks())
        try:
            done.wait()
        except KeyboardInterrupt:
            for task_id in task_ids:
                loading_tasks.cancel_task(task_id)
            done.wait()
            raise
    finally:
        loading_tasks.unsubscribe(on_tasks)
        for tracker, relay in zip(trackers, relays, strict=True):
            tracker.unsubscribe(relay)

    return [
        _import_outcome(tracker, registered.get(task_id, ""))
        for task_id, tracker in zip(task_ids, trackers, strict=True)
    ]


def _import_tracker(task_id: str) -> Any:
    """The progress tracker of the background load *task_id*; it must have been registered."""
    from vtscore.concurrency.progress import loading_tasks  # noqa: PLC0415

    tracker = loading_tasks.get_tracker(task_id)
    if tracker is None:
        raise RuntimeError(f"Import task {task_id} was not registered.")
    return tracker


def _import_outcome(tracker: Any, dataset_id: str) -> tuple[str, str]:
    """``(dataset_id, error)`` for one finished load: the id it registered, else why it has none."""
    error = tracker.get().get("error")
    if error:
        return "", f"Import failed: {error}"
    if not dataset_id:
        return "", "Import finished but the dataset could not be saved to the registry."
    return dataset_id, ""


def _wait_for_import(task_id: str) -> str:
    """Block until the background load *task_id* finishes; return its dataset id.

    The single-dataset form of :func:`_await_import_tasks`: a load that failed,
    or finished without registering a dataset, raises.
    """
    dataset_id, error = _await_import_tasks([task_id])[0]
    if error:
        raise ValueError(error)
    return dataset_id


def _release_imported_context(dataset_id: str) -> None:
    """Drop the in-memory copy the load pipeline left behind for *dataset_id*.

    The GUI keeps a freshly imported dataset resident so the user can browse
    it; a CLI run reads it back from its pickle for scoring, so holding the
    import's copy as well would double the run's peak memory.  The background
    archive-thumbnail warm-up the load kicks off is cancelled for the same
    reason: nobody will browse this process's copy.
    """
    import gc  # noqa: PLC0415

    from vtscore.concurrency.async_jobs import archive_thumbnail_jobs  # noqa: PLC0415
    from vtscore.datasets.registry import remove_loaded_id  # noqa: PLC0415
    from vtscore.state.core import unregister_context  # noqa: PLC0415

    for job in archive_thumbnail_jobs.active_jobs():
        if job.dataset_id == dataset_id:
            job.cancel()
    unregister_context(dataset_id)
    remove_loaded_id(dataset_id)
    gc.collect()


def _save_source_dataset(spec: _SourceSpec) -> tuple[dict[str, Any], bool]:
    """Import *spec*'s source exactly as the GUI would and register the result.

    Runs the dashboard's own load pipeline (clipping, embedding, duplicate
    collapse, coverage atlas, registry save) rather than the CLI's lighter
    scoring loader, so the saved dataset is the one a GUI import of the same
    source would have produced.  A ``--dataset`` pickle goes through the
    ``pickle`` importer, which copies it into the saved-datasets directory: the
    registry deletes a dataset's pickle when the dataset is deleted, so it must
    never adopt a file the user still owns.

    Returns the registry entry and whether this call imported it: ``False``
    when the pickle already was a dashboard dataset, which is then used as is.
    """
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415
    from vtscore.datasets.load_pipeline import _run_importer_in_background  # noqa: PLC0415
    from vtscore.datasets.registry import get_dataset  # noqa: PLC0415

    if spec.kind == "pickle":
        existing = _registered_entry_for_pickle(spec.dataset_path)
        if existing is not None:
            cli_progress.emit(
                "dataset_saved",
                text=(
                    f"Dataset {existing.get('name', '')!r} is already on the dashboard "
                    f"(id {existing['id']}); not importing it again."
                ),
                dataset_id=existing["id"],
                name=existing.get("name", ""),
                num_items=existing.get("num_items", 0),
                pkl_path=existing.get("pkl_path", ""),
                already_saved=True,
            )
            return existing, False
        if not Path(spec.dataset_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {spec.dataset_path}")
        importer_name, field_values = "pickle", {"file": spec.dataset_path}
    else:
        importer_name, field_values = spec.importer_name, dict(spec.field_values)

    importer = get_importer(importer_name)
    if importer is None:
        available = _list_importer_names()
        raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")
    importer.validate_cli_field_values(field_values)

    dataset_id = _wait_for_import(_run_importer_in_background(importer, field_values))
    entry = get_dataset(dataset_id)
    if entry is None:
        raise ValueError(f"Saved dataset {dataset_id} is missing from the registry.")
    _release_imported_context(dataset_id)
    _emit_dataset_saved(entry, dataset_id)
    return entry, True


def _delete_after_detection(entry: dict[str, Any], settings_path: str | None) -> None:
    """Delete the dataset this run imported, if the user asked for that (#4674).

    The per-user ``autofind_cli_delete_dataset`` setting makes a saving run
    clean up after itself once its detectors have scored the dataset and the
    results are exported.  The caller only gets here after such a run, and only
    with a dataset the run imported itself: one that was already on the
    dashboard stays, as does the dataset of a run that detected nothing (there
    the import was the point) or failed (so it can be searched again from the
    dashboard).
    """
    from vtscore.config import get_setting, redirect_settings_path  # noqa: PLC0415
    from vtscore.datasets.registry import unregister_dataset  # noqa: PLC0415

    if settings_path:
        redirect_settings_path(settings_path)
    if not get_setting("autofind_cli_delete_dataset") or not unregister_dataset(entry["id"]):
        return
    cli_progress.emit(
        "dataset_deleted",
        text=(
            f"Deleted dataset {entry.get('name', '')!r} (id {entry['id']}) from the dashboard: "
            "AutoFind has run, and the autofind_cli_delete_dataset setting is on."
        ),
        dataset_id=entry["id"],
        name=entry.get("name", ""),
    )


def _emit_dataset_saved(entry: dict[str, Any], dataset_id: str) -> None:
    """Report a dataset a saving run just registered on the dashboard."""
    cli_progress.emit(
        "dataset_saved",
        text=(
            f"Saved dataset {entry.get('name', '')!r} ({entry.get('num_items', 0)} medias) "
            f"to the dashboard (id {dataset_id})."
        ),
        dataset_id=dataset_id,
        name=entry.get("name", ""),
        num_items=entry.get("num_items", 0),
        pkl_path=entry.get("pkl_path", ""),
        already_saved=False,
    )


@dataclass
class _OutputRun:
    """One dataset of a multi-dataset run (#4707), ready to score - or not.

    *medias* is the dataset's chunk stream (``None`` when its import failed,
    with *error* saying why); *entry* its registry entry when this run saved
    it, so the ``autofind_cli_delete_dataset`` setting can delete it afterwards.
    """

    dataset: _RunDataset
    medias: Iterator[dict[int, dict[str, Any]]] | None
    entry: dict[str, Any] | None = None
    error: str = ""


def _multi_source_importer(spec: _SourceSpec) -> tuple[Any, dict[str, Any]]:
    """Resolve a multi-dataset source's importer and validate its shared field values."""
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415

    importer = get_importer(spec.importer_name)
    if importer is None:
        available = _list_importer_names()
        raise ValueError(f"Unknown importer: {spec.importer_name}. Available: {', '.join(available)}")
    _check_multi_output(importer)
    field_values = dict(spec.field_values)
    importer.validate_cli_field_values(field_values)
    return importer, field_values


def _save_multi_source_datasets(spec: _SourceSpec) -> list[_OutputRun]:
    """Import a multi-dataset source as the GUI would; return one run per output.

    One call to :func:`~vtscore.datasets.load_multi._run_multi_output_load_in_background`
    - the importer acquires its source once, every output is finalized as its
    own dataset - then every output is awaited and reported: a
    ``dataset_saved`` event for each that registered, a ``dataset_failed``
    one for each that did not.  A failure is that output's alone, as on the
    dashboard: the others are returned ready to score, and the caller fails
    the run once they have been.
    """
    from vtscore.datasets.load_multi import _run_multi_output_load_in_background  # noqa: PLC0415
    from vtscore.datasets.registry import get_dataset  # noqa: PLC0415

    importer, field_values = _multi_source_importer(spec)
    outputs = list(spec.outputs)
    names = _output_dataset_names(importer, field_values, outputs)
    task_ids = _run_multi_output_load_in_background(importer, field_values, outputs)
    outcomes = _await_import_tasks(task_ids, names)

    runs: list[_OutputRun] = []
    for index, (output, name, (dataset_id, error)) in enumerate(zip(outputs, names, outcomes, strict=True)):
        dataset = _RunDataset(name, output.media_type, output.category, index=index, count=len(outputs))
        entry = get_dataset(dataset_id) if dataset_id else None
        if not error and entry is None:
            error = f"Saved dataset {dataset_id} is missing from the registry."
        if error:
            cli_progress.emit(
                "dataset_failed", text=f"Import of dataset {name!r} failed: {error}", name=name, error=error
            )
            runs.append(_OutputRun(dataset, None, error=error))
            continue
        assert entry is not None
        _release_imported_context(dataset_id)
        _emit_dataset_saved(entry, dataset_id)
        dataset.dataset_id = dataset_id
        dataset.name = str(entry.get("name") or name)
        medias = _SourceSpec(kind="pickle", dataset_path=entry["pkl_path"], chunk_size=spec.chunk_size).load()
        runs.append(_OutputRun(dataset, medias, entry=entry))
    return runs


def _temporary_output_runs(spec: _SourceSpec) -> Iterator[_OutputRun]:
    """Open a multi-dataset source for a temporary run; yield one run per output as it is produced.

    The importer's first chunk of each output is pulled here, so an output it
    produced nothing for is reported as that output's failure (what the
    dashboard's row would say) rather than as the run's; the rest of the
    output's chunks stay lazy.
    """
    import itertools  # noqa: PLC0415

    importer, field_values = _multi_source_importer(spec)
    outputs = list(spec.outputs)
    names = _output_dataset_names(importer, field_values, outputs)
    for output, chunks in spec.load_outputs():
        index = _output_index(outputs, output)
        dataset = _RunDataset(names[index], output.media_type, output.category, index=index, count=len(outputs))
        first = next((chunk for chunk in chunks if chunk), None)
        if first is None:
            error = f"Import produced no {output.category_label()} media."
            cli_progress.emit(
                "dataset_failed",
                text=f"Import of dataset {dataset.name!r} failed: {error}",
                name=dataset.name,
                error=error,
            )
            yield _OutputRun(dataset, None, error=error)
            continue
        yield _OutputRun(dataset, itertools.chain([first], chunks))
