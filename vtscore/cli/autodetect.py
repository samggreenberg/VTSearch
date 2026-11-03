"""The four public ``autodetect_*_main`` entry points: (pickle / importer) x (whole / chunked).

Each builds the :class:`~vtscore.cli.sources._SourceSpec` for its cell of the
2x2 and hands it to :func:`_autodetect`, which runs it
(:func:`~vtscore.cli.run._run_source`) and turns any failure into an
``Error: ...`` line (or an ``error`` event) and exit status 1.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from vtscore.cli import progress as cli_progress
from vtscore.cli.run import _run_source
from vtscore.cli.sources import _SourceSpec

if TYPE_CHECKING:
    from vtscore.datasets.importers.base import OutputSpec


def _autodetect(
    spec: _SourceSpec,
    *,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    as_labelset: bool = False,
    save_dataset: bool = False,
    override_detectors: list[str] | None = None,
    override_clipper: dict[str, Any] | None = None,
    resume: bool = False,
    max_results: int | None = None,
) -> None:
    """Shared body of the four public ``autodetect_*_main`` entry points.

    Everything that used to be hand-copied across the 2x2 (pickle /
    importer) x (whole / chunked) matrix lives here; *spec* supplies the
    only parts that legitimately differ - the loader call, the dry-run
    source description, and the "nothing loaded" message.
    """
    try:
        _run_source(
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
            resume=resume,
            max_results=max_results,
        )
    except Exception as e:
        cli_progress.emit_error(str(e))
        sys.exit(1)


def autodetect_main(
    dataset_path: str,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    as_labelset: bool = False,
    save_dataset: bool = False,
    override_detectors: list[str] | None = None,
    override_clipper: dict[str, Any] | None = None,
    max_results: int | None = None,
) -> None:
    """CLI entry point: run autodetect with all AutoFind detectors.

    With *save_dataset* the source is first saved to the dashboard and the run
    scores that saved copy; having no applicable detector then ends the run
    with a note rather than an error.  The default leaves nothing behind,
    which is what these entry points always did.

    *override_detectors*, when given, names the detectors to run in place of
    the settings file's ``autofind_detectors`` list (which is then not read
    for this run; the file on disk is never modified).  ``--import-labels-into``
    uses it to run exactly the detector it just imported into.

    *as_labelset*, with *stream_results*, streams the hits as label entries
    (every detector's, each naming its detector, no scores) through the
    exporter's ``export_labelset_streaming``, so the export reads back in
    through a label importer.

    *override_clipper*, when given, is an ``input_spec`` -
    ``{"clipper": "sound_tiling", "clipper_params": {"duration": "5"}}`` - that
    every detector is scored at in place of its own (``--override-clipper``):
    the dataset is re-clipped to it unless already clipped to match, and a
    ``*_default`` clipper scores each media as loaded.  An unknown clipper or
    parameter, or a clipper for another media type than a detector scores,
    fails the run.

    *max_results*, with *stream_results*, exports only each detector's best
    *max_results* records, sorted by score, once the run has scored every
    chunk (``--max-results``); peak memory stays at *max_results* hits per
    detector.  Without it a streamed export is in chunk order.
    """
    _autodetect(
        _SourceSpec(kind="pickle", dataset_path=dataset_path),
        settings_path=settings_path,
        exporter_name=exporter_name,
        exporter_field_values=exporter_field_values,
        dry_run=dry_run,
        stream_results=stream_results,
        keep_negatives=keep_negatives,
        as_labelset=as_labelset,
        save_dataset=save_dataset,
        override_detectors=override_detectors,
        override_clipper=override_clipper,
        max_results=max_results,
    )


def autodetect_importer_main(
    importer_name: str,
    field_values: dict[str, Any],
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    as_labelset: bool = False,
    save_dataset: bool = False,
    override_detectors: list[str] | None = None,
    outputs: list[OutputSpec] | None = None,
    override_clipper: dict[str, Any] | None = None,
    max_results: int | None = None,
) -> None:
    """CLI entry point: run autodetect with a named importer and output results.

    *save_dataset*, *override_detectors*, *override_clipper* and *max_results*
    behave as in :func:`autodetect_main`.
    *outputs*, when given, makes the one importer run produce several datasets
    (#4707), one per :class:`~vtscore.datasets.importers.base.OutputSpec`,
    each saved (unless temporary), scored and exported on its own; see
    :func:`_run_multi_source`.
    """
    _autodetect(
        _SourceSpec(
            kind="importer", importer_name=importer_name, field_values=field_values, outputs=tuple(outputs or ())
        ),
        settings_path=settings_path,
        exporter_name=exporter_name,
        exporter_field_values=exporter_field_values,
        dry_run=dry_run,
        stream_results=stream_results,
        keep_negatives=keep_negatives,
        as_labelset=as_labelset,
        save_dataset=save_dataset,
        override_detectors=override_detectors,
        override_clipper=override_clipper,
        max_results=max_results,
    )


def autodetect_main_chunked(
    dataset_path: str,
    chunk_size: int,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    as_labelset: bool = False,
    save_dataset: bool = False,
    override_detectors: list[str] | None = None,
    override_clipper: dict[str, Any] | None = None,
    resume: bool = False,
    max_results: int | None = None,
) -> None:
    """CLI entry point: chunked autodetect on a pickle dataset.

    *save_dataset*, *override_detectors*, *override_clipper* and *max_results*
    behave as in :func:`autodetect_main`; *chunk_size* bounds the scoring pass
    over the saved copy.  *resume*, with *stream_results*, checkpoints the
    stream after each chunk and continues an interrupted run of the same
    command (``--resume``; see :mod:`vtscore.cli.resume`); it cannot be
    combined with *max_results*.
    """
    _autodetect(
        _SourceSpec(kind="pickle", dataset_path=dataset_path, chunk_size=chunk_size),
        settings_path=settings_path,
        exporter_name=exporter_name,
        exporter_field_values=exporter_field_values,
        dry_run=dry_run,
        stream_results=stream_results,
        keep_negatives=keep_negatives,
        as_labelset=as_labelset,
        save_dataset=save_dataset,
        override_detectors=override_detectors,
        override_clipper=override_clipper,
        resume=resume,
        max_results=max_results,
    )


def autodetect_importer_main_chunked(
    importer_name: str,
    field_values: dict[str, Any],
    chunk_size: int,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    as_labelset: bool = False,
    save_dataset: bool = False,
    override_detectors: list[str] | None = None,
    outputs: list[OutputSpec] | None = None,
    override_clipper: dict[str, Any] | None = None,
    resume: bool = False,
    max_results: int | None = None,
) -> None:
    """CLI entry point: chunked autodetect with a named importer.

    *save_dataset*, *override_detectors*, *override_clipper*, *max_results* and
    *outputs* behave as in :func:`autodetect_importer_main`; *chunk_size* bounds the scoring pass over
    the saved copy (the import itself is held in memory whole, as a GUI import
    is).  *resume* behaves as in :func:`autodetect_main_chunked`.
    """
    _autodetect(
        _SourceSpec(
            kind="importer",
            importer_name=importer_name,
            field_values=field_values,
            chunk_size=chunk_size,
            outputs=tuple(outputs or ()),
        ),
        settings_path=settings_path,
        exporter_name=exporter_name,
        exporter_field_values=exporter_field_values,
        dry_run=dry_run,
        stream_results=stream_results,
        keep_negatives=keep_negatives,
        as_labelset=as_labelset,
        save_dataset=save_dataset,
        override_detectors=override_detectors,
        override_clipper=override_clipper,
        resume=resume,
        max_results=max_results,
    )
