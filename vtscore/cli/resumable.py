"""``--resume``: a streamed run that checkpoints after every chunk and continues where it stopped.

The pipeline that writes through the checkpoint format in
:mod:`vtscore.cli.resume`: a re-run of the same command checks the chunks the
earlier attempt completed against their fingerprints without scoring them
again, keeps the recorded thresholds, and appends the rest to the exporter's
output.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vtscore.cli import progress as cli_progress
from vtscore.cli.detectors import _train_detectors_for_first_chunk
from vtscore.cli.export import _refuse_dated_destination, _resumable_exporter
from vtscore.cli.scoring import _detect_media_type, _score_medias_with_detectors
from vtscore.cli.streaming import _chunk_hit_records, _stream_header
from vtscore.datasets.loader import apply_custom_metadata_md5

if TYPE_CHECKING:
    from vtscore.cli.detectors import _RoutedSnapshots
    from vtscore.cli.resume import StreamCheckpoint
    from vtscore.cli.sources import _SourceSpec
    from vtscore.datasets.chunks import DeferredChunk


def _resume_identity(
    spec: _SourceSpec,
    *,
    detector_names: list[str],
    keep_negatives: bool,
    override_clipper: dict[str, Any] | None,
    exporter_name: str,
    exporter_field_values: dict[str, Any],
) -> dict[str, Any]:
    """The identity a ``--resume`` checkpoint records for this run (see :func:`~vtscore.cli.resume.run_identity`).

    Read before the source is opened: the importer's field values are
    normalised in place once loading starts, and the identity must be what
    the user asked for, the same on every attempt.
    """
    from vtscore.cli.resume import detector_identity, run_identity  # noqa: PLC0415
    from vtscore.config import get_setting  # noqa: PLC0415
    from vtscore.config.core_config import _SETTING_NAMES  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path, _read_detector  # noqa: PLC0415

    if spec.kind == "pickle":
        source: dict[str, Any] = {"kind": "pickle", "dataset": os.path.abspath(spec.dataset_path)}
    else:
        source = {"kind": "importer", "importer": spec.importer_name, "params": spec.field_values}
    return run_identity(
        source=source,
        chunk_size=int(spec.chunk_size or 0),
        detectors={name: detector_identity(_read_detector(_detector_path(name))) for name in detector_names},
        settings={name: get_setting(name) for name in sorted(_SETTING_NAMES)},
        scoring={"keep_negatives": bool(keep_negatives), "override_clipper": override_clipper},
        output={"exporter": exporter_name, "fields": exporter_field_values},
    )


def _run_resumable_streaming_pipeline(
    spec: _SourceSpec,
    *,
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
    override_detectors: list[str] | None,
    autofind_detectors: list[str],
    keep_negatives: bool,
    empty_error: str,
    override_clipper: dict[str, Any] | None = None,
) -> None:
    """Stream scored hits through a resumable exporter, checkpointing after every chunk (``--resume``).

    The resumable twin of :func:`_run_streaming_pipeline`.  A checkpoint
    (:class:`~vtscore.cli.resume.StreamCheckpoint`) beside the output records
    the run's identity, each completed chunk's fingerprint, the detectors'
    thresholds and the exporter's position after the last chunk it made
    durable.  With no checkpoint the run starts fresh and writes one; with a
    matching checkpoint it continues (:func:`_write_resumable_stream`).
    Everything that can refuse the resume without loading media - the
    exporter, the output path, the identity, a missing output - is checked
    here first, and the run holds its output for its whole length
    (:func:`~vtscore.cli.resume.exclusive_run`).
    """
    from vtscore.cli.resume import StreamCheckpoint, checkpoint_path_for, exclusive_run  # noqa: PLC0415

    exporter = _resumable_exporter(exporter_name)
    _refuse_dated_destination(exporter, exporter_field_values or {})
    field_values = dict(exporter_field_values or {})
    exporter.validate_cli_field_values(field_values)
    output = exporter.resumable_stream_path(field_values)
    if not output:
        raise ValueError(f"Exporter '{exporter.name}' names no output file to resume.")

    detector_names = list(override_detectors) if override_detectors is not None else autofind_detectors
    identity = _resume_identity(
        spec,
        detector_names=detector_names,
        keep_negatives=keep_negatives,
        override_clipper=override_clipper,
        exporter_name=exporter.name,
        exporter_field_values=field_values,
    )
    checkpoint_path = checkpoint_path_for(output)
    with exclusive_run(output):
        previous = StreamCheckpoint.load(checkpoint_path)
        if previous is not None:
            previous.check_identity(identity)
            if not Path(output).exists():
                raise previous.refuse(f"its output {output} is missing.")
            cli_progress.emit(
                "resume_start",
                text=(
                    f"Resuming from {checkpoint_path}: {previous.chunks_done} chunk(s) already written; "
                    "walking past them to check the source, without scoring them again."
                ),
                checkpoint=str(checkpoint_path),
                chunks_done=previous.chunks_done,
                resumed=True,
            )
        else:
            cli_progress.emit(
                "resume_start",
                text=f"Resumable run: checkpointing to {checkpoint_path} after each chunk.",
                checkpoint=str(checkpoint_path),
                chunks_done=0,
                resumed=False,
            )
        chunks = _write_resumable_stream(
            spec,
            exporter,
            field_values,
            StreamCheckpoint(path=checkpoint_path, identity=identity),
            previous,
            override_detectors=override_detectors,
            autofind_detectors=autofind_detectors,
            keep_negatives=keep_negatives,
            empty_error=empty_error,
            override_clipper=override_clipper,
        )
    cli_progress.emit(
        "resume_complete",
        text=f"Run complete after {chunks} chunk(s); removed the checkpoint {checkpoint_path}.",
        checkpoint=str(checkpoint_path),
        chunks=chunks,
    )


class _ResumableChunks:
    """A resumable run's chunk stream: each chunk fingerprinted, the completed ones checked and skipped.

    Chunks are numbered by their position in the source
    (:meth:`_SourceSpec.load_deferred`), empty ones included, so a chunk that
    scored to nothing still advances the checkpoint.  *previous* is the
    checkpoint of the run this one continues, ``None`` for a fresh run; its
    completed chunks are checked against their fingerprints but never
    embedded or scored, and not even built when the importer can say what
    one holds without building it (a ``key``; see :meth:`check`).
    """

    def __init__(self, spec: _SourceSpec, previous: "StreamCheckpoint | None") -> None:
        self._spec = spec
        self._chunks = iter(spec.load_deferred())
        self._previous = previous
        #: How many chunks the earlier run completed.
        self.done = previous.chunks_done if previous is not None else 0
        #: How many chunks have been pulled so far.
        self.seen = 0
        #: The position of the last chunk handed to the exporter; every chunk
        #: before it was either handed over too or had nothing to write.
        self.handed = self.done
        # The completed chunks' fingerprints are the recorded ones (each is
        # checked as it is passed); the chunks after them are fingerprinted afresh.
        self._completed = list(previous.chunk_digests) if previous is not None else []
        self._new: list[str] = []

    def pull(self) -> DeferredChunk | None:
        """The next chunk, not built yet; ``None`` at the end of the source."""
        deferred = next(self._chunks, None)
        if deferred is None:
            if self._previous is not None:
                self._previous.check_reached(self.seen)
            return None
        self.seen += 1
        return deferred

    def check(self, deferred: DeferredChunk, *, build: bool) -> dict[int, dict[str, Any]] | None:
        """Fingerprint the chunk just pulled, checking it if it was completed; its media if built.

        A chunk with a ``key`` is fingerprinted by it
        (:func:`~vtscore.cli.resume.key_digest`) and built only when *build*
        asks; one without is built either way and fingerprinted by its media
        (:func:`~vtscore.cli.resume.chunk_digest`).  Which of the two a chunk
        gets depends on its importer alone, so a resumed run fingerprints each
        chunk the way the run it continues did.
        """
        from vtscore.cli.resume import chunk_digest, key_digest  # noqa: PLC0415

        raw = deferred.build() if build or deferred.key is None else None
        if deferred.key is not None:
            fingerprint = key_digest(deferred.count, deferred.key())
        else:
            fingerprint = chunk_digest(raw or {})
        if self._previous is not None and self.seen <= self.done:
            self._previous.check_chunk(self.seen, fingerprint)
        else:
            self._new.append(fingerprint)
        return raw

    def first(self, empty_error: str) -> dict[int, dict[str, Any]]:
        """The first non-empty chunk, prepared: the one the detectors train on.

        Built and embedded whether or not it was completed, so a resumed run's
        head is the one the written chunks came from.
        """
        while (deferred := self.pull()) is not None:
            raw = self.check(deferred, build=deferred.count > 0)
            if raw and (chunk := self._spec.prepare(raw)):
                return chunk
        raise ValueError(empty_error)

    def records(
        self,
        first_chunk: dict[int, dict[str, Any]],
        first_position: int,
        detector_mlps: dict[str, dict[str, Any]],
        routed: "_RoutedSnapshots | None",
        keep_negatives: bool,
    ) -> Iterator[list[tuple[str, dict[str, Any]]]]:
        """Yield each chunk's hit records after the completed ones, the first chunk included unless it was."""
        if first_position > self.done:
            results = _score_medias_with_detectors(first_chunk, detector_mlps, routed)
            self.handed = first_position
            yield list(_chunk_hit_records(results, keep_negatives))
        # Only the first chunk's prepared snapshots are reused (see _stream_hit_records).
        routed = None
        while (deferred := self.pull()) is not None:
            if self.seen <= self.done:
                read = self.check(deferred, build=False) is not None
                cli_progress.emit(
                    "chunk_skipped",
                    text=(
                        f"Skipping chunk {self.seen} ({deferred.count} medias): already written"
                        f"{'' if read else '; checked without reading its files'}."
                    ),
                    chunk_num=self.seen,
                    chunk_size=deferred.count,
                    read=read,
                )
                continue
            records = self._score(self.check(deferred, build=True) or {}, detector_mlps, keep_negatives)
            # Before the yield: the exporter commits this chunk before it asks for the next.
            self.handed = self.seen
            yield records

    def _score(
        self, raw: dict[int, dict[str, Any]], detector_mlps: dict[str, dict[str, Any]], keep_negatives: bool
    ) -> list[tuple[str, dict[str, Any]]]:
        """Prepare and score one new chunk; its hit records."""
        chunk = self._spec.prepare(raw) if raw else raw
        if not chunk:
            return []
        apply_custom_metadata_md5(chunk)
        cli_progress.emit(
            "chunk_start",
            text=f"Processing chunk {self.seen} ({len(chunk)} medias)...",
            chunk_num=self.seen,
            chunk_size=len(chunk),
        )
        return list(_chunk_hit_records(_score_medias_with_detectors(chunk, detector_mlps), keep_negatives))

    def save(self, checkpoint: "StreamCheckpoint", position: Any) -> None:
        """Record in *checkpoint* that every chunk handed over is durable at the exporter's *position*."""
        checkpoint.chunks_done = self.handed
        checkpoint.chunk_digests = (self._completed[: self.done] + self._new)[: self.handed]
        checkpoint.position = position
        checkpoint.save()


def _write_resumable_stream(
    spec: _SourceSpec,
    exporter: Any,
    field_values: dict[str, Any],
    checkpoint: "StreamCheckpoint",
    previous: "StreamCheckpoint | None",
    *,
    override_detectors: list[str] | None,
    autofind_detectors: list[str],
    keep_negatives: bool,
    empty_error: str,
    override_clipper: dict[str, Any] | None,
) -> int:
    """Score *spec* chunk by chunk into *exporter*, saving *checkpoint* at every commit; the chunks seen.

    *previous* is the checkpoint of the run this one continues, ``None`` for
    a fresh run.  When it is set, the completed chunks are re-read and
    checked but not scored (:class:`_ResumableChunks`); the detectors are
    retrained on the same first chunk and keep the recorded thresholds
    (:meth:`~vtscore.cli.resume.StreamCheckpoint.pin_thresholds`); and the
    exporter cuts its output back to the recorded position and appends the
    remaining chunks, without a second header.  *checkpoint* is removed once
    the exporter completes.
    """
    from vtscore.exporters.base import StreamResume  # noqa: PLC0415

    stream = _ResumableChunks(spec, previous)
    first_chunk = stream.first(empty_error)
    first_position = stream.seen

    apply_custom_metadata_md5(first_chunk)
    media_type = _detect_media_type(first_chunk)
    detector_mlps, routed = _train_detectors_for_first_chunk(
        first_chunk, media_type, override_detectors, autofind_detectors, override_clipper
    )
    if previous is not None:
        for name, (retrained, recorded) in previous.pin_thresholds(detector_mlps).items():
            cli_progress.emit(
                "resume_threshold_kept",
                text=(
                    f"Detector '{name}' retrained to threshold {retrained:.6f}; keeping the checkpoint's "
                    f"{recorded:.6f} so every chunk is judged by one cut."
                ),
                detector=name,
                retrained=retrained,
                recorded=recorded,
            )
    checkpoint.thresholds = {name: float(info["threshold"]) for name, info in detector_mlps.items()}

    resume = StreamResume(
        position=previous.position if previous is not None else None,
        commit=lambda position: stream.save(checkpoint, position),
    )
    result = exporter.export_cli_streaming_resumable(
        _stream_header(media_type, detector_mlps, keep_negatives),
        stream.records(first_chunk, first_position, detector_mlps, routed, keep_negatives),
        field_values,
        resume,
    )
    checkpoint.clear()
    message = result.get("message", "Export complete.")
    cli_progress.emit("export_complete", text=message, message=message)
    return stream.seen
