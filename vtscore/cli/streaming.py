"""``--stream-results``: score chunk by chunk and hand the hits straight to a streaming exporter.

Nothing accumulates across chunks, so this is the path that scales to a source
with more media (and more hits) than fit in RAM.  ``--as-labelset`` streams the
hits as label entries instead (:func:`_stream_label_records`), and
``--max-results`` keeps each detector's best records (:func:`_top_records`).
The resumable form (``--resume``) lives in :mod:`vtscore.cli.resumable`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from vtscore.cli import progress as cli_progress
from vtscore.cli.detectors import _train_detectors_for_first_chunk
from vtscore.cli.export import _streaming_exporter
from vtscore.cli.scoring import _detect_media_type, _score_medias_with_detectors
from vtscore.datasets.loader import apply_custom_metadata_md5

if TYPE_CHECKING:
    from vtscore.cli.datasets import _RunDataset
    from vtscore.cli.detectors import _RoutedSnapshots


def _stream_hit_records(
    first_chunk: dict[int, dict[str, Any]],
    rest: Iterator[dict[int, dict[str, Any]]],
    detector_mlps: dict[str, dict[str, Any]],
    keep_negatives: bool,
    routed: "_RoutedSnapshots | None" = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Score each chunk and yield ``(detector_name, hit)`` pairs in chunk order.

    No global accumulation and no global sort: each chunk is scored, its hits
    are yielded, and the chunk is dropped before the next one is pulled, so
    peak memory stays bounded by the chunk size regardless of how many hits
    the whole run produces.  *routed* is the first chunk's already-prepared
    scoring snapshots, handed over from training (issue #3647) and released as
    soon as that chunk is scored.  Above-threshold hits carry ``label="good"``;
    below-threshold hits are emitted (with ``label="bad"``) only when
    *keep_negatives* is set.
    """
    import itertools  # noqa: PLC0415

    for chunk_num, chunk in enumerate(itertools.chain([first_chunk], rest), 1):
        if not chunk:
            continue
        if chunk is not first_chunk:
            # The first chunk was already normalised by the caller (it had to
            # be, to train the detectors and build the header).
            apply_custom_metadata_md5(chunk)
            cli_progress.emit(
                "chunk_start",
                text=f"Processing chunk {chunk_num} ({len(chunk)} medias)...",
                chunk_num=chunk_num,
                chunk_size=len(chunk),
            )
        chunk_results = _score_medias_with_detectors(chunk, detector_mlps, routed)
        # Only the first chunk has prepared snapshots to reuse, and holding
        # them past its scoring pass would keep every clip alive for the whole
        # run - the one thing this pipeline exists to avoid.
        routed = None
        yield from _chunk_hit_records(chunk_results, keep_negatives)


def _chunk_hit_records(
    chunk_results: dict[str, dict[str, Any]], keep_negatives: bool
) -> Iterator[tuple[str, dict[str, Any]]]:
    """One scored chunk's ``(detector_name, hit)`` pairs, labelled, in detector order."""
    for det_name, det_result in chunk_results.items():
        for hit in det_result.get("hits", []):
            yield det_name, {**hit, "label": "good"}
        if keep_negatives:
            for hit in det_result.get("negative_hits", []):
                yield det_name, {**hit, "label": "bad"}


def _top_records(
    records: Iterator[tuple[str, dict[str, Any]]],
    max_results: int,
    detector_order: list[str],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Each detector's *max_results* best records, highest score first (``--max-results``, #5279).

    One pass over *records* with a bounded min-heap per detector: a record
    that beats the weakest one kept takes its place, and every other record is
    dropped as soon as it is read, so peak memory is *max_results* hits per
    detector however many the run scores.  Nothing is yielded until *records*
    is exhausted, since the last chunk can still hold the best hit.  The
    detectors come out in *detector_order* (the header's, which names every
    detector a record can carry), each one's records sorted by score, ties in
    the order they arrived.  With ``keep_negatives`` the ``bad`` records
    compete too, ranked below every ``good`` one, so they fill a detector's
    quota only past its last hit.
    """
    import heapq  # noqa: PLC0415

    heaps: dict[str, list[tuple[bool, float, int, dict[str, Any]]]] = {}
    seen: dict[str, int] = {}
    for arrival, (det_name, hit) in enumerate(records):
        seen[det_name] = seen.get(det_name, 0) + 1
        # -arrival makes the earlier of two equal records the stronger, and,
        # being unique, settles every comparison before it reaches the hit.
        entry = (hit["label"] == "good", hit["score"], -arrival, hit)
        heap = heaps.setdefault(det_name, [])
        if len(heap) < max_results:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    for det_name in detector_order:
        kept, total = len(heaps.get(det_name, [])), seen.get(det_name, 0)
        cli_progress.emit(
            "top_results",
            text=(
                f"Detector '{det_name}': {total} result(s) scored; "
                f"writing the top {kept} (--max-results {max_results})."
            ),
            detector=det_name,
            kept=kept,
            seen=total,
            max_results=max_results,
        )
    for det_name in detector_order:
        for *_rank, hit in sorted(heaps.pop(det_name, []), reverse=True):
            yield det_name, hit


def _stream_label_records(records: Iterator[tuple[str, dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    """Turn a ``(detector_name, hit)`` stream into label entries, each naming its detector.

    Each hit becomes the elements a labelset export records for it
    (:func:`~vtscore.datasets.labelset.elements_for_media`, so a dupe set
    expands to its members), serialised and passed through
    :func:`~vtscore.datasets.labelset.label_entry_for_export`.  The hit's
    ``score`` and media ``id`` stay behind: a score is a number this head
    measured, and the id is this run's numbering, gone once it ends.  Lazy, like
    the hit stream it wraps.
    """
    from vtscore.datasets.labelset import elements_for_media, label_entry_for_export  # noqa: PLC0415

    for detector_name, hit in records:
        for element in elements_for_media(hit, hit["label"]):
            yield {**label_entry_for_export(element.to_dict()), "detector": detector_name}


def _run_streaming_pipeline(
    media_source: Iterator[dict[int, dict[str, Any]]],
    *,
    exporter_name: str | None,
    exporter_field_values: dict[str, Any] | None,
    override_detectors: list[str] | None,
    autofind_detectors: list[str],
    keep_negatives: bool,
    empty_error: str,
    dataset: "_RunDataset | None" = None,
    as_labelset: bool = False,
    override_clipper: dict[str, Any] | None = None,
    max_results: int | None = None,
) -> None:
    """Stream scored hits straight to a streaming-capable exporter.

    Trains detectors on the first chunk (so the exporter gets its header
    before any hit), then hands the exporter a lazy record iterator.  Nothing
    accumulates across chunks, so this is the path that scales to a media
    source with more items (and more hits) than fit in RAM.  *dataset*, when
    the run produces several (#4707), rides into the header as a ``dataset``
    block.

    With *as_labelset* the hits go out as label entries instead, through
    ``export_labelset_streaming``: every detector's labels in one export, each
    entry naming its detector, and a header carrying no threshold or balance
    state (see :func:`_stream_label_records`).

    A resumable run (``--resume``) takes
    :func:`~vtscore.cli.resumable._run_resumable_streaming_pipeline` instead,
    which :func:`~vtscore.cli.run._run_pipeline` picks.

    With *max_results* (``--max-results``) the exporter gets each detector's
    best *max_results* records, sorted by score, once the last chunk is
    scored, instead of every record in chunk order (:func:`_top_records`).
    It is never resumable (:func:`~vtscore.cli.run._check_max_results`).
    """
    exporter = _streaming_exporter(exporter_name, as_labelset=as_labelset)
    exporter.validate_cli_field_values(exporter_field_values or {})

    # Pull the first non-empty chunk so we can detect the media type and train
    # the detectors before any hit streams out.
    iterator = iter(media_source)
    first_chunk: dict[int, dict[str, Any]] | None = None
    for chunk in iterator:
        if chunk:
            first_chunk = chunk
            break
    if first_chunk is None:
        raise ValueError(empty_error)

    apply_custom_metadata_md5(first_chunk)
    media_type = _detect_media_type(first_chunk)
    detector_mlps, routed = _train_detectors_for_first_chunk(
        first_chunk, media_type, override_detectors, autofind_detectors, override_clipper
    )

    header = _stream_header(media_type, detector_mlps, keep_negatives, dataset, max_results)
    records = _stream_hit_records(first_chunk, iterator, detector_mlps, keep_negatives, routed)
    if max_results is not None:
        records = _top_records(records, max_results, list(detector_mlps))
    if as_labelset:
        label_header: dict[str, Any] = {
            "media_type": media_type,
            "detectors": list(detector_mlps),
            "keep_negatives": bool(keep_negatives),
        }
        if dataset is not None:
            label_header["dataset"] = dataset.to_dict()
        if max_results is not None:
            label_header["max_results"] = max_results
        result = exporter.export_labelset_streaming(
            label_header, _stream_label_records(records), exporter_field_values or {}
        )
    else:
        result = exporter.export_cli_streaming(header, records, exporter_field_values or {})
    message = result.get("message", "Export complete.")
    cli_progress.emit("export_complete", text=message, message=message)


def _stream_header(
    media_type: str,
    detector_mlps: dict[str, dict[str, Any]],
    keep_negatives: bool,
    dataset: "_RunDataset | None" = None,
    max_results: int | None = None,
) -> dict[str, Any]:
    """The metadata a streamed hit export opens with (see ``export_cli_streaming``).

    ``max_results`` is there only when ``--max-results`` set it: the records
    are then each detector's best, sorted by score, not chunk-ordered.
    """
    header: dict[str, Any] = {
        "media_type": media_type,
        "detectors": [
            {
                "detector_name": name,
                "threshold": round(info["threshold"], 4),
                "balance": info.get("balance"),
            }
            for name, info in detector_mlps.items()
        ],
        "keep_negatives": bool(keep_negatives),
    }
    if dataset is not None:
        header["dataset"] = dataset.to_dict()
    if max_results is not None:
        header["max_results"] = max_results
    return header
