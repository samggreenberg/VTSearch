"""Whole-file CSV and ``.npz`` helpers for detections and face records.

Two files per run of the full pipeline, deliberately:

* ``<prefix>.csv`` holds one :class:`~imagekit.face.records.FaceRecord` row per
  face with the embedding **omitted** (see :meth:`FaceRecord.csv_header`), so
  the run is readable and diffable;
* ``<prefix>.npz`` holds the raw embeddings keyed by ``"<media_id>::<face_index>"``
  (:func:`embedding_key`), plus the run's compat tag under :data:`COMPAT_FIELD`,
  so a loader can refuse to mix it with another space before it reads a vector.

The detection CLI writes a detections-only CSV (:func:`save_detections_csv`)
with no companion file.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from imagekit.face.compat import single_compat
from imagekit.face.records import (
    FaceDetection,
    FaceRecord,
    detection_csv_header,
    detection_from_row,
    detection_to_row,
)

#: The ``.npz`` field holding the compat tag of every vector in the archive.
COMPAT_FIELD = "__compat__"

#: The ``.npz`` field naming this archive layout, for a loader to check.
FORMAT_FIELD = "__format__"
NPZ_FORMAT = "imagekit-face-embeddings:1"

_KEY_SEP = "::"


def embedding_key(media_id: str, face_index: int) -> str:
    """The ``.npz`` key of one face's vector."""
    return f"{media_id}{_KEY_SEP}{int(face_index)}"


def split_embedding_key(key: str) -> tuple[str, int]:
    """Inverse of :func:`embedding_key`."""
    media_id, _, index = key.rpartition(_KEY_SEP)
    if not _ or not index.isdigit():
        raise ValueError(f"not an embedding key: {key!r}")
    return media_id, int(index)


def assign_face_indices(records: Iterable[FaceRecord]) -> list[tuple[int, FaceRecord]]:
    """Number faces ``0, 1, ...`` per ``media_id`` in record order.

    The pipeline emits faces highest-confidence first, so index 0 is a media's
    primary face.  Records are not modified; the index rides beside each.
    """
    counters: dict[str, int] = {}
    out: list[tuple[int, FaceRecord]] = []
    for record in records:
        index = counters.get(record.media_id, 0)
        counters[record.media_id] = index + 1
        out.append((index, record))
    return out


# ---------------------------------------------------------------------------
# Detections CSV
# ---------------------------------------------------------------------------


def save_detections_csv(path: str | Path, rows: Iterable[tuple[str, int, FaceDetection]]) -> int:
    """Write ``(media_id, face_index, detection)`` rows; returns how many."""
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(detection_csv_header())
        for media_id, face_index, detection in rows:
            writer.writerow(detection_to_row(media_id, face_index, detection))
            count += 1
    return count


def load_detections_csv(path: str | Path) -> list[tuple[str, int, FaceDetection]]:
    """Read a file written by :func:`save_detections_csv`."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != detection_csv_header():
            raise ValueError(f"{path}: not a detections CSV (header {header})")
        return [detection_from_row(row) for row in reader if row]


# ---------------------------------------------------------------------------
# Face records CSV
# ---------------------------------------------------------------------------


def save_face_records_csv(path: str | Path, records: Iterable[FaceRecord]) -> int:
    """Write one row per record (embedding omitted), faces numbered per media; returns how many."""
    indexed = assign_face_indices(records)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(FaceRecord.csv_header())
        for index, record in indexed:
            writer.writerow(record.to_row(index))
    return len(indexed)


def load_face_records_csv(path: str | Path) -> list[FaceRecord]:
    """Read a file written by :func:`save_face_records_csv`; embeddings come back empty.

    Pair with :func:`load_embeddings_npz` through :func:`load_face_records` to
    get the vectors back.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != FaceRecord.csv_header():
            raise ValueError(f"{path}: not a face records CSV (header {header})")
        return [FaceRecord.from_row(row) for row in reader if row]


# ---------------------------------------------------------------------------
# Embeddings NPZ
# ---------------------------------------------------------------------------


def save_embeddings_npz(path: str | Path, records: Iterable[FaceRecord]) -> int:
    """Write every record's raw embedding, keyed per face, with the compat tag; returns how many.

    All records must share one compat tag (:func:`~imagekit.face.compat.single_compat`);
    an archive of one space is the whole point of the tag.  An empty record
    list still writes an archive, tagged ``""``, so a run that found no faces
    leaves both files.
    """
    indexed = assign_face_indices(records)
    compat = single_compat((r.compat for _, r in indexed), what="face records") if indexed else ""
    arrays = {embedding_key(r.media_id, i): np.asarray(r.embedding, dtype=np.float32) for i, r in indexed}
    arrays[COMPAT_FIELD] = np.asarray(compat)
    arrays[FORMAT_FIELD] = np.asarray(NPZ_FORMAT)
    # pyright reads the ``**`` splat against savez's keyword parameters; every key here is an array name.
    np.savez_compressed(path, **arrays)  # pyright: ignore[reportArgumentType]
    return len(indexed)


def load_embeddings_npz(path: str | Path) -> tuple[dict[str, np.ndarray], str]:
    """Read a file written by :func:`save_embeddings_npz`: ``({key: vector}, compat)``."""
    with np.load(path, allow_pickle=False) as archive:
        fmt = str(archive[FORMAT_FIELD]) if FORMAT_FIELD in archive.files else ""
        if fmt != NPZ_FORMAT:
            raise ValueError(f"{path}: not an imagekit embeddings archive (format {fmt!r})")
        compat = str(archive[COMPAT_FIELD])
        vectors = {
            key: np.asarray(archive[key], dtype=np.float32)
            for key in archive.files
            if key not in (COMPAT_FIELD, FORMAT_FIELD)
        }
    return vectors, compat


def load_face_records(csv_path: str | Path, npz_path: str | Path) -> list[FaceRecord]:
    """Join a records CSV with its embeddings archive into full records.

    Every CSV row must find its vector (:func:`embedding_key` of its media id
    and face index) and carry the archive's compat tag; a mismatch names the
    row rather than returning a half-filled record.
    """
    records = load_face_records_csv(csv_path)
    vectors, compat = load_embeddings_npz(npz_path)
    out: list[FaceRecord] = []
    for record in records:
        key = embedding_key(record.media_id, record.face_index)
        if key not in vectors:
            raise ValueError(f"{npz_path}: no embedding for {key!r} listed in {csv_path}")
        if record.compat != compat:
            raise ValueError(f"{csv_path}: row {key!r} has compat {record.compat!r} but the archive is {compat!r}")
        record.embedding = vectors[key]
        out.append(record)
    return out
