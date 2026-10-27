"""The face types an application sees, and their CSV row forms.

:class:`FaceRecord` is the **only** face type the rest of an application
handles: a box, five landmarks, a confidence, the raw embedding, its quality,
the embedding-space tag (``compat``) and the id of the media it came from.
Every provider - the open-weights ``oss`` pipeline, the ``ext`` wrapper around
an external backend, a third party's - converts to it at its own boundary, so nothing
provider-specific leaks past :mod:`imagekit.face.providers`.

:class:`FaceDetection` is the detector-only half (box, landmarks,
confidence): what a detection CLI writes, and what an embedding extractor
takes in.

The row helpers here are the CSV *shape* (header and one row); reading and
writing whole files is :mod:`imagekit.face.face_util`.  Embeddings never go in
the CSV - they are saved separately as ``.npz`` keyed by media id and face
index - so a CSV row is readable by eye and the vectors stay binary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The five landmarks, in the order every detector here reports them.
LANDMARK_NAMES: tuple[str, ...] = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")

#: Width of the face embeddings this package produces and accepts.
EMBEDDING_DIM = 512

_BOX_COLUMNS = ("x1", "y1", "x2", "y2")
_LANDMARK_COLUMNS = tuple(f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y"))


def oss_compat(model_name: str, version: int | str) -> str:
    """The compat tag of an open-weights embedder: ``oss-<model>:<version>``."""
    return f"oss-{model_name}:{version}"


def embedding_quality(embedding: np.ndarray) -> float:
    """The L2 norm of a raw embedding: magnitude-aware losses put quality there."""
    return float(np.linalg.norm(np.asarray(embedding, dtype=np.float32)))


def _fmt(value: Any) -> str:
    return f"{float(value):.6f}"


def _landmark_cells(landmarks: np.ndarray | None) -> list[str]:
    if landmarks is None:
        return [""] * len(_LANDMARK_COLUMNS)
    return [_fmt(v) for v in np.asarray(landmarks, dtype=np.float32).reshape(-1)]


def _landmarks_from_cells(cells: Sequence[str]) -> np.ndarray | None:
    if len(cells) != len(_LANDMARK_COLUMNS):
        raise ValueError(f"expected {len(_LANDMARK_COLUMNS)} landmark cells, got {len(cells)}")
    if all(cell.strip() == "" for cell in cells):
        return None
    return np.asarray([float(c) for c in cells], dtype=np.float32).reshape(5, 2)


def as_landmarks(points: Any) -> np.ndarray | None:
    """Coerce *points* to a ``(5, 2)`` ``float32`` array, or ``None`` when absent or unusable."""
    if points is None:
        return None
    try:
        arr = np.asarray(points, dtype=np.float32).reshape(5, 2)
    except (TypeError, ValueError):
        return None
    return arr if bool(np.isfinite(arr).all()) else None


def as_bbox(box: Any) -> tuple[float, float, float, float]:
    """Coerce *box* to an ``(x1, y1, x2, y2)`` tuple of floats."""
    x1, y1, x2, y2 = (float(v) for v in np.asarray(box, dtype=np.float64).reshape(4))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


@dataclass
class FaceDetection:
    """One detected face: where it is and how sure the detector was.

    ``landmarks`` is ``(5, 2)`` in image pixels, in :data:`LANDMARK_NAMES`
    order, or ``None`` for a detector (or a row) that gave none; the aligner
    then falls back to the box.
    """

    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray | None
    confidence: float

    @property
    def has_landmarks(self) -> bool:
        return self.landmarks is not None

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


def detection_csv_header() -> list[str]:
    """Columns of a detections-only CSV."""
    return ["media_id", "face_index", *_BOX_COLUMNS, "confidence", *_LANDMARK_COLUMNS]


def detection_to_row(media_id: str, face_index: int, detection: FaceDetection) -> list[str]:
    """One detections-CSV row, in :func:`detection_csv_header` order."""
    return [
        media_id,
        str(int(face_index)),
        *(_fmt(v) for v in detection.bbox),
        _fmt(detection.confidence),
        *_landmark_cells(detection.landmarks),
    ]


def detection_from_row(row: Sequence[str]) -> tuple[str, int, FaceDetection]:
    """Parse a detections-CSV row back into ``(media_id, face_index, detection)``."""
    header = detection_csv_header()
    if len(row) != len(header):
        raise ValueError(f"detection row has {len(row)} cells; expected {len(header)}")
    media_id, face_index = row[0], int(row[1])
    bbox = as_bbox(row[2:6])
    confidence = float(row[6])
    landmarks = _landmarks_from_cells(row[7:])
    return media_id, face_index, FaceDetection(bbox, landmarks, confidence)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class FaceRecord:
    """A detected, aligned and embedded face.

    Attributes:
        bbox: ``(x1, y1, x2, y2)`` in image pixels.
        landmarks: ``(5, 2)`` pixels in :data:`LANDMARK_NAMES` order, or ``None``.
        confidence: Detector confidence in ``[0, 1]``.
        embedding: ``(512,)`` ``float32``, **unnormalised**, exactly as the model
            produced it.  :meth:`unit_embedding` is the cosine-space copy.
        quality: L2 norm of ``embedding``: magnitude-aware losses (MagFace)
            encode face quality in the norm.
        compat: Embedding-space tag, e.g. ``"oss-magface-ir100:1"``, or
            whatever tag an external backend reports.  Vectors with
            different tags must never be compared (:mod:`imagekit.face.compat`).
        media_id: Identifier of the image this face came from; the CSV and
            ``.npz`` writers key on it.
    """

    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray | None
    confidence: float
    embedding: np.ndarray
    quality: float
    compat: str
    media_id: str = ""
    #: Set by the loaders when a CSV row carried an explicit face index.
    face_index: int = field(default=0, compare=False)

    @classmethod
    def from_detection(
        cls,
        detection: FaceDetection,
        embedding: Any,
        compat: str,
        media_id: str = "",
        face_index: int = 0,
    ) -> "FaceRecord":
        """Pair a detection with the raw embedding its aligned crop produced."""
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return cls(
            bbox=as_bbox(detection.bbox),
            landmarks=as_landmarks(detection.landmarks),
            confidence=float(detection.confidence),
            embedding=vec,
            quality=embedding_quality(vec),
            compat=str(compat),
            media_id=str(media_id),
            face_index=int(face_index),
        )

    @property
    def detection(self) -> FaceDetection:
        """The detector-only half of this record."""
        return FaceDetection(self.bbox, self.landmarks, self.confidence)

    def unit_embedding(self) -> np.ndarray:
        """An L2-normalised copy of :attr:`embedding` (a zero vector stays zero)."""
        vec = np.asarray(self.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec.copy() if norm == 0.0 else vec / norm

    @staticmethod
    def csv_header() -> list[str]:
        """Columns of a face-records CSV; the embedding is deliberately absent."""
        return ["media_id", "face_index", *_BOX_COLUMNS, "confidence", "quality", "compat", *_LANDMARK_COLUMNS]

    def to_row(self, face_index: int | None = None) -> list[str]:
        """One records-CSV row, in :meth:`csv_header` order.

        *face_index* overrides :attr:`face_index`; writers that number faces
        per media pass it in.
        """
        index = self.face_index if face_index is None else int(face_index)
        return [
            self.media_id,
            str(index),
            *(_fmt(v) for v in self.bbox),
            _fmt(self.confidence),
            _fmt(self.quality),
            self.compat,
            *_landmark_cells(self.landmarks),
        ]

    @classmethod
    def from_row(cls, row: Sequence[str], embedding: Any = None) -> "FaceRecord":
        """Parse a records-CSV row; *embedding* supplies the vector the CSV omits.

        Without one the record carries an empty ``(0,)`` vector, so a CSV-only
        load is still a list of :class:`FaceRecord`; :attr:`quality` and
        :attr:`compat` come from the row either way.
        """
        header = cls.csv_header()
        if len(row) != len(header):
            raise ValueError(f"face record row has {len(row)} cells; expected {len(header)}")
        vec = np.zeros((0,), dtype=np.float32) if embedding is None else np.asarray(embedding, dtype=np.float32)
        return cls(
            bbox=as_bbox(row[2:6]),
            landmarks=_landmarks_from_cells(row[9:]),
            confidence=float(row[6]),
            embedding=vec.reshape(-1),
            quality=float(row[7]),
            compat=row[8],
            media_id=row[0],
            face_index=int(row[1]),
        )
