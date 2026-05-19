"""Face localizer using MediaPipe Face Detection."""

from __future__ import annotations

import io
from typing import Any, Optional

from PIL import Image

from vtscore.media.processors import Localizer


class FaceLocalizer(Localizer):
    """Localizes faces in images using MediaPipe Face Detection.

    Running :meth:`localize` on an image clip returns a list of dicts, one per
    detected face whose confidence meets the threshold::

        [
            {
                "confidence": 0.95,
                "bbox": [x1, y1, x2, y2],
            },
            ...
        ]

    Coordinates are in pixel space (float) matching the original image size.
    """

    def __init__(self, name: str, threshold: float = 0.5, model_selection: int = 1) -> None:
        """Create a face localizer.

        Args:
            name: Unique name for this localizer instance.
            threshold: Minimum confidence to count a detection (0-1).
            model_selection: 0 for short-range (within 2m), 1 for full-range (within 5m).
        """
        self._name = name
        self._threshold = threshold
        self._model_selection = model_selection
        self._detector: Optional[Any] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def media_type(self) -> str:
        return "image"

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def model_selection(self) -> int:
        return self._model_selection

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if self._detector is not None:
            return
        import mediapipe as mp  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        self._detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=self._threshold,
            model_selection=self._model_selection,
        )

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------

    def localize(self, clip: dict[str, Any]) -> list[dict[str, Any]]:
        """Detect faces in *clip* and return bounding boxes.

        The *clip* dict must contain ``"media_bytes"`` (raw image bytes).

        Returns a list of dicts, each with keys ``"confidence"`` and ``"bbox"``
        (``[x1, y1, x2, y2]`` in pixels).
        """
        self.load_model()
        assert self._detector is not None

        media_bytes = clip.get("media_bytes")
        if media_bytes is None:
            return []

        import numpy as np  # noqa: PLC0415

        image = Image.open(io.BytesIO(media_bytes)).convert("RGB")
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        results = self._detector.process(img_array)

        hits: list[dict[str, Any]] = []
        if not results.detections:
            return hits

        for detection in results.detections:
            conf = detection.score[0]
            if conf < self._threshold:
                continue
            bbox_rel = detection.location_data.relative_bounding_box
            x1 = bbox_rel.xmin * w
            y1 = bbox_rel.ymin * h
            x2 = (bbox_rel.xmin + bbox_rel.width) * w
            y2 = (bbox_rel.ymin + bbox_rel.height) * h
            hits.append(
                {
                    "confidence": round(float(conf), 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

        hits.sort(key=lambda h: h["confidence"], reverse=True)
        return hits

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["localizer_type"] = "face"
        d["config"] = {
            "threshold": self._threshold,
            "model_selection": self._model_selection,
        }
        return d

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "FaceLocalizer":
        """Reconstruct a ``FaceLocalizer`` from a saved config dict."""
        return cls(
            name=name,
            threshold=config.get("threshold", 0.5),
            model_selection=config.get("model_selection", 1),
        )
