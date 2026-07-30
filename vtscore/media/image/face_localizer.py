"""Face localizer using facenet-pytorch's MTCNN detector."""

from __future__ import annotations

from typing import Any, Optional

from vtscore.media.image.decode import decode_bounded_rgb
from vtscore.media.processors import Localizer


class FaceLocalizer(Localizer):
    """Localizes faces in images using facenet-pytorch's MTCNN detector.

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
    MTCNN ships inside ``facenet-pytorch`` with bundled weights, so it needs no
    separate model download and shares the install used by the FaceNet embedder.
    """

    def __init__(self, name: str, threshold: float = 0.5) -> None:
        """Create a face localizer.

        Args:
            name: Unique name for this localizer instance.
            threshold: Minimum confidence to count a detection (0-1).
        """
        self._name = name
        self._threshold = threshold
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if self._detector is not None:
            return
        from facenet_pytorch import MTCNN  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        from vtscore.config import resolve_device  # noqa: PLC0415

        self._detector = MTCNN(
            keep_all=True,
            select_largest=False,
            post_process=False,
            device=resolve_device(),
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

        # Bounded decode caps the bitmap MTCNN is handed; ``scale`` maps the
        # detected boxes back into the original image's pixel space.
        image, scale = decode_bounded_rgb(media_bytes)

        det_boxes, det_probs = self._detector.detect(image)

        hits: list[dict[str, Any]] = []
        if det_boxes is None:
            return hits

        for box, prob in zip(det_boxes, det_probs):
            if prob is None:
                continue
            conf = float(prob)
            if conf < self._threshold:
                continue
            x1, y1, x2, y2 = (float(v) / scale for v in box)
            hits.append(
                {
                    "confidence": round(conf, 4),
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
        }
        return d

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "FaceLocalizer":
        """Reconstruct a ``FaceLocalizer`` from a saved config dict."""
        return cls(
            name=name,
            threshold=config.get("threshold", 0.5),
        )
