"""OCR extractor using PaddleOCR for text extraction from images."""

from __future__ import annotations

import io
from typing import Any, Optional

from PIL import Image

from vtsearch.media.processors import Extractor


class OCRExtractor(Extractor):
    """Extracts text regions from images using PaddleOCR.

    Each instance is configured with a language and a confidence threshold.
    Running :meth:`extract` on an image clip returns a list of dicts, one per
    detected text region whose confidence meets the threshold::

        [
            {
                "confidence": 0.95,
                "bbox": [x1, y1, x2, y2],
                "label": "detected text here",
            },
            ...
        ]

    Coordinates are in pixel space (float) as a bounding rectangle derived
    from the PaddleOCR polygon output.
    """

    def __init__(self, name: str, language: str = "en", threshold: float = 0.5) -> None:
        self._name = name
        self._language = language
        self._threshold = threshold
        self._model: Optional[Any] = None

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
    def language(self) -> str:
        return self._language

    @property
    def threshold(self) -> float:
        return self._threshold

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if self._model is not None:
            return
        from paddleocr import PaddleOCR  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        self._model = PaddleOCR(use_angle_cls=True, lang=self._language, show_log=False)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, clip: dict[str, Any]) -> list[dict[str, Any]]:
        """Detect text regions in *clip* and return bounding boxes with text.

        The *clip* dict must contain ``"media_bytes"`` (raw image bytes).

        Returns a list of dicts, each with keys ``"confidence"``, ``"bbox"``
        (``[x1, y1, x2, y2]`` in pixels), and ``"label"`` (the detected text).
        """
        self.load_model()
        assert self._model is not None

        media_bytes = clip.get("media_bytes")
        if media_bytes is None:
            return []

        import numpy as np  # noqa: PLC0415

        image = Image.open(io.BytesIO(media_bytes)).convert("RGB")
        img_array = np.array(image)

        results = self._model.ocr(img_array, cls=True)

        hits: list[dict[str, Any]] = []
        if not results:
            return hits

        for line_group in results:
            if not line_group:
                continue
            for line in line_group:
                polygon, (text, conf) = line
                if conf < self._threshold:
                    continue
                # Convert polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] to bbox [xmin,ymin,xmax,ymax]
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                hits.append(
                    {
                        "confidence": round(float(conf), 4),
                        "bbox": [round(c, 2) for c in bbox],
                        "label": text,
                    }
                )

        hits.sort(key=lambda h: h["confidence"], reverse=True)
        return hits

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["extractor_type"] = "ocr"
        d["config"] = {
            "language": self._language,
            "threshold": self._threshold,
        }
        return d

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "OCRExtractor":
        """Reconstruct an ``OCRExtractor`` from a saved config dict."""
        return cls(
            name=name,
            language=config.get("language", "en"),
            threshold=config.get("threshold", 0.5),
        )
