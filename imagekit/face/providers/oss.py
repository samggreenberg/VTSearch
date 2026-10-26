"""``OssFacePipeline``: this package's own detector and embedder behind the provider seam."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from imagekit.face.detection.face_detector import FaceDetector
from imagekit.face.extraction.face_embedding_extractor import FaceEmbeddingExtractor
from imagekit.face.pipeline import DETECTOR_KEY, EMBEDDER_KEY
from imagekit.face.records import FaceRecord
from imagekit.runtime import RuntimeOptions
from imagekit.util import decode_image_buffer, ensure_image_tensor

NOT_INITIALIZED = "Must initialize OssFacePipeline first!"


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = config.get(key) or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"pipeline config: {key!r} must be a JSON object")
    return section


class OssFacePipeline:
    """Detect with :class:`FaceDetector`, embed every detection with :class:`FaceEmbeddingExtractor`.

    Reads ``config["detector"]["model_bundle"]`` (plus optional ``config`` and
    ``confidence_threshold``) and ``config["embedder"]["model"]`` (plus
    optional ``config``); see :func:`imagekit.face.pipeline.resolve_config_paths`.
    """

    def __init__(self) -> None:
        self.detector = FaceDetector()
        self.extractor = FaceEmbeddingExtractor()
        self._threshold: float | None = None
        self._initialized = False

    def initialize(self, opts: RuntimeOptions, config: Mapping[str, Any]) -> None:
        detector = _section(config, DETECTOR_KEY)
        embedder = _section(config, EMBEDDER_KEY)
        bundle = detector.get("model_bundle") or detector.get("model")
        if not bundle:
            raise ValueError("oss provider: config['detector']['model_bundle'] (the detector .onnx) is required")
        model = embedder.get("model")
        if not model:
            raise ValueError("oss provider: config['embedder']['model'] (the embedder .onnx) is required")
        self.detector.initialize(opts, bundle, detector.get("config"))
        self.extractor.initialize(opts, model, embedder.get("config"))
        threshold = detector.get("confidence_threshold")
        self._threshold = None if threshold is None else float(threshold)
        self._initialized = True

    def compatibility(self) -> str:
        if not self._initialized:
            raise RuntimeError(NOT_INITIALIZED)
        return self.extractor.get_compatibility()

    def process(self, image: Any, *, confidence_threshold: float | None = None, media_id: str = "") -> list[FaceRecord]:
        if not self._initialized:
            raise RuntimeError(NOT_INITIALIZED)
        tensor = ensure_image_tensor(image)
        threshold = self._threshold if confidence_threshold is None else confidence_threshold
        detections = self.detector.detect(tensor, threshold)
        vectors = self.extractor.extract_many(detections, tensor)
        compat = self.extractor.get_compatibility()
        return [
            FaceRecord.from_detection(det, vec, compat, media_id, index)
            for index, (det, vec) in enumerate(zip(detections, vectors))
        ]

    def process_buffer(
        self, buffer: bytes, *, confidence_threshold: float | None = None, media_id: str = ""
    ) -> list[FaceRecord]:
        if not self._initialized:
            raise RuntimeError(NOT_INITIALIZED)
        return self.process(decode_image_buffer(buffer), confidence_threshold=confidence_threshold, media_id=media_id)
