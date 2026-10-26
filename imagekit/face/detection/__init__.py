"""Face detection: :class:`FaceDetector` over SCRFD or RetinaFace ONNX weights."""

from imagekit.face.detection.face_detector import (
    DEFAULT_CONFIG_PATH,
    RETINAFACE_CONFIG_PATH,
    DetectorConfig,
    FaceDetector,
)

__all__ = ["DEFAULT_CONFIG_PATH", "RETINAFACE_CONFIG_PATH", "DetectorConfig", "FaceDetector"]
