"""Face embedding: :class:`FaceEmbeddingExtractor` over MagFace / ArcFace ONNX weights."""

from imagekit.face.extraction.face_embedding_extractor import (
    ARCFACE_CONFIG_PATH,
    ARCFACE_MBF_CONFIG_PATH,
    MAGFACE_CONFIG_PATH,
    ExtractorConfig,
    FaceEmbeddingExtractor,
)

__all__ = [
    "ARCFACE_CONFIG_PATH",
    "ARCFACE_MBF_CONFIG_PATH",
    "MAGFACE_CONFIG_PATH",
    "ExtractorConfig",
    "FaceEmbeddingExtractor",
]
