"""The public face API: records, the pipeline seam, the open-weights models, the file helpers.

Everything an application needs is importable from here::

    from imagekit import RuntimeOptions, read_image_tensor
    from imagekit.face import get_pipeline, read_pipeline_config

    config = read_pipeline_config("faces.json")
    pipeline = get_pipeline()                 # IMAGEKIT_FACE_PROVIDER, else "oss"
    pipeline.initialize(RuntimeOptions(), config)
    records = pipeline.process(read_image_tensor("photo.jpg"), media_id="photo.jpg")
"""

from imagekit.face.compat import COMPAT_KEY, CompatibilityError, check_compat, single_compat, stamp_compat
from imagekit.face.detection.face_detector import DetectorConfig, FaceDetector
from imagekit.face.extraction.face_embedding_extractor import ExtractorConfig, FaceEmbeddingExtractor
from imagekit.face.face_align import ARCFACE_TEMPLATE, norm_crop
from imagekit.face.face_util import (
    embedding_key,
    load_detections_csv,
    load_embeddings_npz,
    load_face_records,
    load_face_records_csv,
    save_detections_csv,
    save_embeddings_npz,
    save_face_records_csv,
)
from imagekit.face.pipeline import (
    BUILTIN_PROVIDERS,
    DEFAULT_PROVIDER,
    ENTRY_POINT_GROUP,
    PROVIDER_ENV_VAR,
    FacePipeline,
    available_providers,
    default_provider_name,
    get_pipeline,
    load_pipeline,
    read_pipeline_config,
    register_provider,
    resolve_config_paths,
    runtime_options_from_config,
    unregister_provider,
)
from imagekit.face.records import (
    EMBEDDING_DIM,
    LANDMARK_NAMES,
    FaceDetection,
    FaceRecord,
    embedding_quality,
    oss_compat,
)

__all__ = [
    "ARCFACE_TEMPLATE",
    "BUILTIN_PROVIDERS",
    "COMPAT_KEY",
    "DEFAULT_PROVIDER",
    "EMBEDDING_DIM",
    "ENTRY_POINT_GROUP",
    "LANDMARK_NAMES",
    "PROVIDER_ENV_VAR",
    "CompatibilityError",
    "DetectorConfig",
    "ExtractorConfig",
    "FaceDetection",
    "FaceDetector",
    "FaceEmbeddingExtractor",
    "FacePipeline",
    "FaceRecord",
    "available_providers",
    "check_compat",
    "default_provider_name",
    "embedding_key",
    "embedding_quality",
    "get_pipeline",
    "load_detections_csv",
    "load_embeddings_npz",
    "load_face_records",
    "load_face_records_csv",
    "load_pipeline",
    "norm_crop",
    "oss_compat",
    "read_pipeline_config",
    "register_provider",
    "resolve_config_paths",
    "runtime_options_from_config",
    "save_detections_csv",
    "save_embeddings_npz",
    "save_face_records_csv",
    "single_compat",
    "stamp_compat",
    "unregister_provider",
]
