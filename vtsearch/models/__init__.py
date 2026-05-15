"""Model loading, embeddings, and training utilities.

The detector lifecycle (registry, store, training-glue, label-sync, the
resolve→embed pipeline, and the labeling-session analyzer) lives in
:mod:`vtsearch.detectors`. This package now owns only embedding helpers,
torch model loaders, neural-net training, and the diversity tree —
step 2 of the codebase reorg will split those out too.
"""

from vtsearch.models.diversity_tree import DiversityTree
from vtsearch.models.embeddings import (
    embed_audio_file,
    embed_image_file,
    embed_paragraph_file,
    embed_text_query,
    embed_video_file,
)
from vtsearch.models.loader import (
    get_clap_model,
    get_e5_model,
    get_xclip_model,
    initialize_models,
    preload_autoload_media_types,
)
from vtsearch.models.training import (
    build_model,
    build_model_from_weights,
    calculate_cross_calibration_threshold,
    calculate_gmm_threshold,
    calculate_safe_threshold,
    collect_media_origins,
    find_optimal_threshold,
    train_and_score,
    train_detector_from_origins,
    train_model,
)

__all__ = [
    # Diversity Tree
    "DiversityTree",
    # Embeddings
    "embed_audio_file",
    "embed_video_file",
    "embed_image_file",
    "embed_paragraph_file",
    "embed_text_query",
    # Loader
    "initialize_models",
    "preload_autoload_media_types",
    "get_clap_model",
    "get_xclip_model",
    "get_e5_model",
    # Training
    "build_model",
    "build_model_from_weights",
    "collect_media_origins",
    "train_model",
    "train_and_score",
    "train_detector_from_origins",
    "calculate_gmm_threshold",
    "calculate_safe_threshold",
    "find_optimal_threshold",
    "calculate_cross_calibration_threshold",
]
