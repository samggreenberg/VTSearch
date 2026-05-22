"""Generic learned-sort training primitives (MLP/SVM) and threshold helpers.

Detector-specific glue (vote-aware training, origin-based detector
training, the train→threshold→serialise pipeline) lives in
:mod:`vtscore.detectors`. The neural-net primitives here are media-
agnostic: they take feature matrices and labels in, and return models
and decision thresholds out.
"""

from vtscore.training.mlp import (
    build_model,
    build_model_from_weights,
    train_model,
)
from vtscore.training.thresholds import (
    calculate_cross_calibration_threshold,
    calculate_gmm_threshold,
    calculate_safe_threshold,
    cross_calibration_threshold_cached,
    find_optimal_threshold,
)

__all__ = [
    "build_model",
    "build_model_from_weights",
    "train_model",
    "calculate_gmm_threshold",
    "find_optimal_threshold",
    "calculate_cross_calibration_threshold",
    "cross_calibration_threshold_cached",
    "calculate_safe_threshold",
]
