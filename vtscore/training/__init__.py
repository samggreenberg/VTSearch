"""Generic learned-sort training primitives (MLP/SVM) and threshold helpers.

Detector-specific glue (vote-aware training, origin-based detector
training, the train→threshold→serialise pipeline) lives in
:mod:`vtscore.detectors`. The neural-net primitives in :mod:`~vtscore.training.mlp`
and :mod:`~vtscore.training.svm` are media-agnostic: they take feature
matrices and labels in, and return models and decision thresholds out.

Two modules here sit one level up from that, scoring a *media snapshot*
rather than a bare feature matrix: :mod:`~vtscore.training.region_similarity`
(and :mod:`~vtscore.training.structural_similarity`) rank a snapshot against a
query vector, and :mod:`~vtscore.training.query_sort` composes them into the
whole-dataset sorts driven by an external query — an example media file or a
label file. Neither is re-exported here; import them by module.
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
    calibration_folds,
    calibration_folds_cached,
    conformal_threshold,
    fold_anchored_gmm_threshold,
    threshold_from_folds,
)

__all__ = [
    "build_model",
    "build_model_from_weights",
    "train_model",
    "calculate_gmm_threshold",
    "conformal_threshold",
    "calculate_cross_calibration_threshold",
    "calibration_folds",
    "calibration_folds_cached",
    "threshold_from_folds",
    "fold_anchored_gmm_threshold",
    "calculate_safe_threshold",
]
