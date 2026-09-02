"""Decision-threshold computation for learned-sort scores.

GMM-, cross-calibration-, and safe-threshold helpers. These are media-
agnostic: they take score lists and label lists and return a single
float threshold. Detector-specific glue (sourcing ``X_list`` / ``y_list``
from votes, caching on ``DetectorContext``) lives in
:mod:`vtscore.detectors`.

The implementation is split across five submodules, layered so that each one
only reads from those above it:

* :mod:`~vtscore.training.thresholds.knobs` - what an Inclusion value, a
  Train/Calibrate split and the sentinels *mean*.  Depends on nothing.
* :mod:`~vtscore.training.thresholds.gmm` - the 1-D two-component mixture, its
  anchored variant, and the rules that read a cut off a fit.  Self-contained.
* :mod:`~vtscore.training.thresholds.anchored` - the shipped path: per-fold
  anchored mixtures combined in quantile space, plus the voted-media haystack
  exclusion.
* :mod:`~vtscore.training.thresholds.conformal` - the rival estimator: fold
  splits and split-conformal quantiles over pooled held-out scores.
* :mod:`~vtscore.training.thresholds.blend` - the retired safe-threshold blend,
  kept as the anchored path's small-label fallback.

Everything below is re-exported here, so ``vtscore.training.thresholds.X``
resolves exactly as it did when this was one module.  **Patch targets are the
exception**, because a stub has to be installed where the *caller* looks the
name up, and the split moved some of those lookups:

* A caller outside this package that imports lazily (``from
  vtscore.training.thresholds import fit_fold_anchored_cut`` inside a function,
  as :mod:`vtscore.detectors.training` does) re-reads the attribute off this
  package on every call, so patching the package still reaches it.
* A caller *inside* a submodule resolves its own module global, which this
  package's attribute is only a copy of.  Stubbing ``fit_gmm_threshold`` for
  :func:`~vtscore.training.thresholds.blend.calculate_safe_threshold`, or
  ``compute_fold_orderings`` for
  :func:`~vtscore.training.thresholds.conformal.calculate_cross_calibration_threshold`,
  means patching ``thresholds.blend`` / ``thresholds.conformal`` - not this
  package, where the rebind is silently ignored.
"""

from __future__ import annotations

from vtscore.training.thresholds.anchored import (
    EXCLUSION_MIN_REMAINDER,
    FOLD_ANCHOR_COMBINE,
    FOLD_ANCHOR_CUT_RULE,
    FOLD_ANCHOR_QTILT_STEP,
    FOLD_ANCHOR_WEIGHT,
    FoldAnchoredCut,
    apply_vote_exclusion,
    drop_voted,
    fit_fold_anchored_cut,
    fold_anchored_gmm_threshold,
    rank_transfer,
    resolve_exclusion_floor,
)
from vtscore.training.thresholds.blend import (
    _as_context,
    blend_gmm_threshold,
    calculate_safe_threshold,
    safe_blend_weight,
)
from vtscore.training.thresholds.conformal import (
    _DITHER_SAMPLE_ROWS,
    CALIBRATION_SPLIT_SEED,
    CONFORMAL_BASE_BUDGET,
    CONFORMAL_QPOS_MAX,
    FOLD_CONFORMAL_COMBINES,
    CalibrationFolds,
    _calibration_cache_key,
    _compute_fold_orderings_grouped,
    _dithered_count,
    _group_node_blocks,
    _grouped_folds,
    _per_bag_fit_weights,
    _pooled_group_scores,
    _score_rows_digest,
    _split_dither_rng,
    calculate_cross_calibration_threshold,
    calibration_folds,
    calibration_folds_cached,
    combined_fold_conformal_threshold,
    compute_fold_orderings,
    compute_grouped_fold_node_scores,
    conformal_threshold,
    per_fold_conformal_cuts,
    threshold_from_fold_orderings,
    threshold_from_folds,
)
from vtscore.training.thresholds.gmm import (
    _ANCHOR_MIN_WEIGHT,
    _ANCHOR_VAR_FLOOR_FRAC,
    _GMM_MAX_SAMPLES,
    ANCHOR_WEIGHT_DEFAULT,
    CUT_KIND_CONTINUED,
    CUT_KIND_DEGENERATE_MIDPOINT,
    CUT_KIND_INTERIOR,
    GmmFit1D,
    _anchored_em,
    _quadratic_roots,
    _rate_cut,
    _weighted_gaussian_crossing,
    anchored_gmm_fit,
    calculate_gmm_threshold,
    fit_anchored_score_gmm,
    fit_gmm_threshold,
    fit_score_gmm,
    gmm_cut_from_fit,
    gmm_fit_array,
    scored_ordering,
    snap_cut_to_sample,
)
from vtscore.training.thresholds.knobs import (
    ACQUISITION_INCLUSION_OFFSET,
    INCLUSION_MAX,
    INCLUSION_MIN,
    NO_GOOD_THRESHOLD,
    PRODUCTION_SPLIT,
    PRODUCTION_SPLIT_BY_SPACE,
    acquisition_inclusion,
    classify_threshold_provenance,
    inclusion_cost_weights,
    production_split_for,
)

__all__ = [
    "ACQUISITION_INCLUSION_OFFSET",
    "INCLUSION_MAX",
    "INCLUSION_MIN",
    "NO_GOOD_THRESHOLD",
    "PRODUCTION_SPLIT",
    "PRODUCTION_SPLIT_BY_SPACE",
    "acquisition_inclusion",
    "classify_threshold_provenance",
    "inclusion_cost_weights",
    "production_split_for",
    "ANCHOR_WEIGHT_DEFAULT",
    "CUT_KIND_CONTINUED",
    "CUT_KIND_DEGENERATE_MIDPOINT",
    "CUT_KIND_INTERIOR",
    "GmmFit1D",
    "_ANCHOR_MIN_WEIGHT",
    "_ANCHOR_VAR_FLOOR_FRAC",
    "_GMM_MAX_SAMPLES",
    "_anchored_em",
    "_quadratic_roots",
    "_rate_cut",
    "_weighted_gaussian_crossing",
    "anchored_gmm_fit",
    "calculate_gmm_threshold",
    "fit_anchored_score_gmm",
    "fit_gmm_threshold",
    "fit_score_gmm",
    "gmm_cut_from_fit",
    "gmm_fit_array",
    "scored_ordering",
    "snap_cut_to_sample",
    "EXCLUSION_MIN_REMAINDER",
    "FOLD_ANCHOR_COMBINE",
    "FOLD_ANCHOR_CUT_RULE",
    "FOLD_ANCHOR_QTILT_STEP",
    "FOLD_ANCHOR_WEIGHT",
    "FoldAnchoredCut",
    "apply_vote_exclusion",
    "drop_voted",
    "fit_fold_anchored_cut",
    "fold_anchored_gmm_threshold",
    "rank_transfer",
    "resolve_exclusion_floor",
    "CALIBRATION_SPLIT_SEED",
    "CONFORMAL_BASE_BUDGET",
    "CONFORMAL_QPOS_MAX",
    "CalibrationFolds",
    "FOLD_CONFORMAL_COMBINES",
    "_DITHER_SAMPLE_ROWS",
    "_calibration_cache_key",
    "_compute_fold_orderings_grouped",
    "_dithered_count",
    "_group_node_blocks",
    "_grouped_folds",
    "_per_bag_fit_weights",
    "_pooled_group_scores",
    "_score_rows_digest",
    "_split_dither_rng",
    "calculate_cross_calibration_threshold",
    "calibration_folds",
    "calibration_folds_cached",
    "combined_fold_conformal_threshold",
    "compute_fold_orderings",
    "compute_grouped_fold_node_scores",
    "conformal_threshold",
    "per_fold_conformal_cuts",
    "threshold_from_fold_orderings",
    "threshold_from_folds",
    "_as_context",
    "blend_gmm_threshold",
    "calculate_safe_threshold",
    "safe_blend_weight",
]
