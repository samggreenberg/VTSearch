"""The 1-D two-component mixture, its cuts, its anchored variant, and the text sort's lines.

A re-export facade over the four modules this one was split into (#5424),
each of which reads only those above it:

* :mod:`~vtscore.training.thresholds.gmm_core` - the fit type
  (:class:`GmmFit1D`), the density-crossing solvers and the
  ``cut_fallback_kind`` vocabulary, and the sample handling every fit shares.
* :mod:`~vtscore.training.thresholds.gmm_em` - the plain and label-anchored
  EM fits and their stopping rules.
* :mod:`~vtscore.training.thresholds.gmm_threshold` - the named cut rules
  and :func:`calculate_gmm_threshold`.
* :mod:`~vtscore.training.thresholds.text_sort` - the typed-query sort's
  display and acquisition lines.

``vtscore.training.thresholds.gmm.X`` still resolves, to the same object, for
every name those four define.  **Patch targets are not here**: a stub has to
sit in the module whose function reads the name - ``gmm_em._EM_LOGLIK_TOL``,
``text_sort.TEXT_SORT_CUT_RULE``, ``gmm_core._GMM_MAX_SAMPLES`` - and a rebind
on this facade is silently ignored.
"""

from __future__ import annotations

from vtscore.training.thresholds.gmm_core import (
    _GMM_MAX_SAMPLES,
    CUT_KIND_CONTINUED,
    CUT_KIND_DEGENERATE_MIDPOINT,
    CUT_KIND_INTERIOR,
    GmmFit1D,
    _quadratic_roots,
    _rate_cut,
    _weighted_gaussian_crossing,
    gmm_fit_array,
    scored_ordering,
    snap_cut_to_sample,
)
from vtscore.training.thresholds.gmm_em import (
    _ANCHOR_MIN_WEIGHT,
    _ANCHOR_VAR_FLOOR_FRAC,
    _ANCHORED_EM_LOGLIK_TOL,
    _ANCHORED_EM_MAX_ITER,
    _ANCHORED_EM_TOL,
    _EM_LOGLIK_TOL,
    _EM_MAX_ITER,
    _EM_TOL,
    _KMEANS_MAX_ITER,
    _NO_ANCHORS,
    ANCHOR_WEIGHT_DEFAULT,
    _anchor_loglik_terms,
    _anchored_em,
    _plain_em,
    _record_stopping,
    _two_means_init,
    anchored_gmm_fit,
    fit_anchored_score_gmm,
    fit_score_gmm,
    fit_score_gmm_sklearn,
)
from vtscore.training.thresholds.gmm_threshold import (
    _fit_gmm_threshold_with_array,
    calculate_gmm_threshold,
    fit_gmm_threshold,
    gmm_cut_from_fit,
)
from vtscore.training.thresholds.text_sort import (
    _TEXT_SORT_CONVERGED_MAX_ITER,
    _TEXT_SORT_CONVERGED_TOL,
    TEXT_SORT_COUNT_EXPONENT,
    TEXT_SORT_COUNT_MAX_BETA,
    TEXT_SORT_COUNT_MIN_SCORES,
    TEXT_SORT_COUNT_Z,
    TEXT_SORT_CUT_DEFAULT,
    TEXT_SORT_CUT_RULE,
    TEXT_SORT_CUT_RULES,
    TEXT_SORT_SEPARATION_D,
    TEXT_SORT_TAIL_K,
    TextSortCuts,
    _count_estimate,
    _count_keep_line,
    _count_line,
    _guarded_line_from_fit,
    ashman_d,
    bulk_location_scale,
    converge_score_gmm,
    guarded_text_sort_threshold,
    resolve_text_sort_cut_rule,
    text_sort_acquisition_threshold,
    text_sort_cuts,
    text_sort_threshold,
)

__all__ = [
    # gmm_core
    "_GMM_MAX_SAMPLES",
    "CUT_KIND_CONTINUED",
    "CUT_KIND_DEGENERATE_MIDPOINT",
    "CUT_KIND_INTERIOR",
    "GmmFit1D",
    "_quadratic_roots",
    "_rate_cut",
    "_weighted_gaussian_crossing",
    "gmm_fit_array",
    "scored_ordering",
    "snap_cut_to_sample",
    # gmm_em
    "_ANCHOR_MIN_WEIGHT",
    "_ANCHOR_VAR_FLOOR_FRAC",
    "_ANCHORED_EM_LOGLIK_TOL",
    "_ANCHORED_EM_MAX_ITER",
    "_ANCHORED_EM_TOL",
    "_EM_LOGLIK_TOL",
    "_EM_MAX_ITER",
    "_EM_TOL",
    "_KMEANS_MAX_ITER",
    "_NO_ANCHORS",
    "ANCHOR_WEIGHT_DEFAULT",
    "_anchor_loglik_terms",
    "_anchored_em",
    "_plain_em",
    "_record_stopping",
    "_two_means_init",
    "anchored_gmm_fit",
    "fit_anchored_score_gmm",
    "fit_score_gmm",
    "fit_score_gmm_sklearn",
    # gmm_threshold
    "_fit_gmm_threshold_with_array",
    "calculate_gmm_threshold",
    "fit_gmm_threshold",
    "gmm_cut_from_fit",
    # text_sort
    "_TEXT_SORT_CONVERGED_MAX_ITER",
    "_TEXT_SORT_CONVERGED_TOL",
    "TEXT_SORT_COUNT_EXPONENT",
    "TEXT_SORT_COUNT_MAX_BETA",
    "TEXT_SORT_COUNT_MIN_SCORES",
    "TEXT_SORT_COUNT_Z",
    "TEXT_SORT_CUT_DEFAULT",
    "TEXT_SORT_CUT_RULE",
    "TEXT_SORT_CUT_RULES",
    "TEXT_SORT_SEPARATION_D",
    "TEXT_SORT_TAIL_K",
    "TextSortCuts",
    "_count_estimate",
    "_count_keep_line",
    "_count_line",
    "_guarded_line_from_fit",
    "ashman_d",
    "bulk_location_scale",
    "converge_score_gmm",
    "guarded_text_sort_threshold",
    "resolve_text_sort_cut_rule",
    "text_sort_acquisition_threshold",
    "text_sort_cuts",
    "text_sort_threshold",
]
