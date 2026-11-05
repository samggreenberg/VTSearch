"""The training-domain summary: a domain-shift reference that travels as origins and scalars (#5567).

:func:`~vtscore.coverage.domain_shift.domain_shift_report` compares a dataset
against a reference dataset's own vectors, so the reference has to be on hand.
A detector handed to another user travels without the dataset it was trained
on, and the no-persisted-vectors rule keeps that dataset's vectors from
travelling with it.  This module summarises a reference in a form that can:

- **Anchors.**  The reference's vectors are split in half by the fixed seed the
  report splits by.  The *fit* half is clustered (seeded spherical k-means,
  :data:`SUMMARY_ANCHORS` clusters), and each cluster's anchor is the *member*
  nearest its centroid: an item of the reference, never an average of items,
  so a caller can name it by its origin and re-embed it wherever the summary is
  read.
- **Calibration scores.**  An item's score is its cosine to the nearest anchor.
  Items of the *calibration* half are scored the same way, and their sorted
  scores are kept (at most :data:`SUMMARY_MAX_CAL`).  They are scalars, not
  vectors, but they live in the embedder's geometry, so whatever records them
  must name the embedder and refuse to read them in another.

Reading it back is the report's own split-conformal rule with the fit half
replaced by the anchors: an item's p-value is its score's rank among the
calibration scores (:func:`summary_pvalues`).  The anchors are a function of
the fit half alone, so a calibration item and an item drawn from the reference
are still exchangeable, and the p-value is uniform under "drawn from the
reference" by construction, as the full report's is.  The summary gives up
power, not validity: :data:`SUMMARY_ANCHORS` anchors stand in for up to 20 000
fit vectors.

**Why no per-anchor radius.**  #5567 proposed recording each anchor with the
radius its cluster lived within.  A ball is not a calibrated test: radii set
by a cluster's farthest member cover everything, and radii set at a quantile
flag a share of the reference's own items that depends on the data, measured
on the points the balls were drawn around (the bias #3348 removed from the
coverage atlas's guard).  The calibration scores make the test exact instead,
and a radius as the score's normaliser (distance over radius) measured no
better than the raw cosine on synthetic data, so it is not kept.

Pure geometry over unit-normalized vectors: nothing is persisted here and
nothing needs a GPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vtscore.coverage.domain_shift import (
    _CHUNK,
    _EPS,
    _SEED,
    DOMAIN_SHIFT_ALPHA,
    DOMAIN_SHIFT_MAX_FIT,
    DOMAIN_SHIFT_MIN_CAL,
    ReferenceTooSmall,
    _normalize,
    _report_from_pvalues,
    _sample_queries,
    _topk_mean_cosine,
)

#: Anchors a summary keeps.  Each is an origin the reader resolves and embeds,
#: so this is the summary's cost on the receiving side as well as its power.
#: On five real embedders (#5568), of the full report's excess on a 30 % shift,
#: 64 anchors kept 85 %, 128 kept 94 % (87 % on the worst embedder), and 256
#: 96 %: past 128 the curve is flat and the cost is not.
#: ``docs/experiments/2026-11-05-summary-power-5568/REPORT.md``.
SUMMARY_ANCHORS = 128

#: Calibration scores a summary keeps.  The smallest attainable p-value is
#: ``1 / (n + 1)``, and 1 000 scores put the atypical fraction's calibration
#: error well under the report's own sampling error.
SUMMARY_MAX_CAL = 1_000

#: Spherical k-means iterations when choosing anchors.  The anchors need only
#: be spread over the reference, not converged: any function of the fit half
#: keeps the p-values valid.
_KMEANS_ITERS = 10

#: Decimals the calibration scores are kept to, and an item's score is rounded
#: to before it is ranked, so a summary read back from JSON ranks exactly as
#: the one drawn.
SCORE_DECIMALS = 6


def _nearest_anchor(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the most cosine-similar row of *centers* for each row of *points*."""
    out = np.empty(points.shape[0], dtype=np.int64)
    for start in range(0, points.shape[0], _CHUNK):
        out[start : start + _CHUNK] = np.argmax(points[start : start + _CHUNK] @ centers.T, axis=1)
    return out


def _choose_anchors(fit: np.ndarray, n_anchors: int, rng: np.random.Generator) -> np.ndarray:
    """Rows of *fit* (unit vectors) to anchor on: each cluster's member nearest its centroid, sorted.

    Every row is an anchor when *fit* has no more rows than *n_anchors*.  A
    cluster that ends up empty contributes none, so there can be fewer.
    """
    n = fit.shape[0]
    if n <= n_anchors:
        return np.arange(n)
    centers = fit[rng.choice(n, n_anchors, replace=False)].copy()
    for _ in range(_KMEANS_ITERS):
        assign = _nearest_anchor(fit, centers)
        onehot = np.zeros((n, n_anchors), dtype=np.float32)
        onehot[np.arange(n), assign] = 1.0
        sums = onehot.T @ fit
        norms = np.linalg.norm(sums, axis=1)
        kept = norms > _EPS
        # An emptied cluster keeps its last centre rather than collapsing to 0.
        centers[kept] = sums[kept] / norms[kept, None]
    assign = _nearest_anchor(fit, centers)
    anchors = []
    for j in range(n_anchors):
        members = np.flatnonzero(assign == j)
        if members.size:
            anchors.append(int(members[np.argmax(fit[members] @ centers[j])]))
    return np.unique(np.asarray(anchors, dtype=np.int64))


def summarize_reference(reference: np.ndarray, *, n_anchors: int = SUMMARY_ANCHORS) -> tuple[np.ndarray, np.ndarray]:
    """Summarise *reference* as ``(anchor rows, calibration scores)`` for :func:`summary_pvalues`.

    *anchor rows* are indices into *reference*, ascending: the items whose
    vectors the reader needs, which a caller records by origin.
    *calibration scores* are the sorted nearest-anchor cosines of the
    calibration half (at most :data:`SUMMARY_MAX_CAL`), rounded to
    :data:`SCORE_DECIMALS`.  Deterministic for a given *reference*: the split,
    the subsample and the clustering are all seeded.

    Raises :class:`~vtscore.coverage.domain_shift.ReferenceTooSmall` when the
    calibration half would hold fewer than ``DOMAIN_SHIFT_MIN_CAL`` vectors, as
    the full report does, and :class:`ValueError` for *n_anchors* below 1.
    """
    if n_anchors < 1:
        raise ValueError(f"n_anchors must be at least 1, got {n_anchors}")
    ref = _normalize(reference)
    if ref.shape[0] < 2 * DOMAIN_SHIFT_MIN_CAL:
        raise ReferenceTooSmall(
            f"reference has {ref.shape[0]} vectors; at least {2 * DOMAIN_SHIFT_MIN_CAL} are needed to calibrate"
        )
    # The report's split (``_split_reference``): the permutation is random, so
    # its leading rows are a random draw from either half.
    perm = np.random.default_rng(_SEED).permutation(ref.shape[0])
    half = ref.shape[0] // 2
    fit_rows = perm[:half][:DOMAIN_SHIFT_MAX_FIT]
    cal_rows = perm[half:][:SUMMARY_MAX_CAL]
    chosen = _choose_anchors(ref[fit_rows], n_anchors, np.random.default_rng(_SEED + 2))
    anchor_rows = np.sort(fit_rows[chosen])
    calibration = np.round(np.sort(_topk_mean_cosine(ref[cal_rows], ref[anchor_rows], 1)), SCORE_DECIMALS)
    return anchor_rows, calibration


def summary_pvalues(anchors: np.ndarray, calibration: Sequence[float] | np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """One typicality p-value per row of *matrix*, against a summary's *anchors* and *calibration* scores.

    *anchors* are the anchor items' vectors (re-embedded by the reader, in the
    embedder the summary was drawn in), *calibration* the scores
    :func:`summarize_reference` returned.  Small means "almost none of the
    reference's calibration items sat this far from every anchor".  For a row
    drawn from the reference the p-value is uniform.  Scores every row;
    :func:`summary_shift_report` is the one that subsamples.

    Raises :class:`~vtscore.coverage.domain_shift.ReferenceTooSmall` for fewer
    than ``DOMAIN_SHIFT_MIN_CAL`` calibration scores, and :class:`ValueError`
    for no anchors or a width mismatch.
    """
    anchor_mat = _normalize(anchors)
    if anchor_mat.shape[0] == 0:
        raise ValueError("the summary has no anchors")
    cal = np.sort(np.asarray(calibration, dtype=np.float64).ravel())
    if cal.size < DOMAIN_SHIFT_MIN_CAL:
        raise ReferenceTooSmall(
            f"summary has {cal.size} calibration scores; at least {DOMAIN_SHIFT_MIN_CAL} are needed to calibrate"
        )
    queries = _normalize(matrix)
    if queries.shape[0] == 0:
        return np.ones(0, dtype=np.float64)
    if queries.shape[1] != anchor_mat.shape[1]:
        raise ValueError(f"vector width mismatch: anchors {anchor_mat.shape[1]}, matrix {queries.shape[1]}")
    scores = np.round(_topk_mean_cosine(queries, anchor_mat, 1), SCORE_DECIMALS)
    return (1.0 + np.searchsorted(cal, scores, side="right")) / (cal.size + 1.0)


def summary_shift_report(
    anchors: np.ndarray,
    calibration: Sequence[float] | np.ndarray,
    matrix: np.ndarray,
    alpha: float = DOMAIN_SHIFT_ALPHA,
) -> dict:
    """:func:`~vtscore.coverage.domain_shift.domain_shift_report`, against a summary instead of the reference's vectors.

    The same dict, drawn by the same rule from the same sample of *matrix*'s
    rows, so a reader cannot tell the two apart by shape: only the reference
    differs.  Raises as :func:`summary_pvalues` does.
    """
    return _report_from_pvalues(summary_pvalues(anchors, calibration, _sample_queries(matrix)), alpha)
