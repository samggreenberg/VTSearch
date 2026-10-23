"""Domain shift: how much of a dataset looks unlike a reference dataset.

A detector trained on dataset A and pointed at dataset B should not be trusted
without hands-on verification if B is unlike A.  This module answers "how much
of B is unlike A?" with a **split-conformal kNN** score over A's own vectors:

- A's vectors are split in half by a fixed seed.  One half is the *fit* set,
  the other the *calibration* set.
- An item's score is the mean cosine to its ``k`` nearest fit vectors.  Each
  calibration vector is scored the same way, and an item's p-value is its rank
  among those scores, ``(1 + #{cal <= s}) / (n_cal + 1)``.
- Under the null "the item is drawn from A", the calibration vectors and the
  item are exchangeable, so the p-value is uniform **by construction** -- about
  *alpha* of A's own items fall below *alpha*, on every embedder, patch spaces
  included.  An excess well above *alpha* is the shift signal.

This replaced the Coverage Atlas's typicality guard (#3348).  That guard ranked
a query against leave-one-out scores of the points its k-means cells were
drawn around.  Those points sit more centrally in their cells than any fresh
item, so the guard read A's own held-out items as atypical: KS 0.11 from
uniform on every embedder, and on ``dinov3_patch`` false alarms on 60 % of
A's own held-out splits, with separation 0.36.  The rule here reads KS
0.021-0.033 and 0 false alarms on all five embedders measured, and separation
1.00 on ``dinov3_patch``: ``docs/experiments/2026-10-23-atlas-node-model-3348/REPORT.md``.

Pure geometry over unit-normalized vectors: nothing is persisted and nothing
needs a GPU.
"""

from __future__ import annotations

import math

import numpy as np

#: Neighbour count for the kNN score (the app's conformal default, and the
#: value measured in #3348).
DOMAIN_SHIFT_K = 5

#: Default significance level for the per-item p-values.
DOMAIN_SHIFT_ALPHA = 0.05

#: Caps that keep the check cheap enough to run automatically (the SPA asks
#: for it the moment a detector meets a dataset).  The cost is
#: ``queries x fit x dim``.  Subsampling either side at random keeps the
#: p-values valid: the fit and calibration halves stay disjoint random draws
#: from the reference, and a random subset of the queries estimates the same
#: atypical fraction.  4 000 queries put the fraction's standard error under
#: 0.004.  #3348 measured references up to 9 800 items per half.
DOMAIN_SHIFT_MAX_FIT = 20_000
DOMAIN_SHIFT_MAX_CAL = 10_000
DOMAIN_SHIFT_MAX_QUERY = 4_000

#: Fewest calibration items for a p-value to reach below *alpha* at all: the
#: smallest attainable p is ``1 / (n_cal + 1)``.
DOMAIN_SHIFT_MIN_CAL = 20

#: Seed for the reference split and the subsamples, so a report is
#: reproducible for a given pair of datasets.
_SEED = 0

_EPS = 1e-12
_CHUNK = 2048


class ReferenceTooSmall(ValueError):
    """The reference has too few vectors to calibrate a p-value below alpha."""


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2:
        raise ValueError(f"expected a (n, d) matrix of vectors, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.where(norms > _EPS, matrix / np.maximum(norms, _EPS), 0.0).astype(np.float32)


def _topk_mean_cosine(queries: np.ndarray, refs: np.ndarray, k: int) -> np.ndarray:
    """Mean of the *k* largest cosines from each query row to the rows of *refs*."""
    k = max(1, min(k, refs.shape[0]))
    out = np.empty(queries.shape[0], dtype=np.float64)
    for start in range(0, queries.shape[0], _CHUNK):
        sims = queries[start : start + _CHUNK] @ refs.T
        out[start : start + _CHUNK] = (-np.partition(-sims, k - 1, axis=1)[:, :k]).mean(axis=1)
    return out


def _split_reference(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit and calibration halves of *reference*, by a fixed seed, each capped."""
    perm = np.random.default_rng(_SEED).permutation(reference.shape[0])
    half = reference.shape[0] // 2
    fit = perm[:half][:DOMAIN_SHIFT_MAX_FIT]
    cal = perm[half:][:DOMAIN_SHIFT_MAX_CAL]
    return reference[fit], reference[cal]


def typicality_pvalues(reference: np.ndarray, matrix: np.ndarray, k: int = DOMAIN_SHIFT_K) -> np.ndarray:
    """One split-conformal typicality p-value per row of *matrix*, against *reference*.

    Small means "almost nothing in the reference's calibration half sits this
    far from its fit half".  For a row drawn from the reference's own
    distribution the p-value is uniform, so ``P(p < alpha) <= alpha``.

    Raises :class:`ReferenceTooSmall` when the calibration half would hold
    fewer than ``DOMAIN_SHIFT_MIN_CAL`` vectors (a reference under twice that).
    Scores every row of *matrix*;
    :func:`domain_shift_report` is the one that subsamples.
    """
    ref = _normalize(reference)
    queries = _normalize(matrix)
    if ref.shape[0] < 2 * DOMAIN_SHIFT_MIN_CAL:
        raise ReferenceTooSmall(
            f"reference has {ref.shape[0]} vectors; at least {2 * DOMAIN_SHIFT_MIN_CAL} are needed to calibrate"
        )
    if queries.shape[0] and queries.shape[1] != ref.shape[1]:
        raise ValueError(f"vector width mismatch: reference {ref.shape[1]}, matrix {queries.shape[1]}")
    fit, cal = _split_reference(ref)
    cal_scores = np.sort(_topk_mean_cosine(cal, fit, k))
    if queries.shape[0] == 0:
        return np.ones(0, dtype=np.float64)
    scores = _topk_mean_cosine(queries, fit, k)
    return (1.0 + np.searchsorted(cal_scores, scores, side="right")) / (cal_scores.size + 1.0)


def domain_shift_report(reference: np.ndarray, matrix: np.ndarray, alpha: float = DOMAIN_SHIFT_ALPHA) -> dict:
    """Compare a dataset's vectors (*matrix*) against a reference dataset's and report shift.

    A large excess of atypical items means a detector trained on the
    reference is being pointed at data unlike it, and shouldn't be trusted
    without hands-on verification.  At most ``DOMAIN_SHIFT_MAX_QUERY`` rows of
    *matrix*, drawn at random by a fixed seed, are scored; ``n_items`` is the
    number scored.

    Returns a dict with ``n_items``, ``alpha``, ``frac_atypical`` (observed
    fraction with p-value below alpha), ``expected_atypical`` (= alpha, which
    the p-values' uniformity under no shift makes the measured rate, not just
    the nominal one), ``z_score`` (binomial z of the excess),
    ``median_pvalue``, and ``shifted`` (True when the excess is both
    statistically clear, z > 3, and practically large, at least double
    alpha).  Raises :class:`ReferenceTooSmall` as
    :func:`typicality_pvalues` does.
    """
    queries = _normalize(matrix)
    if queries.shape[0] > DOMAIN_SHIFT_MAX_QUERY:
        pick = np.random.default_rng(_SEED + 1).choice(queries.shape[0], DOMAIN_SHIFT_MAX_QUERY, replace=False)
        queries = queries[np.sort(pick)]
    pvals = typicality_pvalues(reference, queries)
    n = int(pvals.shape[0])
    if n == 0:
        return {
            "n_items": 0,
            "alpha": alpha,
            "frac_atypical": 0.0,
            "expected_atypical": alpha,
            "z_score": 0.0,
            "median_pvalue": 1.0,
            "shifted": False,
        }
    frac = float(np.mean(pvals < alpha))
    se = math.sqrt(alpha * (1.0 - alpha) / n)
    z = (frac - alpha) / se if se > 0 else 0.0
    return {
        "n_items": n,
        "alpha": alpha,
        "frac_atypical": frac,
        "expected_atypical": alpha,
        "z_score": z,
        "median_pvalue": float(np.median(pvals)),
        "shifted": bool(z > 3.0 and frac >= 2.0 * alpha),
    }
