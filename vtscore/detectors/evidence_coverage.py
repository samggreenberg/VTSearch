"""Labelset-kNN evidence coverage: cross-user decision support without an atlas.

The Coverage Atlas (``vtscore/state/coverage_atlas.py``) answers "how typical is
this item of the data the detector was trained on?" — but only against a
*loaded reference dataset* whose atlas has been built.  In the cross-user
handoff this feature exists for (userB has userA's detector JSON, not userA's
haystack), that reference is absent and the atlas/domain-shift endpoint can't
fire.

This module answers the *decision-support* half of the question
(``docs/plans/coverage-atlas.md`` §3, §6.1, phase v0) using only what already
travels inside the detector: its labelset.  The detector's labeled origins are
re-embedded against the active dataset's embedder at load time
(``DetectorContext.label_embeddings``), so the labeled evidence vectors are in
memory, cross-user, **by construction** — no persisted vectors, no atlas, no
rule amendment.

For each active-dataset item ``x`` with predicted class ``ŷ``, two signals:

- **D(x)** — a conformal *support p-value*: how far ``x`` sits from the labeled
  evidence of its predicted class, calibrated against the leave-one-out
  same-class distances *within* the labelset.  Small ``D`` means "the detector
  is calling ``ŷ`` here, but userA labeled nothing like ``x`` as ``ŷ``" — an
  evidence vacuum whose call rests on interpolation, not supervision.  Under
  the null "``x`` is drawn from the class it was assigned", ``D`` is
  approximately uniform, so an excess of small values (well above ``alpha``) is
  the domain-shift signal — the same shape as
  :func:`~vtscore.coverage.atlas.domain_shift_report`, but keyed on
  *evidence* coverage rather than *data* coverage.
- **TS(x)** — the trust-score ratio of Jiang et al. [11]: distance to the
  nearest evidence of the *other* class over distance to the predicted class.
  ``TS < 1`` means ``x`` is closer to what userA labeled as the opposite class
  than to its own predicted class — a suspect call even inside a dense region.

Both are pure geometry over unit-normalized embeddings; nothing here is
persisted and nothing needs a GPU.  §3 of the plan argues decision support is
the *primary* signal and typicality secondary, so this v0 buys the more
important half of the quadrant picture with zero new infrastructure.
"""

from __future__ import annotations

import math

import numpy as np

_EPS = 1e-12

#: Default neighbour rank for the kNN distances.  ``k = 1`` (nearest labeled
#: evidence) is the sharpest choice at the small labelset sizes detectors carry
#: (dozens of votes); larger ``k`` needs more same-class evidence than a typical
#: detector has to stay meaningful.
DEFAULT_K = 1

#: Default significance level for the support p-value.  Matches
#: :func:`~vtscore.coverage.atlas.domain_shift_report` so the two reports
#: read on the same scale.
DEFAULT_ALPHA = 0.05


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Return *matrix* as ``(n, d)`` float32 with each row scaled to unit norm.

    A 1-D input is treated as a single row.  Zero rows stay zero (their cosine
    against anything is 0, i.e. maximal distance).  ``label_embeddings`` are
    already normalized at ingest, but re-normalizing here keeps the module
    correct when handed raw vectors and costs one pass.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[0] == 0:
        # An empty ``(0, d)`` set — no rows to normalize; keep its width so
        # downstream matmuls still see the right dimensionality.
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.where(norms > _EPS, matrix / np.maximum(norms, _EPS), 0.0).astype(np.float32)


def _kth_nn_distance(queries: np.ndarray, refs: np.ndarray, k: int) -> np.ndarray:
    """Cosine distance (``1 - cos``) from each query row to its k-th nearest ref.

    *queries* and *refs* must be unit-normalized ``(n, d)`` / ``(m, d)``.
    Returns an ``(n,)`` array.  ``k`` is clamped to ``m``; when ``m == 0`` every
    query is maximally far (``2.0``, the largest possible cosine distance), so a
    class with no labeled evidence reads as "nothing supports this here".
    """
    n = queries.shape[0]
    m = refs.shape[0]
    if m == 0 or n == 0:
        return np.full(n, 2.0, dtype=np.float32)
    kk = min(k, m)
    dists = 1.0 - (queries @ refs.T)  # (n, m) cosine distances
    if kk >= m:
        return dists.max(axis=1).astype(np.float32)
    return np.partition(dists, kk - 1, axis=1)[:, kk - 1].astype(np.float32)


def _loo_kth_nn_distances(refs: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out k-th nearest-neighbour distance of each ref within *refs*.

    The calibration reference distribution for :func:`support_pvalues`: how far
    apart same-class evidence sits from itself.  Returns an ``(m,)`` array;
    ``k`` is clamped to ``m - 1`` (a point can't be its own neighbour).  A
    single-point class has no other same-class evidence, so its LOO distance is
    ``0`` by convention (every query then reads as at-least-as-far, i.e. maximal
    support p-value — the honest read when one label can't define a spread).
    """
    m = refs.shape[0]
    if m <= 1:
        return np.zeros(m, dtype=np.float32)
    kk = min(k, m - 1)
    dists = 1.0 - (refs @ refs.T)
    np.fill_diagonal(dists, np.inf)  # exclude self
    return np.partition(dists, kk - 1, axis=1)[:, kk - 1].astype(np.float32)


def support_pvalues(queries: np.ndarray, refs: np.ndarray, k: int = DEFAULT_K) -> np.ndarray:
    """Conformal support p-value per query against a class's labeled evidence.

    ``p(x) = (1 + #{i : a_i >= a(x)}) / (m + 1)``, where ``a(x)`` is ``x``'s
    k-th NN distance to *refs* and ``a_i`` is ref ``i``'s leave-one-out k-th NN
    distance within *refs*.  This is the standard inductive-conformal novelty
    p-value [12]: small ``p`` means ``x`` is farther from the class than almost
    all of the class's own members — an evidence vacuum for that class.

    Returns ``0.0`` for every query when *refs* is empty (a class with no
    labeled evidence supports nothing).  Inputs must be unit-normalized.
    """
    n = queries.shape[0]
    m = refs.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if m == 0:
        return np.zeros(n, dtype=np.float32)
    a_query = _kth_nn_distance(queries, refs, k)
    a_cal = np.sort(_loo_kth_nn_distances(refs, k))
    # #{a_i >= a(x)} = m - #{a_i < a(x)}; the latter is searchsorted(left).
    ge = m - np.searchsorted(a_cal, a_query, side="left")
    return ((1 + ge) / (m + 1)).astype(np.float32)


def predicted_support_pvalues(
    queries: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    predicted_positive: np.ndarray,
    k: int = DEFAULT_K,
) -> np.ndarray:
    """Per-item support p-value ``D`` for each item's *predicted* class.

    Routes each query to the positive- or negative-evidence p-value by its
    ``predicted_positive`` flag.  Inputs must be unit-normalized.
    """
    predicted_positive = np.asarray(predicted_positive, dtype=bool)
    p_pos = support_pvalues(queries, pos, k)
    p_neg = support_pvalues(queries, neg, k)
    return np.where(predicted_positive, p_pos, p_neg).astype(np.float32)


def unsupported_counts(
    pos_vectors: np.ndarray,
    neg_vectors: np.ndarray,
    query_matrix: np.ndarray,
    predicted_positive: np.ndarray,
    k: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, int]:
    """How many of *query_matrix*'s items sit in an evidence vacuum (``D < alpha``), by predicted class (#5089).

    The counts behind :func:`evidence_coverage_report`'s ``frac_unsupported``,
    split by the class each item is routed to: ``n_items``, ``n_pred_pos``,
    ``n_pos_unsupported`` (predicted positives with ``D < alpha``) and
    ``n_neg_unsupported`` (predicted negatives with ``D < alpha``).  Inputs need
    not be normalized.
    """
    queries = _normalize(query_matrix)
    predicted_positive = np.asarray(predicted_positive, dtype=bool)
    n = int(queries.shape[0])
    if n == 0:
        return {"n_items": 0, "n_pred_pos": 0, "n_pos_unsupported": 0, "n_neg_unsupported": 0}
    d = predicted_support_pvalues(queries, _normalize(pos_vectors), _normalize(neg_vectors), predicted_positive, k)
    vacuum = d < alpha
    return {
        "n_items": n,
        "n_pred_pos": int(predicted_positive.sum()),
        "n_pos_unsupported": int((vacuum & predicted_positive).sum()),
        "n_neg_unsupported": int((vacuum & ~predicted_positive).sum()),
    }


def reference_vacuum_counts(
    pos_vectors: np.ndarray,
    neg_vectors: np.ndarray,
    query_matrix: np.ndarray,
    predicted_positive: np.ndarray,
    reference_matrix: np.ndarray,
    reference_positive: np.ndarray,
    k: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, int]:
    """How many queries sit farther from their predicted class's evidence than a reference corpus's items do (#5089).

    The support p-value of :func:`support_pvalues` with the calibration set
    swapped: each query's k-th NN distance to its predicted class's labelled
    evidence is ranked against the same distance of every *reference_matrix*
    item called that class (*reference_positive*), ``p = (1 + #{ref >= a(x)}) /
    (n_ref + 1)``.  With the haystack the votes came from as the reference, the
    null is "drawn as the training haystack's items were", which the votes'
    own leave-one-out spread is not: active learning picks the votes, so they
    are no sample of the class.  Returns ``n_pos_beyond`` and ``n_neg_beyond``,
    the predicted positives and negatives with ``p < alpha``.  A class the
    reference never calls has no spread to rank against, and reads ``p = 1``.
    """
    pos = _normalize(pos_vectors)
    neg = _normalize(neg_vectors)
    queries = _normalize(query_matrix)
    refs = _normalize(reference_matrix)
    q_pos = np.asarray(predicted_positive, dtype=bool)
    r_pos = np.asarray(reference_positive, dtype=bool)
    out = {"n_pos_beyond": 0, "n_neg_beyond": 0}
    if queries.shape[0] == 0:
        return out
    for name, evidence, q_mask, r_mask in (("n_pos_beyond", pos, q_pos, r_pos), ("n_neg_beyond", neg, ~q_pos, ~r_pos)):
        if not q_mask.any() or not r_mask.any():
            continue
        a_query = _kth_nn_distance(queries[q_mask], evidence, k)
        a_ref = np.sort(_kth_nn_distance(refs[r_mask], evidence, k))
        ge = a_ref.shape[0] - np.searchsorted(a_ref, a_query, side="left")
        out[name] = int(np.sum((1 + ge) / (a_ref.shape[0] + 1) < alpha))
    return out


def trust_scores(
    queries: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    predicted_positive: np.ndarray,
    k: int = DEFAULT_K,
) -> np.ndarray:
    """Trust-score ratio ``TS`` per item [11]: other-class over predicted-class.

    ``TS(x) = d(x, other class) / d(x, predicted class)``.  ``TS > 1`` — closer
    to its own predicted class than to the other — is trustworthy; ``TS < 1`` is
    suspect.  When the predicted class has no evidence the denominator is the
    maximal distance ``2.0``, so ``TS`` collapses toward ``0`` (maximally
    suspect), which is the intended read.  Inputs must be unit-normalized.
    """
    predicted_positive = np.asarray(predicted_positive, dtype=bool)
    d_pos = _kth_nn_distance(queries, pos, k)
    d_neg = _kth_nn_distance(queries, neg, k)
    d_pred = np.where(predicted_positive, d_pos, d_neg)
    d_other = np.where(predicted_positive, d_neg, d_pos)
    return (d_other / np.maximum(d_pred, _EPS)).astype(np.float32)


def evidence_coverage_report(
    pos_vectors: np.ndarray,
    neg_vectors: np.ndarray,
    query_matrix: np.ndarray,
    predicted_positive: np.ndarray,
    k: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Summarise evidence coverage of *query_matrix* under a detector's labelset.

    *pos_vectors* / *neg_vectors* are the detector's labeled Good / Bad
    embeddings (from ``DetectorContext.label_embeddings``); *query_matrix* holds
    one embedding per scored active-dataset item and *predicted_positive* its
    per-item predicted class (``find_score >= threshold``).

    Returns a dict with ``n_items``, ``k``, ``alpha``, ``n_pos_labels``,
    ``n_neg_labels``, ``frac_unsupported`` (share with support p-value ``D <
    alpha`` — an evidence vacuum for the predicted class), ``expected_unsupported``
    (``= alpha``), ``z_score`` (binomial z of the excess), ``median_support``
    (median ``D``), ``frac_low_trust`` (share with ``TS < 1`` — closer to the
    other class's evidence), ``median_trust`` (median ``TS``), and
    ``unsupported`` (headline: the excess of vacuum items is both statistically
    clear, ``z > 3``, and practically large, at least ``2 * alpha``).  Mirrors
    :func:`~vtscore.coverage.atlas.domain_shift_report` so the two read on
    one scale.
    """
    pos = _normalize(pos_vectors)
    neg = _normalize(neg_vectors)
    queries = _normalize(query_matrix)
    predicted_positive = np.asarray(predicted_positive, dtype=bool)
    n_pos = int(pos.shape[0])
    n_neg = int(neg.shape[0])
    n = int(queries.shape[0])

    if n == 0:
        return {
            "n_items": 0,
            "k": k,
            "alpha": alpha,
            "n_pos_labels": n_pos,
            "n_neg_labels": n_neg,
            "frac_unsupported": 0.0,
            "expected_unsupported": alpha,
            "z_score": 0.0,
            "median_support": 1.0,
            "frac_low_trust": 0.0,
            "median_trust": 1.0,
            "unsupported": False,
        }

    d = predicted_support_pvalues(queries, pos, neg, predicted_positive, k)
    ts = trust_scores(queries, pos, neg, predicted_positive, k)

    frac_unsupported = float(np.mean(d < alpha))
    se = math.sqrt(alpha * (1.0 - alpha) / n)
    z = (frac_unsupported - alpha) / se if se > 0 else 0.0
    return {
        "n_items": n,
        "k": k,
        "alpha": alpha,
        "n_pos_labels": n_pos,
        "n_neg_labels": n_neg,
        "frac_unsupported": frac_unsupported,
        "expected_unsupported": alpha,
        "z_score": z,
        "median_support": float(np.median(d)),
        "frac_low_trust": float(np.mean(ts < 1.0)),
        "median_trust": float(np.median(ts)),
        "unsupported": bool(z > 3.0 and frac_unsupported >= 2.0 * alpha),
    }
