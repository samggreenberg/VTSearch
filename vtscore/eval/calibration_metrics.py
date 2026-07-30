"""Pure-numpy calibration metrics + pooling variants for the #2781 study.

This module has **no torch / vtscore-heavy imports** so it can be unit-tested on
a CPU-only box without the model stack.  It provides:

* :func:`operating_cost` — inclusion-weighted ``FPR + FNR`` cost at a fixed cut
  (the number the trained threshold actually pays on a held-out set).
* :func:`oracle_cut` — the cut that *minimises* that same weighted cost over a
  score/label set (the best a threshold rule could possibly do on this ranking),
  via an O(n log n) sweep over the observed scores.  The gap between the trained
  cost and this is the **calibration regret** of issue #2781.
* :func:`threshold_percentile` / :func:`is_degenerate` — where the trained
  threshold sits in a score distribution, and the ``degenerate`` flag (a cut
  above every score or below every score) that is the #2781 runaway-threshold
  bug's signature.
* Segment pooling variants — :func:`segment_max_pool`,
  :func:`segment_topk_mean_pool`, :func:`segment_pnorm_pool` — that collapse a
  flat per-node score vector (with segment boundaries) to one score per image.
  ``max`` is the production tree/patch pooling; ``topk``/``pnorm`` are the
  remedial arms (re-pool the same model's node scores; see the plan).

Every threshold uses the ``predicted positive iff score >= threshold``
convention, matching :func:`vtscore.eval.voting_iterations._evaluate_on_test`.
"""

from __future__ import annotations

import numpy as np


def inclusion_weights(inclusion: int) -> tuple[float, float]:
    """``(fpr_weight, fnr_weight)`` for an inclusion value (mirror of the harness).

    Kept here (rather than imported) so this module stays torch-free and
    self-contained for unit testing; it is identical to
    :func:`vtscore.eval.voting_iterations._inclusion_weights`.
    """
    if inclusion >= 0:
        return 1.0, 2.0**inclusion
    return 2.0 ** (-inclusion), 1.0


def operating_cost(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    fpr_weight: float,
    fnr_weight: float,
) -> tuple[float, float, float]:
    """Return ``(cost, fpr, fnr)`` at *threshold* (predict positive iff ``>=``).

    ``cost = fpr_weight * FPR + fnr_weight * FNR``.  Empty denominators yield a
    zero rate (no negatives -> FPR 0; no positives -> FNR 0), matching the
    harness's historical behaviour.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    predicted = scores >= threshold
    pos = labels == 1.0
    neg = ~pos
    total_pos = float(pos.sum())
    total_neg = float(neg.sum())
    fp = float(np.count_nonzero(predicted & neg))
    fn = float(np.count_nonzero(~predicted & pos))
    fpr = fp / total_neg if total_neg > 0 else 0.0
    fnr = fn / total_pos if total_pos > 0 else 0.0
    return fpr_weight * fpr + fnr_weight * fnr, fpr, fnr


def oracle_cut(
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_weight: float,
    fnr_weight: float,
) -> tuple[float, float, float, float]:
    """The cost-minimising cut over *scores*: ``(threshold, cost, fpr, fnr)``.

    Sweeps every distinct achievable decision boundary (each observed score,
    plus the "predict nothing" option), respecting the ``>=`` convention and
    never splitting a tie.  This is an *oracle* — it reads the labels of the set
    it is optimised on — so it is a lower bound on achievable cost, not a rule.
    Ties in cost break toward the higher threshold (fewer false positives).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = scores.size
    total_pos = float((labels == 1.0).sum())
    total_neg = float(n - total_pos)

    # Baseline candidate: predict nothing positive (threshold above every score).
    # FPR 0, FNR 1 (when any positives exist).
    none_fnr = 1.0 if total_pos > 0 else 0.0
    above = float(np.nextafter(scores.max(), np.inf)) if n else 1.0
    best = (above, fnr_weight * none_fnr, 0.0, none_fnr)
    if n == 0:
        return best

    order = np.argsort(-scores, kind="mergesort")  # descending, stable
    s = scores[order]
    y = labels[order]
    cum_pos = np.cumsum(y)  # tp when the top-(k) are predicted positive, k = index+1
    k = np.arange(1, n + 1, dtype=np.float64)
    tp = cum_pos
    fp = k - cum_pos
    fn = total_pos - tp
    fpr = np.where(total_neg > 0, fp / total_neg, 0.0)
    fnr = np.where(total_pos > 0, fn / total_pos, 0.0)
    cost = fpr_weight * fpr + fnr_weight * fnr

    # A cut after the k-th highest score is achievable only if it does not split
    # a tie: s[k-1] != s[k].  The last position (predict-all) is always valid.
    valid = np.ones(n, dtype=bool)
    valid[:-1] = s[:-1] != s[1:]
    idx = np.nonzero(valid)[0]
    if idx.size:
        j = int(idx[np.argmin(cost[idx])])
        if cost[j] < best[1]:
            # threshold = k-th highest score -> predicts every score >= s[j].
            best = (float(s[j]), float(cost[j]), float(fpr[j]), float(fnr[j]))
    return best


def threshold_percentile(scores: np.ndarray, threshold: float) -> float:
    """Fraction of *scores* strictly below *threshold* (in ``[0, 1]``).

    1.0 means the cut sits above every score (predict nothing); 0.0 means below
    every score (predict everything).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return float("nan")
    return float(np.count_nonzero(scores < threshold) / scores.size)


def is_degenerate(scores: np.ndarray, threshold: float) -> bool:
    """True when *threshold* sits above the max or below the min of *scores*.

    A degenerate cut classifies every item the same way (all-negative when above
    the max, all-positive when below the min).  Above-the-max degeneracy with
    FNR 1 / FPR 0 is the #2781 runaway-threshold signature.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return False
    return bool(threshold > scores.max() or threshold <= scores.min())


# ------------------------------------------------------------------
# Segment pooling variants (collapse per-node scores -> per-image score)
# ------------------------------------------------------------------


def segment_counts(seg_starts: np.ndarray, total: int) -> np.ndarray:
    """Node count ``N`` per segment given cumulative ``seg_starts`` and row total."""
    seg_starts = np.asarray(seg_starts, dtype=np.int64)
    ends = np.empty_like(seg_starts)
    ends[:-1] = seg_starts[1:]
    ends[-1] = total
    return ends - seg_starts


def segment_max_pool(flat: np.ndarray, seg_starts: np.ndarray) -> np.ndarray:
    """Per-segment max (the production tree / patch pooling)."""
    flat = np.asarray(flat, dtype=np.float64)
    seg_starts = np.asarray(seg_starts, dtype=np.int64)
    if flat.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.maximum.reduceat(flat, seg_starts)


def segment_topk_mean_pool(flat: np.ndarray, seg_starts: np.ndarray, k: int) -> np.ndarray:
    """Per-segment mean of the top-``min(k, N)`` node scores.

    A softer max: an image scores by the average of its *k* best-matching nodes
    rather than its single best, damping the heavy upper tail a max over many
    nodes accrues (the hypothesised over-firing of the raw-patch HAC tree).
    """
    flat = np.asarray(flat, dtype=np.float64)
    seg_starts = np.asarray(seg_starts, dtype=np.int64)
    total = flat.shape[0]
    out = np.empty(seg_starts.size, dtype=np.float64)
    for i in range(seg_starts.size):
        a = int(seg_starts[i])
        b = int(seg_starts[i + 1]) if i + 1 < seg_starts.size else total
        seg = flat[a:b]
        kk = min(k, seg.size)
        if kk <= 0:
            out[i] = float("nan")
        elif kk >= seg.size:
            out[i] = float(seg.mean())
        else:
            # top-kk without full sort
            out[i] = float(np.partition(seg, seg.size - kk)[seg.size - kk :].mean())
    return out


def segment_pnorm_pool(
    flat: np.ndarray,
    seg_starts: np.ndarray,
    null_sorted: np.ndarray,
) -> np.ndarray:
    """Extreme-value normalisation: ``1 - F_neg(max)^N`` per segment.

    *null_sorted* is the ascending-sorted empirical distribution of node scores
    over the calibration **negative** bags.  For an image whose ``N`` nodes have
    best score ``m``, ``F_neg(m)`` is the fraction of null node scores ``<= m``,
    and ``1 - F_neg(m)^N`` is the probability a null image with ``N`` nodes would
    reach a max at least ``m`` — so an image with twice the nodes no longer earns
    an inflated max for free.  Score is a p-value-like quantity in ``[0, 1]``.
    """
    flat = np.asarray(flat, dtype=np.float64)
    seg_starts = np.asarray(seg_starts, dtype=np.int64)
    null_sorted = np.asarray(null_sorted, dtype=np.float64)
    total = flat.shape[0]
    m = null_sorted.size
    out = np.empty(seg_starts.size, dtype=np.float64)
    for i in range(seg_starts.size):
        a = int(seg_starts[i])
        b = int(seg_starts[i + 1]) if i + 1 < seg_starts.size else total
        seg = flat[a:b]
        if seg.size == 0:
            out[i] = float("nan")
            continue
        mx = float(seg.max())
        n_nodes = seg.size
        if m == 0:
            out[i] = 0.0  # no null mass -> cannot exceed; treat as fully un-surprising
            continue
        # fraction of null <= mx
        cdf = float(np.searchsorted(null_sorted, mx, side="right")) / m
        out[i] = 1.0 - cdf**n_nodes
    return out


def pool_blocks(
    blocks: list[np.ndarray],
    variant: str,
    *,
    topk: int = 4,
    null_sorted: np.ndarray | None = None,
) -> list[float]:
    """Pool a list of variable-length node-score *blocks* (one per bag) to scalars.

    The block-list counterpart of :func:`pool_segment`, used to re-pool the
    per-fold held-out calibration groups when recalibrating a variant threshold
    (each block is one calibration bag's node scores).  ``max`` reproduces the
    production grouped-calibration pooling.
    """
    if variant == "max":
        return [float(np.max(b)) for b in blocks]
    if variant == "topk":
        out = []
        for b in blocks:
            b = np.asarray(b, dtype=np.float64)
            kk = min(topk, b.size)
            out.append(float(np.partition(b, b.size - kk)[b.size - kk :].mean()) if kk < b.size else float(b.mean()))
        return out
    if variant == "pnorm":
        if null_sorted is None:
            raise ValueError("pnorm pooling requires null_sorted (calibration negative node scores)")
        null_sorted = np.asarray(null_sorted, dtype=np.float64)
        m = null_sorted.size
        out = []
        for b in blocks:
            b = np.asarray(b, dtype=np.float64)
            mx = float(b.max())
            cdf = (float(np.searchsorted(null_sorted, mx, side="right")) / m) if m else 1.0
            out.append(1.0 - cdf**b.size)
        return out
    raise ValueError(f"unknown pooling variant {variant!r}")


def negative_block_null(blocks: list[np.ndarray], labels: list[float]) -> np.ndarray:
    """Ascending-sorted node scores over the negative-labelled *blocks* (the pnorm null).

    Concatenates every block whose bag label is 0 and sorts, giving the empirical
    null distribution ``F_neg`` for :func:`segment_pnorm_pool` / :func:`pool_blocks`.
    """
    neg = [np.asarray(b, dtype=np.float64) for b, lb in zip(blocks, labels, strict=True) if lb == 0.0]
    if not neg:
        return np.empty(0, dtype=np.float64)
    return np.sort(np.concatenate(neg))


def pool_segment(
    flat: np.ndarray,
    seg_starts: np.ndarray,
    variant: str,
    *,
    topk: int = 4,
    null_sorted: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch to the pooling *variant* (``max`` / ``topk`` / ``pnorm``)."""
    if variant == "max":
        return segment_max_pool(flat, seg_starts)
    if variant == "topk":
        return segment_topk_mean_pool(flat, seg_starts, topk)
    if variant == "pnorm":
        if null_sorted is None:
            raise ValueError("pnorm pooling requires null_sorted (calibration negative node scores)")
        return segment_pnorm_pool(flat, seg_starts, null_sorted)
    raise ValueError(f"unknown pooling variant {variant!r}")
