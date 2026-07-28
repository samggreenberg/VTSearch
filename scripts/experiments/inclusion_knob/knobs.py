"""Score treatments, calibration, and inclusion-knob designs under test.

Everything here is deliberately layered the way production is layered
(``vtscore/training/thresholds.py``): a *treatment* changes how the models are
trained (and therefore what the score distribution looks like), a *knob design*
maps ``inclusion in [-10, 10]`` plus the held-out fold orderings to a decision
threshold in raw score space.  The four designs:

* ``argmin``      - production today: min-weighted-cost search over observed
                    held-out score cuts (:func:`find_optimal_threshold` per fold,
                    aggregated by :func:`threshold_from_fold_orderings`).
* ``bayes``       - the Bayes-optimal cut for the inclusion cost ratio applied
                    directly in probability space: ``p* = fpr_w / (fpr_w + fnr_w)``.
                    Moves by construction, but only *means* anything if scores
                    are honest probabilities.
* ``bayes_temp``  - same, after temperature-scaling the scores with a ``T``
                    fitted on the held-out fold orderings (Platt-style).  The
                    raw-space threshold is ``sigmoid(T * logit(p*))``.
* ``conformal``   - split-conformal quantile rule: positive inclusion buys a
                    false-negative budget taken as a quantile of held-out
                    *positive* scores; negative inclusion a false-positive
                    budget on held-out *negative* scores.  Monotone in the
                    knob by construction.
"""

from __future__ import annotations

import math

import numpy as np

# Budget at |inclusion| = 0 for the conformal rule; halves per knob step.
CONFORMAL_BASE_BUDGET = 0.25

#: Temperature is bounded below at 1.0: with tiny, often perfectly-separated
#: calibration sets the NLL-optimal T collapses toward 0 (sharpening an
#: already-overconfident model).  We only ever want temperature to *soften*.
TEMP_MIN, TEMP_MAX = 1.0, 100.0


def inclusion_weights(inclusion: int) -> tuple[float, float]:
    """``(fpr_weight, fnr_weight)`` exactly as production computes them."""
    if inclusion >= 0:
        return 1.0, 2.0**inclusion
    return 2.0 ** (-inclusion), 1.0


# ---------------------------------------------------------------------------
# Treatments (model training)
# ---------------------------------------------------------------------------


def train_final_model(X_votes: np.ndarray, y_votes: np.ndarray, hidden_dim: int, smooth_eps: float):
    """Train the full-data model for a treatment.

    ``smooth_eps == 0`` is the bit-for-bit production path.  With smoothing we
    still call the production ``train_model`` loop, passing smoothed targets
    plus explicit balanced sample weights (the weights production would have
    computed from the hard labels), so the *only* delta is the target values.
    """
    import torch

    from vtscore.training.mlp import train_model

    X = torch.from_numpy(np.ascontiguousarray(X_votes, dtype=np.float32))
    y_hard = torch.tensor(y_votes, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]
    if smooth_eps == 0.0:
        return train_model(X, y_hard, input_dim, hidden_dim=hidden_dim)

    num_true = int(y_hard.sum().item())
    num_false = len(y_hard) - num_true
    weights = torch.where(y_hard == 1, num_false / num_true, 1.0).squeeze(1)
    y_smooth = y_hard * (1.0 - 2.0 * smooth_eps) + smooth_eps
    return train_model(X, y_smooth, input_dim, hidden_dim=hidden_dim, sample_weights=weights)


def fold_orderings_for_treatment(
    X_votes: np.ndarray,
    y_votes: np.ndarray,
    hidden_dim: int,
    smooth_eps: float,
    calibrate_count: int,
    calibration_fraction: float,
) -> tuple[list[tuple[list[float], list[float]]], float | None]:
    """Held-out ``(scores, labels)`` per calibration fold, per treatment.

    ``smooth_eps == 0`` delegates to production ``compute_fold_orderings``
    (same fresh ``RandomState(42)`` production uses per call).  The smoothed
    variant mirrors that function's row-wise path - same stratified split
    sizing, same RNG consumption pattern - but trains each fold with smoothed
    targets.  Recorded labels stay hard (they are the ground truth the
    threshold search consumes).
    """
    from vtscore.training.thresholds import compute_fold_orderings

    X_list = list(np.asarray(X_votes, dtype=np.float32))
    y_list = [float(v) for v in y_votes]
    input_dim = X_votes.shape[1]
    rng = np.random.RandomState(42)
    if smooth_eps == 0.0:
        return compute_fold_orderings(
            X_list,
            y_list,
            input_dim,
            rng=rng,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
        )

    n = len(X_list)
    if n < 4:
        return [], 0.5
    X_np = np.array(X_list)
    y_np = np.array(y_list)
    n_cal = max(1, round(n * calibration_fraction))
    n_train = n - n_cal
    if n_train < 2 or n_cal < 1:
        return [], 2.0
    pos_idx = np.where(y_np == 1.0)[0]
    neg_idx = np.where(y_np == 0.0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return [], 0.5

    def _per_class_n_train(class_total: int) -> int:
        target = round(class_total * n_train / n)
        return max(1, min(class_total - 1, target))

    n_train_pos = _per_class_n_train(len(pos_idx))
    n_train_neg = _per_class_n_train(len(neg_idx))

    import torch

    from vtscore.utils.scores import sigmoid_to_finite_scores

    orderings: list[tuple[list[float], list[float]]] = []
    for _ in range(max(1, calibrate_count)):
        pos_perm = rng.permutation(pos_idx)
        neg_perm = rng.permutation(neg_idx)
        train_idx = np.concatenate([pos_perm[:n_train_pos], neg_perm[:n_train_neg]])
        cal_idx = np.concatenate([pos_perm[n_train_pos:], neg_perm[n_train_neg:]])
        model = train_final_model(X_np[train_idx], y_np[train_idx], hidden_dim, smooth_eps)
        with torch.no_grad():
            X_cal = torch.from_numpy(np.ascontiguousarray(X_np[cal_idx], dtype=np.float32))
            X_cal = X_cal.to(next(model.parameters()).device)
            scores = sigmoid_to_finite_scores(model(X_cal))
        orderings.append((scores, y_np[cal_idx].tolist()))
    return orderings, None


# ---------------------------------------------------------------------------
# Temperature calibration
# ---------------------------------------------------------------------------

_LOGIT_EPS = 1e-7


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(p / (1.0 - p))


def fit_temperature(orderings: list[tuple[list[float], list[float]]]) -> float:
    """Fit a scalar temperature on the pooled held-out fold scores (NLL).

    Golden-section search over ``log T`` in ``[TEMP_MIN, TEMP_MAX]``.
    Deterministic; returns 1.0 (identity) when the pooled set is single-class.
    """
    scores = np.array([s for ss, _ in orderings for s in ss], dtype=np.float64)
    labels = np.array([lb for _, ll in orderings for lb in ll], dtype=np.float64)
    if len(scores) == 0 or labels.min() == labels.max():
        return 1.0
    logits = _logit(scores)

    def nll(log_t: float) -> float:
        z = logits / math.exp(log_t)
        # log(1 + e^{-z}) stable form
        return float(np.mean(np.logaddexp(0.0, -z) + (1.0 - labels) * z))

    lo, hi = math.log(TEMP_MIN), math.log(TEMP_MAX)
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = nll(d)
    best = (a + b) / 2.0
    # Identity must win ties: if softening doesn't beat T=1's NLL, keep T=1.
    return math.exp(best) if nll(best) < nll(lo) - 1e-9 else 1.0


# ---------------------------------------------------------------------------
# Knob designs: (orderings, T, inclusion) -> raw-score threshold
# ---------------------------------------------------------------------------


def threshold_argmin(orderings, inclusion: int) -> float:
    """Production behavior (min-weighted-cost over observed cuts)."""
    from vtscore.training.thresholds import threshold_from_fold_orderings

    return threshold_from_fold_orderings(orderings, inclusion)


def threshold_bayes(inclusion: int, temperature: float = 1.0) -> float:
    """Bayes-optimal probability cut for the inclusion cost ratio.

    With cost ratio ``fnr_w / fpr_w`` the optimal cut on a *calibrated*
    ``P(good)`` is ``p* = fpr_w / (fpr_w + fnr_w)``.  A calibrated score
    ``sigmoid(logit(s)/T) >= p*`` is equivalent to the raw-space threshold
    ``s >= sigmoid(T * logit(p*))``, which is what this returns.
    """
    fpr_w, fnr_w = inclusion_weights(inclusion)
    p_star = fpr_w / (fpr_w + fnr_w)
    return float(1.0 / (1.0 + math.exp(-temperature * _logit(p_star))))


#: Positive-quantile position the k = -10 end of the knob walks to.  At -10
#: only the region scoring above the 75th percentile of held-out positives is
#: included - "just the most confident matches".
CONFORMAL_QPOS_MAX = 0.75


def threshold_conformal(orderings, inclusion: int) -> float:
    """Split-conformal quantile rule over the pooled held-out fold scores.

    Globally monotone non-increasing in ``inclusion`` by construction (a naive
    two-sided rule - FN quantile for k>0, FP quantile for k<0 - inverts across
    k=0 whenever the calibration scores have a gap between the classes):

    * ``k > 0`` - a **false-negative budget** ``alpha(k) = BASE * 2^-k``:
      threshold = the ``alpha``-quantile of held-out *positive* scores, so
      ~``(1-alpha)`` of true positives land at or above the cut.  Both branch
      arguments shrink with k, so the threshold only ever moves down.
    * ``k <= 0`` - a walk *up* the positive score distribution
      (``q_pos(k) = BASE + (QPOS_MAX - BASE) * |k|/10``: at -10 only the
      top-quartile-of-positives region remains), guarded by a **false-positive
      budget** on held-out negative scores (``max`` with the
      ``1 - BASE * 2^k`` negative quantile) so overlap-heavy tasks keep FPR
      control.  Both branches are monotone in k, so their max is too, and the
      k=0 value dominates the k=1 value, making the whole path monotone.
    """
    pos = np.array([s for ss, ll in orderings for s, lb in zip(ss, ll, strict=True) if lb == 1.0])
    neg = np.array([s for ss, ll in orderings for s, lb in zip(ss, ll, strict=True) if lb == 0.0])
    if len(pos) == 0 or len(neg) == 0:
        return 2.0  # NO_GOOD sentinel: cannot calibrate
    if inclusion > 0:
        alpha = CONFORMAL_BASE_BUDGET * 2.0 ** (-inclusion)
        return float(np.quantile(pos, alpha))
    q_pos = CONFORMAL_BASE_BUDGET + (CONFORMAL_QPOS_MAX - CONFORMAL_BASE_BUDGET) * (-inclusion) / 10.0
    beta = CONFORMAL_BASE_BUDGET * 2.0**inclusion
    return float(max(np.quantile(pos, q_pos), np.quantile(neg, 1.0 - beta)))


def knob_threshold(design: str, orderings, temperature: float, inclusion: int) -> float:
    """Dispatch a knob *design* name to its threshold."""
    if design == "argmin":
        return threshold_argmin(orderings, inclusion)
    if design == "bayes":
        return threshold_bayes(inclusion, temperature=1.0)
    if design == "bayes_temp":
        return threshold_bayes(inclusion, temperature=temperature)
    if design == "conformal":
        return threshold_conformal(orderings, inclusion)
    raise ValueError(f"unknown knob design: {design}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def saturation_stats(scores: np.ndarray) -> dict[str, float]:
    """How saturated is a score distribution?"""
    s = np.asarray(scores, dtype=np.float64)
    return {
        "sat_frac_extreme": float(np.mean((s < 0.001) | (s > 0.999))),
        "sat_frac_mid": float(np.mean((s >= 0.01) & (s <= 0.99))),
        "sat_mean_abs_logit": float(np.mean(np.abs(_logit(s)))),
    }


def pool_metrics(pool_scores: np.ndarray, pool_truth: np.ndarray, threshold: float) -> dict[str, float]:
    """Included-set size + confusion metrics for ``score >= threshold``."""
    included = pool_scores >= threshold
    n_pos = int(pool_truth.sum())
    n_neg = len(pool_truth) - n_pos
    tp = int(np.sum(included & (pool_truth == 1)))
    fp = int(np.sum(included & (pool_truth == 0)))
    fn = n_pos - tp
    return {
        "n_included": int(included.sum()),
        "pool_size": len(pool_truth),
        "recall": tp / n_pos if n_pos else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "fpr": fp / n_neg if n_neg else float("nan"),
        "fnr": fn / n_pos if n_pos else float("nan"),
    }
