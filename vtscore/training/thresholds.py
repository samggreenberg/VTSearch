"""Decision-threshold computation for learned-sort scores.

GMM-, cross-calibration-, and safe-threshold helpers. These are media-
agnostic: they take score lists and label lists and return a single
float threshold. Detector-specific glue (sourcing ``X_list`` / ``y_list``
from votes, caching on ``DetectorContext``) lives in
:mod:`vtscore.detectors`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# Sentinel threshold meaning "predict nothing as Good". Sigmoid scores are
# in [0, 1], so any value > 1.0 makes every ``score >= threshold`` check
# evaluate to False. Kept finite (vs. ``float("inf")``) so it cannot poison
# downstream blends - ``0.0 * inf`` evaluates to NaN, which would then be
# stored on ``DetectorContext.threshold`` and break every comparison.
NO_GOOD_THRESHOLD = 2.0

# False-negative budget of the conformal inclusion rule at inclusion 0; each
# +1 step of inclusion halves it (see :func:`conformal_threshold`).  0.25 means
# the default cutoff may sacrifice at most ~25% of true matches to the
# false-positive guard - a cap, spent only when class overlap forces it.
CONFORMAL_BASE_BUDGET = 0.25

# Positive-score quantile the inclusion = -10 end of the knob walks to: at -10
# only the region scoring above the 75th percentile of held-out positives is
# included - "just the most confident matches".
CONFORMAL_QPOS_MAX = 0.75

# Above this many scores, fit the GMM on a random subsample instead of the full
# set. A 2-component, 1-D GMM only needs to recover the two clusters' means and
# variances, which 50k samples estimate as accurately as the full population -
# so the threshold is statistically indistinguishable while the EM fit stays
# O(50k) instead of O(N). This matters because ``calculate_gmm_threshold`` runs
# on the *full* score distribution on every cosine/text sort (sorting.py) and in
# the safe-threshold blend, where N reaches ~250k (GUI Find) to 2M+ (CLI Find).
_GMM_MAX_SAMPLES = 50_000


def classify_threshold_provenance(fallback: float | None) -> str:
    """Name the code path a trained threshold came from, from its *fallback*.

    :func:`compute_fold_orderings` returns a ``fallback`` that fully discriminates
    which branch produced the threshold: ``None`` means the conformal quantile
    rule ran on real fold orderings; :data:`NO_GOOD_THRESHOLD` (2.0) means the
    "no valid Train/Calibrate split" sentinel; ``0.5`` means a too-few-labels
    early return.  Used by the calibration study (issue #2781) to attribute the
    runaway-threshold bug; the safe-threshold GMM blend is a separate caller and
    is tagged ``"gmm_blend"`` at that site, not here.
    """
    if fallback is None:
        return "conformal"
    if fallback == NO_GOOD_THRESHOLD:
        return "no_good_sentinel"
    if fallback == 0.5:
        return "too_few_default"
    return "unknown"


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    """Real roots of ``a*x^2 + b*x + c``, degenerating gracefully to the linear case.

    Uses the cancellation-free ("citardauq") pairing ``q = -(b + sign(b)*sqrt(D))/2``,
    ``x = {q/a, c/q}`` rather than the textbook formula.  That matters here because
    the near-equal-variance case drives ``a`` toward 0, where ``(-b + sqrt(D)) /
    (2a)`` is catastrophic cancellation over a vanishing denominator while ``c/q``
    stays accurate and converges smoothly to the linear root ``-c/b``.
    """
    if a == 0.0:
        return [] if b == 0.0 else [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    q = -0.5 * (b + math.copysign(math.sqrt(disc), b))
    if q == 0.0:
        # Only reachable with b == 0 and disc == 0, i.e. ``a*x^2 = 0``.
        return [0.0]
    return [q / a, c / q]


def _weighted_gaussian_crossing(
    w_lo: float,
    mu_lo: float,
    var_lo: float,
    w_hi: float,
    mu_hi: float,
    var_hi: float,
) -> float | None:
    """Score between the two means where the weighted component densities cross.

    Solves ``w_lo * N(x; mu_lo, var_lo) == w_hi * N(x; mu_hi, var_hi)``.  Taking
    logs makes the difference a quadratic ``f(x) = a x^2 + b x + c`` (``f > 0``
    means the Bad component owns that score), so the crossing is a root of that
    quadratic - the Bayes decision boundary between the two fitted components
    **with the mixture weights as class priors**, i.e. the cut that minimises
    expected misclassification *count*.

    **Not the shipped cut.**  Production cuts at the midpoint between the means
    (:func:`calculate_gmm_threshold`); this solver stays live only as an eval
    variant (``*_cross`` in :data:`vtscore.eval.voting_iterations._SAFE_GMM_VARIANTS`).
    #2798 shipped it on the geometry argument below and #2799 measured it as a
    small net loss (+0.0036 cost at 6-20 votes, +0.0059 at 2-5), so #2833 reverted
    it.  The geometry argument was right in *direction* - the crossing does sit
    above the midpoint under max-pooling - but the exchange rate is unfavourable:
    it buys ~1 FPR for ~1.3 FNR, and for a needle-finding tool the missed positive
    is the worse error.  #2836 is the open question of why (leading hypothesis:
    we score a *rate* loss, so the prior-odds term in this crossing is the bias).

    The crossing and the midpoint agree **exactly** when the components are
    equal-weight and equal-variance.  They diverge precisely where region voting
    lives: a media's score is the max over ~24 region nodes, so the Bad mode is an
    extreme-value statistic - wider, right-skewed, and far heavier than the Good
    mode.  A wider/heavier low component pushes the crossing *above* the midpoint
    (with equal variances the offset is ``var * ln(w_lo/w_hi) / (mu_hi - mu_lo)``).

    Returns ``None`` - meaning "the caller should fall back to the midpoint" -
    whenever the crossing is not a well-defined boundary: non-positive weights or
    variances, non-ordered/degenerate means, a complex-root fit, no root strictly
    between the means (near-equal variances with an extreme weight ratio push the
    linear root outside the interval), or a fit in which the Bad component still
    out-densities the Good one at the Good mean.  When two roots land inside the
    interval the larger one is taken: above it the Good component dominates all
    the way to its own mean, which is the boundary a threshold wants.
    """
    if not (w_lo > 0.0 and w_hi > 0.0 and var_lo > 0.0 and var_hi > 0.0):
        return None
    if not (mu_hi > mu_lo):
        return None

    # Solve in ``u = x - mu_lo`` so the interval is ``(0, d)``.  Shifting keeps
    # the roots exact while dropping the ``mu^2 / var`` terms that would dominate
    # the coefficients (and their cancellation) for score scales far from zero.
    d = mu_hi - mu_lo
    offset = math.log(w_lo / w_hi) + 0.5 * math.log(var_hi / var_lo)
    a = 0.5 / var_hi - 0.5 / var_lo
    b = -d / var_hi
    c = 0.5 * d * d / var_hi + offset

    # The Good mode must actually be Good-dominated, else "the score above which
    # Good wins" is not something this fit expresses.  Evaluated in closed form
    # rather than as ``a d^2 + b d + c`` (the same value, without the cancellation).
    if offset - 0.5 * d * d / var_lo >= 0.0:
        return None

    inside = [u for u in _quadratic_roots(a, b, c) if math.isfinite(u) and 0.0 < u < d]
    if not inside:
        return None
    return mu_lo + max(inside)


@dataclass(frozen=True)
class GmmFit1D:
    """The two components of a fitted 1-D, 2-component GMM, ordered by mean.

    Carries exactly the parameters the two candidate cut rules need, so one EM
    fit can be re-cut under both rules (the safe-threshold measurement study,
    issue #2799, and its #2836 follow-up) instead of re-fitting per rule.  ``lo``
    is the Bad (low-mean) component, ``hi`` the Good one.
    """

    w_lo: float
    mu_lo: float
    var_lo: float
    w_hi: float
    mu_hi: float
    var_hi: float

    def midpoint(self) -> float:
        """The production cut: the midpoint between the two component means."""
        return (self.mu_lo + self.mu_hi) / 2.0

    def crossing_or_midpoint(self) -> float:
        """Eval-only cut (#2798, reverted by #2833): equal-density crossing, midpoint when none exists."""
        crossing = _weighted_gaussian_crossing(self.w_lo, self.mu_lo, self.var_lo, self.w_hi, self.mu_hi, self.var_hi)
        return self.midpoint() if crossing is None else crossing


def gmm_fit_array(scores: "list[float] | np.ndarray") -> np.ndarray:
    """The (possibly subsampled) float64 array a score-GMM is fitted on.

    Above :data:`_GMM_MAX_SAMPLES` scores, takes a deterministic (seed-42)
    random subsample; below, returns the scores unchanged.  Exposed separately
    from :func:`fit_score_gmm` so a caller that needs the fit's *input* too
    (e.g. for the median fallback, or to transform the same sample into logit
    space) subsamples exactly once.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.shape[0] > _GMM_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, size=_GMM_MAX_SAMPLES, replace=False)
    return arr


def fit_score_gmm(arr: np.ndarray) -> GmmFit1D | None:
    """Fit a deterministic 2-component GMM to a 1-D score array.

    Returns ``None`` when the fit fails (fewer than 2 scores, or an EM
    failure), leaving the fallback policy to the caller -
    :func:`calculate_gmm_threshold` falls back to the median.
    """
    if arr.shape[0] < 2:
        return None

    from sklearn.mixture import GaussianMixture  # noqa: PLC0415

    try:
        gmm: GaussianMixture = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(arr.reshape(-1, 1))

        # The stubs type these ``np.ndarray | None``; all are set after ``fit``.
        assert gmm.means_ is not None
        assert gmm.covariances_ is not None
        assert gmm.weights_ is not None
        means = np.ravel(gmm.means_)
        # ``covariances_`` is (n_components, 1, 1) under the default "full"
        # covariance type; ravel gives the two scalar variances.
        variances = np.ravel(gmm.covariances_)
        weights = np.ravel(gmm.weights_)

        low_idx = 0 if means[0] < means[1] else 1
        high_idx = 1 - low_idx
        return GmmFit1D(
            w_lo=float(weights[low_idx]),
            mu_lo=float(means[low_idx]),
            var_lo=float(variances[low_idx]),
            w_hi=float(weights[high_idx]),
            mu_hi=float(means[high_idx]),
            var_hi=float(variances[high_idx]),
        )
    except Exception:
        return None


def calculate_gmm_threshold(scores: list[float]) -> float:
    """Use a Gaussian Mixture Model to find a threshold between two score distributions.

    Fits a 2-component GMM to the provided scores, assuming a bimodal distribution
    representing Bad (low) and Good (high) classes, and returns the **midpoint
    between the two fitted component means**.

    #2798 replaced this midpoint with the components' equal-density crossing (see
    :func:`_weighted_gaussian_crossing`) on the geometry argument that max-pooling
    fattens the Bad mode, so the midpoint cuts inside Bad mass.  #2799 measured the
    two as paired within-step variants and the crossing lost on cost in every
    max-pooled window (report ``docs/experiments/safe-thresholds/REPORT.md``), so
    #2833 reverted to the midpoint.  The crossing solver is retained for the eval
    variant family and for #2836, which is looking for a third, better-founded cut.

    For score sets larger than :data:`_GMM_MAX_SAMPLES`, fits on a deterministic
    (seed-42) random subsample - the two-Gaussian fit is unchanged in practice
    but the cost no longer grows with the dataset size.

    Args:
        scores: List of model confidence scores, expected to follow a bimodal distribution.

    Returns:
        A float threshold. Scores at or above this value are classified as Good.
        Returns ``0.5`` when fewer than 2 scores are provided; falls back to
        the median of scores if GMM fitting fails.
    """
    if len(scores) < 2:
        return 0.5

    arr = gmm_fit_array(scores)
    fit = fit_score_gmm(arr)
    if fit is None:
        # If GMM fails, return median (of the subsample when one was taken -
        # representative of the full distribution and keeps this path bounded).
        return float(np.median(arr))
    return fit.midpoint()


def _score_rows_digest(score_rows_by_group: dict | None) -> bytes | None:
    """Digest of the per-bag **inference** row stacks, for the calibration key.

    ``None`` when no override is in play, so a call that pools each bag over
    its training rows keeps a distinct cache key from one that pools over the
    scorer's rows - otherwise a cached ordering computed under the old geometry
    would be served after the wiring changed.
    """
    if score_rows_by_group is None:
        return None
    h = hashlib.blake2b()
    for g in sorted(score_rows_by_group, key=repr):
        h.update(repr(g).encode())
        h.update(np.asarray(score_rows_by_group[g], dtype=np.float32).tobytes())
    return h.digest()


def _calibration_cache_key(
    X_list: list,
    y_list: list[float],
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int,
    groups: list | None = None,
    score_rows_by_group: dict | None = None,
) -> tuple:
    """Build a deterministic cache key for the calibration **fold orderings**.

    The orderings (per-fold held-out scores + labels) are a deterministic
    function of these inputs (RNG seeded with 42 at every cached call site)
    and are **inclusion-independent** - ``inclusion`` is deliberately *not* in
    the key, so an Inclusion change hits the cache and only re-runs the cheap
    conformal quantile rule.  The key encodes a hash of the raw training vectors (not
    just label IDs) so a labelset re-resolved to different embeddings - e.g.
    after the embedder changes - invalidates the cache automatically.  See
    docs/plans/find-verification-workflow.md.
    """
    # Hash the raw training vectors rather than embedding them in the key.
    # The full ``(N_labels x D x 4)``-byte string reaches ~150 MB at 100k
    # labels and would live in the calibration cache until the next call
    # invalidates it. blake2b (fast, 128-bit digest) keeps the key tiny while
    # still changing whenever the labelset re-resolves to different embeddings
    # (e.g. after the embedder changes); the cache is already invalidated by any
    # vote change, so the hash is purely for collision resistance.
    X_hash = hashlib.blake2b(np.stack(X_list).astype(np.float32, copy=False).tobytes()).digest()
    y_hash = hashlib.blake2b(np.asarray(y_list, dtype=np.float32).tobytes()).digest()
    # Bag membership changes the fold split and per-group max-pool, so a change
    # in grouping must invalidate the cached orderings even when X/y are equal.
    groups_key = tuple(str(g) for g in groups) if groups is not None else None
    return (
        X_hash,
        y_hash,
        int(calibrate_count),
        float(calibration_fraction),
        int(hidden_dim),
        groups_key,
        _score_rows_digest(score_rows_by_group),
    )


def cross_calibration_threshold_cached(
    X_list: list,
    y_list: list[float],
    input_dim: int,
    inclusion_value: int,
    *,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int,
    det_ctx: Any = None,
    groups: list | None = None,
    score_rows_by_group: dict | None = None,
) -> float:
    """Memoized wrapper around :func:`calculate_cross_calibration_threshold`.

    When *det_ctx* is provided, caches the inclusion-independent **fold
    orderings** on ``det_ctx.calibration_cache`` as ``(key, (orderings,
    fallback))`` and reuses them whenever the (labels, calibrate settings)
    key matches.  This is the common case during interactive sorting: the
    user toggles ``inclusion`` or loads a new media item, the labels stay the
    same, and the only work left is re-running the cheap conformal quantile
    rule over the cached orderings - no ~200-epoch fold fits.

    A real label change produces a different cache key and falls through to a
    fresh calibration - no explicit invalidation needed.  *score_rows_by_group*
    (see :func:`compute_fold_orderings`) enters the key too, so a change in the
    rows a bag is scored over can never be served from a stale ordering.
    """
    payload: tuple[list[tuple[list[float], list[float]]], float | None] | None = None
    key = None
    if det_ctx is not None:
        key = _calibration_cache_key(
            X_list,
            y_list,
            calibrate_count,
            calibration_fraction,
            hidden_dim,
            groups,
            score_rows_by_group,
        )
        cached = getattr(det_ctx, "calibration_cache", None)
        if cached is not None and cached[0] == key:
            payload = cached[1]

    if payload is None:
        rng = np.random.RandomState(42)
        payload = compute_fold_orderings(
            X_list,
            y_list,
            input_dim,
            rng=rng,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            groups=groups,
            score_rows_by_group=score_rows_by_group,
        )
        if det_ctx is not None and key is not None:
            det_ctx.calibration_cache = (key, payload)

    orderings, fallback = payload
    if fallback is not None:
        return fallback
    return threshold_from_fold_orderings(orderings, inclusion_value)


def conformal_threshold(
    scores: list[float],
    labels: list[float],
    inclusion_value: int = 0,
) -> float:
    """Split-conformal quantile threshold over held-out calibration scores.

    Maps ``inclusion_value`` to a decision threshold via quantiles of the
    calibration score distributions rather than a min-cost search over
    observed cuts.  The min-cost argmin this replaced had exactly as many
    distinct optima as the calibration set had ranking errors, so on
    well-separated votes (the common case) the threshold never moved with
    inclusion; quantiles move whenever the scores have any spread (see
    docs/experiments/inclusion-knob/REPORT.md and issue #2693).

    The rule, for ``k = inclusion_value`` (``BASE = CONFORMAL_BASE_BUDGET``):

    * A **false-negative cap** ``alpha(k) = min(1, BASE * 2^-k)``: the
      threshold never exceeds the ``alpha``-quantile of the calibration
      *positive* scores, so an estimated ``1 - alpha`` of true matches land
      at or above the cut.  ``+k`` therefore has a portable, user-facing
      meaning - "the fraction of true matches I'm willing to miss, halving
      per step" - independent of the dataset or detector.  The cap is an
      upper bound, not a target: when the classes separate cleanly the cut
      drops to the lowest calibration positive and the budget goes unspent
      (no match is sacrificed that the negatives don't force).
    * A **false-positive guard** for ``k <= 0``: the threshold stays at or
      above the ``1 - BASE * 2^k`` quantile of the calibration *negative*
      scores, so overlap-heavy tasks keep FPR control, and above a walk *up*
      toward the positive score distribution.  The walk interpolates linearly
      in score space from the **gap midpoint** at ``k = 0`` to the
      ``QPOS_MAX`` quantile of positives at ``k = -10`` (at -10 only the
      top-quartile-of-positives region remains - "just the surest matches").

    The **gap midpoint** is what keeps the default cut usable.  When the
    classes separate cleanly there is an empty band between the top of the
    negatives (``fp_guard``) and the bottom of the positives; *every* cut
    inside that band has identical empirical error on the calibration set, so
    the band's top edge - the single lowest calibration positive - is an
    arbitrary choice among equals, and it is the worst one available:

    * It is an **extreme order statistic** over a handful of held-out votes,
      so it moves violently from one vote to the next (issue #2781's "the
      threshold jumps to the top, then it's normal again one click later").
    * It is measured on the **fold models'** score scale but applied to the
      **final** model's scores.  The fold models train on half the votes and
      saturate, so their lowest held-out positive routinely lands above every
      score the final model produces - a cut that admits nothing at all, not
      even the items the user personally voted Good.

    Sitting in the middle of the band is the max-margin choice among cuts the
    calibration data cannot distinguish, and it costs nothing in FN budget:
    the midpoint is strictly below every calibration positive, so a cleanly
    separated task still spends none of its miss budget.

    Every component is monotone non-increasing in ``k``, so their min/max
    composition is too: the threshold is monotone non-increasing in
    inclusion **by construction**.  Raising inclusion can only grow the
    included set, and the sets are nested (everything included at ``k`` stays
    included at ``k + 1``) - which is what makes "cut off at Inclusion 1,
    verify up to Inclusion 4" a well-defined workflow.

    Args:
        scores: Held-out calibration scores, one per example.  Must come from
            data the scoring model did *not* train on; scores on the training
            votes themselves are optimistically separated and yield a
            too-tight band.
        labels: True binary labels (1.0 for good, 0.0 for bad),
            corresponding to ``scores``.
        inclusion_value: Integer in ``[-10, 10]``; higher includes more.

    Returns:
        A float threshold, always realizable within the calibration score
        range (the rule never abstains).  Defaults to 0.5 when the score
        list is empty or single-class (no quantiles to take).
    """
    if not scores:
        return 0.5

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    pos = scores_arr[labels_arr == 1.0]
    neg = scores_arr[labels_arr != 1.0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    def _threshold_at(k: int) -> float:
        fn_cap = float(np.quantile(pos, min(1.0, CONFORMAL_BASE_BUDGET * 2.0**-k)))
        if k > 0:
            # The k=0 floor keeps the seam monotone: q_pos(alpha) can sit
            # above the k=0 cut when the budget goes unspent there.
            return min(fn_cap, _threshold_at(0))
        fp_guard = float(np.quantile(neg, 1.0 - CONFORMAL_BASE_BUDGET * 2.0**k))
        # Midpoint of the band the calibration data cannot resolve: from the
        # top of the negatives up to the lowest positive.  Collapses to
        # ``fp_guard`` under class overlap (no band), so the FPR-controlled
        # regime is untouched - only the cleanly-separated case moves.
        gap_mid = (fp_guard + max(fp_guard, float(np.min(pos)))) / 2.0
        # Walk from that midpoint at k=0 up to the QPOS_MAX positive quantile
        # at k=-10, linearly in score space.  Interpolating on values rather
        # than on quantile positions keeps the knob's stops evenly spaced even
        # when only a handful of calibration positives exist (a quantile walk
        # over 4 points has 4 stops; this one always has 11).
        top = float(np.quantile(pos, CONFORMAL_QPOS_MAX))
        walk = gap_mid + (-k / 10.0) * max(0.0, top - gap_mid)
        return min(fn_cap, max(fp_guard, walk))

    return _threshold_at(inclusion_value)


def _per_bag_fit_weights(
    y_rows: np.ndarray,
    group_rows: list,
) -> np.ndarray:
    """Per-row loss weights that balance Good votes against Bad **bags**.

    Every Good row weighs ``n_bad_bags / n_good``; every Bad row weighs
    ``1 / (rows in its bag)`` so each Bad image contributes exactly one image's
    worth of negative mass regardless of how many region nodes it flooded in.
    Total Good mass (``n_bad_bags``) equals total Bad mass (``n_bad_bags``), so
    the classes are balanced at the same magnitude as :func:`train_model`'s
    default inverse-frequency weights - only the *unit* changes from row to bag.
    """
    from collections import Counter  # noqa: PLC0415

    n_good = sum(1 for lbl in y_rows if lbl == 1.0)
    bad_groups = {g for g, lbl in zip(group_rows, y_rows, strict=True) if lbl == 0.0}
    n_bad_bags = len(bad_groups)
    bag_sizes = Counter(g for g, lbl in zip(group_rows, y_rows, strict=True) if lbl == 0.0)
    good_w = (n_bad_bags / n_good) if n_good else 1.0
    w = np.ones(len(y_rows), dtype=np.float32)
    for i, (g, lbl) in enumerate(zip(group_rows, y_rows, strict=True)):
        w[i] = good_w if lbl == 1.0 else (1.0 / bag_sizes[g])
    return w


def _pooled_group_scores(
    model: Any,
    cal_groups: list,
    rows_by_group: dict,
    X_np: np.ndarray,
    score_rows_by_group: dict | None,
) -> list[float]:
    """Collapse each calibration group to one max-pooled sigmoid score.

    With *score_rows_by_group* each group pools over the rows the scorer will
    max-pool at **inference**; otherwise it pools over the rows it trained on
    (the historical behaviour every production caller takes).
    """
    import torch  # noqa: PLC0415

    from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

    device = next(model.parameters()).device
    if score_rows_by_group is not None:
        blocks = [np.asarray(score_rows_by_group[g], dtype=np.float32) for g in cal_groups]
        sizes = [b.shape[0] for b in blocks]
        with torch.no_grad():
            X_cal = torch.tensor(np.concatenate(blocks, axis=0), dtype=torch.float32).to(device)
            flat = sigmoid_to_finite_scores(model(X_cal))
        out: list[float] = []
        offset = 0
        for size in sizes:
            out.append(max(flat[offset : offset + size]))
            offset += size
        return out

    cal_idx = [i for g in cal_groups for i in rows_by_group[g]]
    with torch.no_grad():
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32).to(device)
        row_scores = sigmoid_to_finite_scores(model(X_cal))
    by_row = dict(zip(cal_idx, row_scores, strict=True))
    return [max(by_row[i] for i in rows_by_group[g]) for g in cal_groups]


def _group_node_blocks(
    model: Any,
    cal_groups: list,
    rows_by_group: dict,
    X_np: np.ndarray,
    score_rows_by_group: dict | None,
) -> list[np.ndarray]:
    """Per calibration group, the array of that group's per-node sigmoid scores.

    The un-pooled counterpart of :func:`_pooled_group_scores`: it returns each
    group's full node-score vector rather than its max, so a caller can re-pool
    the bag under an alternative rule (top-k, extreme-value) while reusing the
    exact fold model and node scores.  ``max`` over each returned block
    reproduces :func:`_pooled_group_scores` value-for-value.
    """
    import torch  # noqa: PLC0415

    from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

    device = next(model.parameters()).device
    if score_rows_by_group is not None:
        blocks = [np.asarray(score_rows_by_group[g], dtype=np.float32) for g in cal_groups]
        sizes = [b.shape[0] for b in blocks]
        with torch.no_grad():
            X_cal = torch.tensor(np.concatenate(blocks, axis=0), dtype=torch.float32).to(device)
            flat = sigmoid_to_finite_scores(model(X_cal))
        out: list[np.ndarray] = []
        offset = 0
        for size in sizes:
            out.append(np.asarray(flat[offset : offset + size], dtype=np.float64))
            offset += size
        return out

    cal_idx = [i for g in cal_groups for i in rows_by_group[g]]
    with torch.no_grad():
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32).to(device)
        row_scores = sigmoid_to_finite_scores(model(X_cal))
    by_row = dict(zip(cal_idx, row_scores, strict=True))
    return [np.asarray([by_row[i] for i in rows_by_group[g]], dtype=np.float64) for g in cal_groups]


def _grouped_folds(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    groups: list,
    rng: np.random.RandomState | None,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int | None,
) -> tuple[list[tuple[Any, list]], float | None, np.ndarray, dict, dict]:
    """Train the bag-aware calibration folds; return the trained fold models.

    The shared core of :func:`_compute_fold_orderings_grouped` and
    :func:`compute_grouped_fold_node_scores`: both need identical fold splits and
    fold models, differing only in how they collapse each calibration group
    (max-pool vs. keep every node).  Returns
    ``(folds, fallback, X_np, rows_by_group, label_by_group)`` where *folds* is a
    list of ``(model, cal_groups)`` and *fallback* is a sentinel threshold when
    calibration is impossible (empty *folds* then).
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    _rng = rng if rng is not None else np.random
    X_np = np.array(X_list)
    y_np = np.array(y_list)
    grp = list(groups)

    # Rows per group, and each group's (single) label.
    order_groups: list = []
    rows_by_group: dict = {}
    label_by_group: dict = {}
    for i, g in enumerate(grp):
        if g not in rows_by_group:
            rows_by_group[g] = []
            order_groups.append(g)
            label_by_group[g] = y_np[i]
        rows_by_group[g].append(i)

    pos_groups = [g for g in order_groups if label_by_group[g] == 1.0]
    neg_groups = [g for g in order_groups if label_by_group[g] == 0.0]
    n = len(order_groups)
    if n < 4:
        return [], 0.5, X_np, rows_by_group, label_by_group
    if len(pos_groups) < 2 or len(neg_groups) < 2:
        return [], 0.5, X_np, rows_by_group, label_by_group

    n_cal = max(1, round(n * calibration_fraction))
    n_train = n - n_cal
    if n_train < 2 or n_cal < 1:
        return [], NO_GOOD_THRESHOLD, X_np, rows_by_group, label_by_group

    def _per_class_n_train(class_total: int) -> int:
        target = round(class_total * n_train / n)
        return max(1, min(class_total - 1, target))

    n_train_pos = _per_class_n_train(len(pos_groups))
    n_train_neg = _per_class_n_train(len(neg_groups))

    # Index the plain group lists by position - group ids are tuples, and
    # ``np.array(list_of_tuples)`` would build a 2-D array and mangle them.
    folds: list[tuple[Any, list]] = []
    for _ in range(max(1, calibrate_count)):
        pos_perm = _rng.permutation(len(pos_groups))
        neg_perm = _rng.permutation(len(neg_groups))
        train_groups = [pos_groups[i] for i in pos_perm[:n_train_pos]] + [neg_groups[i] for i in neg_perm[:n_train_neg]]
        cal_groups = [pos_groups[i] for i in pos_perm[n_train_pos:]] + [neg_groups[i] for i in neg_perm[n_train_neg:]]

        train_idx = [i for g in train_groups for i in rows_by_group[g]]
        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
        fold_w = torch.tensor(_per_bag_fit_weights(y_np[train_idx], [grp[i] for i in train_idx]), dtype=torch.float32)
        model = train_model(X_train, y_train, input_dim, hidden_dim=hidden_dim, sample_weights=fold_w)
        folds.append((model, cal_groups))

    return folds, None, X_np, rows_by_group, label_by_group


def _compute_fold_orderings_grouped(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    groups: list,
    rng: np.random.RandomState | None,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int | None,
    score_rows_by_group: dict | None = None,
) -> tuple[list[tuple[list[float], list[float]]], float | None]:
    """Bag-aware variant of :func:`compute_fold_orderings`.

    Splits by *group* (a voted image) instead of by row so a Bad bag's flooded
    region negatives never straddle the Train/Calibrate boundary, sizes the
    split over votes not rows, weight-balances each fold fit per-bag, and
    collapses every calibration group to a single max-pooled score (an image
    scores by its best region, as at inference).

    *score_rows_by_group* overrides which rows a calibration group collapses
    over - see :func:`compute_fold_orderings`.
    """
    folds, fallback, X_np, rows_by_group, label_by_group = _grouped_folds(
        X_list, y_list, input_dim, groups, rng, calibrate_count, calibration_fraction, hidden_dim
    )
    if fallback is not None:
        return [], fallback

    orderings: list[tuple[list[float], list[float]]] = []
    for model, cal_groups in folds:
        # Collapse each calibration group to one max-pooled score, so a Good
        # bag and a Bad bag are pooled the same way the scorer pools an image.
        group_scores = _pooled_group_scores(model, cal_groups, rows_by_group, X_np, score_rows_by_group)
        group_labels = [float(label_by_group[g]) for g in cal_groups]
        orderings.append((group_scores, group_labels))

    return orderings, None


def compute_grouped_fold_node_scores(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    groups: list,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
    score_rows_by_group: dict | None = None,
) -> tuple[list[tuple[list[np.ndarray], list[float]]], float | None]:
    """Bag-aware calibration folds, returning each held-out group's node scores.

    Like :func:`_compute_fold_orderings_grouped` but instead of max-pooling every
    calibration group it returns the group's **full node-score vector**, so a
    caller (the #2781 calibration study) can re-pool the same fold models' scores
    under alternative rules (top-k mean, extreme-value ``pnorm``) to recalibrate
    a threshold for a pooling variant without retraining.  ``max`` over each
    returned block reproduces this arm's production threshold exactly.

    Returns ``(fold_node_data, fallback)`` where *fold_node_data* is a list, one
    entry per fold, of ``(group_node_scores, group_labels)`` - *group_node_scores*
    being a list of 1-D float arrays (one per held-out calibration group).
    """
    folds, fallback, X_np, rows_by_group, label_by_group = _grouped_folds(
        X_list, y_list, input_dim, groups, rng, calibrate_count, calibration_fraction, hidden_dim
    )
    if fallback is not None:
        return [], fallback

    fold_node_data: list[tuple[list[np.ndarray], list[float]]] = []
    for model, cal_groups in folds:
        blocks = _group_node_blocks(model, cal_groups, rows_by_group, X_np, score_rows_by_group)
        group_labels = [float(label_by_group[g]) for g in cal_groups]
        fold_node_data.append((blocks, group_labels))
    return fold_node_data, None


def compute_fold_orderings(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
    groups: list | None = None,
    score_rows_by_group: dict | None = None,
) -> tuple[list[tuple[list[float], list[float]]], float | None]:
    """Train the K calibration folds and return their held-out orderings.

    Each ordering is a ``(cal_scores, cal_labels)`` pair: the fold model's
    sigmoid scores on its held-out calibration split, and that split's true
    labels.  Because :func:`train_model` is inclusion-independent, these
    orderings do **not** depend on ``inclusion`` - so they can be cached once
    and re-thresholded at any inclusion via :func:`threshold_from_fold_orderings`
    (and swept across all inclusions for the Stats chart).  See
    docs/plans/find-verification-workflow.md.

    Returns ``(orderings, fallback)``.  When calibration is not possible the
    orderings are empty and ``fallback`` is the sentinel threshold the public
    wrapper must return (mirrors :func:`calculate_cross_calibration_threshold`'s
    historical early-returns); otherwise ``fallback`` is ``None``.

    *groups* activates the **bag-aware** path used when Bad votes are flooded
    into their region nodes: rows sharing a ``groups`` id belong to one voted
    image (a Bad bag's region negatives, or a single Good row) and
    are kept together on one side of every Train/Calibrate split, split counts
    are taken over *groups* (votes) not rows, fold fits are *weights*-balanced,
    and each calibration group collapses to one max-pooled score - matching how
    inference scores an image by its best region.  When *groups* is ``None``
    (every non-flooded caller) the historical row-wise path runs unchanged.

    *score_rows_by_group* (grouped path only) maps each group id to the row
    stack that group should be **scored** over, decoupling "what the fold model
    trains on" from "what a calibration bag collapses to".  It exists because
    the two are not the same whenever a Good vote contributes fewer rows than a
    Bad vote floods: the Good bag then collapses to a max over 1 row while the
    Bad bag - and every image at inference - collapses to a max over N, and
    ``max`` is an upward-biased order statistic, so the calibrated cut
    lands systematically high and the threshold over-rejects positives.  Passing
    each vote's inference rows here puts both classes in the geometry and at the
    width the scorer will actually use.  The production vote / labelset paths
    supply each voted image's full region-node stack
    (:func:`vtscore.detectors.training.inference_score_rows`); ``None`` keeps
    the "collapse over the training rows" behaviour for callers that have no
    inference geometry to offer.
    """
    if groups is not None:
        return _compute_fold_orderings_grouped(
            X_list,
            y_list,
            input_dim,
            groups,
            rng=rng,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            score_rows_by_group=score_rows_by_group,
        )
    n = len(X_list)
    if n < 4:
        return [], 0.5

    _rng = rng if rng is not None else np.random
    X_np = np.array(X_list)
    y_np = np.array(y_list)

    n_cal = max(1, round(n * calibration_fraction))
    n_train = n - n_cal
    if n_train < 2 or n_cal < 1:
        return [], NO_GOOD_THRESHOLD

    pos_idx = np.where(y_np == 1.0)[0]
    neg_idx = np.where(y_np == 0.0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return [], 0.5

    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    calibrate_count = max(1, calibrate_count)

    def _per_class_n_train(class_total: int) -> int:
        target = round(class_total * n_train / n)
        return max(1, min(class_total - 1, target))

    n_train_pos = _per_class_n_train(len(pos_idx))
    n_train_neg = _per_class_n_train(len(neg_idx))

    orderings: list[tuple[list[float], list[float]]] = []
    for _ in range(calibrate_count):
        pos_perm = _rng.permutation(pos_idx)
        neg_perm = _rng.permutation(neg_idx)
        train_idx = np.concatenate([pos_perm[:n_train_pos], neg_perm[:n_train_neg]])
        cal_idx = np.concatenate([pos_perm[n_train_pos:], neg_perm[n_train_neg:]])

        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32)

        model = train_model(X_train, y_train, input_dim, hidden_dim=hidden_dim)

        with torch.no_grad():
            from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

            X_cal = X_cal.to(next(model.parameters()).device)
            # Sanitize non-finite sigmoids (destabilised fold model): the
            # orderings are cached, swept for the Stats chart, and averaged
            # into ``DetectorContext.threshold`` - a NaN here would silently
            # break every downstream ``score >= threshold`` comparison and
            # leak NaN into JSON responses.
            scores = sigmoid_to_finite_scores(model(X_cal))
        orderings.append((scores, y_np[cal_idx].tolist()))

    return orderings, None


def threshold_from_fold_orderings(
    fold_orderings: list[tuple[list[float], list[float]]],
    inclusion_value: int,
) -> float:
    """Apply the conformal inclusion rule to the pooled fold orderings.

    Cheap: pools every fold's cached held-out ``(scores, labels)`` and runs
    :func:`conformal_threshold` once - no fold refits.  Pooling (rather than
    averaging per-fold thresholds) is deliberate: the knob's resolution is
    bounded by the number of calibration scores the quantiles are taken over,
    and per-fold quantiles on a handful of votes each would waste the other
    folds' scores.  All folds' scores live on the same sigmoid scale, so the
    pool is exchangeable enough for the quantile rule.

    Callers must pass a non-empty ``fold_orderings`` (the empty case is
    handled via the ``fallback`` from :func:`compute_fold_orderings`);
    an empty list returns :data:`NO_GOOD_THRESHOLD` defensively.
    """
    if not fold_orderings:
        return NO_GOOD_THRESHOLD
    pooled_scores = [s for scores, _ in fold_orderings for s in scores]
    pooled_labels = [lb for _, labels in fold_orderings for lb in labels]
    return conformal_threshold(pooled_scores, pooled_labels, inclusion_value)


def calculate_cross_calibration_threshold(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    inclusion_value: int = 0,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
    groups: list | None = None,
    score_rows_by_group: dict | None = None,
) -> float:
    """Estimate a decision threshold using k-fold calibration.

    Performs ``calibrate_count`` independent random Train/Calibrate splits.
    For each split, trains a model on the Train portion and scores the
    held-out Calibrate portion.  The pooled held-out scores then feed the
    conformal inclusion rule via :func:`threshold_from_fold_orderings`.

    Algorithm:
        For each of *k* = ``calibrate_count`` rounds:
        1. Stratified random split into Train (``1 - calibration_fraction``)
           and Calibrate (``calibration_fraction``).  Stratification guarantees
           the Train side has at least one of each class, so the per-fold MLP
           fit always has both-class supervision.
        2. Train a model on Train.
        3. Score the Calibrate portion.
        Pool the *k* rounds' held-out (score, label) pairs and apply
        :func:`conformal_threshold` at *inclusion_value*.

    Args:
        X_list: List of embedding arrays (one per labelled example).
        y_list: List of binary labels (1.0 for good, 0.0 for bad),
            aligned with ``X_list``.
        input_dim: Dimensionality of the embeddings.
        inclusion_value: Integer in ``[-10, 10]`` passed to
            :func:`conformal_threshold` to control the miss/false-alarm
            trade-off.  It does **not** enter model training (the fold models
            are inclusion-independent), so the same fold scores can be
            re-thresholded at any inclusion - see
            docs/plans/find-verification-workflow.md.
        rng: Optional seeded RandomState for reproducible splits. Falls back
            to the global ``np.random`` state when ``None``.
        calibrate_count: Number of random Train/Calibrate splits (default 2).
        calibration_fraction: Fraction of data used for calibration in each
            split (default 0.5).  For example, 0.2 means 80% Train / 20%
            Calibrate.  If the fraction is so extreme that a valid split
            cannot be formed (fewer than 2 training or 1 calibration
            examples), returns :data:`NO_GOOD_THRESHOLD` so that nothing
            is predicted as Good.
        hidden_dim: Force a specific hidden-layer width for the fold models.
            When ``None`` (default), each fold model auto-sizes based on its
            own training-set size.  Pass the full-data hidden dim to ensure
            fold models match the final model's architecture.
        score_rows_by_group: Per-group inference row stacks; see
            :func:`compute_fold_orderings`.  Grouped path only.

    Returns:
        A float threshold. Returns 0.5 when calibration is not possible:
        fewer than 4 examples total, or fewer than 2 of either class
        (stratified splitting needs at least one of each class on both
        the train and calibrate sides).  Returns :data:`NO_GOOD_THRESHOLD`
        (a finite sentinel above the sigmoid range) if
        ``calibration_fraction`` makes a valid split impossible.
    """
    orderings, fallback = compute_fold_orderings(
        X_list,
        y_list,
        input_dim,
        rng=rng,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
        groups=groups,
        score_rows_by_group=score_rows_by_group,
    )
    if fallback is not None:
        return fallback
    return threshold_from_fold_orderings(orderings, inclusion_value)


def calculate_safe_threshold(
    xcal_threshold: float,
    all_scores: list[float],
    n_labels: int,
) -> float:
    """Blend cross-calibration and GMM thresholds for robustness with small label counts.

    When few labels are available the cross-calibration threshold can be unreliable.
    This function computes a GMM-based threshold on the full score distribution and
    returns a weighted average of the two, where the weight assigned to x-cal grows
    linearly with the number of labels.

    Blending rules:
        * ``n_labels <= 6``  → pure GMM threshold (the x-cal weight
          ``(n_labels - 6) / 14`` is 0 at exactly 6).
        * ``n_labels >= 20`` → pure x-cal threshold.
        * In between → linear interpolation.

    Args:
        xcal_threshold: The cross-calibrated threshold.
        all_scores: Model output scores for all medias (used for GMM fitting).
        n_labels: Total number of labelled examples (good + bad).

    Returns:
        A finite blended threshold float. If either input is non-finite,
        falls back to the other; if both are non-finite, returns ``0.5``.
        The result is guaranteed finite so it can be safely stored on
        ``DetectorContext.threshold`` without breaking ``score >= threshold``
        comparisons.
    """
    return blend_gmm_threshold(xcal_threshold, calculate_gmm_threshold(all_scores), n_labels)


def safe_blend_weight(n_labels: int) -> float:
    """The x-cal weight of the safe-threshold blend at *n_labels* labels.

    Linear ramp: 0 at 6 labels (pure GMM), 1 at 20 (pure x-cal).
    """
    MIN_LABELS = 6
    MAX_LABELS = 20
    return max(0.0, min(1.0, (n_labels - MIN_LABELS) / (MAX_LABELS - MIN_LABELS)))


def blend_gmm_threshold(xcal_threshold: float, gmm_threshold: float, n_labels: int) -> float:
    """Blend an x-cal and a GMM threshold on the safe-threshold label ramp.

    The blending core of :func:`calculate_safe_threshold`, split out so a
    caller with a pre-computed GMM cut (the #2799 measurement harness re-cuts
    one fitted GMM under several rules) applies the identical ramp and
    finite-guards without re-fitting.
    """
    # Defend against non-finite inputs from either side: an upstream
    # ``calculate_cross_calibration_threshold`` can theoretically still
    # surface inf/NaN, and ``calculate_gmm_threshold`` returns NaN when
    # the model produced non-finite scores. Without these guards the blend
    # below would store NaN on ``DetectorContext.threshold`` and silently
    # break every ``score >= threshold`` comparison downstream.
    xcal_finite = math.isfinite(xcal_threshold)
    gmm_finite = math.isfinite(gmm_threshold)
    if not xcal_finite and not gmm_finite:
        return 0.5
    if not xcal_finite:
        return gmm_threshold
    if not gmm_finite:
        return xcal_threshold

    label_weight = safe_blend_weight(n_labels)
    blended = label_weight * xcal_threshold + (1.0 - label_weight) * gmm_threshold
    if not math.isfinite(blended):
        return 0.5
    return blended
