"""Gumbel(low) + Normal(high) score mixture — the extreme-value cut (issue #2836).

A region-voted media's score is the **max** over ~24 region nodes, so even a
thoroughly Bad image gets ~24 draws at a false positive and its score is an
extreme-value statistic: right-skewed, with a heavy upper tail.  The production
GMM fits that mode with a *Gaussian*, which has neither property, and then solves
for a cut against the resulting fiction — in exactly the region of the axis (the
Bad component's right tail) where the cut lands.

This module fits the same 2-component mixture with the low component replaced by
a **Gumbel**, the limiting distribution of a maximum of light-tailed draws.  It
is deliberately a drop-in sibling of :class:`~vtscore.training.thresholds.GmmFit1D`:
same ordering convention (``lo`` is the Bad mode), same ``lam``-tilted crossing
family (see :meth:`GumbelNormalFit1D.crossing`), so a cut rule can be swapped
between the two families without changing anything else.

The two repairs it enables are related but separable: a heavier, right-skewed low
component also absorbs the "one strong region" Bad images that a Gaussian low
component cannot, which leaves the high component closer to the true positives.
So this addresses the *misspecification* and the *component-identity* hypotheses
at once, while ``lam`` addresses the *loss* hypothesis independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Floors that keep a degenerate M-step from producing a spike component.
_MIN_SCALE = 1e-9
_MIN_VAR = 1e-12
#: Fixed-point / EM iteration budgets.  Both converge in well under these on
#: real score samples; the caps only bound the pathological cases.
_SCALE_ITERS = 200
_EM_ITERS = 200
_EM_TOL = 1e-9
#: Bisection steps for the crossing root.  60 halvings take a unit interval to
#: ~1e-18, i.e. below float64 resolution on a [0, 1] score axis.
_BISECT_ITERS = 60


def _gumbel_logpdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    """Log density of ``Gumbel(loc, scale)`` (maximum convention, mode at *loc*)."""
    z = (x - loc) / scale
    return -math.log(scale) - z - np.exp(-z)


def _normal_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    return -0.5 * (math.log(2.0 * math.pi * var) + (x - mu) ** 2 / var)


def _weighted_gumbel_mle(x: np.ndarray, w: np.ndarray) -> tuple[float, float] | None:
    """Weighted MLE ``(loc, scale)`` for a Gumbel, by the standard fixed point.

    The score equations reduce to a single fixed point in the scale,
    ``scale = xbar_w - sum(w x e^{-x/scale}) / sum(w e^{-x/scale})``, after which
    ``loc = scale * ln(W / sum(w e^{-x/scale}))``.  Both sums are evaluated in a
    max-shifted form: scores live in ``[0, 1]`` and the scale can be small, so
    ``e^{-x/scale}`` underflows to zero for every point unless the largest
    exponent is factored out first (the ratio is shift-invariant, and the shift
    re-enters ``loc`` as an additive term).
    """
    total = float(w.sum())
    if total <= 0.0 or x.size == 0:
        return None
    xbar = float((w * x).sum() / total)
    var = float((w * (x - xbar) ** 2).sum() / total)
    if not (var > 0.0 and math.isfinite(var)):
        return None
    # Method-of-moments start: Var(Gumbel) = pi^2 scale^2 / 6.
    scale = max(math.sqrt(6.0 * var) / math.pi, _MIN_SCALE)

    for _ in range(_SCALE_ITERS):
        u = -x / scale
        u_max = float(u.max())
        e = w * np.exp(u - u_max)
        s = float(e.sum())
        if not (s > 0.0 and math.isfinite(s)):
            return None
        nxt = xbar - float((e * x).sum()) / s
        if not (math.isfinite(nxt) and nxt > 0.0):
            return None
        nxt = max(nxt, _MIN_SCALE)
        converged = abs(nxt - scale) <= 1e-12 * max(1.0, scale)
        scale = nxt
        if converged:
            break

    u = -x / scale
    u_max = float(u.max())
    s = float((w * np.exp(u - u_max)).sum())
    if not (s > 0.0 and math.isfinite(s)):
        return None
    # loc = scale * (ln W - ln sum(w e^{-x/scale})), undoing the max shift.
    loc = scale * (math.log(total) - (u_max + math.log(s)))
    if not (math.isfinite(loc) and math.isfinite(scale)):
        return None
    return loc, scale


@dataclass(frozen=True)
class GumbelNormalFit1D:
    """A fitted ``w_lo*Gumbel(loc_lo, scale_lo) + w_hi*N(mu_hi, var_hi)`` mixture.

    ``mean_loglik`` is the per-point log likelihood of the sample it was fitted
    on, so it can be compared directly against the Gaussian mixture's — the
    likelihood-ratio evidence for the misspecification hypothesis.
    """

    w_lo: float
    loc_lo: float
    scale_lo: float
    w_hi: float
    mu_hi: float
    var_hi: float
    mean_loglik: float

    @property
    def mode_lo(self) -> float:
        """The Bad component's mode (a Gumbel's mode is its location)."""
        return self.loc_lo

    def crossing(self, lam: float = 1.0) -> float | None:
        """Root of ``w_lo*Gumbel(x) == lam*w_hi*N(x)`` between the two modes.

        No closed form here (the Gaussian pair's quadratic does not survive the
        double exponential), so this bisects the log-density difference on
        ``[mode_lo, mu_hi]``.  Returns ``None`` unless that difference actually
        changes sign across the interval — the same "this fit does not express a
        Bad-then-Good boundary" guard :func:`~vtscore.training.thresholds._weighted_gaussian_crossing`
        applies.
        """
        if not (self.w_lo > 0.0 and self.w_hi > 0.0 and lam > 0.0 and self.scale_lo > 0.0 and self.var_hi > 0.0):
            return None
        lo, hi = self.mode_lo, self.mu_hi
        if not (hi > lo):
            return None

        def diff(x: float) -> float:
            arr = np.array([x], dtype=np.float64)
            lo_term = math.log(self.w_lo) + float(_gumbel_logpdf(arr, self.loc_lo, self.scale_lo)[0])
            hi_term = math.log(lam * self.w_hi) + float(_normal_logpdf(arr, self.mu_hi, self.var_hi)[0])
            return lo_term - hi_term

        f_lo, f_hi = diff(lo), diff(hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
            return None
        # Bad must own its own mode and Good its own; otherwise there is no
        # crossing to find and the caller should fall back.
        if not (f_lo > 0.0 > f_hi):
            return None
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            if diff(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def rate_crossing(self, fpr_weight: float = 1.0, fnr_weight: float = 1.0) -> float | None:
        """Prior-free crossing scaled by the cost ratio; see :meth:`GmmFit1D.rate_crossing`."""
        if not (fpr_weight > 0.0 and fnr_weight > 0.0 and self.w_hi > 0.0):
            return None
        return self.crossing(lam=(fnr_weight / fpr_weight) * (self.w_lo / self.w_hi))

    def lo_survival(self, x: float) -> float:
        """``P(Bad component > x)`` — the FPR this cut implies under the fitted Bad mode."""
        if self.scale_lo <= 0.0:
            return float("nan")
        return float(-np.expm1(-math.exp(-(x - self.loc_lo) / self.scale_lo)))


def fit_gumbel_normal_mixture(
    arr: np.ndarray,
    *,
    init_split: float | None = None,
) -> GumbelNormalFit1D | None:
    """EM-fit a Gumbel(low) + Normal(high) mixture to a 1-D score sample.

    *init_split* seeds the responsibilities by a hard split at that score (pass
    the Gaussian GMM's midpoint so both families start from the same place and
    the likelihood comparison is not an artefact of initialisation); without it
    the sample median is used.  Returns ``None`` when the sample is too small,
    an M-step degenerates, or the fit loses the ordering that makes ``lo`` the
    Bad mode — the caller decides the fallback, exactly as with
    :func:`~vtscore.training.thresholds.fit_score_gmm`.
    """
    x = np.asarray(arr, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 4:
        return None

    split = float(np.median(x)) if init_split is None else float(init_split)
    r = np.where(x < split, 1.0, 0.0)  # responsibility for the low component
    # A hard split that puts everything on one side leaves the other component
    # unfittable; nudge it to a soft split instead.
    if r.sum() < 2.0 or (1.0 - r).sum() < 2.0:
        r = np.full(x.shape, 0.5)

    prev_ll = -math.inf
    w_lo = float(np.clip(r.mean(), 1e-6, 1.0 - 1e-6))
    loc_lo = scale_lo = mu_hi = var_hi = float("nan")
    ll = -math.inf
    for _ in range(_EM_ITERS):
        # --- M step ---
        gum = _weighted_gumbel_mle(x, r)
        if gum is None:
            return None
        loc_lo, scale_lo = gum
        scale_lo = max(scale_lo, _MIN_SCALE)
        w_hi_mass = float((1.0 - r).sum())
        if w_hi_mass <= 0.0:
            return None
        mu_hi = float(((1.0 - r) * x).sum() / w_hi_mass)
        var_hi = max(float(((1.0 - r) * (x - mu_hi) ** 2).sum() / w_hi_mass), _MIN_VAR)
        w_lo = float(np.clip(r.mean(), 1e-9, 1.0 - 1e-9))
        w_hi = 1.0 - w_lo

        # --- E step (log-sum-exp for stability) ---
        log_lo = math.log(w_lo) + _gumbel_logpdf(x, loc_lo, scale_lo)
        log_hi = math.log(w_hi) + _normal_logpdf(x, mu_hi, var_hi)
        m = np.maximum(log_lo, log_hi)
        denom = m + np.log(np.exp(log_lo - m) + np.exp(log_hi - m))
        if not np.all(np.isfinite(denom)):
            return None
        r = np.exp(log_lo - denom)
        ll = float(denom.mean())
        if abs(ll - prev_ll) <= _EM_TOL * max(1.0, abs(prev_ll)):
            break
        prev_ll = ll

    if not (math.isfinite(ll) and math.isfinite(loc_lo) and math.isfinite(mu_hi)):
        return None
    # "lo" must still be the low mode; a fit that swapped them is not describing
    # a Bad-then-Good axis and has no cut to offer.
    if not (mu_hi > loc_lo):
        return None
    return GumbelNormalFit1D(
        w_lo=w_lo,
        loc_lo=loc_lo,
        scale_lo=scale_lo,
        w_hi=1.0 - w_lo,
        mu_hi=mu_hi,
        var_hi=var_hi,
        mean_loglik=ll,
    )


def gaussian_mixture_mean_loglik(arr: np.ndarray, fit: object) -> float:
    """Per-point log likelihood of *arr* under a :class:`GmmFit1D`.

    Lives here rather than on ``GmmFit1D`` because its only consumer is the
    likelihood comparison against :class:`GumbelNormalFit1D` — "is the Gaussian
    low component actually a worse description of this sample?".
    """
    x = np.asarray(arr, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    w_lo = float(getattr(fit, "w_lo"))  # noqa: B009 - duck-typed across fit classes
    w_hi = float(getattr(fit, "w_hi"))  # noqa: B009
    if not (w_lo > 0.0 and w_hi > 0.0):
        return float("nan")
    log_lo = math.log(w_lo) + _normal_logpdf(x, float(fit.mu_lo), float(fit.var_lo))  # type: ignore[attr-defined]
    log_hi = math.log(w_hi) + _normal_logpdf(x, float(fit.mu_hi), float(fit.var_hi))  # type: ignore[attr-defined]
    m = np.maximum(log_lo, log_hi)
    return float((m + np.log(np.exp(log_lo - m) + np.exp(log_hi - m))).mean())
