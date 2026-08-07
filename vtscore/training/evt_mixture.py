"""Gumbel + Normal score mixture — the extreme-value cut (issues #2836, #2846).

A region-voted media's score is the **max** over ~24 region nodes, so even a
thoroughly Bad image gets ~24 draws at a false positive and its score is an
extreme-value statistic: right-skewed, with a heavy upper tail.  The production
GMM fits that mode with a *Gaussian*, which has neither property, and then solves
for a cut against the resulting fiction — in exactly the region of the axis (the
Bad component's right tail) where the cut lands.

This module fits the same 2-component mixture with one component replaced by a
**Gumbel**, the limiting distribution of a maximum of light-tailed draws.  It is
deliberately a sibling of :class:`~vtscore.training.thresholds.GmmFit1D`: same
mode-ordering vocabulary (``lo`` is the Bad mode), same ``lam``-tilted crossing
family (see :meth:`GumbelNormalFit1D.crossing`), so a cut rule can be swapped
between the two families without changing anything else.

The two repairs it enables are related but separable: a heavier, right-skewed low
component also absorbs the "one strong region" Bad images that a Gaussian low
component cannot, which leaves the high component closer to the true positives.
So this addresses the *misspecification* and the *component-identity* hypotheses
at once, while ``lam`` addresses the *loss* hypothesis independently.

**Which side the Gumbel lands on is not fixed** (issue #2846).  #2836 pinned the
Gumbel to the *low* component from the region-voting argument and discarded any
fit that converged the other way; on production-like samples that discarded 14 %
of fits outright, and it is the largest single reason the Gumbel arm silently
degraded to the midpoint.  The premise does not survive the arithmetic: a sim set
is 95–99 % negatives, so the right-skewed max-pooled *bulk* **is** the negative
class, and EM putting the Gumbel on the upper mode is EM preferring the better
description rather than a fit to throw away.  So the fit records which component
landed where (:attr:`GumbelNormalFit1D.gumbel_is_low`) and the crossing solves in
whichever orientation it got; ``allow_swapped=False`` reproduces #2836's
behaviour for the incumbent variants that still have to be measured against it.

Both orientations admit the *same* solver, and exactly.  Between the two modes
the log-density difference is monotone::

    d/dx [log g - log n]  =  (e^{-z} - 1)/scale  +  (x - mu)/var,   z = (x-loc)/scale

with ``z >= 0`` and ``x <= mu`` on ``[loc, mu]``, so both terms are ``<= 0``
(and both flip sign together in the swapped orientation).  The difference
therefore crosses zero at most once in the interval: bisection cannot miss a
root, and "no sign change at the endpoints" really does mean "no root here"
rather than "the bracket was unlucky".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

#: Floors that keep a degenerate M-step from producing a spike component.
_MIN_SCALE = 1e-9
_MIN_VAR = 1e-12
#: Fixed-point / EM iteration budgets.  Both converge in well under these on real
#: score samples; the caps only bound the pathological cases.  The tolerances are
#: deliberately looser than float precision: this fit runs once per step per cell
#: (and once per replicate in the theory bench), and the last few digits of a
#: scale parameter cannot move a cut by anything a threshold comparison can see.
_SCALE_ITERS = 100
_SCALE_TOL = 1e-10
_EM_ITERS = 100
_EM_TOL = 1e-8
#: Bisection steps for the crossing root.  60 halvings take a unit interval to
#: ~1e-18, i.e. below float64 resolution on a [0, 1] score axis.
_BISECT_ITERS = 60
#: ``exp(-z)`` overflows float64 at ``z < -709``.  Clip a hair below that: the
#: Gumbel log density is ``-1e300``-ish there, i.e. exactly the ``-inf`` the
#: overflow would produce, without the warning.
_MAX_NEG_Z = 700.0


def _gumbel_logpdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    """Log density of ``Gumbel(loc, scale)`` (maximum convention, mode at *loc*).

    ``z`` is clipped below at ``-_MAX_NEG_Z``: far under the location the density
    is already zero to every digit float64 has, and leaving ``exp(-z)`` to
    overflow to ``inf`` only trades an exact ``-inf`` for the same ``-inf`` plus a
    RuntimeWarning on every fit.
    """
    z = np.clip((x - loc) / scale, -_MAX_NEG_Z, None)
    return -math.log(scale) - z - np.exp(-z)


def _normal_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    return -0.5 * (math.log(2.0 * math.pi * var) + (x - mu) ** 2 / var)


def _weighted_gumbel_mle(x: np.ndarray, w: np.ndarray, init_scale: float | None = None) -> tuple[float, float] | None:
    """Weighted MLE ``(loc, scale)`` for a Gumbel, by the standard fixed point.

    The score equations reduce to a single fixed point in the scale,
    ``scale = xbar_w - sum(w x e^{-x/scale}) / sum(w e^{-x/scale})``, after which
    ``loc = scale * ln(W / sum(w e^{-x/scale}))``.  Both sums are evaluated in a
    max-shifted form: scores live in ``[0, 1]`` and the scale can be small, so
    ``e^{-x/scale}`` underflows to zero for every point unless the largest
    exponent is factored out first (the ratio is shift-invariant, and the shift
    re-enters ``loc`` as an additive term).

    *init_scale* warm-starts the fixed point.  This is not a micro-optimisation:
    the M-step runs inside an EM loop where the scale barely moves between
    iterations, so warm-starting collapses the inner loop to one or two passes
    and takes the whole fit from ``O(EM * SCALE_ITERS * n)`` to about
    ``O(EM * n)`` - the difference between a fit that costs a minute and one
    that costs a moment, at every step of every cell.
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
    if init_scale is not None and math.isfinite(init_scale) and init_scale > _MIN_SCALE:
        scale = init_scale

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
        converged = abs(nxt - scale) <= _SCALE_TOL * max(1.0, scale)
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


#: Why a crossing solve produced no root.  ``"ok"`` means it did.
#:
#: ``modes_swapped`` is the #2846 case: the fit is fine, EM simply put the Gumbel
#: on the upper mode, and only ``allow_swapped=False`` calls reject it.  The two
#: ``owns`` reasons are the genuine "this fit has no Bad-then-Good boundary"
#: verdicts and are named by *which* endpoint failed, because they point at
#: different repairs: ``hi_owns_lo_mode`` is a near-collapsed fit (the components
#: sit on top of each other), ``lo_owns_hi_mode`` is a low component whose tail
#: swamps the high mode.
CROSSING_REASONS: tuple[str, ...] = (
    "ok",
    "degenerate_params",
    "modes_swapped",
    "modes_not_ordered",
    "nonfinite",
    "hi_owns_lo_mode",
    "lo_owns_hi_mode",
    "both_ends_wrong",
)


def _endpoint_reason(f_lo: float, f_hi: float) -> str:
    """Classify the log-density difference at the two modes.

    ``"ok"`` iff it enters positive and leaves negative — Bad owning its own mode
    and Good owning its own.  Because the difference is monotone across the
    interval (see the module docstring), that is not merely a bracketing
    convenience: it is exactly the condition for a root to exist there.
    """
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return "nonfinite"
    if f_lo <= 0.0 and f_hi >= 0.0:
        return "both_ends_wrong"
    if f_lo <= 0.0:
        return "hi_owns_lo_mode"
    if f_hi >= 0.0:
        return "lo_owns_hi_mode"
    return "ok"


@dataclass(frozen=True)
class GumbelNormalFit1D:
    """A fitted ``w_gumbel*Gumbel(loc, scale) + w_normal*N(mu, var)`` mixture.

    Stored **by component**, not by mode: which of the two ends up lower is an
    outcome of the fit (see the module docstring), so the low/high vocabulary the
    cut rules speak is derived from ``loc`` vs ``mu`` rather than baked into the
    field names.

    ``mean_loglik`` is the per-point log likelihood of the sample it was fitted
    on, so it can be compared directly against the Gaussian mixture's — the
    likelihood-ratio evidence for the misspecification hypothesis.
    """

    w_gumbel: float
    loc: float
    scale: float
    w_normal: float
    mu: float
    var: float
    mean_loglik: float

    @property
    def gumbel_is_low(self) -> bool:
        """Whether the Gumbel is the low (Bad) mode — #2836's assumption, not a given."""
        return self.loc < self.mu

    @property
    def mode_lo(self) -> float:
        """The Bad component's mode (a Gumbel's mode is its location)."""
        return min(self.loc, self.mu)

    @property
    def mode_hi(self) -> float:
        """The Good component's mode."""
        return max(self.loc, self.mu)

    @property
    def w_lo(self) -> float:
        return self.w_gumbel if self.gumbel_is_low else self.w_normal

    @property
    def w_hi(self) -> float:
        return self.w_normal if self.gumbel_is_low else self.w_gumbel

    def _log_lo(self, x: np.ndarray) -> np.ndarray:
        """Log density of the low component, whichever family it turned out to be."""
        if self.gumbel_is_low:
            return _gumbel_logpdf(x, self.loc, self.scale)
        return _normal_logpdf(x, self.mu, self.var)

    def _log_hi(self, x: np.ndarray) -> np.ndarray:
        if self.gumbel_is_low:
            return _normal_logpdf(x, self.mu, self.var)
        return _gumbel_logpdf(x, self.loc, self.scale)

    def crossing_state(self, lam: float = 1.0, *, allow_swapped: bool = False) -> tuple[str, float | None]:
        """``(reason, cut)`` for ``w_lo*f_lo(x) == lam*w_hi*f_hi(x)`` between the modes.

        The reason is one of :data:`CROSSING_REASONS`; the cut is ``None`` for
        every reason but ``"ok"``.  Exposed alongside :meth:`crossing` so the
        calibration harness can record *why* a rule declined to fire instead of
        only that it did (issue #2846) — a rule that falls back has to stay
        visible, and "no root" and "wrong orientation" want opposite repairs.

        No closed form here (the Gaussian pair's quadratic does not survive the
        double exponential), so this bisects the log-density difference on
        ``[mode_lo, mode_hi]``.  That difference is monotone across the interval
        in either orientation (see the module docstring), so a sign change at the
        endpoints is necessary *and* sufficient and the bisection is exact.
        """
        if not (self.w_lo > 0.0 and self.w_hi > 0.0 and lam > 0.0 and self.scale > 0.0 and self.var > 0.0):
            return "degenerate_params", None
        if not allow_swapped and not self.gumbel_is_low:
            return "modes_swapped", None
        lo, hi = self.mode_lo, self.mode_hi
        if not (hi > lo):
            return "modes_not_ordered", None

        def diff(x: float) -> float:
            arr = np.array([x], dtype=np.float64)
            lo_term = math.log(self.w_lo) + float(self._log_lo(arr)[0])
            hi_term = math.log(lam * self.w_hi) + float(self._log_hi(arr)[0])
            return lo_term - hi_term

        endpoints = _endpoint_reason(diff(lo), diff(hi))
        if endpoints != "ok":
            return endpoints, None
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            if diff(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return "ok", 0.5 * (lo + hi)

    def crossing(self, lam: float = 1.0, *, allow_swapped: bool = False) -> float | None:
        """Root of ``w_lo*f_lo(x) == lam*w_hi*f_hi(x)``; ``None`` when there is none.

        ``allow_swapped=False`` additionally declines any fit whose Gumbel is the
        *high* component, which is #2836's shipped behaviour and is kept so the
        incumbent ``gumbel_*`` variants stay measurable against the #2846 repair.
        """
        return self.crossing_state(lam, allow_swapped=allow_swapped)[1]

    def rate_crossing(
        self, fpr_weight: float = 1.0, fnr_weight: float = 1.0, *, allow_swapped: bool = False
    ) -> float | None:
        """Prior-free crossing scaled by the cost ratio; see :meth:`GmmFit1D.rate_crossing`."""
        return self.rate_crossing_state(fpr_weight, fnr_weight, allow_swapped=allow_swapped)[1]

    def rate_crossing_state(
        self, fpr_weight: float = 1.0, fnr_weight: float = 1.0, *, allow_swapped: bool = False
    ) -> tuple[str, float | None]:
        """:meth:`crossing_state` at the rate-optimal tilt."""
        if not (fpr_weight > 0.0 and fnr_weight > 0.0 and self.w_hi > 0.0):
            return "degenerate_params", None
        lam = (fnr_weight / fpr_weight) * (self.w_lo / self.w_hi)
        return self.crossing_state(lam, allow_swapped=allow_swapped)

    def lo_survival(self, x: float) -> float:
        """``P(Bad component > x)`` — the FPR this cut implies under the fitted Bad mode.

        Reads whichever component is the low one, so a swapped fit reports the
        Normal's tail rather than silently reporting the Gumbel's from the wrong
        end of the axis.
        """
        if self.gumbel_is_low:
            if self.scale <= 0.0:
                return float("nan")
            z = min(max((x - self.loc) / self.scale, -_MAX_NEG_Z), _MAX_NEG_Z)
            return float(-np.expm1(-math.exp(-z)))
        if self.var <= 0.0:
            return float("nan")
        return float(0.5 * math.erfc((x - self.mu) / math.sqrt(2.0 * self.var)))

    def lo_quantile(self, alpha: float) -> float | None:
        """The ``x`` where :meth:`lo_survival` equals *alpha*; ``None`` if there is none.

        The exact inverse of :meth:`lo_survival`, and deliberately its immediate
        neighbour: the two branch on ``gumbel_is_low`` the same way, and a
        quantile that read the Gumbel's tail off a fit whose low component is the
        *Normal* would be wrong in the silent, plausible way this study line keeps
        producing.  Both branches are closed form, so there is no bracket to get
        wrong and no bisection to converge:

        * Gumbel low — ``S(x) = 1 - exp(-exp(-z))``, ``z = (x - loc)/scale``, so
          ``x = loc - scale*ln(-ln(1 - alpha))``.  At ``alpha = 0.158`` that is
          ``loc + 1.761*scale``.
        * Normal low (the swapped fit) — ``S(x) = Phi((mu - x)/sd)``, so
          ``x = mu + sd*Phi^-1(1 - alpha)``.

        Unlike a crossing, this always exists for a non-degenerate fit: a tail
        quantile needs no Bad-then-Good boundary between the modes, which is the
        entire reason the one-constant rule is worth measuring after the crossing
        family's fallback rate sank it (issues #2846, #2881).  The units are the
        fit's own — logit space, wherever the caller fitted it — so the caller
        maps back exactly as it does for a crossing.
        """
        if not (0.0 < alpha < 1.0):
            return None
        if self.gumbel_is_low:
            if not (self.scale > 0.0 and math.isfinite(self.loc)):
                return None
            x = self.loc - self.scale * math.log(-math.log1p(-alpha))
        else:
            if not (self.var > 0.0 and math.isfinite(self.mu)):
                return None
            x = self.mu + math.sqrt(self.var) * NormalDist().inv_cdf(1.0 - alpha)
        return x if math.isfinite(x) else None


#: Why the EM fit produced nothing.  ``"ok"`` means it produced a fit.
#:
#: Note what is *absent*: #2836's ``mu_hi > loc_lo`` rejection.  That branch was
#: essentially the entire failure rate (14 % of production-like fits against
#: 0.1 % for every numerical cause combined), and it was rejecting sound fits;
#: it now surfaces as :attr:`GumbelNormalFit1D.gumbel_is_low` for the caller to
#: act on, rather than destroying the fit.  See the module docstring.
FIT_FAILURES: tuple[str, ...] = (
    "ok",
    "too_few",
    "gumbel_mle_failed",
    "hi_mass_collapsed",
    "denom_nonfinite",
    "nonfinite_params",
)


def fit_gumbel_normal_mixture(
    arr: np.ndarray,
    *,
    init_split: float | None = None,
) -> GumbelNormalFit1D | None:
    """EM-fit a Gumbel + Normal mixture to a 1-D score sample; ``None`` on failure."""
    return fit_gumbel_normal_mixture_state(arr, init_split=init_split)[1]


def fit_gumbel_normal_mixture_state(
    arr: np.ndarray,
    *,
    init_split: float | None = None,
) -> tuple[str, GumbelNormalFit1D | None]:
    """``(reason, fit)`` — EM-fit a Gumbel + Normal mixture to a 1-D score sample.

    *init_split* seeds the responsibilities by a hard split at that score (pass
    the Gaussian GMM's midpoint so both families start from the same place and
    the likelihood comparison is not an artefact of initialisation); without it
    the sample median is used.  The reason is one of :data:`FIT_FAILURES`, so a
    caller that has to report *why* a step produced no EVT cut can tell an
    unfittable sample from a fit that simply has no crossing (issue #2846).

    The fit is returned whichever way round the components land — the caller
    decides what to do with a swapped one, exactly as it decides the fallback for
    :func:`~vtscore.training.thresholds.fit_score_gmm` returning ``None``.
    """
    x = np.asarray(arr, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 4:
        return "too_few", None

    split = float(np.median(x)) if init_split is None else float(init_split)
    # Seeded as the low component, which is where the region-voting argument
    # expects the Gumbel; EM is free to walk it to the other mode from there.
    r = np.where(x < split, 1.0, 0.0)  # responsibility for the Gumbel
    # A hard split that puts everything on one side leaves the other component
    # unfittable; nudge it to a soft split instead.
    if r.sum() < 2.0 or (1.0 - r).sum() < 2.0:
        r = np.full(x.shape, 0.5)

    prev_ll = -math.inf
    w_gumbel = float(np.clip(r.mean(), 1e-6, 1.0 - 1e-6))
    loc = scale = mu = var = float("nan")
    ll = -math.inf
    for _ in range(_EM_ITERS):
        # --- M step (warm-started from the previous iteration's scale) ---
        gum = _weighted_gumbel_mle(x, r, init_scale=None if math.isnan(scale) else scale)
        if gum is None:
            return "gumbel_mle_failed", None
        loc, scale = gum
        scale = max(scale, _MIN_SCALE)
        normal_mass = float((1.0 - r).sum())
        if normal_mass <= 0.0:
            return "hi_mass_collapsed", None
        mu = float(((1.0 - r) * x).sum() / normal_mass)
        var = max(float(((1.0 - r) * (x - mu) ** 2).sum() / normal_mass), _MIN_VAR)
        w_gumbel = float(np.clip(r.mean(), 1e-9, 1.0 - 1e-9))
        w_normal = 1.0 - w_gumbel

        # --- E step (log-sum-exp for stability) ---
        log_g = math.log(w_gumbel) + _gumbel_logpdf(x, loc, scale)
        log_n = math.log(w_normal) + _normal_logpdf(x, mu, var)
        m = np.maximum(log_g, log_n)
        denom = m + np.log(np.exp(log_g - m) + np.exp(log_n - m))
        if not np.all(np.isfinite(denom)):
            return "denom_nonfinite", None
        r = np.exp(log_g - denom)
        ll = float(denom.mean())
        # ``prev_ll`` starts at -inf, where both sides of the tolerance test are
        # inf and ``inf <= inf`` would declare convergence after a single M/E
        # step - leaving the fit at its initialisation.  Only test once there is
        # a finite previous value to compare against.
        if math.isfinite(prev_ll) and abs(ll - prev_ll) <= _EM_TOL * max(1.0, abs(prev_ll)):
            break
        prev_ll = ll

    if not (math.isfinite(ll) and math.isfinite(loc) and math.isfinite(mu)):
        return "nonfinite_params", None
    return "ok", GumbelNormalFit1D(
        w_gumbel=w_gumbel,
        loc=loc,
        scale=scale,
        w_normal=1.0 - w_gumbel,
        mu=mu,
        var=var,
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
