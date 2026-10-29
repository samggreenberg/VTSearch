"""Fitting the 1-D two-component score mixture by EM, plain and label-anchored.

:func:`fit_score_gmm` fits two Gaussians to a score sample from a deterministic
2-means start (:func:`_two_means_init`); :func:`fit_anchored_score_gmm` refits
that mixture with the voted items' component membership clamped (issue #2852),
and :func:`anchored_gmm_fit` applies the fallback policy around it.  Both run
one loop, :func:`_anchored_em` - the plain fit is that loop with no anchors
(:func:`_plain_em`) - and keeping them in one module is what stops them
drifting into two estimators of the same mixture.
:func:`fit_score_gmm_sklearn` is the pre-#3585 reference the native fit is
measured against.

The stopping rules (``_EM_*``, ``_ANCHORED_EM_*``) are module globals read at
call time, so a study or a test installs another rule by patching **this**
module.  Reads :mod:`~vtscore.training.thresholds.gmm_core` only.
"""

from __future__ import annotations

import math

import numpy as np

from vtscore.training.thresholds.gmm_core import GmmFit1D, gmm_fit_array


#: Iteration cap for the unanchored EM (:func:`fit_score_gmm`), and the
#: parameter-delta tolerance it would stop on if the likelihood rule below were
#: switched off.  Both are sklearn's own values for the fit this replaced, so
#: "how long may it run" did not change with the implementation.  The tolerance
#: is inert on the shipped path - :func:`_plain_em` passes
#: :data:`_EM_LOGLIK_TOL`, and :func:`_anchored_em` reads one rule or the other,
#: never both - and is kept so the parameter rule stays reachable as a measured
#: arm (``scripts/experiments/gmm_init/arms_3585.py``).
_EM_MAX_ITER = 100
_EM_TOL = 1e-8

#: Convergence tolerance for the unanchored fit, on the **mean log-likelihood**
#: rather than on the parameters: stop when an iteration improves the objective
#: by less than this.  Set to sklearn's own default, so the fit this replaced
#: and the fit that replaced it stop at the same place in the same sense -
#: which is what lets the two be compared as implementations of one estimator
#: instead of as two estimators.  See :func:`_anchored_em` for why the
#: parameter-delta rule is the wrong one here (a barely bimodal sort crawls a
#: flat ridge for 200 iterations to buy 0.004 nats).
_EM_LOGLIK_TOL = 1e-3

#: Iteration cap for the 2-means init below.  Lloyd's on a sorted 1-D sample
#: converges in a handful of boundary moves and each one costs a binary search,
#: so this is a runaway guard rather than a budget.
_KMEANS_MAX_ITER = 100

#: The empty anchor arrays :func:`_plain_em` hands :func:`_anchored_em`.  Module
#: level so a fit does not allocate two throwaway arrays; never written to (the
#: loop only reads ``.sum()`` and ``(a - mu) ** 2`` off them).
_NO_ANCHORS = np.empty(0, dtype=np.float64)


def _two_means_init(xs: np.ndarray) -> GmmFit1D | None:
    """Deterministic 2-means split of the **sorted** sample *xs*, as EM's start.

    Exact Lloyd's algorithm for ``k = 2`` in one dimension.  A 2-means
    assignment in 1-D is an *interval* split, so an iteration is a
    :func:`numpy.searchsorted` for the midpoint of the two centres followed by
    two O(1) prefix-sum means - not a pass over the data.  The whole
    initialiser is therefore one sort and one pair of cumulative sums, and its
    cost does not grow with the iteration count the way sklearn's
    ``init_params="kmeans"`` does, where a third of the initialiser's time went
    at 50k scores (issue #3585).

    Started from the sample's **min and max** rather than from sampled points,
    so it is deterministic without an RNG - no ``random_state`` to thread
    through, and the same scores give the same init on every machine.  Scores
    are bounded (a sigmoid or a cosine), so the pair is not an outlier artefact,
    and it is the start that survives a *saturated* distribution: quartiles
    coincide as soon as one class holds under 25% of the sample, which on the
    #3166 shape is exactly the case the split has to find.

    Returns ``None`` only when there is nothing to fit at all: fewer than two
    scores, or a non-finite sample.  A **constant** sample has no split to find,
    but it does have an answer - two identical components sitting on the value -
    and that is what it gets, because the anchored fit initialises from here and
    its anchors are what pull such a haystack apart.  (sklearn's initialiser
    answered a constant sample with a phantom second component at zero, so
    ``calculate_gmm_threshold`` on a haystack of 0.3s returned 0.15.)
    """
    n = int(xs.size)
    if n < 2:
        return None
    lo_c, hi_c = float(xs[0]), float(xs[-1])
    if not (math.isfinite(lo_c) and math.isfinite(hi_c)) or not math.isfinite(float(xs.sum())):
        return None
    if hi_c <= lo_c:
        return GmmFit1D(w_lo=0.5, mu_lo=lo_c, var_lo=1e-12, w_hi=0.5, mu_hi=lo_c, var_hi=1e-12)

    csum = np.concatenate(([0.0], np.cumsum(xs)))
    csq = np.concatenate(([0.0], np.cumsum(xs * xs)))
    var_all = max(float(csq[n] / n - (csum[n] / n) ** 2), 0.0)
    floor = max(1e-12, _ANCHOR_VAR_FLOOR_FRAC * var_all)

    split = 0
    for _ in range(_KMEANS_MAX_ITER):
        # Everything below the midpoint of the two centres belongs to the low
        # cluster.  ``side="left"`` ties with the E-step's ``>=`` convention.
        new_split = int(np.searchsorted(xs, 0.5 * (lo_c + hi_c), side="left"))
        # A boundary that empties a cluster is not a 2-means solution; keep the
        # previous centres and stop rather than divide by zero.
        if new_split <= 0 or new_split >= n or new_split == split:
            split = new_split if 0 < new_split < n else split
            break
        split = new_split
        lo_c = float(csum[split] / split)
        hi_c = float((csum[n] - csum[split]) / (n - split))
    if not 0 < split < n:
        # Lloyd's never found a non-degenerate boundary (two distinct values,
        # one of them a single point, and similar edges).  Split the sample in
        # half instead: the EM below is what has to find the components, and an
        # init that names two non-empty groups is all it needs.
        split = n // 2

    def _moments(a: int, b: int) -> tuple[float, float]:
        """``(mean, variance)`` of ``xs[a:b]`` from the prefix sums."""
        m = float(b - a)
        mean = (csum[b] - csum[a]) / m
        # Cancellation in ``E[x^2] - E[x]^2`` can go slightly negative on a
        # tight cluster far from zero; the floor is what the EM applies anyway.
        return mean, max((csq[b] - csq[a]) / m - mean * mean, floor)

    mu_lo, var_lo = _moments(0, split)
    mu_hi, var_hi = _moments(split, n)
    return GmmFit1D(
        w_lo=split / n,
        mu_lo=mu_lo,
        var_lo=var_lo,
        w_hi=(n - split) / n,
        mu_hi=mu_hi,
        var_hi=var_hi,
    )


def _plain_em(x: np.ndarray, init: GmmFit1D) -> GmmFit1D | None:
    """The unanchored EM: :func:`_anchored_em` with no anchors at all.

    With both anchor arrays empty every anchor term in that loop vanishes
    identically - the M-step masses are the responsibility sums, the total mass
    is ``n``, and the anchored sums are zero - so it *is* the plain
    two-component EM, run in the same preallocated-buffer, BLAS-free form.
    Sharing the body is the point: the unanchored fit and the anchored refit it
    initialises cannot drift into two different estimators of the same mixture.
    """
    return _anchored_em(x, _NO_ANCHORS, _NO_ANCHORS, init, 1.0, _EM_MAX_ITER, _EM_TOL, _EM_LOGLIK_TOL)


def fit_score_gmm(arr: np.ndarray) -> GmmFit1D | None:
    """Fit a deterministic 2-component GMM to a 1-D score array.

    Returns ``None`` when the fit fails - fewer than 2 scores, a non-finite
    sample, or an EM failure - leaving the fallback policy to the caller
    (:func:`calculate_gmm_threshold` falls back to the median).  A **constant**
    sample is not a failure: it comes back as two identical components sitting
    on the value, so the cut is that value (see :func:`_two_means_init`).

    **The estimator, and why it is ours** (issue #3585).  Two Gaussians over one
    dimension, fitted by EM from a deterministic 2-means init
    (:func:`_two_means_init`) with the same loop the anchored fit uses
    (:func:`_plain_em`).  Until #3585 this was sklearn's
    ``GaussianMixture(n_components=2, random_state=42)``, which is the same
    estimator of the same model - and was measured at **48% of a whole
    cosine/text sort** (10% after this change, and the sort as a whole 1.7x
    faster), because it pays ``covariance_type="full"`` machinery and a k-means
    init per call on a problem whose covariance is a scalar.  The fit itself is
    **7.3x** cheaper at the sizes a fold sees and **12x** at
    :data:`_GMM_MAX_SAMPLES`.  It is
    kept, unused by production, as :func:`fit_score_gmm_sklearn`: the swap
    *moves the fit* (a different init lands EM in a different place within its
    tolerance, and can land it in a different basin), so it is licensed by a
    measured equivalence rather than by argument, and the reference has to stay
    in the tree for that measurement to be repeatable.  What the study measured
    is in ``docs/experiments/2026-09-13-gmm-init-3585/REPORT.md``.
    """
    x = np.asarray(arr, dtype=np.float64).ravel()
    if x.size < 2:
        return None
    xs = np.sort(x)
    init = _two_means_init(xs)
    if init is None:
        return None
    fit = _plain_em(x, init)
    if fit is None:
        return None
    # EM keeps the init's component order in every case measured, but nothing in
    # the loop enforces it, and every caller reads ``lo``/``hi`` as an ordering.
    if fit.mu_hi < fit.mu_lo:
        return GmmFit1D(
            w_lo=fit.w_hi, mu_lo=fit.mu_hi, var_lo=fit.var_hi, w_hi=fit.w_lo, mu_hi=fit.mu_lo, var_hi=fit.var_lo
        )
    return fit


def fit_score_gmm_sklearn(arr: np.ndarray) -> GmmFit1D | None:
    """The pre-#3585 sklearn fit, kept as the reference the swap is measured against.

    **Not on any production path.**  This is what :func:`fit_score_gmm` was
    until #3585 replaced it with the native EM above, and it stays so that the
    equivalence can be re-measured rather than believed: the gate is an
    admitted-set comparison over real fold haystacks
    (``scripts/experiments/gmm_init/``), and a gate whose baseline lives only in
    a git history cannot be re-run against the next candidate.
    ``tests_lib/sorting/test_gmm_native_fit.py`` holds the standing form of it.
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


# --- Anchored (semi-supervised) mixture estimation: issue #2852 ---------------
#
# The label-anchored mixture fits the same 2-component 1-D Gaussian mixture as
# ``fit_score_gmm`` but on a *partially labelled* sample: the haystack scores
# are free, while each voted item's score has its component membership fixed by
# its label (Good -> high component, Bad -> low component).  This is classical
# semi-supervised ML estimation for a mixture: EM where the E-step clamps the
# labelled points' responsibilities to one-hot and the M-step counts each
# labelled point ``anchor_weight`` times.  ``anchor_weight`` is therefore the
# single fusion knob: the anchors' M-step mass is ``anchor_weight * n_labels``
# against the haystack's ``N``, so with few labels the population dominates
# (the GMM regime) and as labels accumulate the labelled class-conditionals
# take over smoothly - the schedule the safe-blend hand-tunes, derived instead
# from relative likelihood mass.

#: Default per-anchor multiplicity for the anchored EM: each labelled score
#: counts as this many haystack scores in the M-step.  Chosen so a handful of
#: votes is already visible against the ``_GMM_MAX_SAMPLES``-sized haystack
#: sample without drowning it; the #2852 experiment sweeps this.
ANCHOR_WEIGHT_DEFAULT = 10.0

#: Relative variance floor for the anchored EM, as a fraction of the total
#: sample variance.  A component pinned to (near-)duplicate anchor scores would
#: otherwise collapse its variance to 0 and take the likelihood to infinity.
_ANCHOR_VAR_FLOOR_FRAC = 1e-6

#: Minimum mixture weight either anchored component may end with; below this
#: the fit has effectively deleted a component and the anchored path must fall
#: back to the unanchored fit.
_ANCHOR_MIN_WEIGHT = 1e-6

#: Convergence tolerance for the **anchored refit**, on the log-likelihood
#: rather than on the parameters (issue #3825).  Set to ``None`` to take the
#: parameter rule at :data:`_ANCHORED_EM_TOL` instead, which is what shipped
#: until #3825.
#:
#: **Not** the unanchored fit's :data:`_EM_LOGLIK_TOL`, and the difference is
#: the whole finding of #3825.  1e-3 is sklearn's tolerance, it is right for
#: :func:`fit_score_gmm`, and transferring it here is a **regression**: on 114
#: paired trajectory cells it costs +0.026 +- 0.006, because a barely
#: identified fold mixture stopped that early leaves the components too close
#: together and the midpoint cut lands low (FPR +0.044 to buy FNR -0.018).
#: What ports is the *criterion*, not the number.  1e-8 is numerically the
#: incumbent's own tolerance applied to the likelihood instead of the
#: parameters: 1.9x cheaper, the admitted set moves by a median of zero and a
#: p90 of one media in two thousand, and the share of folds exiting on
#: ``max_iter`` falls from 26.5% to 7.4%.  (An earlier version of this comment
#: said 1.6x and 5.7% -> 1.4%: those came from a harness frame fitted at the
#: library's default anchor mass of 10 rather than the shipped 0.3, and #3825's
#: report retracted them.)  Measured in
#: ``docs/experiments/2026-09-13-anchored-em-stop-3825/REPORT.md``.
#:
#: Read inside :func:`fit_anchored_score_gmm` rather than bound as a default
#: argument, so a study can install a stopping rule for a whole run the way
#: ``scripts/experiments/gmm_init/arms_3825.py`` does.
_ANCHORED_EM_LOGLIK_TOL: "float | None" = 1e-8

#: Iteration cap and parameter-delta tolerance for the anchored refit - the
#: signature defaults of :func:`fit_anchored_score_gmm`, named so the loop's
#: budget is one fact rather than two literals.  The tolerance is inert while
#: :data:`_ANCHORED_EM_LOGLIK_TOL` is set (:func:`_anchored_em` reads one rule
#: or the other, never both).
#:
#: **The budget is 2,000, not 200, because a capped refit is not a nearly
#: converged one** (issue #3839).  At 200, 7.4% of 4,493 real fold refits left on
#: the cap, and finishing them (the same loop run to 1e-13) moved the admitted
#: set on 98% of the cases they belonged to - median 0.73% of the haystack, max
#: 36%.  They are slow migrations, the minority component still walking out to
#: the high mode, and a capped fit's midpoint sits low, so the cut over-admits.
#: The issue's other two answers were priced and rejected: a stall / relative /
#: Aitken rule makes "converged" reachable by stopping *earlier* and moves every
#: capped case further from the converged fit, and treating a capped refit as a
#: degeneracy (fall back to the unanchored fit) moves them furthest of all
#: (median 7% of the haystack from converged).  At 2,000: capped 7.4% -> 0.067%
#: (3 of 4,493), no case more than 2.3% of a haystack from converged, the
#: objective better on every fold it changes and worse on none; a paired
#: trajectory A/B over 513 cells gives Delta cost -0.0021 +- 0.0015 (not a
#: regression).  Price: mean refit +25% (the median fold never reaches 200 and
#: is untouched), and on the slow folds themselves 26 -> 47 ms at fold size,
#: 0.32 -> 0.77 s at ``_GMM_MAX_SAMPLES``.  Measured in
#: ``docs/experiments/2026-09-22-anchored-maxiter-3839/REPORT.md``.  The residual
#: rate is still reported (``FoldAnchoredCut.n_unconverged``, provenance
#: ``fold_anchored_maxiter{u}[a/k]``) - the documented rate is 0.07%.
_ANCHORED_EM_MAX_ITER = 2000
_ANCHORED_EM_TOL = 1e-8


def _anchor_loglik_terms(
    a_lo: np.ndarray,
    a_hi: np.ndarray,
    mu: np.ndarray,
    const: np.ndarray,
    two_var: np.ndarray,
    lam: float,
) -> float:
    """The clamped points' contribution to :func:`_anchored_em`'s objective.

    Their responsibilities are fixed one-hot, so each anchor simply adds its own
    component's log density at the multiplicity *lam* the M-step counts it with;
    *const* already holds ``log w_c - 0.5*log(2*pi*var_c)`` per component.  Zero
    when there are no anchors, which is what makes the anchored and free
    objectives the same arithmetic for :func:`_plain_em`.
    """
    total = 0.0
    if a_lo.size:
        total += lam * (a_lo.size * const[0] - float(np.sum((a_lo - mu[0]) ** 2)) / two_var[0])
    if a_hi.size:
        total += lam * (a_hi.size * const[1] - float(np.sum((a_hi - mu[1]) ** 2)) / two_var[1])
    return total


def _record_stopping(
    stats: "dict[str, float] | None",
    n_iter: int,
    stopped: bool,
    loglik: float,
    loglik_tol: "float | None",
) -> None:
    """Fill an :func:`_anchored_em` caller's *stats* sink, if it passed one.

    ``converged`` is 1.0 only when the loop met its tolerance; a fit that left
    on ``max_iter`` is not the estimator anyone specified, and before #3825
    nothing anywhere reported the difference.  ``loglik`` appears only when
    there was a log-likelihood rule to record - the objective at the parameters
    the final iteration started from, i.e. the value the stopping decision was
    taken on rather than a re-evaluation.
    """
    if stats is None:
        return
    stats["n_iter"] = float(n_iter)
    stats["converged"] = float(stopped)
    if loglik_tol is not None:
        stats["loglik"] = float(loglik)


def _anchored_em(
    x: np.ndarray,
    a_lo: np.ndarray,
    a_hi: np.ndarray,
    init: GmmFit1D,
    anchor_weight: float,
    max_iter: int,
    tol: float,
    loglik_tol: "float | None" = None,
    stats: "dict[str, float] | None" = None,
    anchored_objective: bool = True,
) -> GmmFit1D | None:
    """Run the anchored EM iterations; ``None`` on numerical failure.

    *x* is the free (unlabelled) sample, *a_lo* / *a_hi* the anchor scores
    clamped to the low / high component.  Pure numpy, log-domain E-step, fixed
    iteration order, and no BLAS calls (see the M-step comment) - deterministic
    for fixed inputs, and reproducible across machines with it.  Only
    numerical failure (non-finite parameters) returns ``None`` here; semantic
    degeneracy (inverted means, collapsed component) is judged by the caller so
    it can name the reason.

    **This loop is the shipped threshold's dominant cost, which is why it is
    written the way it is** (issue #3558).  The 2026-08-28 fold-count study
    priced a calibration fold at 0.148 s and put **86% of that in this
    function's caller**, :func:`fit_anchored_score_gmm`; that single term is
    the whole reason ``calibrate_count`` ships at 2 when the study measured
    K=6 as worth −0.0057 ± 0.0012 cost early
    (``docs/experiments/2026-08-28-calibration-fold-count-3310/REPORT.md``).
    At the study's 7,747-media haystack the loop is **overhead-bound, not
    FLOP-bound**: two components over one dimension is ~15k doubles an
    iteration, so what it costs is numpy dispatches and allocation, not
    arithmetic.  So the body works in **preallocated 1-D buffers, one per
    component**, instead of allocating a fresh ``(n, 2)`` array (and half a
    dozen ``(n, 2)`` temporaries) per iteration - measured **5-6x faster** at
    1k-50k free scores, and the gap widens with *n*.

    Every operation is deliberately the *same* operation on the *same* values
    as the two-column form it replaces, so the fit is **bit-for-bit**
    identical, not merely close: the arithmetic is elementwise either way, an
    ``axis=1`` max over two columns is :func:`numpy.maximum` of the two, an
    ``axis=1`` sum over two columns is their sum, and each reduction still
    sees a contiguous array of the same values in the same order.  That
    equivalence is the point - a faster fit that moved the threshold would be
    a calibration change needing a study, not an optimisation - and it is
    pinned by ``TestAnchoredEmEquivalence`` in
    ``tests_lib/sorting/test_anchored_gmm.py``, which runs the pre-#3558
    two-column loop side by side and asserts exact equality of all six
    parameters.  **Keep that test passing rather than "tidying" this body**;
    the readable form is right there in the test to diff against.

    **Two stopping rules, and which one you want depends on the sample.**  By
    default (``loglik_tol=None``) the loop stops when no parameter moved by more
    than *tol* - the anchored path's rule, and the right one there because an
    anchored refit starts from an already-converged unanchored fit and has a
    short way to go.  Passing *loglik_tol* switches to "stop when the mean
    log-likelihood stops improving by that much", which is sklearn's rule and
    what :func:`fit_score_gmm` uses.  *anchored_objective* then says which
    likelihood: the **weighted semi-supervised** one this loop ascends, anchors
    included at weight *anchor_weight* and divided by the total mass (the
    default, and the only one that is the estimator's own objective), or the
    free sample's alone.  They coincide exactly when there are no anchors, so
    the unanchored fit cannot tell the difference; on an anchored refit they do
    not, and #3825 measured which one to stop on.

    The difference is not a detail: on a **barely bimodal** sample - a cosine
    text sort, where the query's matches are a shoulder on one broad mode rather
    than a second mode - the likelihood surface has a flat ridge, and EM crawls
    along it forever.  Measured on real 13k-score sorts, the parameter rule at
    1e-8 runs the full 200 iterations to buy ~0.004 nats, while the likelihood
    rule stops in ~25 with the same cut.  A criterion that cannot tell "still
    converging" from "converged, and now drifting" spends all of its time in the
    second case.  Computing the objective costs an extra reduction and an
    in-place log per iteration, which is why it is opt-in rather than always on:
    the anchored path would pay it for a rule it does not need, and it must stay
    bit-for-bit what #3558 pinned.

    **Which rule the anchored refit uses, and why that changed** (issue #3825).
    Until #3825 the anchored path passed nothing and therefore took the
    parameter rule - and #3585, which made the *initialiser* 5x cheaper, left
    that refit as ~90% of a fold's fit: measured on the captured fold corpus it
    ran **97-200 iterations** where the init took ~15, and on 2 of 5 sampled
    folds it never converged at all, exiting on *max_iter*.  The diagnosis is
    the same one the sort path gave: a barely-identified mixture crawls a flat
    ridge, and the parameter criterion cannot tell that it has stopped learning
    anything.  So the anchored path now passes
    :data:`_ANCHORED_EM_LOGLIK_TOL`, and the choice of that value is a measured
    one - ``docs/experiments/2026-09-13-anchored-em-stop-3825/REPORT.md``.

    Pass *stats* to learn **how** a fit ended: the dict comes back with
    ``n_iter`` (iterations run) and ``converged`` (1.0 when the loop met its
    tolerance, 0.0 when it exhausted *max_iter*).  Nothing reported that before
    #3825, which is the only reason a shipped estimator could spend a month
    exiting on its iteration cap without anyone knowing.  It is filled only on
    the success path; a numerical failure returns ``None`` and leaves it alone.

    The initialiser every anchored fit runs first used to be sklearn's
    ``GaussianMixture`` - the larger half of the remaining cost, and ~5x slower
    per EM iteration than this loop on the same estimation problem.  #3585
    replaced it with :func:`fit_score_gmm`, which is now **this same loop with
    no anchors** (:func:`_plain_em`), so the two halves of a fold's fit no
    longer differ by a factor of five for no reason.  That swap moves the fit
    and was licensed by measurement, not by argument; see
    ``docs/experiments/2026-09-13-gmm-init-3585/REPORT.md``.
    """
    lam = float(anchor_weight)
    n = float(x.size)
    n_lo, n_hi = float(a_lo.size), float(a_hi.size)
    total_mass = n + lam * (n_lo + n_hi)

    w = np.array([init.w_lo, init.w_hi], dtype=np.float64)
    mu = np.array([init.mu_lo, init.mu_hi], dtype=np.float64)
    var = np.array([init.var_lo, init.var_hi], dtype=np.float64)

    pooled = np.concatenate([x, a_lo, a_hi])
    var_floor = max(1e-12, _ANCHOR_VAR_FLOOR_FRAC * float(np.var(pooled)))
    var = np.maximum(var, var_floor)

    sum_a_lo, sum_a_hi = float(a_lo.sum()), float(a_hi.sum())

    # Reused every iteration: ``r_lo`` / ``r_hi`` hold the two columns the
    # two-column form kept in one ``(n, 2)`` array, and ``scratch`` stands in
    # for its per-iteration temporaries.  Allocated once because the loop is
    # dispatch- and allocation-bound at these sizes (see the docstring).
    r_lo = np.empty_like(x)
    r_hi = np.empty_like(x)
    scratch = np.empty_like(x)
    prev_loglik: float | None = None
    loglik = float("nan")
    n_iter = 0
    stopped = False

    for n_iter in range(1, max_iter + 1):
        # E-step over the free sample only (anchors are clamped one-hot).
        # Log-domain: log w_c - 0.5*log(2*pi*var_c) - (x-mu_c)^2 / (2*var_c).
        # ``const`` is that first pair of terms, which do not depend on x.
        const = np.log(np.maximum(w, 1e-300)) - 0.5 * np.log(2.0 * math.pi * var)
        two_var = 2.0 * var
        np.subtract(x, mu[0], out=r_lo)
        np.square(r_lo, out=r_lo)
        np.divide(r_lo, two_var[0], out=r_lo)
        np.subtract(const[0], r_lo, out=r_lo)
        np.subtract(x, mu[1], out=r_hi)
        np.square(r_hi, out=r_hi)
        np.divide(r_hi, two_var[1], out=r_hi)
        np.subtract(const[1], r_hi, out=r_hi)
        # Subtract the per-row max (an ``axis=1`` max over two columns is just
        # their elementwise maximum), exponentiate, and normalise by the row
        # sum (likewise, their elementwise sum).
        np.maximum(r_lo, r_hi, out=scratch)
        # The shift the log-sum-exp is taken around; its own sum is half the
        # log-likelihood, and it is about to be overwritten.
        max_sum = float(np.sum(scratch)) if loglik_tol is not None else 0.0
        np.subtract(r_lo, scratch, out=r_lo)
        np.subtract(r_hi, scratch, out=r_hi)
        np.exp(r_lo, out=r_lo)
        np.exp(r_hi, out=r_hi)
        np.add(r_lo, r_hi, out=scratch)
        np.divide(r_lo, scratch, out=r_lo)
        np.divide(r_hi, scratch, out=r_hi)
        if loglik_tol is not None:
            # ``scratch`` still holds the row sums the responsibilities were
            # normalised by, so the objective costs no second pass over the
            # data: a log-sum-exp is the shift plus the log of that sum.  It is
            # the log-likelihood at the parameters this iteration *started*
            # from, which is exactly the quantity sklearn compares between
            # iterations - so "converged" means the same thing in both, and the
            # stopping decision is taken below, after the M-step, as it is there.
            #
            # ``anchored_objective`` decides *whose* likelihood that is: the
            # free sample's alone, or the weighted semi-supervised objective
            # this EM actually ascends, anchors included (#3825).  With no
            # anchors the two are the same arithmetic on the same values -
            # ``total_mass`` is ``n`` and both anchor terms are skipped - so
            # :func:`_plain_em` is unaffected either way, bit for bit.
            np.log(scratch, out=scratch)
            loglik = max_sum + float(np.sum(scratch))
            if anchored_objective:
                loglik = (loglik + _anchor_loglik_terms(a_lo, a_hi, mu, const, two_var, lam)) / total_mass
            else:
                loglik /= n

        # M-step with the anchors folded in at weight ``lam`` each.
        m_lo = float(r_lo.sum()) + lam * n_lo
        m_hi = float(r_hi.sum()) + lam * n_hi
        if not (m_lo > 0.0 and m_hi > 0.0):
            return None
        # ``np.sum`` rather than ``@``: a dot product is dispatched to BLAS,
        # whose accumulation order depends on the kernel the CPU selects and on
        # the thread count, so the same input can differ in its last bits
        # between two machines.  numpy's own pairwise reduction does not
        # (issue #3166), and it reduces in the same order whatever the stride,
        # which is what lets these run over the 1-D buffers instead of over
        # two strided column views.
        np.multiply(r_lo, x, out=scratch)
        sum_lo_x = float(np.sum(scratch))
        np.multiply(r_hi, x, out=scratch)
        sum_hi_x = float(np.sum(scratch))
        mu_new = np.array(
            [
                (sum_lo_x + lam * sum_a_lo) / m_lo,
                (sum_hi_x + lam * sum_a_hi) / m_hi,
            ]
        )
        np.subtract(x, mu_new[0], out=scratch)
        np.square(scratch, out=scratch)
        np.multiply(scratch, r_lo, out=scratch)
        sum_lo_sq = float(np.sum(scratch))
        np.subtract(x, mu_new[1], out=scratch)
        np.square(scratch, out=scratch)
        np.multiply(scratch, r_hi, out=scratch)
        sum_hi_sq = float(np.sum(scratch))
        var_new = np.array(
            [
                (sum_lo_sq + lam * float(((a_lo - mu_new[0]) ** 2).sum())) / m_lo,
                (sum_hi_sq + lam * float(((a_hi - mu_new[1]) ** 2).sum())) / m_hi,
            ]
        )
        var_new = np.maximum(var_new, var_floor)
        w_new = np.array([m_lo, m_hi]) / total_mass

        if not (np.all(np.isfinite(mu_new)) and np.all(np.isfinite(var_new)) and np.all(np.isfinite(w_new))):
            return None
        if loglik_tol is None:
            delta = max(
                float(np.max(np.abs(mu_new - mu))),
                float(np.max(np.abs(var_new - var))),
                float(np.max(np.abs(w_new - w))),
            )
            mu, var, w = mu_new, var_new, w_new
            stopped = delta < tol
        else:
            mu, var, w = mu_new, var_new, w_new
            stopped = prev_loglik is not None and abs(loglik - prev_loglik) < loglik_tol
            prev_loglik = loglik
        if stopped:
            break

    _record_stopping(stats, n_iter, stopped, loglik, loglik_tol)

    return GmmFit1D(
        w_lo=float(w[0]),
        mu_lo=float(mu[0]),
        var_lo=float(var[0]),
        w_hi=float(w[1]),
        mu_hi=float(mu[1]),
        var_hi=float(var[1]),
    )


def fit_anchored_score_gmm(
    arr: np.ndarray,
    anchor_scores: "list[float] | np.ndarray",
    anchor_labels: "list[float] | np.ndarray",
    *,
    anchor_weight: float = ANCHOR_WEIGHT_DEFAULT,
    max_iter: int = _ANCHORED_EM_MAX_ITER,
    tol: float = _ANCHORED_EM_TOL,
    stats: "dict[str, float] | None" = None,
) -> tuple[GmmFit1D | None, str]:
    """Fit the label-anchored 2-component mixture (issue #2852).

    *arr* is the (possibly :func:`gmm_fit_array`-subsampled) haystack score
    sample; *anchor_scores* / *anchor_labels* are the voted items' scores and
    binary labels (1.0 Good -> high component, else low).  Initialised from the
    **unanchored** :func:`fit_score_gmm` fit (deterministic: a 2-means init and
    this same EM loop with no anchors), then refined by anchored EM (see
    :func:`_anchored_em`).

    Returns ``(fit, provenance)``.  On success the provenance is ``"anchored"``
    and the fit's components are class-identified by construction (``hi`` is
    the Good-anchored component).  On failure the fit is ``None`` and the
    provenance names the reason (``"no_anchors"``, ``"too_few_scores"``,
    ``"unanchored_init_failed"``, ``"em_failed"``, ``"inverted_means"``,
    ``"component_collapse"``) - the caller decides the fallback policy
    (:func:`anchored_gmm_fit` falls back to the unanchored fit, never to 0.5).

    Pass *stats* to receive the refit's ``n_iter`` / ``converged`` (see
    :func:`_anchored_em`); the loop is stopped on
    :data:`_ANCHORED_EM_LOGLIK_TOL`, read here at call time rather than bound as
    a default so a study can install another rule (#3825).

    Anchors force a component ordering rather than inherit one: if the labelled
    scores contradict the population modes (Good votes living in the low mode),
    the anchored means invert or a component collapses, and that is reported as
    a degeneracy instead of silently shipping a backwards cut.
    """
    x = np.asarray(arr, dtype=np.float64).ravel()
    a = np.asarray(anchor_scores, dtype=np.float64).ravel()
    z = np.asarray(anchor_labels, dtype=np.float64).ravel()
    if x.size < 2:
        return None, "too_few_scores"
    if a.size == 0 or a.size != z.size:
        return None, "no_anchors"
    if not (anchor_weight > 0.0):
        return None, "no_anchors"

    init = fit_score_gmm(x)
    if init is None:
        return None, "unanchored_init_failed"

    a_hi = a[z == 1.0]
    a_lo = a[z != 1.0]
    fit = _anchored_em(x, a_lo, a_hi, init, anchor_weight, max_iter, tol, _ANCHORED_EM_LOGLIK_TOL, stats)
    if fit is None:
        return None, "em_failed"
    if not (fit.mu_hi > fit.mu_lo):
        return None, "inverted_means"
    if fit.w_lo < _ANCHOR_MIN_WEIGHT or fit.w_hi < _ANCHOR_MIN_WEIGHT:
        return None, "component_collapse"
    return fit, "anchored"


def anchored_gmm_fit(
    all_scores: "list[float] | np.ndarray",
    anchor_scores: "list[float] | np.ndarray",
    anchor_labels: "list[float] | np.ndarray",
    *,
    anchor_weight: float = ANCHOR_WEIGHT_DEFAULT,
) -> tuple[GmmFit1D | None, str]:
    """Production-shaped anchored fit with the #2852 fallback policy applied.

    Subsamples via :func:`gmm_fit_array`, attempts the anchored fit, and on any
    anchored degeneracy falls back to the **unanchored** GMM fit of the same
    sample - never to 0.5.  Returns ``(fit, provenance)`` where provenance is
    ``"anchored"`` or ``"unanchored:<reason>"``; the fit is ``None`` only when
    the unanchored fit fails too (provenance ``"gmm_failed:<reason>"``, caller
    falls back to its median rule).
    """
    arr = gmm_fit_array(all_scores)
    fit, provenance = fit_anchored_score_gmm(arr, anchor_scores, anchor_labels, anchor_weight=anchor_weight)
    if fit is not None:
        return fit, provenance
    fallback = fit_score_gmm(arr)
    if fallback is not None:
        return fallback, f"unanchored:{provenance}"
    return None, f"gmm_failed:{provenance}"
