"""The candidate stopping rules for the **anchored** refit (#3825), and the swap.

Every arm here fits the *same* model with the *same* loop from the *same*
starting point.  They differ only in **when the loop is allowed to stop**, which
is the whole content of the issue: after #3585 made the initialiser 5x cheaper,
``_anchored_em`` became ~90% of a calibration fold's fit, and it was reaching
that cost by running 97-200 iterations of a parameter-delta criterion at 1e-8 -
on a large minority of real folds never converging at all, and exiting on
``max_iter`` instead.

Why this is not a re-run of #3585's gate.  There, the thing being changed was
the *initialiser* of a fit that then re-converged, and there was a second
estimator (the anchored refit) between the change and the shipped cut.  Here
the anchored refit **is** the shipped threshold: nothing downstream re-fits, and
the only thing between this loop and the green/red line is a quantile and a
snap.  So the arms include two **controls** that perturb the incumbent rather
than replace it - ``param1e-6`` and ``iter400`` - because "the candidate moves
N% of admitted sets" is unreadable without knowing what a nudge of the incumbent
moves.  ``iter400`` is the more interesting of the two: the incumbent does not
converge, so *its own* answer is partly a function of where the cap happens to
be, and lifting the cap says how much of today's threshold is the estimator and
how much is the budget.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from vtscore.training.thresholds import gmm as G


@dataclass(frozen=True)
class Stop:
    """One stopping rule for the anchored EM.

    ``loglik_tol=None`` selects the parameter-delta rule at *tol* (what shipped
    until #3825); a value selects sklearn's rule - stop when an iteration
    improves the mean log-likelihood by less than it.  *anchored_objective*
    then says **whose** likelihood is being watched: the weighted
    semi-supervised objective this EM actually ascends (anchors included), or
    the free sample's alone.  That second axis is here because a criterion that
    is not the estimator's own objective is precisely the failure #3585 found
    on the sort path, in a different costume - and the free-sample likelihood
    is not even guaranteed to increase under an anchored M-step.
    """

    max_iter: int
    tol: float
    loglik_tol: "float | None"
    anchored_objective: bool = True


#: The gate's arms.  ``baseline`` first: every other arm is scored against it.
ARMS: "dict[str, tuple[Stop, str]]" = {
    "baseline": (Stop(200, 1e-8, None), "the parameter delta at 1e-8, capped at 200 - what shipped until #3825"),
    "ll1e-3": (
        Stop(200, 1e-8, 1e-3),
        "the anchored objective at 1e-3 - sklearn's tolerance, this estimator's likelihood",
    ),
    "ll1e-4": (Stop(200, 1e-8, 1e-4), "the anchored objective, ten times tighter"),
    "ll1e-5": (Stop(200, 1e-8, 1e-5), "the anchored objective, a hundred times tighter"),
    "ll1e-6": (Stop(200, 1e-8, 1e-6), "the anchored objective, a thousand times tighter"),
    "ll1e-7": (Stop(200, 1e-8, 1e-7), "the anchored objective at 1e-7"),
    "ll1e-8": (
        Stop(200, 1e-8, 1e-8),
        "the anchored objective at 1e-8 - the incumbent's tolerance, on the right quantity",
    ),
    "ll1e-9": (Stop(200, 1e-8, 1e-9), "the anchored objective at 1e-9 - tighter than the rule it replaces"),
    "free1e-3": (Stop(200, 1e-8, 1e-3, False), "the FREE sample's likelihood at 1e-3 - the sort path's rule, borrowed"),
    "free1e-4": (Stop(200, 1e-8, 1e-4, False), "the free sample's likelihood, ten times tighter"),
    "free1e-5": (Stop(200, 1e-8, 1e-5, False), "the free sample's likelihood, a hundred times tighter"),
    "free1e-6": (Stop(200, 1e-8, 1e-6, False), "the free sample's likelihood, a thousand times tighter"),
    "param1e-6": (Stop(200, 1e-6, None), "CONTROL: the incumbent's own rule, loosened 100x"),
    "iter400": (Stop(400, 1e-8, None), "CONTROL: the incumbent's own rule, allowed twice the iterations"),
    "iter25": (Stop(25, 1e-8, None), "the incumbent's rule capped at 25 iterations - cost without a criterion"),
}

#: Arms that are candidates to *ship*.  ``iter25`` is a cost reference rather
#: than a candidate (a cap is not a convergence criterion, and #3585 already
#: found that arm's shape wanting), and the two controls exist to make the
#: candidates' numbers readable.
#:
#: The 1e-7..1e-9 arms were added after the first gate, and the reason is worth
#: recording: at 1e-6 the candidate was still moving 72% of admitted sets, but
#: the two controls showed the incumbent's answer is *stable* - so the question
#: stopped being "how loose can the tolerance be" and became "does watching the
#: right quantity reach the incumbent's own fit sooner".  That is a different
#: candidate, and it is only visible at tolerances the first grid did not reach.
SHIPPABLE = (
    "ll1e-3",
    "ll1e-4",
    "ll1e-5",
    "ll1e-6",
    "ll1e-7",
    "ll1e-8",
    "ll1e-9",
    "free1e-3",
    "free1e-4",
    "free1e-5",
    "free1e-6",
)

#: The real loop, bound at import.  :func:`swap_anchored_stop` rebinds
#: ``G._anchored_em`` to a wrapper, and a wrapper that looked the name up at
#: call time would call itself - the #3585 harness hit exactly that and was
#: saved by a ``RecursionError`` rather than by design.
_REAL_ANCHORED_EM = G._anchored_em


def _bindings() -> "list[tuple[Any, str]]":
    """Every module attribute that names the anchored loop, right now.

    ``_anchored_em`` is resolved as a module global by both of its callers
    inside :mod:`~vtscore.training.thresholds.gmm`, so patching ``G`` is what
    does the work; the package re-export and any module that imported the name
    are patched too, so an arm cannot be defeated by an import style.
    """
    import vtscore.training.thresholds as pkg

    out: "list[tuple[Any, str]]" = [(G, "_anchored_em"), (pkg, "_anchored_em")]
    for mod in list(sys.modules.values()):
        name = getattr(mod, "__name__", "")
        if not (name.startswith("vtscore") or name.startswith("__main__")):
            continue
        if getattr(mod, "_anchored_em", None) is not None and not any(mod is m for m, _ in out):
            out.append((mod, "_anchored_em"))
    return out


@contextmanager
def swap_anchored_stop(stop: Stop) -> Iterator[None]:
    """Run the **anchored** refit under *stop* for the duration.

    The wrapper substitutes the stopping rule only when the call has anchors.
    That is what keeps the arm honest: :func:`~vtscore.training.thresholds.gmm.
    _plain_em` - the unanchored init every anchored fit runs first - is the same
    loop with both anchor arrays empty, and it must keep the rule *it* ships
    with, or the arm would be measuring two changes at once and attributing both
    to the refit.
    """

    def wrapper(  # noqa: ANN001, PLR0913
        x, a_lo, a_hi, init, anchor_weight, max_iter, tol, loglik_tol=None, stats=None, anchored_objective=True
    ):
        objective = True
        if a_lo.size or a_hi.size:
            max_iter, tol, loglik_tol = stop.max_iter, stop.tol, stop.loglik_tol
            objective = stop.anchored_objective
        return _REAL_ANCHORED_EM(x, a_lo, a_hi, init, anchor_weight, max_iter, tol, loglik_tol, stats, objective)

    bindings = _bindings()
    saved = [(mod, attr, getattr(mod, attr)) for mod, attr in bindings]
    try:
        for mod, attr in bindings:
            setattr(mod, attr, wrapper)
        yield
    finally:
        for mod, attr, old in saved:
            setattr(mod, attr, old)


def assert_swap_installed() -> None:
    """Fail if no arm is installed at some binding that names the anchored loop.

    The check a *run* needs, and a different one from the self-test below: an
    arm like ``baseline`` is deliberately indistinguishable from the shipped
    behaviour, so "did the swap take" cannot be answered by looking at a fit.
    It can be answered by looking at the bindings, and it must be - the #3585
    A/B nearly spent an hour per cell measuring the default twice, which
    produces two identical result directories and a clean, wrong "no
    difference".
    """
    missed = [
        f"{getattr(mod, '__name__', mod)}.{attr}"
        for mod, attr in _bindings()
        if getattr(mod, attr) is _REAL_ANCHORED_EM
    ]
    if missed:
        raise AssertionError(f"the anchored-stop swap did not reach: {', '.join(missed)}")


def assert_swap_reaches_the_refit() -> None:
    """Fail loudly if an installed arm is not what the anchored path calls.

    The #3585 A/B nearly spent an hour per cell measuring the default twice,
    which produces two identical result directories and a clean, wrong "no
    difference".  Cheap to check, so it is checked rather than assumed: fit a
    two-lump sample with one anchor under a 1-iteration arm and confirm the
    recorded iteration count is the arm's, not the shipped 200.
    """
    import numpy as np

    from vtscore.training.thresholds import fit_anchored_score_gmm

    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0.2, 0.05, 400), rng.normal(0.8, 0.05, 400)])
    with swap_anchored_stop(Stop(1, 1e-8, None)):
        stats: dict[str, float] = {}
        fit, prov = fit_anchored_score_gmm(x, [0.85, 0.15], [1.0, 0.0], stats=stats)
    if fit is None or stats.get("n_iter") != 1.0:
        raise AssertionError(f"the anchored-stop swap did not reach the refit: {prov!r} stats={stats!r}")


def objective_value(fit, x, anchor_scores, anchor_labels, anchored: bool, weight: float | None = None) -> float:
    """The value of *fit* under one of the two objectives - the "which is better" column.

    Evaluated here rather than read out of the loop, so every arm's fit is
    scored under *both* objectives on the same sample whatever rule it was
    stopped on.  That is what makes the comparison a ranking instead of a
    tautology: an arm stopped on the free likelihood must still be priced on
    the anchored one, which is the objective its EM was actually climbing.

    Matches :func:`~vtscore.training.thresholds.gmm._anchored_em`'s definition
    exactly - the anchored form adds each clamped point's own component log
    density at multiplicity *weight* and divides by the total mass, so with no
    anchors the two coincide.
    """
    import math

    import numpy as np

    from vtscore.training.thresholds import FOLD_ANCHOR_WEIGHT

    if fit is None:
        return float("nan")
    # The shipped fold path's anchor mass, not ``fit_anchored_score_gmm``'s bare
    # default of 10.0: the objective is a function of the weight, so scoring at
    # one mass a fit produced at another compares two different quantities.
    lam = FOLD_ANCHOR_WEIGHT if weight is None else float(weight)
    x = np.asarray(x, dtype=np.float64).ravel()
    a = np.asarray(anchor_scores, dtype=np.float64).ravel()
    z = np.asarray(anchor_labels, dtype=np.float64).ravel()
    a_hi, a_lo = a[z == 1.0], a[z != 1.0]

    def _logdens(v, w, mu, var):
        return math.log(max(w, 1e-300)) - 0.5 * np.log(2.0 * math.pi * var) - (v - mu) ** 2 / (2.0 * var)

    lo = _logdens(x, fit.w_lo, fit.mu_lo, fit.var_lo)
    hi = _logdens(x, fit.w_hi, fit.mu_hi, fit.var_hi)
    m = np.maximum(lo, hi)
    total = float(np.sum(m + np.log(np.exp(lo - m) + np.exp(hi - m))))
    if not anchored:
        return total / float(x.size)
    if a_lo.size:
        total += lam * float(np.sum(_logdens(a_lo, fit.w_lo, fit.mu_lo, fit.var_lo)))
    if a_hi.size:
        total += lam * float(np.sum(_logdens(a_hi, fit.w_hi, fit.mu_hi, fit.var_hi)))
    return total / (float(x.size) + lam * float(a_lo.size + a_hi.size))
