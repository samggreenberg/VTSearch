"""What to do about the anchored refits that still exit on ``max_iter`` (#3839).

#3825 moved the anchored refit's stopping rule onto its own objective at 1e-8
and cut the share of folds leaving on the iteration cap from 26.5% to 7.4%.
#3839 asks what the remaining 7.4% *mean*, and names three answers without
pricing any of them:

1. **accept and document a rate** - a tolerance or budget whose
   non-convergence is known and stated, so the monitor has a target;
2. **give the loop a guarantee** - a relative tolerance, or a stall detector
   ("stop when an iteration buys less than a fraction of the climb so far"),
   so "converged" is reachable on a barely identified mixture;
3. **treat a capped refit as a degeneracy** - fall back to the fold's
   unanchored fit, as ``fit_anchored_score_gmm`` already does for inverted
   means and a collapsed component.

Every arm here is scored against the rule that ships today (``shipped``: the
anchored objective at 1e-8, capped at 200) and against a **limit** - the same
loop run to 1e-13 with a 20,000-iteration budget - because "the fit ran out of
iterations" only matters to the extent that finishing would have moved the
line, and nothing else in the gate measures that.

**How the new rules are run.**  The shipped loop has exactly two rules in it.
The stall, relative and Aitken rules are not in the shipped code, and adding
them there to *measure* them would put an unmeasured rule on the production
path.  So they are run by :func:`_drive`, which steps the real loop one
iteration at a time - ``_anchored_em(..., max_iter=1)`` returns the next
parameters and, with a log-likelihood rule set, the objective at the ones it
started from - and applies the rule between steps.  EM is a fixed-point map
with no state beyond the parameters, so stepping it reproduces the loop
exactly; ``drv_shipped`` (the shipped rule, driven) is the control that proves
it, and the gate asserts it bit for bit rather than arguing it.  The driver
pays a per-call setup the real loop pays once, so driven arms are priced in
**iterations**, not seconds.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import arms_3825 as A25

#: The real loop, bound by ``arms_3825`` at its import - before anything here
#: can patch it.  A wrapper that looked the name up at call time would call
#: itself once installed (#3585).
_REAL = A25._REAL_ANCHORED_EM

#: The shipped rule's own tolerance, which every driven rule also honours as a
#: floor (see :class:`Rule`).
SHIPPED_TOL = 1e-8


@dataclass(frozen=True)
class Rule:
    """One way of ending the anchored refit.

    *kind*:

    ``"loop"``
        the shipped loop itself, stopping on the anchored objective at *tol*,
        capped at *max_iter* - ``shipped`` and its budget variants.
    ``"abs"``
        the same rule, driven one step at a time (the equivalence control).
    ``"stall"``
        stop when an iteration improves the objective by less than *frac* of
        the total climb since the init, **or** by less than 1e-8.  The OR is
        deliberate: it can only ever stop *sooner* than ``shipped``, so every
        fold it changes is a fold the shipped rule was still running on - which
        is the population #3839 is about and nothing else.
    ``"rel"``
        stop when the improvement is below *tol* times the objective's own
        magnitude - the relative tolerance the issue names.
    ``"aitken"``
        stop when the Aitken projection of the *remaining* gain,
        ``d_k * c / (1 - c)`` with ``c = d_k / d_{k-1}``, is below *tol*.  The
        classical EM criterion (it bounds the distance to the limit rather than
        the last step), included because it is the one rule whose "converged"
        states something about the fit instead of about the last iteration.

    *fallback* turns a refit that exhausts *max_iter* into a degeneracy: the
    wrapper returns ``None`` and the caller falls back to the fold's unanchored
    fit, exactly as it does for ``em_failed``.
    """

    max_iter: int
    kind: str = "loop"
    tol: float = SHIPPED_TOL
    frac: float = 0.0
    fallback: bool = False


#: ``shipped`` first: every other arm is scored against it.
ARMS: "dict[str, tuple[Rule, str]]" = {
    "shipped": (Rule(200), "the anchored objective at 1e-8, capped at 200 - what ships since #3825"),
    "drv_shipped": (Rule(200, "abs"), "CONTROL: the shipped rule, driven a step at a time - must equal `shipped`"),
    # The reference every arm is also read against.
    "limit": (Rule(20000, "loop", 1e-13), "REFERENCE: the same loop run to 1e-13 with a 20,000-iteration budget"),
    # Option 1: accept a rate - by budget, or by tolerance.
    "cap400": (Rule(400), "option 1: the shipped rule, twice the budget"),
    "cap1000": (Rule(1000), "option 1: the shipped rule, five times the budget"),
    "cap2000": (Rule(2000), "option 1: the shipped rule, ten times the budget"),
    "cap5000": (Rule(5000), "option 1: the shipped rule, twenty-five times the budget"),
    "ll1e-7": (
        Rule(200, "loop", 1e-7),
        "option 1: the shipped rule at 1e-7 (#3825 measured 4.5% capped vs the old rule)",
    ),
    "ll1e-6": (Rule(200, "loop", 1e-6), "option 1: the shipped rule at 1e-6 (#3825: 1.5% capped)"),
    # Option 2: a rule that can be met.
    "stall1e-3": (Rule(200, "stall", frac=1e-3), "option 2: stop when an iteration buys < 1e-3 of the climb so far"),
    "stall1e-4": (Rule(200, "stall", frac=1e-4), "option 2: stall at 1e-4 of the climb"),
    "stall1e-5": (Rule(200, "stall", frac=1e-5), "option 2: stall at 1e-5 of the climb"),
    "rel1e-8": (Rule(200, "rel", 1e-8), "option 2: relative tolerance, 1e-8 of |objective|"),
    "aitken1e-7": (Rule(200, "aitken", 1e-7), "option 2: Aitken-projected remaining gain < 1e-7"),
    "aitken1e-6": (Rule(200, "aitken", 1e-6), "option 2: Aitken-projected remaining gain < 1e-6"),
    # Option 3: a capped refit is a degeneracy.
    "fallback": (Rule(200, fallback=True), "option 3: a refit that hits the cap falls back to the unanchored fit"),
}


def _drive(x, a_lo, a_hi, init, weight, rule: Rule, stats):  # noqa: ANN001, PLR0912
    """Step the real loop one iteration at a time and apply *rule* between steps.

    Call *k* returns theta_k and reports L(theta_{k-1}) - exactly the pair the
    shipped loop compares on iteration *k* - so the ``abs`` kind reproduces
    ``shipped`` to the bit (asserted by the gate, not assumed here).
    """
    cur = init
    first = prev = None
    d_prev = None
    stopped = False
    loglik = float("nan")
    k = 0
    for k in range(1, rule.max_iter + 1):
        st: "dict[str, float]" = {}
        nxt = _REAL(x, a_lo, a_hi, cur, weight, 1, 1e-30, 1e30, st, True)
        if nxt is None:
            return None
        loglik = st["loglik"]
        cur = nxt
        if first is None:
            first = loglik
        if prev is not None:
            d = loglik - prev
            ad = abs(d)
            if rule.kind == "abs":
                stopped = ad < rule.tol
            elif rule.kind == "stall":
                stopped = ad < SHIPPED_TOL or d < rule.frac * (loglik - first)
            elif rule.kind == "rel":
                stopped = ad < rule.tol * abs(loglik)
            elif rule.kind == "aitken":
                if ad < 1e-14:
                    stopped = True
                elif d_prev is not None and d_prev > 0.0 and 0.0 < d < d_prev:
                    c = d / d_prev
                    stopped = d * c / (1.0 - c) < rule.tol
            else:
                raise ValueError(f"unknown rule kind {rule.kind!r}")
            d_prev = d
        prev = loglik
        if stopped:
            break
    if stats is not None:
        stats["n_iter"] = float(k)
        stats["converged"] = float(stopped)
        stats["loglik"] = float(loglik)
    return cur


@contextmanager
def swap_anchored_stop(rule: Rule) -> Iterator[None]:
    """Run the **anchored** refit under *rule* for the duration.

    Substitutes only when the call has anchors, for ``arms_3825``'s reason:
    ``_plain_em`` - the unanchored init every anchored fit runs first, and the
    fallback's fit - is the same loop with both anchor arrays empty and must
    keep the rule it ships with.
    """

    def wrapper(  # noqa: ANN001, PLR0913
        x, a_lo, a_hi, init, anchor_weight, max_iter, tol, loglik_tol=None, stats=None, anchored_objective=True
    ):
        if not (a_lo.size or a_hi.size):
            return _REAL(x, a_lo, a_hi, init, anchor_weight, max_iter, tol, loglik_tol, stats, anchored_objective)
        sink: "dict[str, float]" = {} if stats is None else stats
        if rule.kind == "loop":
            fit = _REAL(x, a_lo, a_hi, init, anchor_weight, rule.max_iter, tol, rule.tol, sink, True)
        else:
            fit = _drive(x, a_lo, a_hi, init, anchor_weight, rule, sink)
        if rule.fallback and fit is not None and not sink.get("converged", 0.0):
            return None
        return fit

    bindings = A25._bindings()
    saved = [(mod, attr, getattr(mod, attr)) for mod, attr in bindings]
    try:
        for mod, attr in bindings:
            setattr(mod, attr, wrapper)
        yield
    finally:
        for mod, attr, old in saved:
            setattr(mod, attr, old)


def assert_swap_installed() -> None:
    """Fail if some binding of the anchored loop is still the real one (see ``arms_3825``)."""
    missed = [f"{getattr(mod, '__name__', mod)}.{attr}" for mod, attr in A25._bindings() if getattr(mod, attr) is _REAL]
    if missed:
        raise AssertionError(f"the anchored-stop swap did not reach: {', '.join(missed)}")


def assert_swap_reaches_the_refit() -> None:
    """A 1-iteration rule must show up as ``n_iter == 1`` in a real anchored fit, for both kinds."""
    import numpy as np

    from vtscore.training.thresholds import fit_anchored_score_gmm

    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0.2, 0.05, 400), rng.normal(0.8, 0.05, 400)])
    for kind in ("loop", "abs"):
        with swap_anchored_stop(Rule(1, kind)):
            stats: dict[str, float] = {}
            fit, prov = fit_anchored_score_gmm(x, [0.85, 0.15], [1.0, 0.0], stats=stats)
        if fit is None or stats.get("n_iter") != 1.0:
            raise AssertionError(f"the {kind} swap did not reach the refit: {prov!r} stats={stats!r}")


def objective_value(*args, **kwargs) -> float:  # noqa: ANN002, ANN003
    """``arms_3825.objective_value`` - the same scorer, so the two studies' columns compare."""
    return A25.objective_value(*args, **kwargs)
