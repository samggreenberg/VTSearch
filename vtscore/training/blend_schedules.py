"""Mix-in schedules for the safe-threshold blend (issue #2841).

Safe-thresholds trades a **GMM cut** (fitted on the score distribution, needs no
labels, never wild) against a **cross-calibration cut** (conformal, uses the
labels, unreliable when there are few).  #2799 settled *whether* to blend; this
module owns *how much, for how long*.

The historical rule was a single hard-coded line - x-cal weight
``clip((n - 6) / 14, 0, 1)``, i.e. pure GMM at ≤6 labels, pure x-cal at ≥20,
linear between.  Three independent choices were baked into it and none had been
measured: the **endpoints** (6, 20), the **shape** (linear), and the
**statistic** the schedule reads (total labels).  A fourth question - whether a
weighted average is even the right combiner - is not expressible as a weight at
all.  Each is a family here.

A schedule is anything that maps *(x-cal cut, GMM cut, label counts, GMM fit)*
to a final threshold.  Most do it through a weight, so :class:`WeightSchedule`
covers them; :class:`CorridorSchedule` does not (it clamps rather than averages)
and overrides :meth:`BlendSchedule.combine` directly.

Registry lookups go through :func:`get_schedule`.  :data:`PRODUCTION_SCHEDULE`
is the shipped default and reproduces the historical ramp exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "PRODUCTION_SCHEDULE",
    "SAFE_BLEND_SCHEDULES",
    "BlendContext",
    "BlendSchedule",
    "CorridorSchedule",
    "WeightSchedule",
    "get_schedule",
    "schedule_names",
]


@dataclass(frozen=True)
class BlendContext:
    """The label counts a schedule may read.

    Counts are in **votes** (bags), not training rows: region flooding turns one
    Bad vote into many rows, and a schedule that read rows would hand off to
    x-cal the instant a single Bad vote flooded.  See
    :func:`vtscore.detectors.training._flood_context`.

    ``n_good``/``n_bad`` are carried alongside ``n_labels`` because the binding
    constraint on conformal calibration is the **rarer** class, not the total:
    a 19-bad/1-good labelset has 20 labels and one positive, and #2790 traced
    the deep threshold spikes to exactly that starvation.  Schedules that ignore
    them (the historical family) simply never read the fields.
    """

    n_labels: int
    n_good: int
    n_bad: int

    @property
    def n_rare(self) -> int:
        """Labels in the rarer class - the count conformal calibration is limited by."""
        return min(self.n_good, self.n_bad)

    @classmethod
    def from_labels(cls, y_list: list[float], groups: list | None = None) -> "BlendContext":
        """Build a context from a training labelset, collapsing flooded bags.

        With *groups* the counts are per distinct bag (one vote = one bag, whose
        rows share a label); without, every row is its own vote.
        """
        if groups is None:
            good = sum(1 for v in y_list if v == 1.0)
            return cls(n_labels=len(y_list), n_good=good, n_bad=len(y_list) - good)
        by_bag: dict[object, float] = {}
        for label, group in zip(y_list, groups, strict=True):
            by_bag.setdefault(group, label)
        good = sum(1 for v in by_bag.values() if v == 1.0)
        return cls(n_labels=len(by_bag), n_good=good, n_bad=len(by_bag) - good)


class BlendSchedule:
    """Base: how a schedule turns two candidate cuts into one threshold."""

    # Annotated without values on purpose: a class-attribute default here would
    # be picked up by the ``@dataclass`` subclasses as a *field* default via
    # ``getattr``, which then forces every following field to carry one too.
    #: Registry key.
    name: str
    #: One line for the report / settings help.
    description: str

    def weight(self, ctx: BlendContext) -> float:
        """The weight on the **x-cal** cut in ``[0, 1]``; 0 means pure GMM.

        Combiners that are not weighted averages still define this, because the
        harness's schedule-variant screen reports it per step
        (:func:`vtscore.eval.arms_schedule._schedule_variant_rows`) and
        because it is what makes a schedule's shape legible at all.
        """
        raise NotImplementedError

    def combine(self, xcal: float, cut: float, ctx: BlendContext, fit: object | None = None) -> float:
        """Final threshold from the x-cal cut, the GMM cut, and the label counts.

        *fit* is the :class:`~vtscore.training.thresholds.GmmFit1D` behind *cut*
        when one is available (``None`` on the median/degenerate fallbacks);
        only schedules that need the component geometry read it.  Callers must
        have already resolved non-finite inputs.
        """
        w = self.weight(ctx)
        return w * xcal + (1.0 - w) * cut


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0→1 ramp over ``[lo, hi]``, clipped outside."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


@dataclass(frozen=True)
class WeightSchedule(BlendSchedule):
    """A schedule expressed as an x-cal weight over a ramped statistic.

    ``stat`` names the label count the ramp reads (``"labels"``, ``"good"``,
    ``"rare"``); ``lo``/``hi`` are its pure-GMM and pure-x-cal endpoints;
    ``shape`` warps the ramp's ``[0, 1]`` progress; ``cap`` bounds the resulting
    weight so a schedule can decline to ever hand off completely.
    """

    name: str
    description: str
    lo: float
    hi: float
    stat: str = "labels"
    shape: str = "linear"
    cap: float = 1.0

    def _stat(self, ctx: BlendContext) -> int:
        if self.stat == "labels":
            return ctx.n_labels
        if self.stat == "good":
            return ctx.n_good
        if self.stat == "rare":
            return ctx.n_rare
        raise ValueError(f"unknown schedule statistic: {self.stat!r}")

    def weight(self, ctx: BlendContext) -> float:
        t = _ramp(self._stat(ctx), self.lo, self.hi)
        return min(self.cap, _SHAPES[self.shape](t))


#: Warps a ramp's ``[0, 1]`` progress.  ``convex`` holds the GMM longer and then
#: hands off fast; ``concave`` does the reverse; ``step`` is a hard switch at the
#: ramp's midpoint; ``logistic`` is a smooth step, renormalised so it still hits
#: exactly 0 and 1 at the endpoints.
_LOGISTIC_K = 8.0


def _logistic(t: float) -> float:
    raw = 1.0 / (1.0 + math.exp(-_LOGISTIC_K * (t - 0.5)))
    lo = 1.0 / (1.0 + math.exp(_LOGISTIC_K * 0.5))
    hi = 1.0 / (1.0 + math.exp(-_LOGISTIC_K * 0.5))
    return (raw - lo) / (hi - lo)


_SHAPES: dict[str, Callable[[float], float]] = {
    "linear": lambda t: t,
    "convex": lambda t: t * t,
    "concave": lambda t: math.sqrt(t),
    "step": lambda t: 1.0 if t >= 0.5 else 0.0,
    "logistic": _logistic,
}


@dataclass(frozen=True)
class CappedThenReleaseSchedule(BlendSchedule):
    """Hold a permanent GMM share for a while, then hand over completely.

    The cap family (:class:`WeightSchedule` with ``cap < 1``) never hands over,
    which cannot be right in the limit: the GMM midpoint is an **inconsistent**
    estimator of the decision cut - it reads no labels, so its error floors out
    at whatever its two-component symmetry assumption gets wrong, no matter how
    much data arrives - whereas the cross-calibration cut is consistent and
    keeps tightening.  Asymptotically, pure x-cal must win.

    So the real question is not *whether* to hand over but *when*, and this
    schedule makes that an arm: hold ``cap`` through the early ramp, then ramp
    the rest of the way to full trust over ``[release_lo, release_hi]``.
    ``cap50``'s apparent "never hand over" verdict was measured over 30 votes,
    which never reached the regime where the handoff should pay.

    The weight is the max of the two ramps, so it is monotone by construction
    and reduces to the capped schedule below ``release_lo``.
    """

    name: str
    description: str
    lo: float = 6.0
    hi: float = 20.0
    cap: float = 0.5
    release_lo: float = 50.0
    release_hi: float = 200.0

    def weight(self, ctx: BlendContext) -> float:
        held = min(self.cap, _ramp(ctx.n_labels, self.lo, self.hi))
        released = _ramp(ctx.n_labels, self.release_lo, self.release_hi)
        return max(held, released)


@dataclass(frozen=True)
class CorridorSchedule(BlendSchedule):
    """Bound the x-cal cut instead of averaging it away.

    A weighted average taxes *every* x-cal cut, including the good ones, to
    defend against the rare wild one - and the pathology safe-thresholds
    actually fixes is wild: #2788's cold-start "admit nothing" cuts, which #2799
    showed the blend eliminates outright on the whole-image arm.  A clamp is the
    targeted version of that: it is a no-op whenever x-cal is sensible and only
    bites when it leaves the corridor the fitted GMM considers plausible.

    The corridor is the interval between the two component means.  Outside it a
    cut is nearly always degenerate - below ``mu_lo`` it admits the entire Bad
    mode, above ``mu_hi`` it rejects the entire Good mode - while the midpoint
    (the production GMM cut) is its exact centre.  Unramped, that corridor
    applies at every label count: the family's own thesis is that a wild cut is
    never acceptable, however many labels back it.

    With *ramped* the corridor instead opens from the midpoint at ``lo`` labels
    (zero width == pure GMM) to the full interval at ``hi``, and then **releases
    entirely** - past ``hi`` the x-cal cut is returned unclamped, matching the
    production philosophy that enough labels earn full trust.  Note this makes
    ``ramped`` discontinuous at ``hi`` for a wild x-cal: just below, the cut is
    clamped to almost the full corridor; at ``hi`` it is not clamped at all.
    That is deliberate (it is what "ramped" means here) but it is also in
    tension with the unramped variant's thesis, which is why both are measured.
    See the #2841 report for which one the data prefers.

    Falls back to the plain blend when no GMM fit is available (the median
    fallback path), so the corridor never silently becomes a no-op cut.
    """

    name: str
    description: str
    lo: float = 6.0
    hi: float = 20.0
    ramped: bool = True

    def weight(self, ctx: BlendContext) -> float:
        # A corridor always consults the x-cal cut (it is the value being
        # clamped), so the fold calibration is never skippable - except where
        # the corridor has collapsed to a point and the answer is the GMM cut.
        #
        # This is a *skip* predicate, not a mixing weight: a clamp has no
        # weighted-average interpretation, so the 1.0 returned here should be
        # read as "x-cal is consulted", and the ``blend_weight`` column of a
        # corridor row means nothing more than that.
        if self.ramped and _ramp(ctx.n_labels, self.lo, self.hi) <= 0.0:
            return 0.0
        return 1.0

    def combine(self, xcal: float, cut: float, ctx: BlendContext, fit: object | None = None) -> float:
        mu_lo = getattr(fit, "mu_lo", None)
        mu_hi = getattr(fit, "mu_hi", None)
        if mu_lo is None or mu_hi is None:
            return super().combine(xcal, cut, ctx, fit)
        lo_edge, hi_edge = (mu_lo, mu_hi) if mu_lo <= mu_hi else (mu_hi, mu_lo)
        if self.ramped:
            openness = _ramp(ctx.n_labels, self.lo, self.hi)
            if openness >= 1.0:
                return xcal
            lo_edge = cut + (lo_edge - cut) * openness
            hi_edge = cut + (hi_edge - cut) * openness
        return max(lo_edge, min(hi_edge, xcal))


#: Every mix-in strategy #2841 measures.  ``prod`` is the shipped ramp and must
#: stay bit-identical to the historical ``clip((n - 6) / 14, 0, 1)``.
#:
#: The two extremes are controls, not proposals: ``pure_gmm`` is the issue's
#: straw man (ignore the learned threshold forever) and ``pure_xcal`` is
#: safe-thresholds OFF, which #2799 already measured as the loser.
_SCHEDULES: tuple[BlendSchedule, ...] = (
    # --- controls: the two ends of the axis ---
    WeightSchedule("pure_gmm", "Never trust the learned cut", lo=0, hi=0, cap=0.0),
    WeightSchedule("pure_xcal", "Never blend (= safe-thresholds off)", lo=0, hi=0),
    # --- family A: endpoints, linear shape ---
    WeightSchedule("prod", "Shipped ramp: pure GMM ≤6, pure x-cal ≥20", lo=6, hi=20),
    WeightSchedule("fast", "Hand off by 12 labels", lo=6, hi=12),
    WeightSchedule("slow", "Hand off by 40 labels", lo=6, hi=40),
    WeightSchedule("vslow", "Hand off by 80 labels", lo=6, hi=80),
    WeightSchedule("early", "Start trusting x-cal at 2 labels", lo=2, hi=20),
    WeightSchedule("late", "Hold pure GMM until 10 labels", lo=10, hi=30),
    # --- family B: shape at the production endpoints ---
    WeightSchedule("convex", "Hold the GMM, then hand off fast", lo=6, hi=20, shape="convex"),
    WeightSchedule("concave", "Hand off early, then crawl", lo=6, hi=20, shape="concave"),
    WeightSchedule("step", "Hard switch at 13 labels", lo=6, hi=20, shape="step"),
    WeightSchedule("logistic", "Smooth step centred at 13 labels", lo=6, hi=20, shape="logistic"),
    # --- family C: schedule on the class that actually limits calibration ---
    WeightSchedule("rare", "Ramp on the rarer class (1→8)", lo=1, hi=8, stat="rare"),
    WeightSchedule("pos", "Ramp on the positive count (1→8)", lo=1, hi=8, stat="good"),
    # --- family D: never hand off completely ---
    # `slow_cap50` is the synthesis the #2841 long run implies: `slow` won the
    # early window by holding more GMM than `cap50` does there, then collapsed
    # past 40 labels because that is where it becomes pure x-cal.  Keep its
    # gentler ramp, cap it so it never hands over.
    WeightSchedule("slow_cap50", "Slow ramp to 40, capped at half GMM", lo=6, hi=40, cap=0.5),
    WeightSchedule("cap80", "Production ramp, but keep 20% GMM forever", lo=6, hi=20, cap=0.8),
    WeightSchedule("cap50", "Production ramp, but keep 50% GMM forever", lo=6, hi=20, cap=0.5),
    # --- family F: cap, then hand over (issue #2841 follow-up) ---
    # The cap family's "never hand over" was measured over 30 votes, which never
    # reaches the regime where the consistent estimator should overtake the
    # inconsistent one.  These bracket the handoff point.
    CappedThenReleaseSchedule(
        "cap50_release_early", "Half GMM, then full x-cal over 30->100 labels", release_lo=30, release_hi=100
    ),
    CappedThenReleaseSchedule(
        "cap50_release", "Half GMM, then full x-cal over 50->200 labels", release_lo=50, release_hi=200
    ),
    CappedThenReleaseSchedule(
        "cap50_release_late", "Half GMM, then full x-cal over 150->400 labels", release_lo=150, release_hi=400
    ),
    # --- family E: bound the x-cal cut instead of averaging it ---
    CorridorSchedule("corridor", "Clamp x-cal between the component means", ramped=False),
    CorridorSchedule("corridor_ramp", "Corridor opening from the midpoint over 6→20", ramped=True),
)

SAFE_BLEND_SCHEDULES: dict[str, BlendSchedule] = {s.name: s for s in _SCHEDULES}

#: The shipped schedule per **voting mode** (issue #2841 measured them
#: separately and they want different curves; see
#: ``docs/experiments/2026-08-04-mixin-schedule/REPORT.md``).
#:
#: * ``region`` - a patch dataset, which always scores by max-pooling over
#:   regions.  Its x-cal cut needs far longer to become trustworthy *and* never
#:   becomes trustworthy enough to trust alone: over a 200-vote horizon the
#:   plain 6->40 ramp decays to nothing once it reaches pure x-cal (+0.008 by
#:   101-200 votes), while capping at half keeps improving (-0.082).  So the
#:   shipped curve is the slow ramp **with** the cap - best or tied in every
#:   vote band and strictly better than ``cap50`` at every positive count.
#: * ``binary`` - one vector per media.  Here a longer ramp wins only by cutting
#:   lower, which reverses under reweighting; what survives is keeping a
#:   permanent half-share of the label-free GMM cut, which reduces the *spread*
#:   of the threshold rather than relocating it (−0.0173, p=7.6e-43).
#:
#: The old single ramp (``prod``) is retained in the registry as the measurement
#: baseline and as the thing to compare against if this is ever revisited.
PRODUCTION_SCHEDULE_BY_MODE: dict[str, str] = {
    "region": "slow_cap50",
    "binary": "cap50",
}

#: Fallback when the voting mode is unknown.  ``cap50`` is the safe default: it
#: is the only schedule #2841 found that improves **both** modes under **every**
#: cost weighting tested, so a caller that cannot say which mode it is in still
#: gets a strict improvement over the old ramp.
PRODUCTION_SCHEDULE = "cap50"


def production_schedule_for(*, region_voting: bool | None) -> str:
    """The shipped schedule name for a detector that does (or doesn't) region-vote.

    ``None`` means "unknown", which takes :data:`PRODUCTION_SCHEDULE`.
    """
    if region_voting is None:
        return PRODUCTION_SCHEDULE
    return PRODUCTION_SCHEDULE_BY_MODE["region" if region_voting else "binary"]


def get_schedule(name: str | None = None) -> BlendSchedule:
    """Look up a schedule by name; ``None`` yields the mode-agnostic default.

    Prefer passing an explicit name resolved through
    :func:`production_schedule_for` - the default here cannot know the voting
    mode, and the two modes want different curves.
    """
    key = name or PRODUCTION_SCHEDULE
    try:
        return SAFE_BLEND_SCHEDULES[key]
    except KeyError:
        raise ValueError(f"unknown safe-threshold schedule {key!r}; known: {', '.join(schedule_names())}") from None


def schedule_names() -> list[str]:
    """Registry keys, in declaration order."""
    return list(SAFE_BLEND_SCHEDULES)
