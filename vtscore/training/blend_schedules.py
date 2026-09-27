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
    "parse_parametric_schedule",
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

    The corridor is centred on the GMM cut and reaches ``width`` of the way from
    it to each component mean: ``width=1`` is the full interval between the two
    means (the #2841 arm), ``width=0`` collapses it onto the GMM cut (pure GMM).
    #2841 measured the full interval as a no-op on region voting (p=0.62)
    because it is far wider than the x-cal error, so the width is the knob
    (#3551).  Outside the full interval a cut is nearly always degenerate - below
    ``mu_lo`` it admits the entire Bad mode, above ``mu_hi`` it rejects the entire
    Good mode - while the midpoint (the GMM cut) is its exact centre.

    Unramped, the corridor applies at every label count: the family's own thesis
    is that a wild cut is never acceptable, however many labels back it.  With
    *ramped* it opens from a point at ``lo`` labels to its full ``width`` at
    ``hi`` and **stays there**.  It used to release entirely past ``hi`` (return
    the x-cal cut unclamped), which made the threshold jump from nearly the
    corridor edge to the raw cut between ``hi - 1`` and ``hi`` labels for any
    wild x-cal - most visibly on the fold fallback, where the x-cal side is the
    ``NO_GOOD_THRESHOLD`` sentinel and the jump is from ``mu_hi`` to "admit
    nothing".  That discontinuity confounded any sweep of the corridor (#3551),
    and holding the corridor is also what the unramped variant's thesis says.

    With no GMM fit (the median/degenerate fallbacks) there are no component
    means to clamp between.  ``no_fit`` then names the schedule to combine
    under instead; unset, the corridor falls back to a plain blend at its own
    skip weight, i.e. the x-cal cut unclamped - which is what the #2841 arms
    measured, but on the fold fallback that cut is the ``NO_GOOD_THRESHOLD``
    sentinel, so a shipped corridor must name a real schedule here.
    """

    name: str
    description: str
    lo: float = 6.0
    hi: float = 20.0
    ramped: bool = True
    width: float = 1.0
    no_fit: str | None = None

    def openness(self, ctx: BlendContext) -> float:
        """Fraction of the way from the GMM cut to each component mean, in ``[0, width]``."""
        if self.ramped:
            return self.width * _ramp(ctx.n_labels, self.lo, self.hi)
        return self.width

    def weight(self, ctx: BlendContext) -> float:
        # A corridor always consults the x-cal cut (it is the value being
        # clamped), so the fold calibration is never skippable - except where
        # the corridor has collapsed to a point and the answer is the GMM cut.
        #
        # This is a *skip* predicate, not a mixing weight: a clamp has no
        # weighted-average interpretation, so the 1.0 returned here should be
        # read as "x-cal is consulted", and the ``blend_weight`` column of a
        # corridor row means nothing more than that.
        return 0.0 if self.openness(ctx) <= 0.0 else 1.0

    def combine(self, xcal: float, cut: float, ctx: BlendContext, fit: object | None = None) -> float:
        mu_lo = getattr(fit, "mu_lo", None)
        mu_hi = getattr(fit, "mu_hi", None)
        if mu_lo is None or mu_hi is None:
            if self.no_fit is not None:
                return get_schedule(self.no_fit).combine(xcal, cut, ctx, fit)
            return super().combine(xcal, cut, ctx, fit)
        lo_mean, hi_mean = (mu_lo, mu_hi) if mu_lo <= mu_hi else (mu_hi, mu_lo)
        f = self.openness(ctx)
        lo_edge = cut + (lo_mean - cut) * f
        hi_edge = cut + (hi_mean - cut) * f
        if lo_edge > hi_edge:  # a cut outside its own means; keep the interval valid
            lo_edge, hi_edge = hi_edge, lo_edge
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
    CorridorSchedule("corridor_ramp", "Corridor opening from the midpoint over 6→20, then held", ramped=True),
    # --- the #3551 ship: the binary-voting fold fallback ---
    # A corridor at a fifth of the way to each component mean, constant.  On a
    # fold-fallback step the x-cal side is the NO_GOOD_THRESHOLD sentinel, so
    # this is "the GMM midpoint, raised 20% of the way toward the Good mean";
    # it never lets a ramp blend the sentinel into an admit-nothing cut, which
    # `cap50` did on 75-100% of the fallback steps it reached past 6 votes.
    # Without a fit it combines exactly as `cap50` does.
    CorridorSchedule(
        "corridor20", "Clamp x-cal to 0.2 of the way to each component mean", ramped=False, width=0.2, no_fit="cap50"
    ),
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
#: * ``binary`` - one vector per media.  #2841 shipped ``cap50`` here when the
#:   blend was the whole threshold.  Since #2861 the blend is only the fused
#:   cut's fold *fallback* (0.75-1.1% of steps, all before vote 20), where the
#:   x-cal side is the ``NO_GOOD_THRESHOLD`` sentinel - and ``cap50`` blends it
#:   in once its ramp starts, admitting nothing on 75-100% of the fallback
#:   steps past 6 votes.  #3551 re-tuned the fallback on that stack and
#:   ``corridor20`` is the one candidate that cleared the pre-registered ship
#:   rule in the A/B (pooled −0.00018 ± 0.00002 cost, every environment
#:   resolvable, neither reweighting worse).  The effect is small because the
#:   fallback is rare; see
#:   ``docs/experiments/2026-09-22-blend-endpoints-3551/``.  Its region
#:   counterpart (``corridor:w=0.05``) failed the rule, so region is unchanged.
#:
#: The old single ramp (``prod``) is retained in the registry as the measurement
#: baseline and as the thing to compare against if this is ever revisited.
PRODUCTION_SCHEDULE_BY_MODE: dict[str, str] = {
    "region": "slow_cap50",
    "binary": "corridor20",
}

#: Fallback when the voting mode is unknown.  ``cap50`` is the safe default: it
#: is the only schedule #2841 found that improves **both** modes under **every**
#: cost weighting tested, so a caller that cannot say which mode it is in still
#: gets a strict improvement over the old ramp.  (#3551's ``corridor20`` is not
#: a candidate here: on region voting the corridor was resolvably worse at
#: ``fnr x4`` in the screen.)
PRODUCTION_SCHEDULE = "cap50"


def production_schedule_for(*, region_voting: bool | None) -> str:
    """The shipped schedule name for a detector that does (or doesn't) region-vote.

    ``None`` means "unknown", which takes :data:`PRODUCTION_SCHEDULE`.
    """
    if region_voting is None:
        return PRODUCTION_SCHEDULE
    return PRODUCTION_SCHEDULE_BY_MODE["region" if region_voting else "binary"]


#: Families a **parametric** schedule name may instantiate (#3551), and the keys
#: each accepts.  A parametric name is ``family:key=value:key=value``, e.g.
#: ``rare:lo=1:hi=16`` or ``corridor:w=0.2``.  They exist so a tuning sweep can
#: name any point of a family's grid without every point becoming a registry
#: entry; they are never shipped (production resolves through
#: :data:`PRODUCTION_SCHEDULE_BY_MODE`, whose values must be registry names).
#: ``:`` is the separator because schedule lists travel comma-separated in
#: ``CALIB_SCHEDULE_VARIANTS``.
_PARAMETRIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "labels": ("lo", "hi", "cap", "shape"),
    "good": ("lo", "hi", "cap", "shape"),
    "rare": ("lo", "hi", "cap", "shape"),
    "corridor": ("w",),
    "corridor_ramp": ("w", "lo", "hi"),
}


def parse_parametric_schedule(name: str) -> BlendSchedule:
    """Build the schedule a ``family:key=value:...`` name describes.

    Raises ``ValueError`` on an unknown family, an unknown or repeated key, a
    missing ramp endpoint, or a value out of range - a typo in a sweep grid must
    fail the cell, not quietly measure a different schedule.
    """
    family, _, rest = name.partition(":")
    if family not in _PARAMETRIC_FAMILIES or not rest:
        raise ValueError(f"not a parametric schedule name: {name!r}")
    params: dict[str, str] = {}
    for part in rest.split(":"):
        key, eq, value = part.partition("=")
        if not eq or key not in _PARAMETRIC_FAMILIES[family] or key in params:
            raise ValueError(f"bad parameter {part!r} in schedule {name!r}")
        params[key] = value
    if family.startswith("corridor"):
        width = float(params.get("w", "1"))
        if not 0.0 <= width <= 1.0:
            raise ValueError(f"corridor width must lie in [0, 1]: {name!r}")
        if family == "corridor":
            return CorridorSchedule(name, f"Corridor at {width:g} of the mean gap", ramped=False, width=width)
        lo, hi = float(params.get("lo", "6")), float(params.get("hi", "20"))
        return CorridorSchedule(name, f"Corridor opening to {width:g} over {lo:g}->{hi:g}", lo=lo, hi=hi, width=width)
    if "lo" not in params or "hi" not in params:
        raise ValueError(f"schedule {name!r} needs both lo and hi")
    lo, hi = float(params["lo"]), float(params["hi"])
    cap = float(params.get("cap", "1"))
    shape = params.get("shape", "linear")
    if hi < lo or not 0.0 <= cap <= 1.0 or shape not in _SHAPES:
        raise ValueError(f"out-of-range parameter in schedule {name!r}")
    return WeightSchedule(
        name, f"Ramp on {family} {lo:g}->{hi:g}, cap {cap:g}", lo=lo, hi=hi, stat=family, shape=shape, cap=cap
    )


def get_schedule(name: str | None = None) -> BlendSchedule:
    """Look up a schedule by name; ``None`` yields the mode-agnostic default.

    Prefer passing an explicit name resolved through
    :func:`production_schedule_for` - the default here cannot know the voting
    mode, and the two modes want different curves.  A name containing ``:`` is
    a parametric point of a family (:func:`parse_parametric_schedule`), which is
    how the #3551 sweep names its tuning grid.
    """
    key = name or PRODUCTION_SCHEDULE
    if ":" in key:
        return parse_parametric_schedule(key)
    try:
        return SAFE_BLEND_SCHEDULES[key]
    except KeyError:
        raise ValueError(f"unknown safe-threshold schedule {key!r}; known: {', '.join(schedule_names())}") from None


def schedule_names() -> list[str]:
    """Registry keys, in declaration order."""
    return list(SAFE_BLEND_SCHEDULES)
