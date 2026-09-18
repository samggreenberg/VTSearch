"""The app's Autopilot phase machine, ported for the voting-iterations harness.

The eval simulates a VTSearch user driving Autopilot.  For that simulation to
mean anything, the simulated user has to move through the *same phases*, on the
same triggers, as the real one — otherwise a study measures a flow nobody takes.
This module is the port of that logic, kept deliberately small and pure so it
can be diffed against its two sources by eye:

* :func:`next_phase` mirrors ``AutopilotStateService.checkPhaseTransition``
  (``frontend/src/app/services/autopilot-state.service.ts``).
* :func:`smart_status`, :func:`stable_status`, and :func:`span_status` mirror
  ``_compute_smart_status`` / ``_compute_stable_status`` / ``_compute_span_status``
  in :mod:`vtscore.detectors.labeling_progress`, which is what the app's
  ``/api/labeling-status`` poll feeds into the phase machine.  Only Span is
  still a copy: Smart and Stable are one-line wrappers over
  :mod:`vtscore.detectors.cost_trend` (issue #3832) and
  :mod:`vtscore.detectors.stability` (issue #3831), which the app calls too, so
  neither rule can drift from the light the user reads.

Why a port rather than a call: the phase machine itself lives in TypeScript, so
there is nothing to import; and the indicator functions in
``labeling_progress`` are wrapped in a module-level, lock-guarded MLP cache
built for an interactive session (one detector, advancing one vote at a time).
A simulation runs thousands of independent trajectories and already holds every
per-step model it needs, so it feeds those inputs in directly.  The rules —
every threshold and constant below — are copied verbatim, and
``tests_lib/detectors/test_autopilot_flow.py`` pins them against the sources.

Because this is a copy, it can go stale silently: change a phase trigger in the
app and the simulated user keeps taking the old route, so every study run after
that measures a flow nobody takes.  ``scripts/check-eval-app-sync.py`` — a
``./run-tests.sh`` gate — digests both sources and fails when either moves; see
"The Eval Default Arm IS the App" in ``docs/EVAL.md``.

The phase ordering the app implements, and the harness therefore reproduces:

1. ``good`` until ``good_target`` positives exist (text sort, take the top).
2. ``bad`` until ``bad_target`` negatives exist — still on the **text/example
   sort**, taking the item nearest that sort's cutoff.  No detector is trained
   in this phase, which is why the app never computes a threshold from a
   1-vs-1 fit (see issue #2788).
3. ``hard`` — the first learned sort.  Refine the boundary until the detector
   is both *smart* (error cost has levelled off) and *stable* (predictions have
   stopped flipping).
4. ``new`` — explore the coverage atlas until *span* goes green too.
5. ``done`` (all three green) or ``exhausted`` (nothing left to label).
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

from vtscore.detectors.cost_trend import (
    SMART_FLAT_THRESHOLD,
    SMART_MIN_POINTS,
    SMART_SLOPE_T,
    SMART_WINDOW,
    smart_status_from_costs,
)
from vtscore.detectors.stability import (
    MIN_PER_CLASS,
    STABLE_FALLING_RATIO,
    STABLE_MAX_THRESHOLD,
    STABLE_MIN_ENTRIES,
    STABLE_RATE_THRESHOLD,
    STABLE_WINDOW,
    ScoredSnapshot,
    stability_entry,
    stable_status_from_entries,
)
from vtscore.eval.startup_schedule import StartupState, is_startup_phase

Phase = Literal["idle", "good", "bad", "hard", "new", "done", "exhausted"]
Status = Literal["red", "yellow", "green"]

# Autopilot's initial-phase vote targets (``INITIAL_STATE.goodToStart`` /
# ``badToStart``).  The first learned sort therefore happens at 3 + 4 = 7 votes,
# which always clears the calibrator's >=2-per-class fold-split guard.
GOOD_TARGET = 3
BAD_TARGET = 4

# ``_compute_smart_status`` / ``_compute_stable_status``: both indicators stay
# red until the labelset has at least this many of each class.  ``MIN_PER_CLASS``
# is owned by ``vtscore.detectors.stability`` and re-exported here, with the
# Stable constants below, so a study can name every gate from one place.

# ``_compute_smart_status``: the window of recent models the error-cost trend
# is regressed over, the minimum number of points that makes a trend
# meaningful, the relative-slope cutoff below which the cost is still falling,
# and how many standard errors below zero that slope must sit before the
# decline is believed rather than read as noise (issue #3832).  All four are
# owned by ``vtscore.detectors.cost_trend`` and re-exported here, with the
# Stable constants below, so a study can name every gate from one place.

# ``_compute_stable_status``: the flip-rate window and its cutoffs — the
# confident-flip average must be under 0.5% of the pool, no single recent step
# at 1%, and the raw rate must have stopped falling (the later half of the
# window not below ``STABLE_FALLING_RATIO`` of the earlier half).  Re-exported
# from ``vtscore.detectors.stability`` so a study naming a gate, or measuring
# how far a run sat from one, has every threshold from one place.
__all__ = [
    "MIN_PER_CLASS",
    "SMART_FLAT_THRESHOLD",
    "SMART_MIN_POINTS",
    "SMART_SLOPE_T",
    "SMART_WINDOW",
    "STABLE_FALLING_RATIO",
    "STABLE_MAX_THRESHOLD",
    "STABLE_MIN_ENTRIES",
    "STABLE_RATE_THRESHOLD",
    "STABLE_WINDOW",
]

# ``_compute_span_status``: yellow once this many atlas nodes carry evidence;
# green at ``autopilot_goal_diversity`` (default 40), capped at the tree size.
SPAN_YELLOW = 10
SPAN_GREEN_DEFAULT = 40


def smart_detail(recent_error_costs: list[float], good: int, bad: int) -> dict[str, Any]:
    """The app's Smart sub-object, *whole*: the light and the numbers behind it.

    *recent_error_costs* are the per-step costs of the last :data:`SMART_WINDOW`
    cached models, each scored against the **current** labelset (never the
    held-out test split — the app has no test labels, and using them here would
    leak into the vote order).  Not a port: the rule is
    :func:`~vtscore.detectors.cost_trend.smart_status_from_costs`, which the app
    calls too, so the harness reads the same light as the user.

    :func:`smart_status` keeps only ``status``.  This keeps ``slope`` and
    ``slope_t`` as well, which is what lets a study say how *close* a yellow was
    to green rather than only that it was not green (issue #3560).
    """
    return smart_status_from_costs(recent_error_costs, good, bad)


def smart_status(recent_error_costs: list[float], good: int, bad: int) -> Status:
    """The app's ``_compute_smart_status``: has the error cost levelled off?

    The light alone, for callers that only need the phase; :func:`smart_detail`
    is the same call with its margins kept.
    """
    return smart_detail(recent_error_costs, good, bad)["status"]  # type: ignore[return-value]


def stable_detail(stability_entries: list[dict[str, Any]], good: int, bad: int) -> dict[str, Any]:
    """The app's Stable sub-object, *whole*: the light and the numbers behind it.

    Each entry is one :func:`vtscore.detectors.stability.stability_entry`
    record for one retraining - raw and *confident* flip counts over the
    still-unlabeled pool, with the whole pool as denominator.  Not a port: the
    rule is :func:`~vtscore.detectors.stability.stable_status_from_entries`,
    which the app calls too, so the harness reads the same light as the user.

    :func:`stable_status` keeps only ``status``.  This keeps the three rates
    Stable's three gates turn on - the confident average and its worst step,
    and the two halves of the raw window - so a study can say which gate was
    binding and by how much (issue #3560).
    """
    return stable_status_from_entries(stability_entries, good, bad)


def stable_status(stability_entries: list[dict[str, Any]], good: int, bad: int) -> Status:
    """The app's ``_compute_stable_status``: have predictions stopped flipping?

    The light alone, for callers that only need the phase; :func:`stable_detail`
    is the same call with its margins kept.
    """
    return stable_detail(stability_entries, good, bad)["status"]


def span_target(depth: int, green_at: int = SPAN_GREEN_DEFAULT) -> int:
    """How many consecutive evidence-bearing nodes Span needs to read green.

    The goal, capped at what the tree can supply: a 12-node atlas cannot show
    40 nodes of evidence, so on it green means 12.  ``0`` on a degenerate tree,
    which :func:`span_status` reads as green outright.

    Split out of :func:`span_status` because it is also the *denominator* a
    study wants beside :attr:`AutopilotFlow.span_level`: "31 nodes" says
    nothing without the bar it is short of, and that bar is per-run (the green
    target is a knob) and per-step (the atlas grows).  One definition, two
    readers, so the reported target cannot disagree with the light.
    """
    return min(green_at, depth) if depth > 0 else 0


def span_status(level: int, depth: int, green_at: int = SPAN_GREEN_DEFAULT) -> Status:
    """Port of ``_compute_span_status``: how much of the atlas carries evidence?

    *level* is the atlas's count of consecutive BFS-order evidence-bearing
    nodes and *depth* its total node count (i.e.
    :meth:`CoverageAtlas.span_info`'s ``level`` / ``depth``).  A degenerate tree
    (no nodes) reads green, matching the app — there is no diversity left to
    chase.
    """
    if depth <= 0:
        return "green"
    green = span_target(depth, green_at)
    yellow = min(SPAN_YELLOW, green)
    if level >= green:
        return "green"
    if level >= yellow:
        return "yellow"
    return "red"


def next_phase(
    good_count: int,
    bad_count: int,
    *,
    remaining_unlabeled: float,
    smart: Status,
    stable: Status,
    span: Status,
    good_target: int = GOOD_TARGET,
    bad_target: int = BAD_TARGET,
) -> Phase:
    """Port of ``AutopilotStateService.checkPhaseTransition``.

    *remaining_unlabeled* is how many pool items still carry no vote; pass
    ``math.inf`` when the collection size is unknown (the app's "size unknown"
    case, which leaves the targets uncapped and never exhausts).

    The phase is derived from counts and indicator statuses rather than
    accumulated, so it can move backwards — un-toggling votes regresses the
    phase, exactly as it does in the app.
    """
    # Cap each target at the most votes of that class the collection could still
    # yield, so a tiny dataset can still advance past the initial phases instead
    # of stranding in ``good`` forever.
    eff_good_target = min(good_target, good_count + remaining_unlabeled)
    eff_bad_target = min(bad_target, bad_count + remaining_unlabeled)

    if good_count < eff_good_target:
        return "good"
    if bad_count < eff_bad_target:
        return "bad"
    if smart == "green" and stable == "green" and span == "green":
        return "done"
    if remaining_unlabeled == 0:
        return "exhausted"
    if smart == "green" and stable == "green":
        return "new"
    return "hard"


#: Phases in which the app has run a learned sort, so a trained detector's
#: scores and threshold are on screen and driving the next pick.  Before this
#: the user is still on the text/example sort (see the module docstring).
TRAINED_PHASES: frozenset[str] = frozenset({"hard", "new", "done", "exhausted"})


#: The phase in which the app has told the user it is done: all three quality
#: indicators are green, and the panel reads "All quality indicators are green.
#: You can continue labeling or export your results."  Spelled here so that
#: analysis asking "where did the stopping rule fire?" compares against a name
#: rather than a string literal.
#:
#: ``exhausted`` is deliberately **not** here.  It is the app running out of
#: pool with the indicators still amber - the opposite result - and folding the
#: two together would report a starved run as a converged one.
STOPPING_PHASE: str = "done"


def stopping_rule_fired(phase: str) -> bool:
    """Whether *phase* is the one the app's stopping rules produce.

    The app announces completion the **first** time this becomes true and never
    re-announces it (``AutopilotStateService.completionAnnounced``), while the
    phase itself is *derived* every step and can therefore go back to ``hard``
    on the next vote.  So "where did the stopping rule fire" is the first step
    this holds, not the last, and the two are routinely far apart - see
    ``scripts/experiments/calibration/stopping.py``, which reports both.
    """
    return phase == STOPPING_PHASE


def _num(v: Any) -> float:
    """A status dict's numeric extra as a float, ``nan`` when the rule omitted it.

    The indicator rules drop their margins entirely on the branches that refuse
    to fit one - too few votes of a class, too little history - so a missing key
    means "not measured".  Reading it as 0.0 would put a run that has cast six
    votes at the exact centre of the Smart flatness test.
    """
    return float("nan") if v is None else float(v)


def app_has_detector(phase: str) -> bool:
    """Whether the app would have a trained detector on screen in *phase*.

    The harness records this per step as ``app_trained``: a threshold computed
    at a step where this is false is a number no user ever sees, and studies
    about threshold quality (issue #2788) must filter on it rather than
    counting every simulated step.
    """
    if is_startup_phase(phase):
        # A schedule's rounds are on the seed sort by construction, so the app
        # would have no detector on screen however many votes have been cast.
        return False
    return phase in TRAINED_PHASES


class AutopilotFlow:
    """Per-trajectory phase state for one simulated Autopilot session.

    Holds the two inputs the indicators need — the Smart window of error costs
    and the accumulated per-step prediction-flip counts (Stable) — and
    recomputes the phase after every vote.  The atlas span is read from the
    harness's atlas, which it labels in lock-step with the votes.

    After every :meth:`update` the three indicator lights that produced the
    phase are left on :attr:`smart` / :attr:`stable` / :attr:`span` (with the
    raw :attr:`span_level` / :attr:`span_depth` behind the last of them), which
    is what lets a run say *which* rule held it short of ``done``.

    Note the asymmetry between the two, which is the app's: Stable's flip counts
    are a genuine history (each entry compares two consecutive models and can
    never be recomputed later), whereas Smart's costs are *not* a history at
    all.  The app re-scores its last :data:`SMART_WINDOW` cached models against
    the current labelset on every poll, so the whole window is replaced each
    step; accumulating one frozen ``(model_t, labelset_t)`` cost per step would
    confound model improvement with labelset growth (issue #2923).
    """

    def __init__(
        self,
        *,
        good_target: int = GOOD_TARGET,
        bad_target: int = BAD_TARGET,
        span_green: int | None = None,
        startup: Optional[StartupState] = None,
    ):
        self.good_target = good_target
        self.bad_target = bad_target
        self.span_green = SPAN_GREEN_DEFAULT if span_green is None else span_green
        #: A parameterised opening (issue #3267).  ``None`` - the default - is
        #: the app's own: the Good/Bad targets above, resolved by
        #: :func:`next_phase`.  When set, the schedule owns every phase until it
        #: runs out and the app's machine takes over from ``hard``.
        self.startup = startup
        self.phase: Phase = "good" if startup is None else startup.phase_name()  # type: ignore[assignment]
        #: The most recent Smart window: the last :data:`SMART_WINDOW` models'
        #: costs, all against the *current* labelset.  Replaced, never appended.
        self.recent_error_costs: list[float] = []
        self.stability: list[dict[str, Any]] = []
        self._prev_snapshot: Optional[ScoredSnapshot] = None
        #: The three indicator lights behind the phase, as of the last
        #: :meth:`update`.  The phase alone cannot say which of Smart and Stable
        #: is holding a trajectory in ``hard`` - both gate that transition
        #: together - so a study asking *why* a run never stopped needs the
        #: lights themselves.  Computing them is already paid for by
        #: :meth:`update`; keeping them costs three attribute writes.  Blank
        #: until the first :meth:`update`, and left blank throughout a startup
        #: schedule's rounds, which own the phase without consulting them.
        self.smart: Status | str = ""
        self.stable: Status | str = ""
        self.span: Status | str = ""
        #: The raw span counts the Span light is a threshold on: how many
        #: consecutive BFS-order atlas nodes carry evidence, out of how many the
        #: tree has.  ``-1`` where there is no atlas (a non-autopilot run) or no
        #: update yet.  Reported alongside the light because "red" and "green"
        #: cannot say how far short a run fell, and the gap is the actionable
        #: number when Span turns out to be the binding rule.
        self.span_level: int = -1
        self.span_depth: int = -1
        #: Consecutive evidence-bearing nodes Span needs for green *on this
        #: step's atlas* - the run's green target capped at the tree size, via
        #: :func:`span_target`.  ``-1`` where no atlas was read.  The light is
        #: exactly ``span_level >= span_target``, so the pair is the Span
        #: margin: how far short the run fell, in the unit the rule counts in.
        self.span_target: int = -1
        #: The continuous quantities the Smart and Stable gates are thresholds
        #: **on**, as of the last :meth:`update`, and the other half of issue
        #: #3560.  The lights above say whether each rule fired; these say how
        #: close it came, which is what separates a run that sat one noisy
        #: window short of stopping from one that was never going to stop.
        #:
        #: ``nan`` until the rule has enough history to compute them - the
        #: per-class minimum for either, three model steps for Smart, five
        #: entries for Stable - which is "not measured", not "far from green".
        #: Every one of them is read off the dict the rule already builds
        #: (:func:`smart_detail` / :func:`stable_detail`); nothing here is a
        #: second derivation that could disagree with the light beside it.
        #:
        #: Smart is green when ``smart_slope >= SMART_FLAT_THRESHOLD`` **or**
        #: ``smart_slope_t > -SMART_SLOPE_T`` - a disjunction, which is why both
        #: are kept rather than one summary distance.  ``smart_slope_t`` is
        #: ``±inf`` on a window with no residual at all.
        self.smart_slope: float = math.nan
        self.smart_slope_t: float = math.nan
        #: Stable is green when the confident average is under
        #: :data:`STABLE_RATE_THRESHOLD`, its worst step under
        #: :data:`STABLE_MAX_THRESHOLD`, and the raw rate has stopped falling
        #: (the later half of the window not below
        #: ``STABLE_FALLING_RATIO`` of the earlier half).  Three gates, so five
        #: numbers: the two the first two cap, the raw average the UI quotes,
        #: and the two halves the third compares.
        self.stable_flip_rate: float = math.nan
        self.stable_confident_flip_rate: float = math.nan
        self.stable_max_confident_flip_rate: float = math.nan
        self.stable_flip_rate_early: float = math.nan
        self.stable_flip_rate_late: float = math.nan

    def record_step(
        self,
        error_costs: list[float] | None,
        scores: dict[int, float] | None,
        threshold: float | None = None,
        *,
        num_pool: int | None = None,
    ) -> None:
        """Fold one step's model into the Smart / Stable inputs.

        *error_costs* is the freshly re-scored Smart window — the weighted
        FPR/FNR of each of the last :data:`SMART_WINDOW` models against the
        current labelset, oldest first — and replaces the previous window.
        ``None`` means no model was trained this step, leaving the window as it
        stands; an empty list means there was nothing usable to score against
        (either vote class empty), which the app also reports as an empty
        window.  *scores* maps each still-unlabeled pool id to this step's
        served score and *threshold* is the cut applied to them; the flip
        counts against the previous step are derived from the pair exactly as
        the app derives them (:func:`vtscore.detectors.stability.stability_entry`).
        *num_pool* is the whole haystack size the rates are taken over; it
        defaults to the scored ids plus nothing, which is right only for a
        caller that scores every pool item including the labeled ones.
        """
        if error_costs is not None:
            self.recent_error_costs = list(error_costs)
        if scores is None:
            return
        if threshold is None:
            raise ValueError("a threshold is required alongside the pool scores")
        snapshot = ScoredSnapshot.from_scores(scores, threshold)
        pool_size = len(snapshot.scores) if num_pool is None else num_pool
        if self._prev_snapshot is not None:
            self.stability.append(stability_entry(self._prev_snapshot, snapshot, pool_size))
        self._prev_snapshot = snapshot

    def update(self, good_count: int, bad_count: int, remaining_unlabeled: float, span: dict[str, Any] | None) -> Phase:
        """Recompute and return the phase after a vote.

        With a :class:`~vtscore.eval.startup_schedule.StartupState` attached the
        opening is the schedule's, not :data:`GOOD_TARGET` / :data:`BAD_TARGET`;
        once it is spent the app's machine resumes with both targets already
        met, so the trajectory continues into ``hard`` / ``new`` / ``done``
        exactly as it would have.
        """
        if self.startup is not None:
            self.startup.on_click()
            self.startup.advance(good_count, bad_count, remaining_unlabeled)
            if not self.startup.done:
                self.phase = self.startup.phase_name()  # type: ignore[assignment]
                return self.phase
        smart_d = smart_detail(self.recent_error_costs, good_count, bad_count)
        stable_d = stable_detail(self.stability, good_count, bad_count)
        smart: Status = smart_d["status"]
        stable: Status = stable_d["status"]
        sp: Status = span_status(int(span["level"]), int(span["depth"]), self.span_green) if span is not None else "red"
        self.smart, self.stable, self.span = smart, stable, sp
        self.span_level = int(span["level"]) if span is not None else -1
        self.span_depth = int(span["depth"]) if span is not None else -1
        self.span_target = span_target(int(span["depth"]), self.span_green) if span is not None else -1
        # The margins behind the two computed lights.  Absent from the dict
        # exactly when the rule refused to fit one (too few votes, too little
        # history), which is `nan` here and not a zero.
        self.smart_slope = _num(smart_d.get("slope"))
        self.smart_slope_t = _num(smart_d.get("slope_t"))
        self.stable_flip_rate = _num(stable_d.get("avg_flip_rate"))
        self.stable_confident_flip_rate = _num(stable_d.get("avg_confident_flip_rate"))
        self.stable_max_confident_flip_rate = _num(stable_d.get("max_confident_flip_rate"))
        self.stable_flip_rate_early = _num(stable_d.get("flip_rate_early"))
        self.stable_flip_rate_late = _num(stable_d.get("flip_rate_late"))
        self.phase = next_phase(
            good_count,
            bad_count,
            remaining_unlabeled=remaining_unlabeled,
            smart=smart,
            stable=stable,
            span=sp,
            good_target=0 if self.startup is not None else self.good_target,
            bad_target=0 if self.startup is not None else self.bad_target,
        )
        return self.phase
