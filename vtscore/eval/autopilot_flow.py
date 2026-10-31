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
``tests_lib/eval/test_autopilot_flow.py`` pins them against the sources.

Because this is a copy, it can go stale silently: change a phase trigger in the
app and the simulated user keeps taking the old route, so every study run after
that measures a flow nobody takes.  ``scripts/gates/check-eval-app-sync.py`` — a
``./run-tests.sh`` gate — digests both sources and fails when either moves; see
"The Eval Default Arm IS the App" in ``docs/EVAL.md``.

The phase ordering the app implements, and the harness therefore reproduces:

1. ``good`` until ``good_target`` positives exist (text sort, take the top), or
   until the walk runs dry: :data:`MORE_DRY_RUN` of its picks in a row held no
   positive, with one in hand (issue #4731).  The ``more`` walk is then spent.
2. ``bad`` until ``bad_target`` negatives exist — still on the **text/example
   sort**, taking the item nearest that sort's cutoff.  No detector is trained
   in this phase, which is why the app never computes a threshold from a
   1-vs-1 fit (see issue #2788).
2b. ``more`` — back to the top of the same sort until ``more_target`` positives
   exist **or** the walk runs dry: :data:`MORE_DRY_RUN` of its picks in a row
   held no positive (issue #4282, from the #4222 dry-stop study).  Once the walk
   ends it never resumes.
3. ``hard`` — the first learned sort.  Refine the boundary until the detector
   is both *smart* (F-beta has stopped rising) and *stable* (predictions have
   stopped flipping).
4. ``new`` — explore the coverage atlas until *span* goes green too.
5. ``spot`` — the spot check before Done: the first time all three lights are
   green the app checks the line once (a band walk, answered as clicks), then
   reports Done.  A run latches it once the check closes, or is skipped
   because one ended moments ago (:func:`~vtscore.training.thresholds.done_check_due`).
   A check Autopilot runs mid-run for weak separation (#4496) holds the phase
   at ``spot`` too, while it runs.
6. ``done`` (all three green) or ``exhausted`` (nothing left to label).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal, Optional

from vtscore.detectors.cost_trend import (
    SMART_FLAT_THRESHOLD,
    SMART_MIN_POINTS,
    SMART_RISE_THRESHOLD,
    SMART_SLOPE_T,
    SMART_WINDOW,
    cost_trend,
    rising_status_from_scores,
    smart_status_from_costs,
)
from vtscore.detectors.stability import (
    HARD_DRY_RUN,
    MIN_PER_CLASS,
    STABLE_FALLING_RATIO,
    STABLE_MAX_THRESHOLD,
    STABLE_MIN_ENTRIES,
    STABLE_RATE_THRESHOLD,
    STABLE_WINDOW,
    HardDryRun,
    ScoredSnapshot,
    goods_minimum,
    stability_entry,
    stable_status_from_entries,
)
from vtscore.eval.startup_schedule import StartupState, is_startup_phase

Phase = Literal["idle", "good", "bad", "more", "hard", "new", "spot", "done", "exhausted"]
Status = Literal["red", "yellow", "green"]

# Autopilot's initial-phase vote targets (``INITIAL_STATE.goodToStart`` /
# ``badToStart``).  The first learned sort therefore happens at 3 + 4 = 7 votes,
# which always clears the calibrator's >=2-per-class fold-split guard.
GOOD_TARGET = 3
BAD_TARGET = 4

# The ``more`` walk (``INITIAL_STATE.moreToStart`` / ``moreDryRun``, issue #4282):
# after the Good and Bad quorum, keep taking the top of the seed sort until this
# many positives exist in total, or until this many walk picks in a row held no
# positive.  On photos the walk is OFF since #4740: ``MORE_TARGET = 0``, so the
# opening is Good, Bad, then Hard again (``g3@top,b4@mid``).  #4222 had shipped it
# at 20 on AP at vote 150; #4671 scored it on the objective against today's app
# and the walk lost after vote 23 at every preset (the old opening +0.016 /
# +0.013 / +0.014 over votes 51-150, AP +0.009 / +0.010 / +0.008), winning only
# votes 1-25.  ``MORE_DRY_RUN`` still ends a document dataset's walk (#4488).
MORE_TARGET = 0
MORE_DRY_RUN = 16

# ``_compute_smart_status`` / ``_compute_stable_status``: both indicators stay
# red until the labelset has at least this many of each class.  ``MIN_PER_CLASS``
# is owned by ``vtscore.detectors.stability`` and re-exported here, with the
# Stable constants below, so a study can name every gate from one place.

# ``_compute_smart_status``: the window of recent models the F-beta trend is
# regressed over, the minimum number of points that makes a trend meaningful,
# the relative-slope cutoff above which F-beta is still rising (``-
# SMART_FLAT_THRESHOLD``, the cost's cutoff before #5134 with its sign turned
# over), and how many standard errors from zero that slope must sit before the
# rise is believed rather than read as noise (issue #3832).  All four are owned
# by ``vtscore.detectors.cost_trend`` and re-exported here, with the Stable
# constants below, so a study can name every gate from one place.

# ``_compute_stable_status``: the flip-rate window and its cutoffs — the
# confident-flip average must be under 0.5% of the pool, no single recent step
# at 1%, and the raw rate must have stopped falling (the later half of the
# window not below ``STABLE_FALLING_RATIO`` of the earlier half).  Re-exported
# from ``vtscore.detectors.stability`` so a study naming a gate, or measuring
# how far a run sat from one, has every threshold from one place.
__all__ = [
    "HARD_DRY_PHASES",
    "HARD_DRY_RUN",
    "MIN_PER_CLASS",
    "SMART_FLAT_THRESHOLD",
    "SMART_MIN_POINTS",
    "SMART_RISE_THRESHOLD",
    "SMART_SLOPE_T",
    "SMART_WINDOW",
    "STABLE_FALLING_RATIO",
    "STABLE_MAX_THRESHOLD",
    "STABLE_MIN_ENTRIES",
    "STABLE_RATE_THRESHOLD",
    "STABLE_WINDOW",
]

# ``_compute_span_status``: yellow once this many atlas nodes carry evidence;
# green at ``autopilot_goal_diversity``, capped at the tree size.  The default
# is 0 since #5473: Span reads green from the first vote and Autopilot skips its
# Diversity (``new``) step.  ``SPAN_GREEN_ON`` is the goal with the step on, the
# default before #5473.
SPAN_YELLOW = 10
SPAN_GREEN_DEFAULT = 0
SPAN_GREEN_ON = 40


def smart_detail(recent_fbetas: list[float], good: int, bad: int, min_good: int = MIN_PER_CLASS) -> dict[str, Any]:
    """The app's Smart sub-object, *whole*: the light and the numbers behind it.

    *recent_fbetas* are the last :data:`SMART_WINDOW` cached models' best
    F-beta at the session's balance, each scored against the **current**
    labelset (never the held-out test split — the app has no test labels, and
    using them here would leak into the vote order; issue #5134).  Not a port:
    the rule is :func:`~vtscore.detectors.cost_trend.rising_status_from_scores`,
    which the app calls too, so the harness reads the same light as the user.

    :func:`smart_status` keeps only ``status``.  This keeps ``slope`` and
    ``slope_t`` as well, which is what lets a study say how *close* a yellow was
    to green rather than only that it was not green (issue #3560).  *min_good*
    is the Goods the light needs (:attr:`AutopilotFlow.min_good_now`, #5477).
    """
    return rising_status_from_scores(recent_fbetas, good, bad, min_good=min_good)


def smart_status(recent_fbetas: list[float], good: int, bad: int) -> Status:
    """The app's ``_compute_smart_status``: has F-beta stopped rising?

    The light alone, for callers that only need the phase; :func:`smart_detail`
    is the same call with its margins kept.
    """
    return smart_detail(recent_fbetas, good, bad)["status"]  # type: ignore[return-value]


def stable_detail(
    stability_entries: list[dict[str, Any]], good: int, bad: int, min_good: int = MIN_PER_CLASS
) -> dict[str, Any]:
    """The app's Stable sub-object, *whole*: the light and the numbers behind it.

    Each entry is one :func:`vtscore.detectors.stability.stability_entry`
    record for one retraining - raw and *confident* flip counts over the
    still-unlabeled pool, with the whole pool as denominator.  Not a port: the
    rule is :func:`~vtscore.detectors.stability.stable_status_from_entries`,
    which the app calls too, so the harness reads the same light as the user.

    :func:`stable_status` keeps only ``status``.  This keeps the three rates
    Stable's three gates turn on - the confident average and its worst step,
    and the two halves of the raw window - so a study can say which gate was
    binding and by how much (issue #3560).  *min_good* as for :func:`smart_detail`.
    """
    return stable_status_from_entries(stability_entries, good, bad, min_good=min_good)


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
    chase.  So does a goal of 0, the Diversity step off (#5473), with or without a
    tree.
    """
    if depth <= 0 or green_at <= 0:
        return "green"
    green = span_target(depth, green_at)
    yellow = min(SPAN_YELLOW, green)
    if level >= green:
        return "green"
    if level >= yellow:
        return "yellow"
    return "red"


#: The Smart gate's arms (#4359): ``"app"`` reads the light; ``"never"`` holds it yellow.
SMART_GATES = ("app", "never")

#: What the Smart light is read off (#5134).  ``"fbeta_best"`` is the app since
#: #5134: the best F-beta at the session's beta any cut through each of the last
#: :data:`SMART_WINDOW` models' scores reaches, re-scored on the current votes,
#: green once it stops rising
#: (:func:`~vtscore.detectors.cost_trend.rising_status_from_scores`).  The rest
#: are #5134's A/B arms:
#:
#: * ``cost``, the app before #5134: FPR + FNR at each model's own Inclusion 0 cut,
#:   green once it stops *falling* (:func:`~vtscore.detectors.cost_trend.smart_status_from_costs`);
#: * the rest of :data:`SMART_WINDOW_READINGS`, the same re-scored window in other
#:   units, green once it stops rising: ``ap`` (average precision) and ``fbeta``
#:   (F-beta at each model's own Smart cut);
#: * :data:`SMART_HISTORY_READINGS`, one value per retrain, kept as a history:
#:   ``heldout_ap`` and ``heldout_fbeta`` (the same two on the calibration folds'
#:   held-out scores of the votes) and ``line`` (the labels line's expected F-beta,
#:   :func:`~vtscore.training.thresholds.labels_line.stop_level_reading`, from the
#:   first retrain its class model cuts).
#:
#: The caller hands :meth:`AutopilotFlow.record_step` the readings; none reads the
#: served line.
SMART_WINDOW_READINGS: tuple[str, ...] = ("fbeta_best", "cost", "ap", "fbeta")
SMART_HISTORY_READINGS: tuple[str, ...] = ("heldout_ap", "heldout_fbeta", "line")
SMART_READINGS: tuple[str, ...] = (*SMART_WINDOW_READINGS, *SMART_HISTORY_READINGS)
SMART_SIGNALS: tuple[str, ...] = SMART_READINGS
#: The app's Smart since #5134.
APP_SMART_SIGNAL = "fbeta_best"
#: What each reading's trend is recorded as (#5134): the window's relative slope,
#: its mean, the slope's t, and the newest value.  ``rel <= SMART_RISE_THRESHOLD``
#: or ``t < SMART_SLOPE_T`` is green, so any threshold on either can be re-read
#: offline, and ``rel * mean`` is the slope in the reading's own units.
SMART_READING_PARTS: tuple[str, ...] = ("slope", "mean", "t", "last")
SMART_READING_COLUMNS: tuple[str, ...] = tuple(
    f"smart_{name}_{part}" for name in SMART_READINGS for part in SMART_READING_PARTS
)


def reading_columns_nan() -> dict[str, float]:
    """:data:`SMART_READING_COLUMNS`, every one NaN: no reading, or none fitted."""
    return dict.fromkeys(SMART_READING_COLUMNS, math.nan)


#: What tells the user they can stop (#4878).  ``"lights|level"`` is the app since
#: #4878: Done once Smart, Stable and Span are all green, or once the labels line
#: expects at least ``STOP_LEVEL`` F-beta at the session's beta
#: (:func:`~vtscore.training.thresholds.labels_line.stop_level_met`, which the
#: caller reads and passes as *level_met*).  The rest are arms: ``"lights"`` (the
#: app before #4878), ``"level"`` (the level alone) and ``"lights+level"`` (both).
#: :func:`stop_lights` hands :func:`next_phase` the lights that make it decide so;
#: a session the rule does not stop with every light green reads Hard (its pick is
#: Done's), not New, whose atlas walk is over.
STOP_RULES = ("lights|level", "lights", "level", "lights+level")

#: The click cap (#4903): whatever the stop rule, a session that holds this many
#: votes goes to the check before Done and to Done, as one the rule stops does.
#: Every rule leaves some sessions unstopped (a third of small-object sessions
#: under the lights at 150 clicks), and the app has no horizon: a user it never
#: tells to stop either quits when they like or labels on through the collection.
#: ``None`` (an arm) is no cap.
STOP_CAP = 200


def stop_lights(
    smart: Status, stable: Status, span: Status, rule: str, level_met: bool, at_cap: bool = False
) -> tuple[Status, Status, Status]:
    """The lights :func:`next_phase` reads under stop *rule*, given the real ones and whether the level is met.

    ``"lights"`` passes them through.  A rule that says stop passes all three
    green, so the phase goes to the check before Done and to Done as the app's
    own all-green does; one that does not, with every real light green, holds
    Smart yellow, so the phase reads Hard.  Otherwise the real lights stand.
    *at_cap* (the session holds :data:`STOP_CAP` votes, #4903) says stop under
    every rule.
    """
    if rule not in STOP_RULES:
        raise ValueError(f"stop rule must be one of {STOP_RULES}, got {rule!r}")
    if at_cap:
        return "green", "green", "green"
    if rule == "lights":
        return smart, stable, span
    lights = smart == "green" and stable == "green" and span == "green"
    if rule == "level":
        stop = level_met
    elif rule == "lights+level":
        stop = lights and level_met
    else:
        stop = lights or level_met
    if stop:
        return "green", "green", "green"
    if lights:
        return "yellow", stable, span
    return smart, stable, span


def next_phase(  # noqa: C901 - a line-for-line port of checkPhaseTransition, kept diffable by eye
    good_count: int,
    bad_count: int,
    *,
    remaining_unlabeled: float,
    smart: Status,
    stable: Status,
    span: Status,
    good_target: int = GOOD_TARGET,
    bad_target: int = BAD_TARGET,
    more_target: int = MORE_TARGET,
    more_done: bool = False,
    dry_run_stop: bool = False,
    ran_dry: bool = False,
    good_ran_dry: bool = False,
    checking: bool = False,
    done_checked: bool = True,
) -> Phase:
    """Port of ``AutopilotStateService.checkPhaseTransition``.

    *remaining_unlabeled* is how many pool items still carry no vote; pass
    ``math.inf`` when the collection size is unknown (the app's "size unknown"
    case, which leaves the targets uncapped and never exhausts).

    The phase is derived from counts and indicator statuses rather than
    accumulated, so it can move backwards — un-toggling votes regresses the
    phase, exactly as it does in the app.  The one exception is the ``more``
    walk: whether it has run dry is history, not a count, so the caller holds
    it and passes *more_done* (see :class:`AutopilotFlow`).

    *dry_run_stop* is a document dataset (the labeling status's
    ``stop_rule == "dry_run"``, #4488): the walk has no Good target, runs in
    retrain mode too, and *ran_dry* (its run of misses complete) is ``done``.
    The indicators are not read.

    *good_ran_dry* is the Good walk's dry run (#4731): :data:`MORE_DRY_RUN` of
    its picks in a row held no Good, with a Good in hand, so the Goods found so
    far meet its target.  History, like *more_done*: the caller holds it
    (:class:`AutopilotFlow`), and the walk down the same sort is then spent.

    *checking* is a spot check running inside the run (the app's
    ``spotCheck``): the phase is ``spot`` whatever the counts.  *done_checked*
    is whether the run's check before Done has closed (the app's
    ``doneChecked``): until it has, all three lights green read ``spot``, the
    check the app runs before it reports Done.  The default, ``True``, reads
    all green as ``done``, as before the check existed; :class:`AutopilotFlow`
    holds both and passes them.
    """
    # Cap each target at the most votes of that class the collection could still
    # yield, so a tiny dataset can still advance past the initial phases instead
    # of stranding in ``good`` forever.
    eff_good_target = min(good_target, good_count + remaining_unlabeled)
    # A Good walk that ran dry has met its target with the Goods it found (#4731).
    if good_ran_dry and good_count >= 1:
        eff_good_target = min(eff_good_target, good_count)
    eff_bad_target = min(bad_target, bad_count + remaining_unlabeled)

    if checking:
        return "spot"
    if good_count < eff_good_target:
        return "good"
    if bad_count < eff_bad_target:
        return "bad"
    if dry_run_stop:
        if ran_dry:
            return "done"
        if remaining_unlabeled == 0:
            return "exhausted"
        return "more"
    if not more_done and good_count < min(more_target, good_count + remaining_unlabeled):
        return "more"
    if smart == "green" and stable == "green" and span == "green":
        # Check the line once before reporting Done; nothing left unlabeled has nothing to check.
        return "done" if done_checked or remaining_unlabeled == 0 else "spot"
    if remaining_unlabeled == 0:
        return "exhausted"
    if smart == "green" and stable == "green":
        return "new"
    return "hard"


#: Phases in which the app has run a learned sort, so a trained detector's
#: scores and threshold are on screen and driving the next pick.  Before this
#: the user is still on the text/example sort (see the module docstring).
TRAINED_PHASES: frozenset[str] = frozenset({"hard", "new", "spot", "done", "exhausted"})

#: The phases whose Bads count toward #5477's dry run (:data:`HARD_DRY_RUN`):
#: the ones where Autopilot is still picking off a learned sort.  Past Done the
#: app records a vote as list review, not Autopilot's, so ``done`` and
#: ``exhausted`` do not count.
HARD_DRY_PHASES: frozenset[str] = frozenset({"hard", "new", "spot"})


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

#: Autopilot's opening: the phases before its first learned sort.
OPENING_PHASES: frozenset[str] = frozenset({"good", "bad", "more"})


def capped_in_opening(phase: str, good_count: int, bad_count: int, stop_cap: int | None = STOP_CAP) -> bool:
    """The app's ``cappedInOpening`` (#4903): the run holds *stop_cap* votes still in its opening.

    The click cap's stop for a session with no detector: the app says Autopilot
    has stopped and offers a different sort.  It is display only there, so the
    phase - and every pick - is :func:`next_phase`'s, and a session that finds
    its Goods later goes to the check before Done and Done at once.  A reader
    asking where such a session was told to stop reads it here, beside
    :data:`STOPPING_PHASE`.
    """
    return stop_cap is not None and phase in OPENING_PHASES and good_count + bad_count >= stop_cap


#: What the app records for a vote Autopilot surfaced, per labelling phase:
#: the ranking on screen and how the item was drawn off it.  Ported from the
#: label view's phase subscription (``label-view.component.ts``): ``good`` and
#: ``bad`` draw off the seed sort (``top`` then ``hard``), ``hard`` off the
#: learned sort's cut band, and ``new`` defers to the coverage atlas with the
#: learned sort on screen.  The harness has no retrain mode, where the app draws
#: ``good`` / ``bad`` off the learned sort instead, and its seed sort is a text
#: sort or an example sort; both are ``text`` here, which is all a reader of the
#: record distinguishes (learned or not).  Read only by the deprecated
#: :func:`pick_provenance`, and no longer checked against the app (#4362).
_PHASE_PICKS: dict[str, tuple[str, str]] = {
    "good": ("text", "top"),
    "bad": ("text", "hard"),
    "more": ("text", "top"),
    "hard": ("learned", "hard"),
    "new": ("learned", "new"),
}


def pick_provenance(phase: Optional[str]) -> Optional[dict[str, str]]:
    """Deprecated (#4362): the surfacing provenance the app would record for a vote Autopilot surfaced in *phase*.

    The harness recorded it on every simulated click so it could filter the
    #4220 estimate's evidence with the app's own
    :func:`~vtscore.datasets.vote_provenance.calibrates_precision` (#4245).
    Neither side builds that estimate any more, so no step records it, and
    ``scripts/gates/check-eval-app-sync.py`` no longer checks :data:`_PHASE_PICKS`
    against the label view.  It still answers from that table, with a
    ``DeprecationWarning``, until the name is removed.  ``None`` outside the
    labelling phases.
    """
    warnings.warn(
        "pick_provenance() is deprecated: the harness no longer records vote provenance, since neither it nor "
        "the app builds the #4220 precision estimate any more (#4362). It will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    if phase not in _PHASE_PICKS:
        return None
    sort_kind, select_mode = _PHASE_PICKS[phase]
    return {"flow": "autopilot", "phase": phase, "select_mode": select_mode, "sort_kind": sort_kind}


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


def app_has_detector(phase: str, *, more_shown: bool = False) -> bool:
    """Whether the app would have a trained detector on screen in *phase*.

    The harness records this per step as ``app_trained``: a threshold computed
    at a step where this is false is a number no user ever sees, and studies
    about threshold quality (issue #2788) must filter on it rather than
    counting every simulated step.

    *more_shown* is #4637's arm, not the app: the ``more`` walk draws off the
    detector's ranking, so the detector is on screen from the end of the Bad
    phase (#4604).
    """
    if is_startup_phase(phase):
        # A schedule's rounds are on the seed sort by construction, so the app
        # would have no detector on screen however many votes have been cast.
        return False
    if more_shown and phase == "more":
        return True
    return phase in TRAINED_PHASES


class AutopilotFlow:
    """Per-trajectory phase state for one simulated Autopilot session.

    Holds the two inputs the indicators need — the Smart window of F-beta
    points and the accumulated per-step prediction-flip counts (Stable) — and
    recomputes the phase after every vote.  The atlas span is read from the
    harness's atlas, which it labels in lock-step with the votes.

    After every :meth:`update` the three indicator lights that produced the
    phase are left on :attr:`smart` / :attr:`stable` / :attr:`span` (with the
    raw :attr:`span_level` / :attr:`span_depth` behind the last of them), which
    is what lets a run say *which* rule held it short of ``done``.

    Note the asymmetry between the two, which is the app's: Stable's flip counts
    are a genuine history (each entry compares two consecutive models and can
    never be recomputed later), whereas Smart's points are *not* a history at
    all.  The app re-scores its last :data:`SMART_WINDOW` cached models against
    the current labelset on every poll, so the whole window is replaced each
    step; accumulating one frozen ``(model_t, labelset_t)`` point per step would
    confound model improvement with labelset growth (issue #2923).
    """

    def __init__(
        self,
        *,
        good_target: int = GOOD_TARGET,
        bad_target: int = BAD_TARGET,
        more_target: int = MORE_TARGET,
        more_dry_run: int = MORE_DRY_RUN,
        span_green: int | None = None,
        startup: Optional[StartupState] = None,
        dry_run_stop: bool = False,
        smart_gate: str = "app",
        good_dry_run: int | None = MORE_DRY_RUN,
        done_check: bool = True,
        stop_rule: str = "lights|level",
        stop_cap: int | None = STOP_CAP,
        smart_signal: str = APP_SMART_SIGNAL,
        record_readings: bool = False,
        min_good: int = MIN_PER_CLASS,
        hard_dry_run: int | None = HARD_DRY_RUN,
    ):
        self.good_target = good_target
        self.bad_target = bad_target
        #: The Goods Smart and Stable need before either leaves red (#5477):
        #: :data:`MIN_PER_CLASS`, the app's, unless an arm lowers it.  The Bads'
        #: minimum stays :data:`MIN_PER_CLASS`.
        if isinstance(min_good, bool) or not isinstance(min_good, int) or not 1 <= min_good <= MIN_PER_CLASS:
            raise ValueError(f"min_good must be an integer from 1 to {MIN_PER_CLASS}, got {min_good!r}")
        self.min_good = min_good
        #: #5477's dry run past the opening: once this many votes in a row in
        #: :data:`HARD_DRY_PHASES` held no Good, the Goods minimum drops to the
        #: Goods in hand - the target has shown what it can supply.  Latched.
        #: The app's is :data:`HARD_DRY_RUN`; ``None`` never waives it.
        if hard_dry_run is not None and (
            isinstance(hard_dry_run, bool) or not isinstance(hard_dry_run, int) or hard_dry_run < 1
        ):
            raise ValueError(f"hard_dry_run must be an integer >= 1 or None, got {hard_dry_run!r}")
        self.hard_dry_run = hard_dry_run
        self.dry_run = HardDryRun(target=hard_dry_run)
        self.more_target = more_target
        self.more_dry_run = more_dry_run
        #: The Good walk's dry run (#4731, the app's ``moreDryRun``): the Good
        #: phase also ends once this many of its picks in a row held no positive,
        #: with a Good in hand, and the ``more`` walk down the same sort is then
        #: spent.  ``None`` is the pre-#4731 arm, which waits for ``good_target``
        #: Goods however long the walk.  A startup schedule owns its own opening,
        #: so the run is off under one.
        if good_dry_run is not None and (
            isinstance(good_dry_run, bool) or not isinstance(good_dry_run, int) or good_dry_run < 1
        ):
            raise ValueError(f"good_dry_run must be an integer >= 1 or None, got {good_dry_run!r}")
        self.good_dry_run = None if startup is not None else good_dry_run
        #: The Good walk's misses since its latest Good, and whether it has run
        #: dry (latched, as the ``more`` walk's end is).
        self.good_misses = 0
        self.good_dry = False
        #: A document dataset (#4488): the walk is the rest of the run and its
        #: dry run is ``done`` (see :func:`next_phase`).
        self.dry_run_stop = dry_run_stop
        #: #4359's bound: ``"never"`` holds Smart yellow for the phase decision
        #: (the light is still computed and reported) and reads no stop level
        #: (#4878), so a session stays in ``hard`` until the run ends.  ``"app"``
        #: is the app.
        if smart_gate not in SMART_GATES:
            raise ValueError(f"smart_gate must be one of {SMART_GATES}, got {smart_gate!r}")
        self.smart_gate = smart_gate
        #: What the Smart light is read off (:data:`SMART_SIGNALS`, #5134):
        #: :data:`APP_SMART_SIGNAL` is the app.  Its inputs come through
        #: :meth:`record_step`.
        if smart_signal not in SMART_SIGNALS:
            raise ValueError(f"smart_signal must be one of {SMART_SIGNALS}, got {smart_signal!r}")
        self.smart_signal = smart_signal
        #: Whether every reading's trend is kept for the row (#5134's
        #: ``smart_readings``); off, :attr:`smart_reading_columns` stays NaN.
        self.record_readings = record_readings
        #: What tells the user they can stop (:data:`STOP_RULES`).  The app's,
        #: ``"lights|level"``, and the level arms read *level_met*, which the
        #: caller passes to :meth:`update` from the step's labels line; without
        #: it the app's rule is the lights alone.
        if stop_rule not in STOP_RULES:
            raise ValueError(f"stop_rule must be one of {STOP_RULES}, got {stop_rule!r}")
        self.stop_rule = stop_rule
        #: The click cap (:data:`STOP_CAP`, #4903): Done at this many votes
        #: whatever the rule; ``None`` is no cap.
        if stop_cap is not None and (isinstance(stop_cap, bool) or not isinstance(stop_cap, int) or stop_cap < 1):
            raise ValueError(f"stop_cap must be an integer >= 1 or None, got {stop_cap!r}")
        self.stop_cap = stop_cap
        #: The ``more`` walk's history, as the app keeps it: consecutive walk
        #: picks without a positive, and whether the walk has ended (met its
        #: target or ran dry).  A vote's outcome is read the way the app reads
        #: it, from which count rose since the last update.
        self.more_misses = 0
        self.more_done = False
        #: The spot check running inside the run, as the app's ``spotCheck``:
        #: ``"weak"`` for weak separation (#4496), ``"done"`` for the check
        #: before Done, ``"early"`` for the harness's #4664 arm, or ``None``.  The caller drives it
        #: (:meth:`begin_spot_check` / :meth:`end_spot_check`), since the
        #: check's picks are the caller's.
        self.spot_check: str | None = None
        #: Whether the run's check before Done has closed (the app's
        #: ``doneChecked``), latched for the run.  ``done_check=False`` starts
        #: it closed: all green reads ``done`` at once, the app before the
        #: check existed (and a run that never checks, ``spot_check="off"``).
        self.done_checked = not done_check
        #: The last :meth:`update`'s inputs to :func:`next_phase`, so a check
        #: starting or ending re-derives the phase without a vote.
        self._last_args: dict[str, Any] | None = None
        self._counts: tuple[int, int] = (0, 0)
        self.span_green = SPAN_GREEN_DEFAULT if span_green is None else span_green
        #: A parameterised opening (issue #3267).  ``None`` - the default - is
        #: the app's own: the Good/Bad targets above, resolved by
        #: :func:`next_phase`.  When set, the schedule owns every phase until it
        #: runs out and the app's machine takes over from ``hard``.
        self.startup = startup
        self.phase: Phase = "good" if startup is None else startup.phase_name()  # type: ignore[assignment]
        #: The ``cost`` arm's window (#5134): the last :data:`SMART_WINDOW` models'
        #: FPR + FNR, all against the *current* labelset.  Replaced, never appended.
        self.recent_error_costs: list[float] = []
        #: The Smart window (#5134): the last :data:`SMART_WINDOW` models' readings
        #: against the *current* labelset, ``fbeta_best`` the app's (replaced, never
        #: appended), and the per-retrain histories (appended, the last
        #: :data:`SMART_WINDOW` finite values kept).  Whatever the caller hands
        #: :meth:`record_step`.
        self.smart_windows: dict[str, list[float]] = {}
        self.smart_histories: dict[str, list[float]] = {}
        #: Each reading's trend as of the last :meth:`update`, by ``smart_<reading>_<part>``
        #: (:data:`SMART_READING_PARTS`); NaN where too few points, or too few
        #: votes of either class, to fit one.
        self.smart_reading_columns: dict[str, float] = reading_columns_nan()
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
        #: Smart is green when ``smart_slope <= SMART_RISE_THRESHOLD`` **or**
        #: ``smart_slope_t < SMART_SLOPE_T`` - a disjunction, which is why both
        #: are kept rather than one summary distance (#5134; under the ``cost``
        #: arm, ``>= SMART_FLAT_THRESHOLD`` or ``> -SMART_SLOPE_T``).  They are the
        #: margins of the light the run reads.  ``smart_slope_t`` is ``±inf`` on
        #: a window with no residual at all.
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
        windows: dict[str, list[float]] | None = None,
        history: dict[str, float | None] | None = None,
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

        *windows* and *history* are #5134's readings (:data:`SMART_READINGS`):
        the re-scored window in other units, which replaces the last one as the
        costs do, and this retrain's value of each history reading, appended
        where finite (``None`` or NaN adds nothing: no class model, no folds).
        """
        if error_costs is not None:
            self.recent_error_costs = list(error_costs)
        if windows is not None:
            self.smart_windows = {name: list(values) for name, values in windows.items()}
        for name, value in (history or {}).items():
            if value is not None and math.isfinite(value):
                kept = self.smart_histories.setdefault(name, [])
                kept.append(float(value))
                del kept[:-SMART_WINDOW]
        if scores is None:
            return
        if threshold is None:
            raise ValueError("a threshold is required alongside the pool scores")
        snapshot = ScoredSnapshot.from_scores(scores, threshold)
        pool_size = len(snapshot.scores) if num_pool is None else num_pool
        if self._prev_snapshot is not None:
            self.stability.append(stability_entry(self._prev_snapshot, snapshot, pool_size))
        self._prev_snapshot = snapshot

    def update(
        self,
        good_count: int,
        bad_count: int,
        remaining_unlabeled: float,
        span: dict[str, Any] | None,
        *,
        last_vote_good: Optional[bool] = None,
        level_met: bool = False,
    ) -> Phase:
        """Recompute and return the phase after a vote.

        With a :class:`~vtscore.eval.startup_schedule.StartupState` attached the
        opening is the schedule's, not :data:`GOOD_TARGET` / :data:`BAD_TARGET`;
        once it is spent the app's machine resumes with both targets already
        met, so the trajectory continues into ``hard`` / ``new`` / ``done``
        exactly as it would have.  *last_vote_good* is the vote just cast; only
        a schedule round with a dry stop (#4222) reads it, and such a round
        refuses a vote without it.  *level_met* is whether the step's labels line
        expects at least the run's stop level (#4878); every ``stop_rule`` but
        ``"lights"`` reads it.
        """
        if self.startup is not None:
            self.startup.on_click(last_vote_good)
            self.startup.advance(good_count, bad_count, remaining_unlabeled)
            if not self.startup.done:
                self.phase = self.startup.phase_name()  # type: ignore[assignment]
                return self.phase
        else:
            self._note_hard_vote(good_count, bad_count)
            self._note_more_vote(good_count, bad_count)
        min_good = self.min_good_now(good_count)
        if self.smart_signal == "cost":
            # #5134's arm: the light before #5134, FPR + FNR going flat.
            smart_d = smart_status_from_costs(self.recent_error_costs, good_count, bad_count, min_good=min_good)
        else:
            smart_d = smart_detail(self.smart_reading(self.smart_signal), good_count, bad_count, min_good)
        stable_d = stable_detail(self.stability, good_count, bad_count, min_good)
        smart: Status = smart_d["status"]
        self.smart_reading_columns = (
            self._reading_columns(good_count, bad_count, min_good) if self.record_readings else reading_columns_nan()
        )
        stable: Status = stable_d["status"]
        # No atlas reading is red, as in the app, unless the Diversity step is off (#5473).
        sp: Status = (
            span_status(int(span["level"]), int(span["depth"]), self.span_green)
            if span is not None
            else ("green" if self.span_green <= 0 else "red")
        )
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
        # What the phase decision reads: the lights as computed, under #4359's
        # Smart gate and #4878's stop rule (both the app's by default).
        # #4359's bound holds the session in Hard to the end, so it reads no level, nor the cap (#4903), either.
        never = self.smart_gate == "never"
        at_cap = not never and self.stop_cap is not None and good_count + bad_count >= self.stop_cap
        read_smart, read_stable, read_span = stop_lights(
            "yellow" if never else smart, stable, sp, self.stop_rule, level_met and not never, at_cap
        )
        self._last_args = {
            "good_count": good_count,
            "bad_count": bad_count,
            "remaining_unlabeled": remaining_unlabeled,
            "smart": read_smart,
            "stable": read_stable,
            "span": read_span,
            "good_target": 0 if self.startup is not None else self.good_target,
            "bad_target": 0 if self.startup is not None else self.bad_target,
            "more_target": 0 if self.startup is not None else self.more_target,
            "dry_run_stop": self.dry_run_stop,
            "ran_dry": self.more_misses >= self.more_dry_run,
            "good_ran_dry": self.good_dry,
        }
        return self._derive()

    def smart_reading(self, name: str) -> list[float]:
        """Reading *name*'s current window, oldest first (:data:`SMART_READINGS`, #5134)."""
        if name == "cost":
            return list(self.recent_error_costs)
        if name in SMART_WINDOW_READINGS:
            return list(self.smart_windows.get(name, []))
        return list(self.smart_histories.get(name, []))

    @property
    def hard_dry(self) -> bool:
        """#5477's dry run past the opening has run its course (latched)."""
        return self.dry_run.dry

    def min_good_now(self, good_count: int) -> int:
        """The Goods Smart and Stable need at this vote (#5477): :attr:`min_good`, or those in hand once dry."""
        return min(self.min_good, goods_minimum(good_count, self.hard_dry))

    def _reading_columns(self, good: int, bad: int, min_good: int = MIN_PER_CLASS) -> dict[str, float]:
        """Every reading's trend, as :data:`SMART_READING_PARTS` (#5134); NaN where the rule would not fit one."""
        out = reading_columns_nan()
        if good < min_good or bad < MIN_PER_CLASS:
            return out
        for name in SMART_READINGS:
            window = [v for v in self.smart_reading(name) if math.isfinite(v)][-SMART_WINDOW:]
            if len(window) < SMART_MIN_POINTS:
                continue
            trend = cost_trend(window)
            out[f"smart_{name}_slope"] = trend.relative_slope
            out[f"smart_{name}_mean"] = sum(window) / len(window)
            out[f"smart_{name}_t"] = trend.t_stat
            out[f"smart_{name}_last"] = window[-1]
        return out

    def _derive(self) -> Phase:
        """The phase from the last :meth:`update`'s inputs and the history held here."""
        assert self._last_args is not None
        self.phase = next_phase(
            **self._last_args,
            more_done=self.more_done,
            checking=self.spot_check is not None,
            done_checked=self.done_checked,
        )
        if self.phase not in ("good", "bad", "more"):
            # The walk ends once the machine has moved past it - its target met
            # or its run of misses complete - and never resumes, as a schedule
            # round never does.
            self.more_done = True
        return self.phase

    def begin_spot_check(self, kind: str) -> Phase:
        """A spot check starts inside the run (the app's ``beginSpotCheck``): the phase is ``spot`` until it ends.

        *kind* is ``"weak"`` (weak separation, #4496), ``"done"`` (the check
        before Done) or ``"early"`` (the eval harness's check at a fixed vote
        count, #4664, which the app does not run).  Before any :meth:`update`
        there is nothing to derive from, and the phase is left as it is.
        """
        if kind not in ("weak", "done", "early"):
            raise ValueError(f"kind must be 'weak', 'done' or 'early', got {kind!r}")
        if self.spot_check is not None:
            return self.phase
        self.spot_check = kind
        return self._derive() if self._last_args is not None else self.phase

    def end_spot_check(self) -> Phase:
        """The running check closed, finished or cut short (the app's ``endSpotCheck``); the check before Done latches."""
        if self.spot_check is None:
            return self.phase
        if self.spot_check == "done":
            self.done_checked = True
        self.spot_check = None
        return self._derive() if self._last_args is not None else self.phase

    def skip_done_check(self) -> Phase:
        """The check before Done does not run (the app's ``skipDoneCheck``): latch it, and the phase goes on."""
        self.done_checked = True
        return self._derive() if self._last_args is not None else self.phase

    def _note_hard_vote(self, good_count: int, bad_count: int) -> None:
        """Fold the vote just cast into #5477's dry run past the opening, read off the counts as the walks are.

        Called before :meth:`_note_more_vote`, which moves the counts on.  The
        phase held here is the one the vote was cast in.
        """
        prev_good, prev_bad = self._counts
        if good_count > prev_good:
            self.dry_run.note("good", True)
        elif bad_count > prev_bad:
            self.dry_run.note("bad", self.phase in HARD_DRY_PHASES)

    def _note_more_vote(self, good_count: int, bad_count: int) -> None:
        """Fold the vote just cast into the ``more`` walk's run of misses.

        The app's phase check sees only vote counts, so it reads the outcome off
        them: a rise in the Good count is a hit, a rise in the Bad count alone is
        a miss.  Only votes cast *in* the walk count; on a document dataset the
        walk counts even once ``more_done`` has latched, since there it is the stop.
        """
        prev_good, prev_bad = self._counts
        self._counts = (good_count, bad_count)
        if self.phase == "good" and self.good_dry_run is not None and not self.good_dry:
            # #4731: the same reading for the Good walk, as checkPhaseTransition takes it.  A miss counts only
            # with a Good already in hand, so the run is the misses since the
            # latest Good, and a walk that has found nothing never runs dry.
            if good_count > prev_good:
                self.good_misses = 0
            elif bad_count > prev_bad and prev_good >= 1:
                self.good_misses += 1
                if self.good_misses >= self.good_dry_run:
                    self.good_dry = True
                    self.more_done = True
            return
        if self.phase != "more" or (self.more_done and not self.dry_run_stop):
            return
        if good_count > prev_good:
            self.more_misses = 0
        elif bad_count > prev_bad:
            self.more_misses += 1
            if self.more_misses >= self.more_dry_run:
                self.more_done = True
