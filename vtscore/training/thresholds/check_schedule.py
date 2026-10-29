"""The balance's knobs and the check's schedule: its presets, its bands and what a check costs (#4413, #4383).

What the rest of the spot check reads: the balance's presets and range
(F-beta's beta), the cap a balance holds, the bands a walk cuts the ranking
into (:func:`band_edges`), the schedule a check at one balance runs
(:func:`balance_schedule`) and the shape it takes (:func:`check_shape`), the
states a check and a balance report, F-beta itself (:func:`fbeta_score`),
the grid a line is placed on (:func:`line_under`), and how the eval harness
resolves an unpinned balance (:func:`resolve_line_knobs`).  Plain Python: no
numpy, and no sibling module.

Split out of :mod:`~vtscore.training.thresholds.spot_check` (#5425), which
re-exports every name here.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import Sequence

#: The level the likely range is drawn at: each band's tail (#4257's alpha).
#: #4272 split it over the bands in the set; #4483 dropped the split.
CHECK_ALPHA = 0.05

#: The balance's cap at beta <= 1: the top 32 (#4257's ``a:top32``, the #4267
#: ruling).  Above beta 1 it is :data:`CHECK_RECALL_CANDIDATE`.
CHECK_BASE_CANDIDATE = 32
#: The balance's cap above beta 1, where recall weighs more: the top 128.
CHECK_RECALL_CANDIDATE = 128

#: The first band's size, and the smallest set a walk can keep (#4383).
BAND_BASE = 8

#: The picks a band is audited with (owner, 2026-09-30: 5 picks a band).
CHECK_MIN_PICKS = 5

#: The balance (#4413): the user's precision/recall preference as F-beta's beta.
#: The presets are precision-leaning, balanced and recall-leaning; any beta in
#: ``[BETA_MIN, BETA_MAX]`` is accepted.  The ends are the range's own ends, 1/4
#: and 4 (the owner's pick of 2026-10-03 on #4448, priced on #4452's line: past
#: 1/3 and 3 each end buys at most 0.02 more of what it leans toward); they were
#: 0.5 and 2.  A line no check has measured is ``unchecked``; a finished walk is
#: ``checked``.
BALANCE_PRESETS: tuple[float, ...] = (0.25, 1.0, 4.0)
DEFAULT_BETA = 1.0
BETA_MIN, BETA_MAX = 0.25, 4.0
#: How a check treats the line at a balance (:func:`check_shape`):
#: ``advisory``, the app's at every preset (#4452) - the walk runs and its
#: votes train, but the line stays where the labels (or, with no trained head,
#: the unchecked count rule) put it - and ``trim`` (#4427's pricing), which the
#: eval harness can force - the walk audits the bands holding the line, may
#: only step shallower, and the line takes its end.
CHECK_ADVISORY = "advisory"
CHECK_TRIM = "trim"
CHECK_SHAPES: tuple[str, ...] = (CHECK_ADVISORY, CHECK_TRIM)
BALANCE_UNCHECKED = "unchecked"
BALANCE_CHECKED = "checked"
#: A structural detector's line (#4505): the verification gate's boundary, not a
#: cut on a ranking, so no check applies to it and it is neither unchecked nor
#: checked.  Its count is what the gate passes (:func:`gate_balance_state`).
BALANCE_GATE = "gate"
BALANCE_STATES = (BALANCE_UNCHECKED, BALANCE_CHECKED, BALANCE_GATE)

#: The surfacing provenance a check's vote is recorded with
#: (:mod:`vtscore.datasets.vote_provenance`): the app's route and the eval
#: harness both write this one dict, so the two cannot drift.
CHECK_PROVENANCE: dict[str, str] = {"flow": "check"}

#: A check's status while its bands are still being voted on, and once it was
#: abandoned (its votes so far stay ordinary votes; the line's state is as it was).
CHECK_RUNNING = "running"
CHECK_CANCELLED = "cancelled"

#: Which way the walk last moved: where it started, one band deeper, or one
#: band shallower.
WALK_START = "start"
WALK_DEEPER = "deeper"
WALK_SHALLOWER = "shallower"

_EPS = 1e-9

#: Every response rounds a score to this many decimals (``round(score, 4)``),
#: and a threshold to the same.  The line is placed on that grid so the last
#: item of the set it keeps clears it whether a consumer compares the raw score
#: or the rounded one.
LINE_DECIMALS = 4


def line_under(score: float) -> float:
    """*score* floored to :data:`LINE_DECIMALS`: the highest response-grid value the item still clears.

    ``math.floor(score * 1e4) / 1e4`` can land a decimal short (``0.4657``
    reads as ``4656.999...``), so this goes through :class:`~decimal.Decimal`.
    """
    quantum = Decimal(1).scaleb(-LINE_DECIMALS)
    return float(Decimal(repr(float(score))).quantize(quantum, rounding=ROUND_FLOOR))


# ------------------------------------------------------------------ the bands


def band_edges(n: int, base: int = BAND_BASE) -> tuple[int, ...]:
    """``(0, 8, 16, 32, 64, ..., n)``: where the bands of a ranking of *n* items start and end.

    Band ``b`` is the ranks ``[edges[b], edges[b + 1])``.  The last band runs
    to the end of the ranking, however short that leaves it; a ranking of
    fewer than *base* items is one band.  The reference's ``band_edges``.
    """
    if n <= 0:
        return (0,)
    edges = [0]
    e = base
    while e < n:
        edges.append(e)
        e *= 2
    edges.append(n)
    return tuple(edges)


def band_edges_fine(n: int, start: int, base: int = BAND_BASE) -> tuple[int, ...]:
    """:func:`band_edges` with each band past *start* split in two (#4427's ``fine`` walk arm).

    The bands up to the edge holding *start* are the shipped ones; beyond it
    the midpoint of every band is an edge too, so a walk from 32 steps 48, 64,
    96, 128, ... and can end between the shipped edges, where the best cut's
    size sits on the bench (median ~46 at beta 1).
    """
    edges = list(band_edges(n, base))
    out: list[int] = []
    for a, b in zip(edges, edges[1:]):
        out.append(a)
        if a >= start and b - a >= 2:
            out.append(a + (b - a) // 2)
    out.append(edges[-1])
    return tuple(out)


def bands_for(count: int, edges: Sequence[int]) -> int:
    """How many bands from the top it takes to hold the top *count*: at least one."""
    for b in range(1, len(edges)):
        if edges[b] >= count:
            return b
    return max(1, len(edges) - 1)


# ------------------------------------------------------------------ the schedule


@dataclass(frozen=True)
class CheckSchedule:
    """What a check at one balance costs: its starting candidate, rounds and picks a round."""

    candidate: int
    rounds: int
    picks: int

    def as_dict(self) -> dict[str, int]:
        return {"candidate": self.candidate, "rounds": self.rounds, "picks": self.picks}


def balance_schedule(beta: float, alpha: float = CHECK_ALPHA) -> CheckSchedule:
    """``(K, bands, m)`` for a balance *beta* (#4413): the unchecked cap, the bands that hold it, the picks a band.

    *K* is :data:`CHECK_BASE_CANDIDATE` (the top 32) at beta <= 1 and
    :data:`CHECK_RECALL_CANDIDATE` (the top 128) above it: the counts the
    precision floor's schedule kept at 50% and at 10% (the #4267 ruling),
    which the balance inherited for the preset it leans toward.  It is what
    holds the mixture's F-beta argmax on a large sparse corpus, where the
    mixture over-counts the positives and would return 2-4x too much (#4411);
    the walk starts from the same bands.  ``rounds`` is how many bands the walk
    audits before its first verdict, the bands that hold *K*
    (:func:`rounds_for`); ``picks`` is what each band costs,
    :data:`CHECK_MIN_PICKS`.
    """
    if not (BETA_MIN - _EPS <= float(beta) <= BETA_MAX + _EPS):
        raise ValueError(f"beta must be in [{BETA_MIN}, {BETA_MAX}], got {beta!r}")
    candidate = CHECK_BASE_CANDIDATE if float(beta) <= 1.0 + _EPS else CHECK_RECALL_CANDIDATE
    return CheckSchedule(candidate, rounds_for(candidate), CHECK_MIN_PICKS)


def check_shape(beta: float) -> str:
    """How a check at *beta* treats the line: ``advisory`` at every preset since #4452.

    Since #4452 the app's line comes from the labelset alone (the class model
    the calibration folds imply, at the prevalence the detector's evidence
    pools; :mod:`~vtscore.training.thresholds.labels_line`), so a walk's end -
    a count on the corpus it walked - cannot be the line: an exported labelset
    would not reproduce it.  The check audits and reports, its picks are
    ordinary votes the next retrain learns from (uniform picks within bands,
    the least biased evidence the labels hold), and the line stays where the
    labels put it.  ``trim`` remains a shape the eval harness can force.

    #4427's pricing, which set ``trim`` above beta 1 for the count-based line:

    Priced on the objective (the withheld set's F-beta above the app's
    threshold; Binary, 5 seeds, every preset): the full walk that moved the
    line to its peak lost at beta 0.5 and 1 (-0.06 and -0.03 against the
    unchecked line) by buying recall with bands that were mostly wrong.  An
    advisory check - its votes train, the line stays at the unchecked rule -
    was the best there (+0.09 and +0.04 over the walk) and the worst at beta
    2 (-0.02), where its votes tighten the model and the re-drawn line
    returns too few for a recall-leaning balance.  A walk that may only trim
    the line was never worse than the full walk and the best at beta 2
    (+0.01).  So the check's direction follows the preset the user chose.
    """
    del beta  # every preset (#4452)
    return CHECK_ADVISORY


#: How an eval arm whose ``None`` already means "the shipped default" spells
#: *no balance*: the Inclusion arm, whose line is the fold-anchored cut at the
#: arm's ``inclusion``.
NO_BALANCE = "off"


def resolve_line_knobs(beta: float | str | None) -> float | None:
    """The balance an eval arm draws its line at; ``None`` (unpinned) is the app's default (#4413).

    The harness's counterpart of :func:`vtscore.state.line_knobs`, which is
    what the app hands every retrain and re-cut.  ``None`` - the default arm -
    resolves to :data:`DEFAULT_BETA`, what an unset user setting resolves to;
    :data:`NO_BALANCE` is the Inclusion arm (``None``: no balance draws the
    line); a number pins the balance arm.  Validated here, so a malformed arm
    fails before anything expensive runs.
    """
    if beta is None:
        return DEFAULT_BETA
    if beta == NO_BALANCE:
        return None
    if isinstance(beta, (str, bool)):
        raise ValueError(f"beta must be a number in [{BETA_MIN}, {BETA_MAX}], None or {NO_BALANCE!r}; got {beta!r}")
    balance_schedule(float(beta))
    return float(beta)


def fbeta_score(tp: float, k: float, n_pos: float, beta: float) -> float:
    """F-beta of a set of *k* items holding *tp* positives, out of *n_pos* in the corpus."""
    denominator = beta * beta * n_pos + k
    return (1.0 + beta * beta) * tp / denominator if denominator > 0 else 0.0


def rounds_for(candidate: int, base: int = BAND_BASE) -> int:
    """How many bands a walk audits before its first verdict: the bands from the top that hold *candidate*.

    3 for the top 32 (8, 8, 16), 4 for 64, 5 for 128; a corpus smaller than
    the candidate is its own last band.
    """
    return bands_for(candidate, band_edges(max(candidate, 1), base))
