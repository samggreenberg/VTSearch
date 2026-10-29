"""The test sample's vocabulary: its constants and value types (#4527).

Everything :mod:`~vtscore.training.thresholds.line_test`'s sample reads and
reports, but not the sample itself: the level and draws the ranges are taken
at, the sides, phases and stop reasons, the provenance a test vote carries,
the budgets (:class:`LineBudgets`), the bands (:class:`Band`,
:func:`line_bands`), the estimates (:class:`Estimate`, :class:`EdgeEstimate`,
:class:`LineEstimates`, :func:`found_words`), the phase report
(:class:`PhaseReport`) and the round dealt ahead (:class:`RoundAhead`).  The
app's routes and schemas, the kept verdicts and the eval harness read these
without needing :class:`~vtscore.training.thresholds.line_test.LineTest`, and
nothing here reads it.

:mod:`~vtscore.training.thresholds.line_test` re-exports every name here, so
``thresholds.line_test.X`` resolves as it did when this was one module.  The
pool's knobs (``POOLED_WEIGHT``, ``POOL_RADIUS``, ``FIRST_PASS_BEFORE_BUDGET``
and ``pooled_weight``) stay there, beside the sample that reads them, so a
study that rebinds one (#4560) patches that module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from vtscore.training.thresholds.check_schedule import CHECK_MIN_PICKS, band_edges
from vtscore.training.thresholds.ranges import LikelyRange

#: The level the ranges are drawn at: a central ``1 - alpha`` posterior range.
TEST_ALPHA = 0.05

#: Monte Carlo draws behind every range; one set per state, shared by all.
TEST_DRAWS = 4000

#: The seed the draws are taken at.  Fixed, so the ranges are a function of
#: the labels alone (a resumed test reports what the original did) and every
#: state shares one random stream, which steadies the look-ahead's comparisons.
TEST_DRAW_SEED = 0

#: The Jeffreys prior on the pooled share right above the line: ``Beta(1/2, 1/2)``.
JEFFREYS = 0.5

#: The floor under the model's prior on a band below the line, in pseudo-picks
#: either way: enough that a band the model counts empty stays correctable by
#: a pick that finds a match, too little to count as evidence of anything.
MODEL_FLOOR = 0.02

#: The surfacing provenance a test's vote is recorded with
#: (:mod:`vtscore.datasets.vote_provenance`): a test vote never trains.
TEST_PROVENANCE: dict[str, str] = {"flow": "test"}

#: Which side of the line a band is on.
ABOVE = "above"
BELOW = "below"
SIDES = (ABOVE, BELOW)

#: The phases a test moves through, derived from its state.  ``score`` is the
#: app's pass before a sample exists (listed so the vocabulary is whole);
#: ``matches`` checks precision above the line, ``misses`` recall below it;
#: ``done`` is the verdict; ``nothing`` says the line keeps fewer items than
#: one round, so there is nothing to test.
PHASE_SCORE = "score"
PHASE_MATCHES = "matches"
PHASE_MISSES = "misses"
PHASE_DONE = "done"
PHASE_NOTHING = "nothing"
PHASES = (PHASE_SCORE, PHASE_MATCHES, PHASE_MISSES, PHASE_DONE, PHASE_NOTHING)

#: Why a phase ended: its range got narrower than its target, its pick budget
#: was spent, its bands ran out of unlabelled items, or (below the line) the
#: walk hit a dry band.
STOP_WIDTH = "width"
STOP_BUDGET = "budget"
STOP_EXHAUSTED = "exhausted"
STOP_DRY_RUN = "dry_run"
STOP_REASONS = (STOP_WIDTH, STOP_BUDGET, STOP_EXHAUSTED, STOP_DRY_RUN)

#: The recall range in words, by its midpoint: the spot check's *found* words
#: (``foundWords`` in ``frontend/src/app/utils/line-balance.ts``), cut at 15,
#: 37.5, 62.5 and 87.5 percent.
FOUND_WORDS: tuple[tuple[float, str], ...] = (
    (0.15, "few of them found"),
    (0.375, "about a quarter of them found"),
    (0.625, "about half of them found"),
    (0.875, "about three quarters of them found"),
    (math.inf, "nearly all of them found"),
)

_EPS = 1e-9


def found_words(recall: LikelyRange | Estimate | float) -> str:
    """The recall range in words, from its midpoint: how many of all the matches the line likely ships.

    Words rather than a second percentage, as the spot check reads its recall
    range: the tail below the line is model-assisted, and words are its
    honest resolution.  Takes a range (its midpoint) or a bare share.
    """
    if isinstance(recall, (LikelyRange, Estimate)):
        mid = (recall.lo + recall.hi) / 2.0
    else:
        mid = float(recall)
    for cut, words in FOUND_WORDS:
        if mid < cut:
            return words
    return FOUND_WORDS[-1][1]


# ------------------------------------------------------------------ the budgets


@dataclass(frozen=True)
class LineBudgets:
    """What a test may spend and when a phase is narrow enough (Test mode's stop rule, #4523).

    *matches_width* is the precision range's target width and *misses_width*
    the recall range's; a phase ends on width once its range is at or under
    the target after at least one round of its own.  *matches_picks* and
    *misses_picks* cap each phase's picks, *picks_per_round* is a round (the
    spot check's 5), *dry_run_share* is the posterior mass, as a share of
    the positives estimated above the line, below which a band with no match
    in its picks ends the walk below the line, and *model_weight* is what the
    model's share of a band below the line is worth in picks before the
    band's own picks correct it (one round).

    *misses_width* and *dry_run_share* govern only a test with no class model
    (:attr:`LineTest.posteriors` is ``None``: a structural or document
    detector), whose walk stops at the first band with no match, every
    band's mass being zero; with a class model the walk below the line runs
    to *misses_picks*, because its recall range only holds once the walk
    has corrected the model's tail band by band.  The defaults are the values
    #4523 priced on 192,660 replayed Tests
    (``docs/experiments/2026-10-05-line-test-4523/REPORT.md``) and #4540
    re-priced with the pooled prior and the deeper walk
    (``docs/experiments/2026-10-06-test-budget-presets-4540/REPORT.md``): a
    width of 0.20 and 20 picks above the line, where the precision range holds
    the truth in 94-99% of sessions, as well as at 40 picks or better, for up
    to 13 fewer picks at beta 4; 40 picks below it, where walking to the
    budget raised the recall range's coverage from 13-38% to 81-98%.
    """

    matches_width: float = 0.20
    misses_width: float = 0.25
    matches_picks: int = 20
    misses_picks: int = 40
    picks_per_round: int = CHECK_MIN_PICKS
    dry_run_share: float = 0.05
    model_weight: float = CHECK_MIN_PICKS
    alpha: float = TEST_ALPHA
    draws: int = TEST_DRAWS

    def __post_init__(self) -> None:
        if not 0.0 < self.matches_width <= 1.0 or not 0.0 < self.misses_width <= 1.0:
            raise ValueError("a width target must be in (0, 1]")
        if self.matches_picks < 1 or self.misses_picks < 0:
            raise ValueError("the matches budget must be >= 1 and the misses budget >= 0")
        if self.picks_per_round < 1:
            raise ValueError("a round needs at least one pick")
        if self.dry_run_share < 0.0:
            raise ValueError("dry_run_share must be >= 0")
        if self.model_weight < 0.0:
            raise ValueError("model_weight must be >= 0")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.draws < 2:
            raise ValueError("draws must be >= 2")

    def as_dict(self) -> dict[str, Any]:
        return {
            "matches_width": self.matches_width,
            "misses_width": self.misses_width,
            "matches_picks": self.matches_picks,
            "misses_picks": self.misses_picks,
            "picks_per_round": self.picks_per_round,
            "dry_run_share": self.dry_run_share,
            "model_weight": self.model_weight,
            "alpha": self.alpha,
        }


DEFAULT_BUDGETS = LineBudgets()


# ------------------------------------------------------------------ the bands


@dataclass(frozen=True)
class Band:
    """One band of the sample: the ranks ``[lo, hi)`` of the frozen ranking, on one side of the line."""

    index: int
    side: str
    lo: int
    hi: int

    @property
    def size(self) -> int:
        return self.hi - self.lo

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "side": self.side, "lo": self.lo + 1, "hi": self.hi}


def line_bands(n: int, line_count: int) -> tuple[Band, ...]:
    """The bands of a ranking of *n* items whose line keeps the top *line_count*.

    Above the line, the spot check's bands from the top (8, 8, 16, 32, ...,
    cut at the line); below it, the same doubling from the line downward to
    the corpus's end.  Either side may be empty.
    """
    k = max(0, min(int(line_count), int(n)))
    bands: list[Band] = []
    if k > 0:
        edges = band_edges(k)
        for lo, hi in zip(edges, edges[1:]):
            bands.append(Band(len(bands), ABOVE, int(lo), int(hi)))
    if n - k > 0:
        edges = band_edges(n - k)
        for lo, hi in zip(edges, edges[1:]):
            bands.append(Band(len(bands), BELOW, k + int(lo), k + int(hi)))
    return tuple(bands)


# ------------------------------------------------------------------ the estimates


@dataclass(frozen=True)
class Estimate:
    """A point and its central ``1 - alpha`` range, from the joint draws."""

    point: float
    lo: float
    hi: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def holds(self, truth: float) -> bool:
        return self.lo - _EPS <= truth <= self.hi + _EPS

    def as_dict(self) -> dict[str, Any]:
        return {"point": round(self.point, 4), "lo": round(self.lo, 4), "hi": round(self.hi, 4)}


@dataclass(frozen=True)
class EdgeEstimate:
    """What the line would ship if it kept the top *count* (a band edge): the ranges from the same draws."""

    count: int
    side: str
    precision: Estimate
    recall: Estimate
    fbeta: Estimate

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "side": self.side,
            "precision": self.precision.as_dict(),
            "recall": self.recall.as_dict(),
            "fbeta": self.fbeta.as_dict(),
            "found": found_words(self.recall),
        }


@dataclass(frozen=True)
class LineEstimates:
    """Every number a test reports, from one set of joint draws.

    *precision*, *recall* and *fbeta* are the line's; *positives_above* and
    *positives_below* the counts they come from (the latter model-assisted:
    each reached band's model share, corrected by its picks); *tail_positives*
    is the model's count in the bands below the line the walk never reached,
    which is folded into *positives_below* as a point and flagged by
    *tail_from_model*; *at_edges* re-estimates the line at every band edge
    on both sides.  *labelled* is the picks the ranges rest on.
    """

    beta: float
    precision: Estimate
    recall: Estimate
    fbeta: Estimate
    positives_above: Estimate
    positives_below: Estimate
    tail_positives: float
    tail_from_model: bool
    at_edges: tuple[EdgeEstimate, ...]
    labelled: int

    @property
    def found(self) -> str:
        return found_words(self.recall)

    def as_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "precision": self.precision.as_dict(),
            "recall": self.recall.as_dict(),
            "fbeta": self.fbeta.as_dict(),
            "found": self.found,
            "positives_above": self.positives_above.as_dict(),
            "positives_below": self.positives_below.as_dict(),
            "tail_positives": round(self.tail_positives, 2),
            "tail_from_model": self.tail_from_model,
            "labelled": self.labelled,
            "at_edges": [e.as_dict() for e in self.at_edges],
        }


def _estimate(draws: np.ndarray, alpha: float) -> Estimate:
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Estimate(float(np.mean(draws)), float(lo), float(hi))


# ------------------------------------------------------------------ the phase


@dataclass(frozen=True)
class PhaseReport:
    """Where a test stands: its phase, and why each finished phase ended.

    *matches_stop* and *misses_stop* are ``None`` while that phase is still
    owed, else one of :data:`STOP_REASONS`.  *matches_width* and
    *misses_width* are the ranges' current widths, for the lights.
    """

    phase: str
    matches_stop: str | None
    misses_stop: str | None
    matches_width: float | None
    misses_width: float | None
    picks_above: int
    picks_below: int

    @property
    def done(self) -> bool:
        return self.phase in (PHASE_DONE, PHASE_NOTHING)

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "matches_stop": self.matches_stop,
            "misses_stop": self.misses_stop,
            "matches_width": None if self.matches_width is None else round(self.matches_width, 4),
            "misses_width": None if self.misses_width is None else round(self.misses_width, 4),
            "picks_above": self.picks_above,
            "picks_below": self.picks_below,
        }


@dataclass(frozen=True)
class RoundAhead:
    """The round the vote on a round's last pending pick would deal, each way it can go (#5116).

    *id* is that pick; *good* the next round's picks if it is voted a match,
    *bad* if it is not, in the order :meth:`LineTest.draw` deals them, and
    ``()`` where that vote ends the test.  Both ways name the same picks when
    both outcomes send the next round to the same band.  A client may fetch
    their media ahead of the vote, but not show them: only the round the vote
    names is dealt, and the other is discarded unseen.
    """

    id: int
    good: tuple[int, ...]
    bad: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "good": list(self.good), "bad": list(self.bad)}
