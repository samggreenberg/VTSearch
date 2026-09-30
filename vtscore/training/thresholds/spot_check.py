"""The spot check that decides the precision floor's line, and how close it got (#4272).

The floor *X* is a share of what the line returns that should be right.  The
owner's ruling on #4267 (2026-09-29) moved the promise off the #4220 estimator
(:mod:`~vtscore.training.thresholds.precision_floor`, which stays as public
library API that no app path reads since #4360 and #4362) and onto a **spot check**: the user
votes on uniform random picks from a candidate of the top unvoted items, and a
Clopper-Pearson bound on those picks decides.  #4256 showed that model-chosen
votes break 83% of the estimator's X = 50% promises once the reference pool is
consistent; a uniform pick has no such bias, so its bound holds at every
prevalence measured (#4257).

**Do your best, and say how close we got.**  The promise is not make-or-break:

* The line always keeps a set.  Before any check it keeps the schedule's
  unchecked starting candidate; after one, the set the check ended on.  Nothing
  falls back to the old Inclusion 0 cut (this replaces #4247's fallback).
* A checked set carries a **likely range** for how much of it is right: a
  Clopper-Pearson interval from the check's labels, with each tail at
  ``alpha / R``, the level each round is tested at (:func:`range_tail`).  So a
  check confirms *X* iff the range's lower end clears *X*.

**The rule**, priced in ``docs/experiments/2026-09-29-floor-candidate-4267/``
(``scripts/experiments/calibration/analyze_floor_candidate_4267.py`` is the
reference this module matches; ``scripts/check-eval-app-sync.py`` pins the two
against each other):

* **Candidate.**  The top ``K = 32 * 2**max(0, floor(log2(0.5 / X)))`` *unvoted*
  items of the current ranking: the top 128 at 10%, the top 64 at 25%, and the
  top 32 at 50% and above (:func:`check_schedule`).  Voted items are left out,
  because a model chose them.  A corpus with fewer unvoted items is its own
  candidate.  The candidate's ids are fixed when the check starts; its rounds
  sample that one list while the model retrains behind it.
* **Rounds.**  ``R = log2(K / 32) + 1``: 3 at 10%, 2 at 25%, 1 at 50% and above.
* **Picks.**  Each round draws ``m = max(5, ceil(ln(alpha / R) / ln X))`` fresh
  items uniformly from the current candidate: 5 at 10-50%, 11 at 75% and 29 at
  90%.  Labels already seen inside the current candidate are kept, which keeps
  every round's sample uniform (#4257's rule ``b``).
* **Confirm or halve.**  The round confirms *X* iff the range's lower end is at
  least *X*.  Otherwise the candidate halves, down to 32, and the next round
  runs.  A check that ends below *X* is ``short``, and there is no redraw on the
  same candidate.
* **The range is exact** (``s / K``) once the labels cover the candidate.

The three states a floor reports (:data:`FLOOR_STATES`): ``unchecked`` (the
starting candidate, no range), ``confirmed`` (the set the check confirmed) and
``short`` (the top 32 the check ended on).  A finished result is kept and goes
**stale** quietly once the ranking under it moves: later votes retrain the
model, the line follows the new ranking at the result's count, and the range
stays with a ``stale`` flag.

Everything here is scores, ids and labels - never a vector - so a
:class:`SpotCheck` and a :class:`LineRanking` may live on a detector context
for the life of the process (CLAUDE.md, "No Persisted Vectors").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: The check's level: a valid rule confirms a set that is really below *X* in at
#: most this share of sessions (#4257).
CHECK_ALPHA = 0.05

#: The smallest candidate: the top 32 (#4257's ``a:top32``, the #4267 ruling).
CHECK_BASE_CANDIDATE = 32

#: The fewest picks a round draws (owner, 2026-09-29: 5 picks a round).
CHECK_MIN_PICKS = 5

#: Floor states.  ``unchecked``: no check has run on the current floor, and the
#: line is the schedule's starting candidate.  ``confirmed``: the check's range
#: clears the floor.  ``short``: the check ended with its range below the floor,
#: and the line keeps the top 32 it ended on.
FLOOR_UNCHECKED = "unchecked"
FLOOR_CONFIRMED = "confirmed"
FLOOR_SHORT = "short"
FLOOR_STATES = (FLOOR_UNCHECKED, FLOOR_CONFIRMED, FLOOR_SHORT)

#: The surfacing provenance a check's vote is recorded with
#: (:mod:`vtscore.datasets.vote_provenance`): the app's route and the eval
#: harness both write this one dict, so the two cannot drift.
CHECK_PROVENANCE: dict[str, str] = {"flow": "check"}

#: A check's status while its rounds are still being voted on, and once it was
#: abandoned (its votes so far stay ordinary votes; the floor's state is as it was).
CHECK_RUNNING = "running"
CHECK_CANCELLED = "cancelled"

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


# ------------------------------------------------------------------ the schedule


@dataclass(frozen=True)
class CheckSchedule:
    """What a check at one floor costs: its starting candidate, rounds and picks a round."""

    candidate: int
    rounds: int
    picks: int

    def as_dict(self) -> dict[str, int]:
        return {"candidate": self.candidate, "rounds": self.rounds, "picks": self.picks}


def check_schedule(min_precision: float, alpha: float = CHECK_ALPHA) -> CheckSchedule:
    """``(K, R, m)`` for floor *min_precision*: the reference's ``schedule_for``.

    At ``X >= 1`` no finite sample bounds a proportion at 1, so the only check
    that reaches it is a census of the candidate: ``picks`` is the candidate
    itself there.
    """
    x = float(min_precision)
    if not 0.0 < x <= 1.0:
        raise ValueError(f"precision floor must be in (0, 1], got {min_precision!r}")
    doublings = max(0, math.floor(math.log2(0.5 / x) + _EPS))
    candidate = CHECK_BASE_CANDIDATE * 2**doublings
    rounds = doublings + 1
    if x >= 1.0 - _EPS:
        picks = candidate
    else:
        picks = max(CHECK_MIN_PICKS, math.ceil(math.log(alpha / rounds) / math.log(x) - _EPS))
    return CheckSchedule(candidate, rounds, picks)


def rounds_for(candidate: int, base: int = CHECK_BASE_CANDIDATE) -> int:
    """How many rounds a check starting at *candidate* items can run: halvings down to *base*, plus one.

    The schedule's ``R`` for a schedule-sized candidate; for a corpus too small
    to hold one (its own candidate), the halvings that candidate really has.
    """
    if candidate <= base:
        return 1
    return 1 + math.ceil(math.log2(candidate / base) - _EPS)


def range_tail(rounds: int, alpha: float = CHECK_ALPHA) -> float:
    """Each tail of the likely range: the level every round of the check is tested at.

    So the range's lower end is exactly the bound the check tested, and a check
    confirms *X* iff that lower end is at least *X*.  A plain 90% range covers
    too rarely after an early pass: at 10% and 0.44%, a check confirmed in its
    first round showed a range above the truth 11% of the time (winner's
    curse), against 4.0% with this tail.
    """
    return alpha / rounds


# ---------------------------------------------------------------- the bound


def clopper_pearson_lower(right: int, labelled: int, level: float) -> float:
    """One-sided Clopper-Pearson lower bound on a proportion at confidence ``1 - level``."""
    if labelled <= 0 or right <= 0:
        return 0.0
    from scipy.stats import beta  # noqa: PLC0415

    return float(beta.ppf(level, right, labelled - right + 1))


def clopper_pearson_upper(right: int, labelled: int, level: float) -> float:
    """One-sided Clopper-Pearson upper bound on a proportion at confidence ``1 - level``."""
    if labelled <= 0 or right >= labelled:
        return 1.0
    from scipy.stats import beta  # noqa: PLC0415

    return float(beta.ppf(1.0 - level, right + 1, labelled - right))


@dataclass(frozen=True)
class LikelyRange:
    """How much of a set is likely right, from the check's labels inside it.

    ``lo`` / ``hi`` bound the set's precision; ``labelled`` is how many of the
    set's items the check labelled and ``right`` how many of those were right.
    """

    lo: float
    hi: float
    labelled: int
    right: int

    def as_dict(self) -> dict[str, Any]:
        return {"lo": round(self.lo, 4), "hi": round(self.hi, 4), "labelled": self.labelled, "right": self.right}


def likely_range(right: int, labelled: int, candidate: int, tail: float) -> LikelyRange:
    """The range shown for a set of *candidate* items whose *labelled* uniform labels hold *right*.

    Exact (``right / candidate``) for a census; a Clopper-Pearson interval with
    each tail at *tail* otherwise.  The reference's ``likely_range``.
    """
    if candidate > 0 and labelled >= candidate:
        exact = right / candidate
        return LikelyRange(exact, exact, labelled, right)
    lo = clopper_pearson_lower(right, labelled, tail)
    hi = clopper_pearson_upper(right, labelled, tail)
    return LikelyRange(lo, hi, labelled, right)


# ---------------------------------------------------------------- the ranking


def _rng(seed: int | None = None) -> np.random.Generator:
    """The generator a check's picks are drawn from; tests pin *seed*."""
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class LineRanking:
    """The ranking a detector's line is drawn over: the haystack the last retrain scored, sorted.

    *ids* and *scores* are aligned and sorted by score descending (ties by id),
    with unscorable media (:func:`~vtscore.utils.scores.scored_mask`) left out.
    *voted* names the items the trainer held as voted - the labelset's items
    resolved into the haystack - which never enter a candidate; the caller adds
    the live votes cast since (:meth:`unvoted_ids`).

    Score arrays and ids only: process-scoped, never serialised.
    """

    ids: np.ndarray
    scores: np.ndarray
    voted: frozenset[int]

    @classmethod
    def from_scores(
        cls, ids: Sequence[int], scores: Sequence[float] | np.ndarray, voted: Iterable[int] | None = None
    ) -> "LineRanking":
        from vtscore.utils.scores import scored_mask  # noqa: PLC0415

        id_arr = np.asarray(list(ids), dtype=np.int64)
        score_arr = np.asarray(scores, dtype=np.float64)
        if id_arr.shape != score_arr.shape:
            raise ValueError("ids and scores must align")
        keep = scored_mask(score_arr)
        id_arr, score_arr = id_arr[keep], score_arr[keep]
        order = np.lexsort((id_arr, -score_arr))
        return cls(id_arr[order], score_arr[order], frozenset(int(v) for v in (voted or ())))

    @property
    def size(self) -> int:
        return int(self.ids.size)

    def unvoted_ids(self, also_voted: Iterable[int] = ()) -> np.ndarray:
        """The ranking's ids in rank order, less the trainer's voted items and *also_voted*."""
        excluded = self.voted.union(int(v) for v in also_voted)
        if not excluded:
            return self.ids
        mask = np.fromiter((int(i) not in excluded for i in self.ids), dtype=bool, count=self.ids.size)
        return self.ids[mask]

    def candidate(self, count: int, also_voted: Iterable[int] = ()) -> tuple[int, ...]:
        """The top *count* unvoted ids, in rank order (fewer when the corpus has fewer)."""
        return tuple(int(i) for i in self.unvoted_ids(also_voted)[: max(0, count)])

    def threshold_for(self, count: int, also_voted: Iterable[int] = ()) -> float | None:
        """Where a line keeping the top *count* unvoted items sits: just under the last of them.

        The *count*-th unvoted item's score, floored to the grid responses
        round scores to (:func:`line_under`), so that item clears its own line
        however it is compared - the exact score the trainer saw, or the
        rounded one a result row carries.  ``None`` when nothing is unvoted
        (there is no set to keep).  A *count* past the end of the ranking keeps
        the whole unvoted remainder.
        """
        top = self.candidate(count, also_voted)
        if not top:
            return None
        return line_under(self.score_of(top[-1]))

    def score_of(self, media_id: int) -> float:
        idx = np.flatnonzero(self.ids == media_id)
        if idx.size == 0:
            raise KeyError(media_id)
        return float(self.scores[idx[0]])

    def above(self, threshold: float) -> int:
        """How many items of the whole ranking, voted or not, score at or above *threshold*."""
        return int(np.count_nonzero(self.scores >= threshold))

    def fingerprint(self, count: int, also_voted: Iterable[int] = ()) -> tuple[int, ...]:
        """The identity of the top-*count* unvoted set, for telling a stale result from a live one."""
        return self.candidate(count, also_voted)


# ------------------------------------------------------------------ the check


@dataclass
class SpotCheck:
    """One check of a detector's line at one floor: its fixed candidate, rounds, labels and verdict.

    Built by :meth:`start` on the candidate the ranking holds at that moment.
    :meth:`draw` deals a round's picks; :meth:`record` takes the user's labels
    on them and, once the round is complete, confirms the floor, halves the
    candidate into the next round, or ends the check ``short``.  The candidate
    ids never change after :meth:`start`, so every round samples one list
    however the model moves behind it.
    """

    min_precision: float
    alpha: float
    candidate_ids: tuple[int, ...]
    rounds: int
    picks: int
    round: int = 1
    k: int = 0
    labels: dict[int, bool] = field(default_factory=dict)
    pending: tuple[int, ...] = ()
    status: str = CHECK_RUNNING
    #: The top-*k* unvoted set as it stood when the check finished (its own
    #: votes already cast), for :meth:`is_stale`.  ``None`` while running.
    fingerprint: tuple[int, ...] | None = None
    _generator: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def start(
        cls,
        candidate_ids: Sequence[int],
        min_precision: float,
        *,
        alpha: float = CHECK_ALPHA,
        seed: int | None = None,
    ) -> "SpotCheck":
        """A running check over *candidate_ids* (the top-K unvoted ids, rank order), with round 1 drawn.

        The rounds and the picks a round follow the floor's schedule, sized to
        the candidate the corpus really has: a corpus with fewer unvoted items
        than the schedule's K is its own candidate, with the halvings it has.
        """
        ids = tuple(int(i) for i in candidate_ids)
        if not ids:
            raise ValueError("a check needs at least one unvoted item to draw from")
        schedule = check_schedule(min_precision, alpha)
        rounds = rounds_for(len(ids))
        picks = schedule.picks
        if rounds != schedule.rounds and min_precision < 1.0 - _EPS:
            # A truncated candidate has fewer rounds, so each is tested at a
            # looser level and may need fewer picks; the reference's m(X, R).
            picks = max(CHECK_MIN_PICKS, math.ceil(math.log(alpha / rounds) / math.log(min_precision) - _EPS))
        check = cls(
            min_precision=float(min_precision),
            alpha=float(alpha),
            candidate_ids=ids,
            rounds=rounds,
            picks=int(picks),
            k=len(ids),
            _generator=_rng(seed),
        )
        check.draw()
        return check

    # ---- what the check is looking at

    @property
    def start_k(self) -> int:
        return len(self.candidate_ids)

    @property
    def current(self) -> tuple[int, ...]:
        """The current candidate: the top *k* of the fixed list."""
        return self.candidate_ids[: self.k]

    @property
    def running(self) -> bool:
        return self.status == CHECK_RUNNING

    @property
    def finished(self) -> bool:
        return self.status in (FLOOR_CONFIRMED, FLOOR_SHORT)

    @property
    def tail(self) -> float:
        return range_tail(self.rounds, self.alpha)

    def counts(self) -> tuple[int, int]:
        """``(labelled, right)`` inside the current candidate."""
        labelled = right = 0
        for cid in self.current:
            if cid in self.labels:
                labelled += 1
                right += int(self.labels[cid])
        return labelled, right

    def range(self) -> LikelyRange:
        """The likely range of the current candidate's precision, from the labels inside it."""
        labelled, right = self.counts()
        return likely_range(right, labelled, self.k, self.tail)

    # ---- the rounds

    def draw(self) -> tuple[int, ...]:
        """Deal this round's picks: fresh items drawn uniformly from the current candidate.

        Fewer than ``picks`` when the candidate has fewer unlabelled items left
        (the round is then a census of it).  Returned in draw order, which is
        random, so a client can show them in that order.
        """
        if not self.running:
            raise ValueError("the check has finished")
        fresh = [cid for cid in self.current if cid not in self.labels]
        m = min(self.picks, len(fresh))
        if m <= 0:
            # Every item is labelled already: the round is decided as a census.
            self.pending = ()
            self._evaluate()
            return ()
        chosen = self._generator.choice(len(fresh), size=m, replace=False)
        self.pending = tuple(fresh[int(i)] for i in chosen)
        return self.pending

    def record(self, votes: Mapping[int, bool]) -> bool:
        """Take the user's labels on this round's picks (``True`` = right / Good).

        Returns ``True`` when the round completed and was evaluated.  A label on
        an item that was not dealt this round is refused; a partial round waits
        for the rest of its picks.
        """
        if not self.running:
            raise ValueError("the check has finished")
        stray = [cid for cid in votes if cid not in self.pending]
        if stray:
            raise ValueError(f"not this round's picks: {stray}")
        for cid, right in votes.items():
            self.labels[int(cid)] = bool(right)
        self.pending = tuple(cid for cid in self.pending if cid not in self.labels)
        if self.pending:
            return False
        self._evaluate()
        return True

    def _evaluate(self) -> None:
        """Confirm, halve into the next round, or end short."""
        if self.range().lo >= self.min_precision - _EPS:
            self.status = FLOOR_CONFIRMED
            return
        if self.k > CHECK_BASE_CANDIDATE:
            self.k = max(CHECK_BASE_CANDIDATE, self.k // 2)
            self.round += 1
            self.draw()
            return
        self.status = FLOOR_SHORT

    def cancel(self) -> None:
        """Abandon a running check; its votes so far stay ordinary votes."""
        if self.running:
            self.status = CHECK_CANCELLED

    # ---- after the check

    def is_stale(self, ranking: LineRanking, also_voted: Iterable[int] = ()) -> bool:
        """Whether the ranking under the result has moved since the check finished."""
        if self.fingerprint is None:
            return False
        return ranking.fingerprint(self.k, also_voted) != self.fingerprint

    def as_dict(self) -> dict[str, Any]:
        """The check as a client sees it: the round, its pending picks, the labels so far and the range."""
        labelled, right = self.counts()
        rng = self.range() if labelled else None
        return {
            "status": self.status,
            "min_precision": self.min_precision,
            "round": self.round,
            "rounds": self.rounds,
            "picks_per_round": self.picks,
            "candidate": self.k,
            "start_candidate": self.start_k,
            "picks": list(self.pending),
            "labelled": labelled,
            "right": right,
            "range": rng.as_dict() if rng is not None else None,
        }


# ------------------------------------------------------------------- the line


@dataclass(frozen=True)
class FloorState:
    """What the floor says about a detector's line: its state, the set's size and how close it got.

    Built by :func:`floor_state`; the wire shape every response that carries a
    line carries beside its ``threshold`` (:func:`vtscore.state.core.detector_floor_state`).
    """

    min_precision: float
    status: str
    count: int
    range: LikelyRange | None
    stale: bool
    schedule: CheckSchedule

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_precision": self.min_precision,
            "status": self.status,
            "count": self.count,
            "range": None if self.range is None else {**self.range.as_dict(), "stale": self.stale},
            "schedule": self.schedule.as_dict(),
        }


def applicable_result(min_precision: float, result: SpotCheck | None) -> SpotCheck | None:
    """*result* when it is a finished check of *min_precision*; ``None`` otherwise.

    A result is about the floor it was run at: a different floor is unchecked
    until it is checked itself (the old result waits, and shows again if the
    floor moves back).
    """
    if result is None or not result.finished:
        return None
    if abs(result.min_precision - min_precision) > _EPS:
        return None
    return result


def floor_count(min_precision: float, result: SpotCheck | None) -> int:
    """The count the line keeps at *min_precision*: the finished check's set, else the starting candidate.

    The corpus may hold fewer unvoted items; :class:`LineRanking` truncates.
    """
    applicable = applicable_result(min_precision, result)
    if applicable is not None:
        return applicable.k
    return check_schedule(min_precision).candidate


def floor_line(
    ranking: LineRanking | None,
    min_precision: float,
    result: SpotCheck | None = None,
    also_voted: Iterable[int] = (),
) -> float | None:
    """The threshold the line sits at: the last item of the set the floor keeps.

    ``None`` when there is no ranking, or nothing in it is unvoted; the caller
    keeps whatever inclusion-blind fallback it has, as it did before the floor.
    """
    if ranking is None:
        return None
    return ranking.threshold_for(floor_count(min_precision, result), also_voted)


def floor_state(
    min_precision: float,
    result: SpotCheck | None = None,
    ranking: LineRanking | None = None,
    also_voted: Iterable[int] = (),
) -> FloorState:
    """The floor's state at *min_precision*, given the detector's last finished check and its ranking."""
    schedule = check_schedule(min_precision)
    applicable = applicable_result(min_precision, result)
    if applicable is None:
        count = schedule.candidate
        if ranking is not None:
            count = min(count, len(ranking.candidate(count, also_voted)))
        return FloorState(float(min_precision), FLOOR_UNCHECKED, count, None, False, schedule)
    stale = ranking is not None and applicable.is_stale(ranking, also_voted)
    return FloorState(float(min_precision), applicable.status, applicable.k, applicable.range(), stale, schedule)


__all__ = [
    "CHECK_ALPHA",
    "CHECK_BASE_CANDIDATE",
    "CHECK_CANCELLED",
    "CHECK_MIN_PICKS",
    "CHECK_PROVENANCE",
    "CHECK_RUNNING",
    "FLOOR_CONFIRMED",
    "FLOOR_SHORT",
    "FLOOR_STATES",
    "FLOOR_UNCHECKED",
    "CheckSchedule",
    "FloorState",
    "LikelyRange",
    "LineRanking",
    "SpotCheck",
    "applicable_result",
    "check_schedule",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "floor_count",
    "floor_line",
    "floor_state",
    "likely_range",
    "line_under",
    "range_tail",
    "rounds_for",
]
