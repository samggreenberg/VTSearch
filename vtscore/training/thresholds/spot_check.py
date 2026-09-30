"""The spot check that decides the precision floor's line, and how close it got (#4272, #4388).

The floor *P* is a share of what the line returns that should be right.  The
owner's ruling on #4267 (2026-09-29) moved the promise off the #4220 estimator
(:mod:`~vtscore.training.thresholds.precision_floor`, which stays as public
library API that no app path reads since #4360 and #4362) and onto a **spot
check**: the user votes on uniform random picks, and their labels decide.
#4256 showed that model-chosen votes break 83% of the estimator's P = 50%
promises once the reference pool is consistent; a uniform pick has no such
bias, so what it says holds at every prevalence measured (#4257).

**The check walks the ranking in bands (#4383, the owner's ruling of
2026-09-30).**  The first version of the check (#4272) drew its picks from a
fixed candidate, the top 32 unvoted at P >= 50% and the top 128 at 10%, and
could only halve it.  That count was right on exactly one corpus, the 11k-image
bench it was priced on: on a corpus ten times larger it kept 5% of the
positives, and on a 320-image Find set it returned all 32 at 4% right
(``docs/experiments/2026-09-30-line-estimate-4383/REPORT.md``).  The walk lets
the line follow the corpus:

* **Bands.**  The unvoted ranking is cut into bands from the top: the top 8,
  the next 8, then 16, 32, 64, ... doubling to the corpus's end
  (:func:`band_edges`).  A band is audited with :data:`CHECK_MIN_PICKS` items
  drawn uniformly from it (a census when it holds no more than that), and a
  band is never drawn from twice.
* **The union under test** is the top ``b`` bands.  Its estimate is
  band-stratified: each band's share of right answers weighted by the band's
  size, which is unbiased for the union whatever the picks per band
  (:meth:`SpotCheck.estimate`).
* **The walk** starts at the bands that hold today's count (the floor's
  schedule: 32 at P >= 50%, 128 at 10%; :func:`check_schedule`), auditing
  each.  While the union's estimate is at or above *P* it goes one band deeper
  (drawing that band); while it is below, one band shallower (no new picks).
  It stops on the first reversal, or at either end, and the line keeps the
  deepest union that met *P*: ``confirmed``.  When no union met it, the first
  band, ``short``.
* **How close.**  A checked set carries a **likely range**: each audited
  band's Clopper-Pearson interval, exact where the band was censused, with
  each tail at ``alpha / bands`` (the union bound over the bands in the set),
  weighted by band size (:meth:`SpotCheck.range`).  The walk decides on the
  point estimate (the owner's "do your best", #4267, and the rule priced), so
  a confirmed set's range can straddle *P*; the range says how close.

**Do your best, and say how close we got** (#4267).  The line always keeps a
set.  Before any check it keeps the schedule's unchecked starting candidate;
after one, the set the walk ended on.  Nothing falls back to the old Inclusion
0 cut.  The states a floor reports (:data:`FLOOR_STATES`): ``unchecked``,
``confirmed`` and ``short``.  A finished result is kept and goes **stale**
quietly once the ranking under it moves: later votes retrain the model, the
line follows the new ranking at the result's count, and the range stays with a
``stale`` flag.

The rule the walk implements is ``grow-fine`` in
``scripts/experiments/calibration/analyze_line_estimate_4383.py``
(``band_edges``, ``draw_audits``, ``Audits.union_estimate``, ``rule_grow``),
which ``scripts/check-eval-app-sync.py`` pins against this module.

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

#: The level the likely range is drawn at: each audited band's tail is
#: ``alpha / bands`` over the bands in the set (#4257's alpha).
CHECK_ALPHA = 0.05

#: The schedule's base: the starting candidate at P >= 50% is the top 32
#: (#4257's ``a:top32``, the #4267 ruling); it doubles as the floor falls.
CHECK_BASE_CANDIDATE = 32

#: The first band's size, and the smallest set a walk can keep (#4383).
BAND_BASE = 8

#: The picks a band is audited with (owner, 2026-09-30: 5 picks a band).
CHECK_MIN_PICKS = 5

#: Floor states.  ``unchecked``: no check has run on the current floor, and the
#: line is the schedule's starting candidate.  ``confirmed``: the walk ended on
#: a set whose estimate met the floor.  ``short``: no set met it, and the line
#: keeps the first band the walk ended on.
FLOOR_UNCHECKED = "unchecked"
FLOOR_CONFIRMED = "confirmed"
FLOOR_SHORT = "short"
FLOOR_STATES = (FLOOR_UNCHECKED, FLOOR_CONFIRMED, FLOOR_SHORT)

#: The surfacing provenance a check's vote is recorded with
#: (:mod:`vtscore.datasets.vote_provenance`): the app's route and the eval
#: harness both write this one dict, so the two cannot drift.
CHECK_PROVENANCE: dict[str, str] = {"flow": "check"}

#: A check's status while its bands are still being voted on, and once it was
#: abandoned (its votes so far stay ordinary votes; the floor's state is as it was).
CHECK_RUNNING = "running"
CHECK_CANCELLED = "cancelled"

#: Which way the walk last moved: where it started, one band deeper (the last
#: union met the floor), or one band shallower (it did not).
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


def bands_for(count: int, edges: Sequence[int]) -> int:
    """How many bands from the top it takes to hold the top *count*: at least one."""
    for b in range(1, len(edges)):
        if edges[b] >= count:
            return b
    return max(1, len(edges) - 1)


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
    """``(K, bands, m)`` for floor *min_precision*: what a walk starts from and costs a band.

    *K* is the unchecked starting candidate, the count today's line keeps
    before any check: ``32 * 2**max(0, floor(log2(0.5 / P)))``, the top 128 at
    10%, the top 64 at 25%, and the top 32 at 50% and above (the #4267 ruling;
    the reference's ``schedule_for``).  ``rounds`` is how many bands the walk
    audits before its first verdict, the bands that hold *K*
    (:func:`rounds_for`); ``picks`` is what each band costs,
    :data:`CHECK_MIN_PICKS`.  At ``P >= 1`` no sample can vouch for every
    item, so each band is censused: ``picks`` is the first band's size.
    """
    x = float(min_precision)
    if not 0.0 < x <= 1.0:
        raise ValueError(f"precision floor must be in (0, 1], got {min_precision!r}")
    doublings = max(0, math.floor(math.log2(0.5 / x) + _EPS))
    candidate = CHECK_BASE_CANDIDATE * 2**doublings
    picks = BAND_BASE if x >= 1.0 - _EPS else CHECK_MIN_PICKS
    return CheckSchedule(candidate, rounds_for(candidate), picks)


def rounds_for(candidate: int, base: int = BAND_BASE) -> int:
    """How many bands a walk audits before its first verdict: the bands from the top that hold *candidate*.

    3 for the top 32 (8, 8, 16), 4 for 64, 5 for 128; a corpus smaller than
    the candidate is its own last band.
    """
    return bands_for(candidate, band_edges(max(candidate, 1), base))


def range_tail(rounds: int, alpha: float = CHECK_ALPHA) -> float:
    """Each tail of a band's interval when *rounds* bands make up the set: ``alpha / rounds``.

    The union bound: with every band's interval at this tail, the set's range
    holds at ``1 - alpha``.  A plain 90% range covers too rarely after an early
    pass (the same labels decide the walk and draw the range): at 10% and
    0.44%, a check confirmed at its first verdict showed a range above the
    truth 11% of the time, against 4.0% with this tail (#4272).
    """
    return alpha / max(1, rounds)


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
    #: :func:`mixture_count`'s one-slot memo: the mixture fitted on this
    #: ranking (``[fit]``, ``[None]`` when none fits), empty until first asked.
    #: Contents only; the ranking itself stays immutable.
    mixture: list[Any] = field(default_factory=list, compare=False, repr=False)

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
    """One walk of a detector's line at one floor: its bands, the picks in each, and where it ended.

    Built by :meth:`start` on the unvoted ranking as it stands at that moment
    (ids in rank order), cut into bands.  :meth:`draw` deals the next band's
    picks; :meth:`record` takes the user's labels on them and, once the band is
    audited, either deals the next band the union under test still owes, or
    decides: the walk goes deeper, shallower, or ends.  The ranking's ids never
    change after :meth:`start`, so every band samples the one list fixed then,
    however the model moves behind it.
    """

    min_precision: float
    alpha: float
    #: The unvoted ranking the walk is over, in rank order, fixed at the start.
    ranking_ids: tuple[int, ...]
    edges: tuple[int, ...]
    picks: int
    #: The bands the walk started from: those that hold the floor's candidate.
    start_bands: int
    #: The union under test: the top *bands* bands, ``edges[bands]`` items.
    bands: int = 1
    #: The deepest union whose estimate met the floor, if any.
    best: int | None = None
    direction: str = WALK_START
    #: Bands audited so far (each is one round of picks).
    round: int = 0
    #: The union under test's size while running; the kept set's once finished.
    k: int = 0
    labels: dict[int, bool] = field(default_factory=dict)
    pending: tuple[int, ...] = ()
    #: The band the pending picks were drawn from; ``None`` between bands.
    band: int | None = None
    status: str = CHECK_RUNNING
    #: The top-*k* unvoted set as it stood when the check finished (its own
    #: votes already cast), for :meth:`is_stale`.  ``None`` while running.
    fingerprint: tuple[int, ...] | None = None
    _generator: Any = field(default=None, repr=False, compare=False)
    _rank: dict[int, int] = field(default_factory=dict, repr=False, compare=False)
    _audited: set[int] = field(default_factory=set, repr=False, compare=False)

    @classmethod
    def start(
        cls,
        ranking_ids: Sequence[int],
        min_precision: float,
        *,
        alpha: float = CHECK_ALPHA,
        seed: int | None = None,
        start_count: int | None = None,
    ) -> "SpotCheck":
        """A running walk over *ranking_ids* (the unvoted ranking, rank order), with its first band drawn.

        The walk starts at the bands that hold *start_count* items: the
        floor's schedule by default (32 at P >= 50%, 128 at 10%), or a
        caller's own proposal (#4389).  A ranking shorter than that starts at
        its last band.
        """
        ids = tuple(int(i) for i in ranking_ids)
        if not ids:
            raise ValueError("a check needs at least one unvoted item to draw from")
        if len(set(ids)) != len(ids):
            raise ValueError("the ranking's ids must be distinct")
        schedule = check_schedule(min_precision, alpha)
        edges = band_edges(len(ids))
        start = bands_for(min(start_count if start_count is not None else schedule.candidate, len(ids)), edges)
        check = cls(
            min_precision=float(min_precision),
            alpha=float(alpha),
            ranking_ids=ids,
            edges=edges,
            picks=int(schedule.picks),
            start_bands=start,
            bands=start,
            k=int(edges[start]),
            _generator=_rng(seed),
            _rank={cid: i for i, cid in enumerate(ids)},
        )
        check.draw()
        return check

    # ---- what the check is looking at

    @property
    def n_bands(self) -> int:
        return len(self.edges) - 1

    @property
    def start_k(self) -> int:
        """The count the walk started from, as the ranking really has it."""
        return int(self.edges[self.start_bands])

    @property
    def current(self) -> tuple[int, ...]:
        """The union under test: the top *k* of the fixed ranking."""
        return self.ranking_ids[: self.k]

    @property
    def running(self) -> bool:
        return self.status == CHECK_RUNNING

    @property
    def finished(self) -> bool:
        return self.status in (FLOOR_CONFIRMED, FLOOR_SHORT)

    @property
    def tail(self) -> float:
        """Each band's tail in the current set's range: ``alpha`` split over its bands."""
        return range_tail(self.bands, self.alpha)

    def band_ids(self, b: int) -> tuple[int, ...]:
        """The ids in band *b*, in rank order."""
        return self.ranking_ids[self.edges[b] : self.edges[b + 1]]

    def band_counts(self, b: int) -> tuple[int, int, int]:
        """``(size, labelled, right)`` for band *b*."""
        lo, hi = self.edges[b], self.edges[b + 1]
        labelled = right = 0
        for cid, ok in self.labels.items():
            r = self._rank.get(cid)
            if r is not None and lo <= r < hi:
                labelled += 1
                right += int(ok)
        return hi - lo, labelled, right

    def counts(self) -> tuple[int, int]:
        """``(labelled, right)`` inside the current set."""
        labelled = right = 0
        for cid, ok in self.labels.items():
            r = self._rank.get(cid)
            if r is not None and r < self.k:
                labelled += 1
                right += int(ok)
        return labelled, right

    def estimate(self) -> float | None:
        """The band-stratified share of the current set that is right, or ``None`` before any of it was audited.

        Each band's share of right picks, weighted by the band's size: the
        reference's ``Audits.union_estimate``.  A band in the set with no
        picks (only possible while the walk is still auditing its start)
        leaves the estimate ``None``.
        """
        if self.k <= 0:
            return None
        total = 0.0
        for b in range(self.bands):
            size, labelled, right = self.band_counts(b)
            if size and not labelled:
                return None
            total += size * (right / labelled if labelled else 0.0)
        return total / self.k

    def range(self) -> LikelyRange:
        """The likely range of the current set's precision: its bands' intervals, weighted by band size.

        Each band's interval is Clopper-Pearson at :attr:`tail` (exact where
        the band was censused); the set's range is their size-weighted mean,
        which holds at ``1 - alpha`` by the union bound over the bands.
        """
        if self.k <= 0:
            return LikelyRange(0.0, 1.0, 0, 0)
        lo = hi = 0.0
        labelled_all = right_all = 0
        for b in range(self.bands):
            size, labelled, right = self.band_counts(b)
            part = likely_range(right, labelled, size, self.tail)
            lo += size * part.lo
            hi += size * part.hi
            labelled_all += labelled
            right_all += right
        return LikelyRange(lo / self.k, hi / self.k, labelled_all, right_all)

    # ---- the rounds

    def draw(self) -> tuple[int, ...]:
        """Deal the next band's picks, or decide the walk when the set under test is fully audited.

        The picks are fresh items drawn uniformly from the lowest band of the
        set that has not been audited yet, in draw order, which is random, so
        a client can show them in that order.  Fewer than ``picks`` when the
        band has fewer items (a census of it).  Returns ``()`` once the walk
        has decided.
        """
        if not self.running:
            raise ValueError("the check has finished")
        while self.running:
            owed = next((b for b in range(self.bands) if b not in self._audited), None)
            if owed is None:
                self._evaluate()
                continue
            fresh = [cid for cid in self.band_ids(owed) if cid not in self.labels]
            m = min(self.picks, len(fresh))
            if m <= 0:
                # Every item in the band is labelled already: audited as a census.
                self._audited.add(owed)
                continue
            chosen = self._generator.choice(len(fresh), size=m, replace=False)
            self.pending = tuple(fresh[int(i)] for i in chosen)
            self.band = owed
            return self.pending
        return ()

    def record(self, votes: Mapping[int, bool]) -> bool:
        """Take the user's labels on this band's picks (``True`` = right / Good).

        Returns ``True`` when the band's round completed and the walk moved on
        (to its next band, or to a decision).  A label on an item that was
        not dealt is refused; a partial round waits for the rest of its picks.
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
        if self.band is not None:
            self._audited.add(self.band)
        self.band = None
        self.round += 1
        self.draw()
        return True

    def _evaluate(self) -> None:
        """Deeper while the set meets the floor, shallower while it does not; stop on the first reversal."""
        est = self.estimate()
        if est is not None and est >= self.min_precision - _EPS:
            self.best = self.bands
            if self.bands >= self.n_bands:
                self._finish(FLOOR_CONFIRMED, self.bands)
                return
            self.direction = WALK_DEEPER
            self.bands += 1
            self.k = int(self.edges[self.bands])
            return
        if self.best is not None:
            # The set under test fell short: step back to the deepest that did not.
            self.direction = WALK_SHALLOWER
            self._finish(FLOOR_CONFIRMED, self.best)
            return
        if self.bands <= 1:
            self._finish(FLOOR_SHORT, 1)
            return
        self.direction = WALK_SHALLOWER
        self.bands -= 1
        self.k = int(self.edges[self.bands])

    def _finish(self, status: str, bands: int) -> None:
        self.status = status
        self.bands = bands
        self.k = int(self.edges[bands])
        self.pending = ()
        self.band = None

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
        """The check as a client sees it: the band being audited, the set under test, the labels so far and the range."""
        labelled, right = self.counts()
        rng = self.range() if labelled else None
        est = self.estimate()
        band = None
        if self.band is not None:
            band = {"index": self.band, "lo": int(self.edges[self.band]) + 1, "hi": int(self.edges[self.band + 1])}
        return {
            "status": self.status,
            "min_precision": self.min_precision,
            "round": self.round + (1 if self.pending else 0),
            "rounds": self.n_bands,
            "picks_per_round": self.picks,
            "candidate": self.k,
            "start_candidate": self.start_k,
            "bands": self.bands,
            "band": band,
            "direction": self.direction,
            "estimate": None if est is None else round(est, 4),
            "picks": list(self.pending),
            "labelled": labelled,
            "right": right,
            "range": rng.as_dict() if rng is not None else None,
        }


# ------------------------------------------------------------ the mixture's count


def mixture_count(
    ranking: LineRanking | None,
    min_precision: float,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
) -> int | None:
    """How many unvoted items the vote-anchored mixture says are at least *min_precision* right (#4389).

    The owner's ruling on #4383 for the line with no audit votes: the smaller
    of today's count and this.  A 2-component mixture is fitted on the
    ranking's scores (:func:`~vtscore.training.thresholds.gmm.anchored_gmm_fit`,
    which subsamples a large ranking), anchored by the scores of the items in
    *labels* (``True`` = Good) that the ranking holds; the high component's
    posterior is read at each unvoted score, best first, and the count is the
    deepest top *k* whose mean posterior is at or above the floor (the
    ``gmm`` rule of ``analyze_line_estimate_4383.py``), or the *k* with the
    highest mean when none is (best effort, at least one).  ``None`` when
    there is no ranking, nothing unvoted, or no mixture fits, and the caller
    keeps today's count.

    Priced in ``docs/experiments/2026-09-30-line-estimate-4383/REPORT.md``
    (``gmm`` and ``min-fixed-gmm``): right-sized on small corpora and at
    moderate prevalence, 5-66x too deep on large sparse ones, which is why
    it only ever *lowers* the count.  **Fitted once per ranking**, on the
    anchors of the first caller - the retrain's training labels, or the
    human votes for a ranking a cold Find built - and memoised on it, so
    the re-cut and every response that reports the floor's state read the
    same fit; only the unvoted set the count is taken over follows the live
    votes, as the ranking's own line does.  A retrain parks a new ranking,
    and with it a new fit on the votes it trained on.
    """
    if ranking is None or ranking.size == 0:
        return None
    excluded = ranking.voted.union(int(v) for v in also_voted)
    mask = np.fromiter((int(i) not in excluded for i in ranking.ids), dtype=bool, count=ranking.size)
    scores = ranking.scores[mask]  # already best first
    if scores.size == 0:
        return None
    from scipy.stats import norm  # noqa: PLC0415

    if not ranking.mixture:
        from vtscore.training.thresholds.gmm import anchored_gmm_fit  # noqa: PLC0415

        anchor_scores: list[float] = []
        anchor_labels: list[float] = []
        if labels:
            index = {int(cid): i for i, cid in enumerate(ranking.ids.tolist())}
            for cid, is_good in labels.items():
                row = index.get(int(cid))
                if row is not None:
                    anchor_scores.append(float(ranking.scores[row]))
                    anchor_labels.append(1.0 if is_good else 0.0)
        fit, _provenance = anchored_gmm_fit(ranking.scores, anchor_scores, anchor_labels)
        ranking.mixture.append(fit if (fit is not None and fit.var_hi > 0 and fit.var_lo > 0) else None)
    fit = ranking.mixture[0]
    if fit is None:
        return None
    hi = fit.w_hi * norm.pdf(scores, fit.mu_hi, math.sqrt(fit.var_hi))
    lo = fit.w_lo * norm.pdf(scores, fit.mu_lo, math.sqrt(fit.var_lo))
    with np.errstate(invalid="ignore", divide="ignore"):
        posterior = np.nan_to_num(hi / (hi + lo), nan=0.5)
    cum = np.cumsum(posterior) / np.arange(1, scores.size + 1)
    ok = np.flatnonzero(cum >= float(min_precision) - _EPS)
    return int(ok.max()) + 1 if ok.size else int(np.argmax(cum)) + 1


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


def floor_count(min_precision: float, result: SpotCheck | None, proposal: int | None = None) -> int:
    """The count the line keeps at *min_precision*: the set the finished walk ended on, else the unchecked count.

    The unchecked count is the schedule's starting candidate, lowered to
    *proposal* when the caller has one: the mixture's count
    (:func:`mixture_count`), the owner's rule for the line before any check
    (#4389).  The corpus may hold fewer unvoted items; :class:`LineRanking`
    truncates.
    """
    applicable = applicable_result(min_precision, result)
    if applicable is not None:
        return applicable.k
    count = check_schedule(min_precision).candidate
    if proposal is not None:
        count = max(1, min(count, int(proposal)))
    return count


def floor_line(
    ranking: LineRanking | None,
    min_precision: float,
    result: SpotCheck | None = None,
    also_voted: Iterable[int] = (),
    proposal: int | None = None,
) -> float | None:
    """The threshold the line sits at: the last item of the set the floor keeps.

    *proposal* is the mixture's count for the unchecked line, if the caller
    has one (:func:`mixture_count`, #4389).  ``None`` when there is no
    ranking, or nothing in it is unvoted; the caller keeps whatever
    inclusion-blind fallback it has, as it did before the floor.
    """
    if ranking is None:
        return None
    return ranking.threshold_for(floor_count(min_precision, result, proposal), also_voted)


def floor_state(
    min_precision: float,
    result: SpotCheck | None = None,
    ranking: LineRanking | None = None,
    also_voted: Iterable[int] = (),
    proposal: int | None = None,
) -> FloorState:
    """The floor's state at *min_precision*, given the detector's last finished check, its ranking and the mixture's count."""
    schedule = check_schedule(min_precision)
    applicable = applicable_result(min_precision, result)
    if applicable is None:
        count = floor_count(min_precision, None, proposal)
        if ranking is not None:
            count = min(count, len(ranking.candidate(count, also_voted)))
        return FloorState(float(min_precision), FLOOR_UNCHECKED, count, None, False, schedule)
    stale = ranking is not None and applicable.is_stale(ranking, also_voted)
    return FloorState(float(min_precision), applicable.status, applicable.k, applicable.range(), stale, schedule)


__all__ = [
    "BAND_BASE",
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
    "WALK_DEEPER",
    "WALK_SHALLOWER",
    "WALK_START",
    "CheckSchedule",
    "FloorState",
    "LikelyRange",
    "LineRanking",
    "SpotCheck",
    "applicable_result",
    "band_edges",
    "bands_for",
    "check_schedule",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "floor_count",
    "floor_line",
    "floor_state",
    "likely_range",
    "line_under",
    "mixture_count",
    "range_tail",
    "rounds_for",
]
