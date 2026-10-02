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

from vtscore.training.thresholds.knobs import ACQUISITION_ARGMAX_FACTOR

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

#: The balance (#4413): the user's precision/recall preference as F-beta's beta.
#: The presets are precision-leaning, balanced and recall-leaning; any beta in
#: ``[BETA_MIN, BETA_MAX]`` is accepted.  A finished balance walk is ``checked``:
#: a balance has nothing to fall short of, so there is no ``short``.
BALANCE_PRESETS: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_BETA = 1.0
BETA_MIN, BETA_MAX = 0.25, 4.0
#: How a check treats the line at a balance (#4427's pricing, 2026-10-02):
#: ``advisory`` at beta <= 1 - the walk runs and its votes train, but the line
#: keeps the unchecked rule's count - and ``trim`` above - the walk audits the
#: bands holding the line, may only step shallower, and the line takes its end.
CHECK_ADVISORY = "advisory"
CHECK_TRIM = "trim"
CHECK_SHAPES: tuple[str, ...] = (CHECK_ADVISORY, CHECK_TRIM)
BALANCE_CHECKED = "checked"
BALANCE_STATES = (FLOOR_UNCHECKED, BALANCE_CHECKED)

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


def balance_schedule(beta: float, alpha: float = CHECK_ALPHA) -> CheckSchedule:
    """The unchecked cap and the walk's start for a balance *beta* (#4413).

    The cap is the floor's schedule count for the preset the balance leans
    toward: 32 at beta <= 1 (the 50% schedule), 128 above it (the 10% one).
    It is what holds the mixture's F-beta argmax on a large sparse corpus,
    where the mixture over-counts the positives and would return 2-4x too
    much (#4411); the walk starts from the same bands.
    """
    if not (BETA_MIN - _EPS <= float(beta) <= BETA_MAX + _EPS):
        raise ValueError(f"beta must be in [{BETA_MIN}, {BETA_MAX}], got {beta!r}")
    return check_schedule(0.5 if float(beta) <= 1.0 + _EPS else 0.1, alpha)


def check_shape(beta: float) -> str:
    """How a check at *beta* treats the line (#4427): ``advisory`` at beta <= 1, ``trim`` above.

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
    return CHECK_ADVISORY if float(beta) <= 1.0 + _EPS else CHECK_TRIM


def resolve_line_knobs(min_precision: float | str | None, beta: float | None) -> tuple[float | None, float | None]:
    """The preference an eval arm draws its line at, as ``(floor, beta)``; neither pinned is the app's default (#4413).

    The harness's counterpart of :func:`vtscore.state.line_knobs`, which is
    what the app hands every retrain and re-cut: a pinned *beta* is the
    balance arm (``(None, beta)``; the floor is unused whatever it says);
    neither pinned resolves to the app's default preference
    (:data:`~vtscore.config.runtime.DEFAULT_LINE_PREFERENCE`) - the balance
    at :data:`DEFAULT_BETA`, or the floor at
    :data:`~vtscore.training.thresholds.DEFAULT_MIN_PRECISION`; a pinned floor
    is the floor arm (deprecated with the floor); ``"off"`` is the Inclusion
    arm (``(None, None)``).  Validated here, so a malformed arm fails before
    anything expensive runs.
    """
    from vtscore.config.runtime import DEFAULT_LINE_PREFERENCE  # noqa: PLC0415
    from vtscore.training.thresholds.precision_floor import resolve_min_precision  # noqa: PLC0415

    if beta is not None:
        balance_schedule(beta)
        return None, float(beta)
    if min_precision is None and DEFAULT_LINE_PREFERENCE == "balance":
        return None, DEFAULT_BETA
    return resolve_min_precision(min_precision), None


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
    #: A balance walk's beta and its count of the ranking's positives (the
    #: mixture's, fixed at the start); ``None`` on a floor walk (#4413).
    beta: float | None = None
    n_pos: float | None = None
    #: The best F-beta estimate the balance walk has seen (the peak's).
    best_estimate: float | None = None
    #: A balance walk's tolerance (#4427's ``tol`` arm): a deeper step whose
    #: estimate is within this of the best is flat, not a fall, and the walk
    #: looks one band further before deciding; 0 is the shipped strict rise.
    tol: float = 0.0
    #: The precision guard (#4427's ``guard`` arm): a deeper band whose audited
    #: share right is below ``guard`` times the start set's ends the walk at
    #: the best set so far, whatever the F-beta estimate says.  ``None`` is the
    #: shipped walk, which deepens on the estimate alone.
    guard: float | None = None
    #: The start set's band-weighted share right, read once its bands are
    #: audited; what the guard compares a deeper band against.
    start_share: float | None = None
    #: The shallower-only walk (#4427's ``shallow`` arm): the start set is the
    #: deepest the walk may keep; from it, shallower while the estimate does
    #: not fall, on the start's own audits.  Off, the shipped walk, it tries
    #: one band deeper first.
    shallow_only: bool = False
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
        picks: int | None = None,
        edges: Sequence[int] | None = None,
    ) -> "SpotCheck":
        """A running walk over *ranking_ids* (the unvoted ranking, rank order), with its first band drawn.

        The walk starts at the bands that hold *start_count* items: the
        floor's schedule by default (32 at P >= 50%, 128 at 10%), or a
        caller's own proposal (#4389).  A ranking shorter than that starts at
        its last band.  *picks* (a band's picks) and *edges* (the bands) are
        the schedule's unless an arm overrides them (#4427).
        """
        ids = tuple(int(i) for i in ranking_ids)
        if not ids:
            raise ValueError("a check needs at least one unvoted item to draw from")
        if len(set(ids)) != len(ids):
            raise ValueError("the ranking's ids must be distinct")
        schedule = check_schedule(min_precision, alpha)
        edges = band_edges(len(ids)) if edges is None else tuple(int(e) for e in edges)
        start = bands_for(min(start_count if start_count is not None else schedule.candidate, len(ids)), edges)
        check = cls(
            min_precision=float(min_precision),
            alpha=float(alpha),
            ranking_ids=ids,
            edges=edges,
            picks=int(schedule.picks if picks is None else picks),
            start_bands=start,
            bands=start,
            k=int(edges[start]),
            _generator=_rng(seed),
            _rank={cid: i for i, cid in enumerate(ids)},
        )
        check.draw()
        return check

    @classmethod
    def start_balance(
        cls,
        ranking_ids: Sequence[int],
        beta: float,
        n_pos: float,
        *,
        alpha: float = CHECK_ALPHA,
        seed: int | None = None,
        start_count: int | None = None,
        picks: int | None = None,
        tol: float = 0.0,
        fine: bool = False,
        guard: float | None = None,
        shallow_only: bool | None = None,
    ) -> "SpotCheck":
        """A running balance walk (#4413): the same bands and picks, stopped at the F-beta peak.

        *n_pos* is the walk's count of the ranking's positives, the
        mixture's over the unvoted ranking (:func:`mixture_positives`), fixed
        at the start: the audits read each band's share, the count turns
        that into recall.  It starts at the bands holding *start_count*, the
        balance's unchecked count by default (:func:`balance_count`).

        The arms of #4427, all off in the app: *picks* a band (the
        schedule's 5), *tol* (a deeper step within it of the best is flat and
        the walk looks one band further; 0 is the strict rise), *fine* (every
        band past the start split in two, :func:`band_edges_fine`) and *guard*
        (a deeper band whose audited share right is below *guard* times the
        start set's ends the walk at the best set so far: the walk may not buy
        recall with a band that is mostly wrong, whatever the estimate says)
        and *shallow_only* (the walk never tries a deeper band: the start set
        is audited and the walk steps shallower while the estimate does not
        fall, so it costs the start's picks alone and can only cut the line;
        ``None``, the app's, follows :func:`check_shape`: shallower-only under
        the ``trim`` shape, the full walk under ``advisory``).
        """
        schedule = balance_schedule(beta, alpha)
        if not (n_pos > 0):
            raise ValueError("a balance walk needs a positive count of the ranking's positives")
        if tol < 0:
            raise ValueError(f"tol must be >= 0, got {tol!r}")
        if guard is not None and guard < 0:
            raise ValueError(f"guard must be >= 0, got {guard!r}")
        if picks is not None and picks < 1:
            raise ValueError(f"picks must be >= 1, got {picks!r}")
        start = start_count if start_count is not None else schedule.candidate
        n = len(tuple(ranking_ids))
        check = cls.start(
            ranking_ids,
            schedule.candidate and 0.5,  # unused by a balance walk; a valid floor for the schedule's picks
            alpha=alpha,
            seed=seed,
            start_count=start,
            picks=picks,
            edges=band_edges_fine(n, min(start, n)) if fine else None,
        )
        check.beta = float(beta)
        check.n_pos = float(n_pos)
        check.min_precision = float("nan")
        check.tol = float(tol)
        check.guard = None if guard is None else float(guard)
        check.shallow_only = check_shape(beta) == CHECK_TRIM if shallow_only is None else bool(shallow_only)
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
        return self.status in (FLOOR_CONFIRMED, FLOOR_SHORT, BALANCE_CHECKED)

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

    def positives_estimate(self, bands: int | None = None) -> float | None:
        """The band-stratified count of positives in the top *bands* bands (the current set by default)."""
        bands = self.bands if bands is None else bands
        if bands <= 0:
            return 0.0
        total = 0.0
        for b in range(bands):
            size, labelled, right = self.band_counts(b)
            if size and not labelled:
                return None
            total += size * (right / labelled if labelled else 0.0)
        return total

    def fbeta_estimate(self, bands: int | None = None) -> float | None:
        """A balance walk's F-beta estimate for the top *bands* bands: audited positives over the walk's count."""
        if self.beta is None or self.n_pos is None:
            return None
        bands = self.bands if bands is None else bands
        tp = self.positives_estimate(bands)
        if tp is None:
            return None
        return fbeta_score(tp, float(self.edges[bands]), self.n_pos, self.beta)

    def recall_range(self) -> LikelyRange | None:
        """A balance walk's likely range for the current set's recall: the positives' range over the walk's count."""
        if self.n_pos is None or self.k <= 0:
            return None
        lo = hi = 0.0
        labelled_all = right_all = 0
        for b in range(self.bands):
            size, labelled, right = self.band_counts(b)
            part = likely_range(right, labelled, size, self.tail)
            lo += size * part.lo
            hi += size * part.hi
            labelled_all += labelled
            right_all += right
        return LikelyRange(min(1.0, lo / self.n_pos), min(1.0, hi / self.n_pos), labelled_all, right_all)

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
        if self.beta is not None:
            self._evaluate_balance()
            return
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

    def _evaluate_balance(self) -> None:
        """Deeper while the F-beta estimate rises; from a start whose first deeper step does not, shallower while it does not fall.

        A start on the ranking's last band has no deeper step, so it goes
        shallower straight away, as the reference's ``rule_fb_walk`` does:
        on a corpus no larger than the cap that is every walk, and ending
        there kept the whole ranking whatever its audits said.  The peak's
        band edge is the kept set; on a tie the smaller set wins (fewer items,
        and no further picks).  A shallower set is a subset of an audited
        one, so stepping back costs no picks.
        """
        est = self.fbeta_estimate()
        if est is None:
            self._finish(BALANCE_CHECKED, max(1, self.bands))
            return
        if self.direction == WALK_START:
            self.start_share = self.estimate()
        elif self._guard_stops():
            return
        if self.direction in (WALK_START, WALK_DEEPER):
            self._walk_deeper_or_turn(est)
        else:
            self._walk_shallower(est)

    def _walk_deeper_or_turn(self, est: float) -> None:
        """From the start or a deeper step: deeper on a rise (or a flat step within the tolerance), else the peak or a turn."""
        if self._shallow_only_turns(est):
            return
        if self.best is None or est > (self.best_estimate or 0.0) + _EPS:
            self.best, self.best_estimate = self.bands, est
            if self.bands < self.n_bands:
                self.direction = WALK_DEEPER
                self.bands += 1
                self.k = int(self.edges[self.bands])
                return
            # The last band.  Reached by rising, it is the peak; started
            # on, it has not been compared with anything yet.
            if self.direction == WALK_DEEPER or self.bands <= 1:
                self._finish(BALANCE_CHECKED, self.bands)
                return
        elif self._flat_within_tolerance(est):
            # Flat within the tolerance (#4427): look one band further
            # before calling it the peak.  The best stays where it was.
            self.direction = WALK_DEEPER
            self.bands += 1
            self.k = int(self.edges[self.bands])
            return
        elif self.best != self.start_bands or self.best <= 1:
            # The estimate fell after a rise: the peak was the last set.
            self._finish(BALANCE_CHECKED, self.best)
            return
        # The first step from the start fell, or there was none: try the other way.
        self.direction = WALK_SHALLOWER
        self.bands = self.best - 1
        self.k = int(self.edges[self.bands])

    def _shallow_only_turns(self, est: float) -> bool:
        """The shallower-only walk (#4427): the audited start set is the best so far and the walk turns
        shallower at once, never trying a deeper band; a start on the first band is the kept set."""
        if not self.shallow_only or self.direction != WALK_START:
            return False
        self.best, self.best_estimate = self.bands, est
        if self.bands <= 1:
            self._finish(BALANCE_CHECKED, self.bands)
            return True
        self.direction = WALK_SHALLOWER
        self.bands = self.best - 1
        self.k = int(self.edges[self.bands])
        return True

    def _flat_within_tolerance(self, est: float) -> bool:
        return (
            self.tol > 0
            and self.best is not None
            and est >= (self.best_estimate or 0.0) - self.tol
            and self.bands - self.best < 2
            and self.bands < self.n_bands
        )

    def _walk_shallower(self, est: float) -> None:
        """Shallower while the estimate does not fall; the peak is the kept set."""
        if self.best is not None and est >= (self.best_estimate or 0.0) - _EPS:
            self.best, self.best_estimate = self.bands, est
            if self.bands <= 1:
                self._finish(BALANCE_CHECKED, self.bands)
                return
            self.bands -= 1
            self.k = int(self.edges[self.bands])
            return
        self._finish(BALANCE_CHECKED, self.best if self.best is not None else 1)

    def _guard_stops(self) -> bool:
        """The precision guard (#4427): end the walk at the best set so far when the band just audited on a
        deeper step is below ``guard`` times the start set's share right, whatever the estimate says."""
        if self.direction != WALK_DEEPER or self.guard is None or self.start_share is None:
            return False
        _size, labelled, right = self.band_counts(self.bands - 1)
        if not labelled or right / labelled >= self.guard * self.start_share:
            return False
        self._finish(BALANCE_CHECKED, self.best if self.best is not None else max(1, self.bands - 1))
        return True

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
        balance = {}
        if self.beta is not None:
            fb = self.fbeta_estimate()
            recall = self.recall_range() if labelled else None
            balance = {
                "beta": self.beta,
                "fbeta": None if fb is None else round(fb, 4),
                "recall": recall.as_dict() if recall is not None else None,
            }
        return {
            "status": self.status,
            "min_precision": None if self.beta is not None else self.min_precision,
            **balance,
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
    posterior = mixture_posterior(ranking, labels, also_voted)
    if posterior is None:
        return None
    cum = np.cumsum(posterior) / np.arange(1, posterior.size + 1)
    ok = np.flatnonzero(cum >= float(min_precision) - _EPS)
    return int(ok.max()) + 1 if ok.size else int(np.argmax(cum)) + 1


#: A mixture component whose spread is under this share of the ranking's
#: score range has collapsed onto a few near-duplicate scores (#4419): its
#: posterior puts no mass on anything else, so the fit is no estimate and the
#: counts that read it keep the schedule's cap instead.
MIXTURE_MIN_STD_SHARE = 1e-2


def _mixture_is_sound(fit: Any, scores: np.ndarray) -> bool:
    """True when both components of *fit* spread over a real share of *scores*' range (#4419)."""
    if fit is None or not (fit.var_hi > 0 and fit.var_lo > 0):
        return False
    spread = float(scores.max() - scores.min()) if scores.size else 0.0
    if not spread > 0:
        return False
    least = MIXTURE_MIN_STD_SHARE * spread
    return math.sqrt(float(fit.var_hi)) >= least and math.sqrt(float(fit.var_lo)) >= least


def mixture_posterior(
    ranking: LineRanking | None,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
) -> np.ndarray | None:
    """The mixture's high-component posterior at each unvoted item of *ranking*, best first; ``None`` when none fits.

    The fit is :func:`mixture_count`'s, memoised on the ranking; this is the
    read every mixture-based count shares.  A fit with a collapsed component
    (:data:`MIXTURE_MIN_STD_SHARE`, #4419) is no fit: on a tiny ranking the
    high component can land on the top two scores alone, and its posterior
    then says nothing about the rest.
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
        ranking.mixture.append(fit if _mixture_is_sound(fit, ranking.scores) else None)
    fit = ranking.mixture[0]
    if fit is None:
        return None
    hi = fit.w_hi * norm.pdf(scores, fit.mu_hi, math.sqrt(fit.var_hi))
    lo = fit.w_lo * norm.pdf(scores, fit.mu_lo, math.sqrt(fit.var_lo))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(hi / (hi + lo), nan=0.5)


def mixture_positives(
    ranking: LineRanking | None,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
) -> float | None:
    """The mixture's count of positives among the unvoted items: the posterior's total (#4413).

    What a balance walk turns audited shares into recall with.  Over-counts
    on large sparse corpora (10-28x at 0.1%, #3827), which is why the
    unchecked balance line is capped and the walk audits.
    """
    posterior = mixture_posterior(ranking, labels, also_voted)
    return None if posterior is None else float(posterior.sum())


def walk_positives(
    ranking: LineRanking | None,
    beta: float,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
) -> float:
    """The positives a balance walk reads recall against: the mixture's count, else the balance's cap (#4419).

    The mixture (:func:`mixture_positives`) is the estimate when it has one.
    When it does not - nothing fits, or the fit collapsed - the walk still
    has to start, so the denominator is the set the unchecked line would keep
    anyway: the balance's cap (:func:`balance_schedule`), lowered to what is
    unvoted.  Against that denominator the walk goes deeper while the audited
    share right holds up, which is the floor's instinct with the balance's
    stop.  A sound fit that counts fewer than one positive (every positive
    already voted Good, say) takes the cap too: the walk's recall is its
    audited positives over this count, so a near-zero count would report
    every check as having found them all.  Always at least one.  The app's
    check route and the harness's end-of-run check both read this, so a
    check starts in the same cases.
    """
    n_pos = mixture_positives(ranking, labels, also_voted)
    if n_pos is not None and n_pos >= 1.0:
        return n_pos
    cap = balance_schedule(beta).candidate
    unvoted = ranking.unvoted_ids(also_voted).size if ranking is not None else 0
    return float(max(1, min(cap, unvoted) if unvoted else cap))


def acquisition_count(
    ranking: LineRanking | None,
    beta: float,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
    factor: float | None = ACQUISITION_ARGMAX_FACTOR,
) -> int | None:
    """How deep Autopilot's acquisition cut sits under a balance: *factor* of the F-beta argmax's depth (#4409).

    The argmax is :func:`fbeta_count`'s, uncapped; the count is at least one.
    ``None`` with no *factor* (the shipped value since the #4427 revert: the
    caller keeps the line - 4 cut), no ranking or no mixture estimate.
    :data:`~vtscore.training.thresholds.knobs.ACQUISITION_ARGMAX_FACTOR` is
    what ships; the harness's ``acq_p_crossing`` arm passes a number.
    """
    if factor is None:
        return None
    k = fbeta_count(ranking, beta, labels, also_voted)
    return None if k is None else max(1, round(float(factor) * k))


def acquisition_threshold(
    ranking: LineRanking | None,
    beta: float,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
    factor: float | None = ACQUISITION_ARGMAX_FACTOR,
) -> float | None:
    """The score the acquisition cut sits at under a balance (#4409): the last unvoted item :func:`acquisition_count` keeps."""
    k = acquisition_count(ranking, beta, labels, also_voted, factor)
    if k is None or ranking is None:
        return None
    cut = ranking.threshold_for(k, also_voted)
    return None if cut is None or not math.isfinite(float(cut)) else float(cut)


def fbeta_count(
    ranking: LineRanking | None,
    beta: float,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
) -> int | None:
    """How many unvoted items the mixture says maximise F-*beta*: the ``fb-gmm`` rule of #4411.

    The posterior's cumulative sum is the positives in the top *k*, its total
    the corpus's; the count is the argmax of the F-beta those give.  ``None``
    when nothing fits.  Right-sized at 5% and on small corpora, 2-4x too deep
    on large sparse ones, so the unchecked line caps it (:func:`balance_count`).
    """
    posterior = mixture_posterior(ranking, labels, also_voted)
    if posterior is None:
        return None
    tp = np.cumsum(posterior)
    b2 = float(beta) * float(beta)
    est = (1.0 + b2) * tp / (b2 * float(tp[-1]) + np.arange(1, posterior.size + 1))
    return int(np.argmax(est)) + 1


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
    if result is None or not result.finished or result.beta is not None:
        return None  # a balance walk's result never serves a floor (#4413); its min_precision is NaN
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


def aim_words(state: Mapping[str, Any]) -> str:
    """The preference a line's state was drawn at, in words for a log line (#4413).

    ``aiming at 25% right`` for a floor's state (one that carries
    ``min_precision``), ``at F1`` for a balance's (one that carries ``beta``):
    what a headless run says beside the unchecked set it exports.
    """
    beta = state.get("beta")
    if beta is not None:
        return f"at F{beta:g}"
    return f"aiming at {100 * state['min_precision']:.0f}% right"


@dataclass(frozen=True)
class BalanceState:
    """What the balance says about a detector's line: its state, the set's size and what the check estimated (#4413).

    The wire shape beside ``threshold`` under a balance, as :class:`FloorState`
    is under a floor: ``status`` is ``unchecked`` or ``checked``; ``precision``
    and ``recall`` are the audited set's likely ranges; ``fbeta`` its estimate.
    Under the ``advisory`` shape (#4427) the audited set is the walk's end and
    the kept ``count`` is the unchecked rule's; under ``trim`` they coincide.
    """

    beta: float
    status: str
    count: int
    precision: LikelyRange | None
    recall: LikelyRange | None
    fbeta: float | None
    stale: bool
    schedule: CheckSchedule
    #: How a check treats the line at this beta (:func:`check_shape`, #4427):
    #: ``advisory`` (the ranges inform, the count stays the unchecked rule's)
    #: or ``trim`` (the walk's end is the count).
    shape: str = CHECK_TRIM
    #: The set the last check audited (the walk's end), which under ``advisory``
    #: is not the set the line keeps; ``None`` while unchecked.
    audited: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "status": self.status,
            "count": self.count,
            "precision": None if self.precision is None else {**self.precision.as_dict(), "stale": self.stale},
            "recall": None if self.recall is None else {**self.recall.as_dict(), "stale": self.stale},
            "fbeta": None if self.fbeta is None else round(self.fbeta, 4),
            "schedule": self.schedule.as_dict(),
            "shape": self.shape,
            "audited": self.audited,
        }


def applicable_balance(beta: float, result: SpotCheck | None) -> SpotCheck | None:
    """*result* when it is a finished balance walk at *beta*; ``None`` otherwise (a floor walk never applies)."""
    if result is None or not result.finished or result.beta is None:
        return None
    if abs(result.beta - float(beta)) > _EPS:
        return None
    return result


def balance_count(beta: float, result: SpotCheck | None, proposal: int | None = None, shape: str | None = None) -> int:
    """The count the line keeps at *beta*: the finished walk's end where the check's shape lets it move the line, else the unchecked count.

    The unchecked count is the balance's cap (:func:`balance_schedule`),
    lowered to *proposal* when the caller has one: the mixture's F-beta argmax
    (:func:`fbeta_count`).  Under the ``advisory`` shape (:func:`check_shape`,
    beta <= 1; #4427) a finished walk informs the line and never moves it;
    *shape* overrides the preset's (the harness's full-walk arm).
    """
    applicable = applicable_balance(beta, result)
    if applicable is not None and (shape or check_shape(beta)) != CHECK_ADVISORY:
        return applicable.k
    count = balance_schedule(beta).candidate
    if proposal is not None:
        count = max(1, min(count, int(proposal)))
    return count


def balance_line(
    ranking: LineRanking | None,
    beta: float,
    result: SpotCheck | None = None,
    also_voted: Iterable[int] = (),
    proposal: int | None = None,
    shape: str | None = None,
) -> float | None:
    """The threshold the line sits at under a balance: the last item of the set it keeps (``None``: no ranking)."""
    if ranking is None:
        return None
    return ranking.threshold_for(balance_count(beta, result, proposal, shape), also_voted)


def balance_state(
    beta: float,
    result: SpotCheck | None = None,
    ranking: LineRanking | None = None,
    also_voted: Iterable[int] = (),
    proposal: int | None = None,
    shape: str | None = None,
) -> BalanceState:
    """The balance's state at *beta*, given the detector's last finished walk, its ranking and the mixture's count.

    *shape* overrides the preset's check shape (:func:`check_shape`); the
    harness's full-walk arm passes ``trim`` so the walk's end is the count.
    """
    schedule = balance_schedule(beta)
    shape = shape or check_shape(beta)
    applicable = applicable_balance(beta, result)
    count = balance_count(beta, result, proposal, shape)
    if ranking is not None and (applicable is None or shape == CHECK_ADVISORY):
        # The unchecked rule's count, capped by what is unvoted; a walk's end
        # under ``trim`` is a band edge of the ranking it walked, kept as is.
        count = min(count, len(ranking.candidate(count, also_voted)))
    if applicable is None:
        return BalanceState(float(beta), FLOOR_UNCHECKED, count, None, None, None, False, schedule, shape, None)
    stale = ranking is not None and applicable.is_stale(ranking, also_voted)
    return BalanceState(
        float(beta),
        BALANCE_CHECKED,
        count,
        applicable.range(),
        applicable.recall_range(),
        applicable.fbeta_estimate(),
        stale,
        schedule,
        shape,
        applicable.k,
    )


__all__ = [
    "BALANCE_CHECKED",
    "BALANCE_PRESETS",
    "BALANCE_STATES",
    "BETA_MAX",
    "BETA_MIN",
    "BAND_BASE",
    "CHECK_ADVISORY",
    "CHECK_ALPHA",
    "CHECK_BASE_CANDIDATE",
    "CHECK_CANCELLED",
    "CHECK_SHAPES",
    "CHECK_TRIM",
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
    "check_shape",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "floor_count",
    "floor_line",
    "floor_state",
    "likely_range",
    "line_under",
    "mixture_count",
    "mixture_positives",
    "mixture_posterior",
    "fbeta_count",
    "fbeta_score",
    "balance_schedule",
    "balance_count",
    "balance_line",
    "balance_state",
    "applicable_balance",
    "BalanceState",
    "DEFAULT_BETA",
    "range_tail",
    "rounds_for",
]
