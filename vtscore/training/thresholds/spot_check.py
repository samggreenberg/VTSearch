"""The spot check that audits the balance's line, and what it estimated (#4272, #4388, #4413).

The user's preference is a **balance** of precision and recall: F-beta's
*beta* (presets 1/4, 1 and 4).  The line under a balance is the labels'
(:mod:`~vtscore.training.thresholds.labels_line`, #4452); a context with no
trained head falls back to the **count rule** here, the mixture's F-beta
argmax under the balance's cap (:func:`balance_count`).  A **spot check**
audits the ranking with the user's votes on uniform random picks, whose
labels say how much of a set is right without the bias model-chosen votes
carry (#4256, #4257).

**The check walks the ranking in bands (#4383, the owner's ruling of
2026-09-30).**

* **Bands.**  The unvoted ranking is cut into bands from the top: the top 8,
  the next 8, then 16, 32, 64, ... doubling to the corpus's end
  (:func:`band_edges`).  A band is audited with :data:`CHECK_MIN_PICKS` items
  drawn uniformly from it (a census when it holds no more than that), and a
  band is never drawn from twice.
* **The union under test** is the top ``b`` bands.  Its share right is
  band-stratified: each band's share of right answers weighted by the band's
  size, which is unbiased for the union whatever the picks per band
  (:meth:`SpotCheck.estimate`); its positives over the mixture's count of
  the ranking's positives is its recall, and the two give its F-beta
  (:meth:`SpotCheck.fbeta_estimate`).
* **The walk** starts at the bands that hold the balance's cap (the top 32 at
  beta <= 1, 128 above it; :func:`balance_schedule`), auditing each.  It goes
  one band deeper while the F-beta estimate rises and, from a start whose
  first deeper step does not, shallower while it does not fall; the peak is
  the set it ends on (#4411's ``fb-walk``).  The walk never moves the app's
  line: :func:`check_shape` is ``advisory`` at every preset (#4452), and
  ``trim`` (#4427) is a shape only the eval harness forces.
* **How close.**  A check carries a **likely range** for the precision and
  the recall of the set the line kept when it started (#4483, the owner's
  ruling of 2026-10-15; the walk's own set where no count was recorded): each
  band's part of that set is bounded exactly for its known size from the picks
  that fell in it (hypergeometric, each tail at ``alpha``; exact where the part
  was censused, 0-100% where no pick reached it), weighted by size
  (:meth:`SpotCheck.range_over`).

The count rule always keeps a set; the labels' line may keep none.  The
states a balance reports (:data:`BALANCE_STATES`): ``unchecked``,
``checked``, and ``gate`` on a structural detector's line
(:func:`gate_balance_state`).  A finished result
is kept and goes **stale** quietly once the ranking under it moves: later
votes retrain the model, the line follows the new ranking, and the ranges
stay with a ``stale`` flag.

The bands, the picks and the likely range are the ``grow-fine`` reference in
``scripts/experiments/calibration/analyze_line_estimate_4383.py``
(``band_edges``, ``draw_audits``, ``Audits.union_estimate``), and the walk's
stop is ``rule_fb_walk`` in ``analyze_fbeta_line_4411.py``;
``scripts/gates/check-eval-app-sync.py`` pins the schedule, the likely range
and the walk against them.

The spot check is four modules, split along its seams in #5425, each reading
only the ones before it: :mod:`~vtscore.training.thresholds.check_schedule`
(the knobs, the bands and the schedule),
:mod:`~vtscore.training.thresholds.ranges` (the likely range), this one (the
ranking and the walk), and :mod:`~vtscore.training.thresholds.balance` (the
mixture's counts and the balance's state; it reads this module's two classes
as types only).  Every public name of the four still imports from here.  A
stub goes on the module that defines the name: the other three never read
the copy here.

Everything here is scores, ids and labels - never a vector - so a
:class:`SpotCheck` and a :class:`LineRanking` may live on a detector context
for the life of the process (CLAUDE.md, "No Persisted Vectors").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vtscore.training.thresholds.balance import (
    MIXTURE_MIN_STD_SHARE,
    BalanceState,
    acquisition_count,
    acquisition_threshold,
    aim_words,
    applicable_balance,
    balance_count,
    balance_line,
    balance_state,
    fbeta_count,
    gate_balance_state,
    mixture_positives,
    mixture_posterior,
    walk_positives,
)
from vtscore.training.thresholds.check_schedule import (
    _EPS,
    BALANCE_CHECKED,
    BALANCE_GATE,
    BALANCE_PRESETS,
    BALANCE_STATES,
    BALANCE_UNCHECKED,
    BAND_BASE,
    BETA_MAX,
    BETA_MIN,
    CHECK_ADVISORY,
    CHECK_ALPHA,
    CHECK_BASE_CANDIDATE,
    CHECK_CANCELLED,
    CHECK_MIN_PICKS,
    CHECK_PROVENANCE,
    CHECK_RECALL_CANDIDATE,
    CHECK_RUNNING,
    CHECK_SHAPES,
    CHECK_TRIM,
    DEFAULT_BETA,
    LINE_DECIMALS,
    NO_BALANCE,
    WALK_DEEPER,
    WALK_SHALLOWER,
    WALK_START,
    CheckSchedule,
    balance_schedule,
    band_edges,
    band_edges_fine,
    bands_for,
    check_shape,
    fbeta_score,
    line_under,
    resolve_line_knobs,
    rounds_for,
)
from vtscore.training.thresholds.ranges import (
    LikelyRange,
    clopper_pearson_lower,
    clopper_pearson_upper,
    hypergeometric_lower,
    hypergeometric_upper,
    likely_range,
    range_tail,
)


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
    #: :func:`mixture_posterior`'s one-slot memo: the mixture fitted on this
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
    """One walk of a detector's line at one balance: its bands, the picks in each, and where it ended.

    Built by :meth:`start_balance` on the unvoted ranking as it stands at that
    moment (ids in rank order), cut into bands.  :meth:`draw` deals the next band's
    picks; :meth:`record` takes the user's labels on them and, once the band is
    audited, either deals the next band the union under test still owes, or
    decides: the walk goes deeper, shallower, or ends.  The ranking's ids never
    change after :meth:`start`, so every band samples the one list fixed then,
    however the model moves behind it.
    """

    alpha: float
    #: The unvoted ranking the walk is over, in rank order, fixed at the start.
    ranking_ids: tuple[int, ...]
    edges: tuple[int, ...]
    picks: int
    #: The bands the walk started from: those that hold the balance's cap.
    start_bands: int
    #: The walk's beta and its count of the ranking's positives (the
    #: mixture's, fixed at the start; #4413).
    beta: float
    n_pos: float
    #: The best F-beta estimate the walk has seen (the peak's).
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
    #: The peak so far: the union with the best F-beta estimate, if any.
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
    #: votes already cast), for :meth:`is_stale`; *k* is :attr:`range_count`.
    #: ``None`` while running.
    fingerprint: tuple[int, ...] | None = None
    #: How many unvoted items the line kept when the check started: the set
    #: its ranges describe (#4483), the top of :attr:`ranking_ids`.  ``None``
    #: where no count was recorded (a forced ``trim`` arm, whose line is the
    #: walk's end): the ranges describe the walk's set.
    kept: int | None = None
    _generator: Any = field(default=None, repr=False, compare=False)
    _rank: dict[int, int] = field(default_factory=dict, repr=False, compare=False)
    _audited: set[int] = field(default_factory=set, repr=False, compare=False)

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
        kept: int | None = None,
    ) -> "SpotCheck":
        """A running walk over *ranking_ids* (the unvoted ranking, rank order) at *beta*, with its first band drawn (#4413).

        *n_pos* is the walk's count of the ranking's positives, the
        mixture's over the unvoted ranking (:func:`mixture_positives`), fixed
        at the start: the audits read each band's share, the count turns
        that into recall.  It starts at the bands holding *start_count*, the
        balance's cap by default (:func:`balance_schedule`), or a caller's own
        proposal (#4389); a ranking shorter than that starts at its last band.

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

        *kept* is how many of *ranking_ids* the line keeps as the check
        starts (the top of the ranking): the set the finished check's ranges
        describe (#4483).  ``None`` leaves them on the walk's own set.
        """
        ids = tuple(int(i) for i in ranking_ids)
        if not ids:
            raise ValueError("a check needs at least one unvoted item to draw from")
        if len(set(ids)) != len(ids):
            raise ValueError("the ranking's ids must be distinct")
        schedule = balance_schedule(beta, alpha)
        if not (n_pos > 0):
            raise ValueError("a balance walk needs a positive count of the ranking's positives")
        if tol < 0:
            raise ValueError(f"tol must be >= 0, got {tol!r}")
        if guard is not None and guard < 0:
            raise ValueError(f"guard must be >= 0, got {guard!r}")
        if picks is not None and picks < 1:
            raise ValueError(f"picks must be >= 1, got {picks!r}")
        if kept is not None and kept < 0:
            raise ValueError(f"kept must be >= 0, got {kept!r}")
        start_count = schedule.candidate if start_count is None else start_count
        edges = band_edges_fine(len(ids), min(start_count, len(ids))) if fine else band_edges(len(ids))
        start = bands_for(min(start_count, len(ids)), edges)
        check = cls(
            alpha=float(alpha),
            ranking_ids=ids,
            edges=edges,
            picks=int(schedule.picks if picks is None else picks),
            start_bands=start,
            beta=float(beta),
            n_pos=float(n_pos),
            tol=float(tol),
            guard=None if guard is None else float(guard),
            shallow_only=check_shape(beta) == CHECK_TRIM if shallow_only is None else bool(shallow_only),
            bands=start,
            k=int(edges[start]),
            kept=None if kept is None else min(int(kept), len(ids)),
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
        return self.status == BALANCE_CHECKED

    @property
    def tail(self) -> float:
        """Each band's tail in a range: ``alpha`` (#4483; #4272 split it over the set's bands)."""
        return range_tail(self.bands, self.alpha)

    @property
    def range_count(self) -> int:
        """The size of the set a finished check's ranges describe: :attr:`kept`, or the walk's set without one."""
        return self.k if self.kept is None else self.kept

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
        """The F-beta estimate for the top *bands* bands: audited positives over the walk's count."""
        bands = self.bands if bands is None else bands
        tp = self.positives_estimate(bands)
        if tp is None:
            return None
        return fbeta_score(tp, float(self.edges[bands]), self.n_pos, self.beta)

    def parts_over(self, top: int) -> list[tuple[int, int, int]]:
        """``(size, labelled, right)`` for each band's part of the top *top* ranks, from every pick the walk drew.

        A band wholly inside the set is one part; the band the set's edge cuts
        counts only its ranks above the edge and the picks that fell there,
        which are uniform over that part; a band the walk never drew from is
        a part with no picks.
        """
        top = max(0, min(int(top), len(self.ranking_ids)))
        out: list[tuple[int, int, int]] = []
        for b in range(self.n_bands):
            lo, hi = self.edges[b], min(self.edges[b + 1], top)
            if lo >= top:
                break
            labelled = right = 0
            for cid, ok in self.labels.items():
                r = self._rank.get(cid)
                if r is not None and lo <= r < hi:
                    labelled += 1
                    right += int(ok)
            out.append((hi - lo, labelled, right))
        return out

    def _bounds_over(self, top: int) -> tuple[float, float, int, int, int]:
        """The summed bounds on the right items in the top *top*: ``(lo, hi, size, labelled, right)``."""
        lo = hi = 0.0
        size_all = labelled_all = right_all = 0
        for size, labelled, right in self.parts_over(top):
            part = likely_range(right, labelled, size, self.tail)
            lo += size * part.lo
            hi += size * part.hi
            size_all += size
            labelled_all += labelled
            right_all += right
        return lo, hi, size_all, labelled_all, right_all

    def range_over(self, top: int) -> LikelyRange | None:
        """The likely range of the top *top*'s precision: its parts' bounds, weighted by size; ``None`` for an empty set.

        Each part is bounded exactly for its size (:func:`likely_range` at
        :attr:`tail`): 0-100% where no pick reached it.  The set's range is the
        parts' size-weighted sum, which holds wherever every part's does.
        """
        lo, hi, size, labelled, right = self._bounds_over(top)
        if size <= 0:
            return None
        return LikelyRange(lo / size, hi / size, labelled, right)

    def recall_range_over(self, top: int) -> LikelyRange | None:
        """The likely range of the top *top*'s recall: its right items' bounds over the walk's count of positives."""
        lo, hi, size, labelled, right = self._bounds_over(top)
        if size <= 0:
            return None
        return LikelyRange(min(1.0, lo / self.n_pos), min(1.0, hi / self.n_pos), labelled, right)

    def recall_range(self) -> LikelyRange | None:
        """The likely range for the current set's recall: the positives' range over the walk's count."""
        return self.recall_range_over(self.k) if self.k > 0 else None

    def range(self) -> LikelyRange:
        """The likely range of the current set's precision (the walk's set under test; :meth:`range_over`)."""
        rng = self.range_over(self.k) if self.k > 0 else None
        return rng if rng is not None else LikelyRange(0.0, 1.0, 0, 0)

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
        """Whether the ranking under the result has moved since the check finished: its top :attr:`range_count` changed."""
        if self.fingerprint is None:
            return False
        return ranking.fingerprint(self.range_count, also_voted) != self.fingerprint

    def as_dict(self) -> dict[str, Any]:
        """The check as a client sees it: the band being audited, the set under test, the labels so far and the range."""
        labelled, right = self.counts()
        rng = self.range() if labelled else None
        est = self.estimate()
        band = None
        if self.band is not None:
            band = {"index": self.band, "lo": int(self.edges[self.band]) + 1, "hi": int(self.edges[self.band + 1])}
        fb = self.fbeta_estimate()
        recall = self.recall_range() if labelled else None
        return {
            "status": self.status,
            "beta": self.beta,
            "fbeta": None if fb is None else round(fb, 4),
            "recall": recall.as_dict() if recall is not None else None,
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


__all__ = [
    "BALANCE_CHECKED",
    "BALANCE_GATE",
    "BALANCE_PRESETS",
    "BALANCE_STATES",
    "BALANCE_UNCHECKED",
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
    "CHECK_RECALL_CANDIDATE",
    "CHECK_RUNNING",
    "WALK_DEEPER",
    "WALK_SHALLOWER",
    "WALK_START",
    "CheckSchedule",
    "LikelyRange",
    "LineRanking",
    "NO_BALANCE",
    "SpotCheck",
    "band_edges",
    "bands_for",
    "check_shape",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "hypergeometric_lower",
    "hypergeometric_upper",
    "likely_range",
    "line_under",
    "mixture_positives",
    "mixture_posterior",
    "fbeta_count",
    "fbeta_score",
    "balance_schedule",
    "balance_count",
    "balance_line",
    "balance_state",
    "gate_balance_state",
    "applicable_balance",
    "BalanceState",
    "DEFAULT_BETA",
    "range_tail",
    "rounds_for",
    "LINE_DECIMALS",
    "MIXTURE_MIN_STD_SHARE",
    "acquisition_count",
    "acquisition_threshold",
    "aim_words",
    "band_edges_fine",
    "resolve_line_knobs",
    "walk_positives",
]
