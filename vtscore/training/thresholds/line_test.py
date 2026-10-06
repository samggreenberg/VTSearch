"""The test sample: what Test mode measures about a detector's line, and how (#4527).

Test mode (``vtscore/docs/packages/training.md``) asks one question of a detector on a
corpus it never trained on: *if this line went to AutoRun, what share of what
it ships would be right, and what share of the real matches would it ship?*
The answer is the line's precision and recall on that corpus, each as a
likely range, and F-beta at the user's balance as the headline, all from
uniform picks within rank bands: every number here is a function of those
picks and of nothing the model chose (#4257: model-chosen votes broke 83% of
the old estimator's promises).  This module is the sample, its estimators,
the rule that says where the next round of picks goes, and the rule that
says when a phase is done.  It is pure statistics over ids, ranks and labels,
Flask-clean, and holds no vector; the app's routes and the eval harness both
call it, so the two cannot disagree.

**The sample** (:class:`LineTest`) is a frozen ranking and a line: the top
``line_count`` items are what the line keeps.  Both sides of the line are cut
into bands - the spot check's doubling bands from the top for the matches
(:func:`~vtscore.training.thresholds.spot_check.band_edges`: 8, 8, 16, 32,
... to the line) and the same doubling from the line downward for the misses
- and a round of picks is drawn uniformly without replacement from one band
(a census when the band holds no more than a round).  Which band each pick
came from is recorded, because the estimators weight by band.

**The estimators** are joint Monte Carlo ranges from per-band posteriors,
drawn once per state and shared by every number (:meth:`LineTest.estimates`):

* Each band's share right has a Beta posterior, and the band's unlabelled
  items are drawn binomially at that share, so a censused band is exact and
  a band with no picks is as uncertain as its prior says.  Above the line a
  band's prior is the *pooled* share of every pick above the line (Jeffreys
  on the pool) at :data:`POOLED_WEIGHT` picks' worth, drawn once per column
  and shared by every band (#4539): independent Jeffreys priors read each
  empty band of a big sparse line as about 8% right, and summed over a
  thousand items that put the range above the truth every time (#4523).
* **Above the line** the draws give the band-weighted precision of any
  top-*b* union, so the same draws re-estimate the line at every band edge
  (:attr:`LineEstimates.at_edges`), which the verdict's *Lean the Threshold*
  exit reads.
* **Below the line** the count of positives is model-assisted: the labels
  line's item posteriors are the auxiliary (what the model expects each band
  to hold), corrected band by band by the picks.  A band's prior is the
  model's own share for it, at the weight of one round of picks
  (:attr:`LineBudgets.model_weight`, floored by :data:`MODEL_FLOOR`), and
  the picks update it as observations, so the band's count is its size times
  that posterior - the band design's Horvitz-Thompson weights - centred on
  the model where the picks agree with it and moved by the matches they
  find.  Uniform picks from a 10,000-item tail at 0.4% prevalence can bound
  nothing on their own: five picks that find nothing leave a Jeffreys
  posterior admitting an 8% rate, which read as hundreds of positives.
  Bands the walk never reached are taken from the model alone, as a point,
  and flagged (:attr:`LineEstimates.tail_from_model`).  With no model, the
  bands below the line take the Jeffreys prior and the tail counts nothing.
* Recall and F-beta come from the same draws: recall is the line's positives
  over the corpus's, F-beta the textbook one at the sample's beta.
* Each estimate is a point (the posterior mean) and a central range at
  ``1 - alpha`` (:data:`TEST_ALPHA`, 95%), plus the *found* words the spot
  check already reads a recall range as (:func:`found_words`).

**The allocation rule** (:meth:`LineTest.next_band`) is a pure function of
the sample.  Above the line the first rounds audit every band once, from the
band holding the line upward (where the detector is least sure and the set
is largest); later rounds go to the band whose next round would shrink the
F-beta range most in expectation - the greedy face of Neyman allocation
(``docs/plans/coverage-atlas.md`` §6.3), computed on the draws by
pre-posterior analysis over the round's possible outcomes.  Below the line
the walk takes the first band under the line first, then one band deeper a
round.  With a class model it walks to its pick budget: every band it
reaches is the model's count corrected by picks, and a walk that stops
early leaves the tail to the model alone, as a point, whose recall range
held the truth in 13-38% of #4523's sessions against 75-94% for the walk to
the budget.  Without one it stops at a dry run - a band with no match in
its picks - since a deeper band would be read on the Jeffreys prior alone,
and on bands of thousands that counts hundreds of phantom positives.

**The phase machine and stop rule** (:func:`line_phase`) are a pure function
of the sample and its budgets (:class:`LineBudgets`), so the app's view and
the harness derive the phase from state on every poll rather than
accumulating it.  The phases are the matches (precision), the misses
(recall) and ``done``.  The matches phase ends when its range is narrower
than its target, at its pick budget, or when its bands are exhausted; the
misses phase at its pick budget or when its bands are exhausted, and,
without a class model, on its width target or a dry run.  A line that keeps
fewer items than one round is ``nothing``: nothing to test.  The targets and
budgets are parameters whose defaults are the values #4523 priced
(``docs/experiments/2026-10-05-line-test-4523/REPORT.md``).

Test votes never train the detector: they are recorded with
:data:`TEST_PROVENANCE` so a later merge can tell them from a check's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vtscore.training.thresholds.spot_check import (
    BETA_MAX,
    BETA_MIN,
    CHECK_MIN_PICKS,
    LikelyRange,
    band_edges,
    likely_range,
)

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

#: What the pooled share above the line is worth to one band, in picks (#4539).
#: Each band's prior is drawn from the pooled posterior of every pick above the
#: line at this weight, so one round of a band's own picks weighs as much as the
#: pool and a rich band still reads richer; but empty bands no longer read as 8%
#: each (the Jeffreys mean of 0 of 5) summed over a thousand items, which is how
#: a sparse line's precision range came to sit above its truth (#4523).
POOLED_WEIGHT = float(CHECK_MIN_PICKS)


def pooled_weight(depth: int, n_above: int) -> float:
    """What the pooled share is worth, in picks, to the band *depth* bands up from the line (0 holds the line).

    *n_above* is how many bands sit above the line.  The one place a band's
    pull toward the pool is decided, so a study can price another taper by
    replacing it (#4560).
    """
    del depth, n_above
    return POOLED_WEIGHT


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


# ------------------------------------------------------------------ the sample


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class LineTest:
    """A test of one line on one frozen ranking: its bands, the picks in each, and what they say.

    Built by :meth:`start` on the ranking as the Score pass left it (ids in
    rank order, the top *line_count* kept by the line, the labels line's
    item posteriors aligned with the ids when the line has them).  The
    ranking never changes afterwards - no retrain, no re-sort, no line move
    between votes - which is what makes the band design valid.
    :meth:`draw` deals the next round from the band :meth:`next_band` picks;
    :meth:`record` takes the labels on it.  :meth:`estimates` and
    :meth:`phase` read the state.
    """

    ranking_ids: tuple[int, ...]
    line_count: int
    beta: float
    budgets: LineBudgets = DEFAULT_BUDGETS
    #: Each item's chance of being a positive by the labels line, aligned with
    #: :attr:`ranking_ids`; ``None`` when the line has no corpus fit (no class
    #: model), in which case the unreached tail counts nothing and the walk
    #: below the line stops at its first dry band, so recall is unmeasured
    #: below the bands it reached.
    posteriors: np.ndarray | None = field(default=None, repr=False, compare=False)
    labels: dict[int, bool] = field(default_factory=dict)
    #: The band each labelled (or pending) pick was drawn from.
    pick_band: dict[int, int] = field(default_factory=dict)
    pending: tuple[int, ...] = ()
    #: The band the pending picks were drawn from; ``None`` between rounds.
    band: int | None = None
    #: Rounds recorded so far.
    rounds: int = 0
    seed: int | None = None
    #: When the picks restored by :meth:`start` were taken, for a test resumed
    #: from a verdict kept on the detector (#4526); ``None`` for a fresh test.
    kept_at: float | None = None
    #: Bumped on every label change; the estimates cache is keyed on it.
    _version: int = field(default=0, repr=False, compare=False)
    _bands: tuple[Band, ...] = field(default=(), repr=False, compare=False)
    _rank: dict[int, int] = field(default_factory=dict, repr=False, compare=False)
    _generator: Any = field(default=None, repr=False, compare=False)
    _cache: list[Any] = field(default_factory=list, repr=False, compare=False)
    #: The joint draws every estimate rests on, keyed on :attr:`_version`.
    _counts: list[Any] = field(default_factory=list, repr=False, compare=False)

    @classmethod
    def start(
        cls,
        ranking_ids: Sequence[int],
        line_count: int,
        beta: float,
        *,
        posteriors: Sequence[float] | np.ndarray | None = None,
        budgets: LineBudgets = DEFAULT_BUDGETS,
        seed: int | None = None,
        labels: Mapping[int, bool] | None = None,
    ) -> "LineTest":
        """A test over *ranking_ids* (rank order) whose line keeps the top *line_count*, at balance *beta*.

        *labels* restores picks already taken on this ranking (a resumed
        test, #4526): each is placed in the band its rank falls in.  The
        first round is not drawn; call :meth:`draw`.
        """
        ids = tuple(int(i) for i in ranking_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("the ranking's ids must be distinct")
        if not (BETA_MIN - _EPS <= float(beta) <= BETA_MAX + _EPS):
            raise ValueError(f"beta must be in [{BETA_MIN}, {BETA_MAX}], got {beta!r}")
        post = None
        if posteriors is not None:
            post = np.clip(np.nan_to_num(np.asarray(posteriors, dtype=np.float64), nan=0.0), 0.0, 1.0)
            if post.shape != (len(ids),):
                raise ValueError("posteriors must align with the ranking")
        k = max(0, min(int(line_count), len(ids)))
        test = cls(
            ranking_ids=ids,
            line_count=k,
            beta=float(beta),
            budgets=budgets,
            posteriors=post,
            seed=seed,
            _bands=line_bands(len(ids), k),
            _rank={cid: i for i, cid in enumerate(ids)},
            _generator=_rng(seed),
        )
        for cid, right in (labels or {}).items():
            b = test.band_of(int(cid))
            if b is None:
                raise ValueError(f"not on this ranking: {cid}")
            test.labels[int(cid)] = bool(right)
            test.pick_band[int(cid)] = b
        test._invalidate()
        return test

    # ---- the bands

    @property
    def bands(self) -> tuple[Band, ...]:
        return self._bands

    @property
    def above(self) -> tuple[Band, ...]:
        return tuple(b for b in self._bands if b.side == ABOVE)

    @property
    def below(self) -> tuple[Band, ...]:
        return tuple(b for b in self._bands if b.side == BELOW)

    @property
    def size(self) -> int:
        return len(self.ranking_ids)

    @property
    def nothing_to_test(self) -> bool:
        """The line keeps fewer items than one round: no sample can say anything about it."""
        return self.line_count < self.budgets.picks_per_round

    def band_of(self, media_id: int) -> int | None:
        """The index of the band *media_id*'s rank falls in; ``None`` off the ranking."""
        r = self._rank.get(int(media_id))
        if r is None:
            return None
        for b in self._bands:
            if b.lo <= r < b.hi:
                return b.index
        return None

    def band_ids(self, b: int) -> tuple[int, ...]:
        band = self._bands[b]
        return self.ranking_ids[band.lo : band.hi]

    def band_counts(self, b: int) -> tuple[int, int, int]:
        """``(size, labelled, right)`` for band *b*."""
        band = self._bands[b]
        labelled = right = 0
        for cid, ok in self.labels.items():
            if self.pick_band.get(cid) == b:
                labelled += 1
                right += int(ok)
        return band.size, labelled, right

    def band_mass(self, b: int) -> float:
        """What the model expects band *b* to hold: the sum of its items' posteriors (0 with no model)."""
        if self.posteriors is None:
            return 0.0
        band = self._bands[b]
        return float(np.sum(self.posteriors[band.lo : band.hi]))

    def audited(self, b: int) -> bool:
        _, labelled, _ = self.band_counts(b)
        return labelled > 0

    def exhausted(self, b: int) -> bool:
        """Band *b* has no unlabelled item left to draw."""
        size, labelled, _ = self.band_counts(b)
        return labelled >= size

    def picks_on(self, side: str) -> int:
        return sum(1 for cid in self.labels if self._bands[self.pick_band[cid]].side == side)

    def band_range(self, b: int) -> LikelyRange:
        """Band *b*'s own Clopper-Pearson range at :attr:`LineBudgets.alpha`, for a per-band display."""
        size, labelled, right = self.band_counts(b)
        return likely_range(right, labelled, size, self.budgets.alpha / 2.0)

    # ---- the estimators

    def _invalidate(self) -> None:
        self._version += 1
        self._cache.clear()
        self._counts.clear()

    def _pooled_counts(self, extra: dict[int, tuple[int, int]] | None = None) -> tuple[int, int]:
        """``(picks, right)`` over every band above the line, *extra*'s hypothetical picks included."""
        labelled = right = 0
        for band in self.above:
            _, n_lab, n_right = self.band_counts(band.index)
            if extra and band.index in extra:
                n_lab += extra[band.index][0]
                n_right += extra[band.index][1]
            labelled += n_lab
            right += n_right
        return labelled, right

    def _pooled_weight(self, b: int) -> float:
        """Band *b*'s pull toward the pool (:func:`pooled_weight`), counted in bands up from the line."""
        above = self.above
        return pooled_weight(len(above) - 1 - b, len(above))

    def _pooled_share(
        self, rng: np.random.Generator, n: int, extra: dict[int, tuple[int, int]] | None = None
    ) -> np.ndarray:
        """*n* draws of the pooled share right above the line: Jeffreys' prior on every pick above it."""
        labelled, right = self._pooled_counts(extra)
        return rng.beta(JEFFREYS + right, JEFFREYS + (labelled - right), size=n)

    def _band_draws(
        self,
        b: int,
        rng: np.random.Generator,
        n: int,
        extra: tuple[int, int] | None = None,
        pooled: np.ndarray | None = None,
    ) -> np.ndarray:
        """*n* draws of band *b*'s count of positives: its picks, plus its unlabelled items at a drawn share.

        Above the line the share's prior is the pooled share of every pick
        above the line (*pooled*, one draw per column, shared by every band
        so the draws stay joint) at :data:`POOLED_WEIGHT` picks' worth
        (#4539); below it, with a model, it is the model's share of the band
        at :attr:`LineBudgets.model_weight` picks' worth (floored by
        :data:`MODEL_FLOOR`), which the picks then correct; below it with no
        model, Jeffreys'.  *extra* adds ``(picks, right)`` hypothetical picks
        to the band, for the allocation rule's pre-posterior look-ahead.
        """
        band = self._bands[b]
        size, labelled, right = self.band_counts(b)
        if extra is not None:
            labelled += extra[0]
            right += extra[1]
        rest = size - labelled
        if rest <= 0:
            return np.full(n, float(right))
        a: float | np.ndarray
        c: float | np.ndarray
        if band.side == ABOVE:
            if pooled is None:
                pooled = self._pooled_share(rng, n, None if extra is None else {b: extra})
            w = self._pooled_weight(b)
            a, c = MODEL_FLOOR + w * pooled, MODEL_FLOOR + w * (1.0 - pooled)
        elif self.posteriors is not None:
            mean = self.band_mass(b) / size
            weight = self.budgets.model_weight
            a, c = MODEL_FLOOR + weight * mean, MODEL_FLOOR + weight * (1.0 - mean)
        else:
            a = c = JEFFREYS
        share = rng.beta(a + right, c + (labelled - right), size=n)
        return right + rng.binomial(rest, share).astype(np.float64)

    def _draw_all(
        self, rng: np.random.Generator, n: int, extra: dict[int, tuple[int, int]] | None = None
    ) -> np.ndarray:
        """A ``(bands, n)`` matrix of per-band positive counts; unreached bands below the line are the model's point."""
        out = np.zeros((len(self._bands), n))
        pooled = self._pooled_share(rng, n, extra)
        for band in self._bands:
            if band.side == BELOW and not self.audited(band.index) and not (extra and band.index in extra):
                out[band.index] = self.band_mass(band.index)
                continue
            out[band.index] = self._band_draws(
                band.index, rng, n, None if extra is None else extra.get(band.index), pooled
            )
        return out

    def _tail(self) -> tuple[float, bool]:
        """The model's count in the unreached bands below the line, and whether there are any."""
        unreached = [b for b in self.below if not self.audited(b.index)]
        return sum(self.band_mass(b.index) for b in unreached), bool(unreached)

    def _summarise(self, counts: np.ndarray, beta: float, alpha: float) -> tuple[Estimate, Estimate, Estimate]:
        """``(precision, recall, fbeta)`` of the line from a per-band count matrix."""
        above = [b.index for b in self.above]
        tp = counts[above].sum(axis=0) if above else np.zeros(counts.shape[1])
        total = counts.sum(axis=0)
        k = float(self.line_count)
        precision = tp / k if k > 0 else np.zeros_like(tp)
        recall = np.where(total > 0, tp / np.maximum(total, _EPS), 0.0)
        fb = np.where(
            beta * beta * total + k > 0, (1.0 + beta * beta) * tp / np.maximum(beta * beta * total + k, _EPS), 0.0
        )
        return _estimate(precision, alpha), _estimate(recall, alpha), _estimate(fb, alpha)

    def estimates(self, beta: float | None = None) -> LineEstimates:
        """Every number the test reports, from one set of joint draws (cached until the next label)."""
        beta = self.beta if beta is None else float(beta)
        key = (self._version, beta)
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        n, alpha = self.budgets.draws, self.budgets.alpha
        counts = self._base_draws()
        precision, recall, fbeta = self._summarise(counts, beta, alpha)
        above = [b.index for b in self.above]
        below = [b.index for b in self.below]
        tp_above = counts[above].sum(axis=0) if above else np.zeros(n)
        tp_below = counts[below].sum(axis=0) if below else np.zeros(n)
        total = tp_above + tp_below
        tail, from_model = self._tail()
        edges: list[EdgeEstimate] = []
        running = np.zeros(n)
        for band in self._bands:
            running = running + counts[band.index]
            k = float(band.hi)
            p = running / k
            r = np.where(total > 0, running / np.maximum(total, _EPS), 0.0)
            fb = (1.0 + beta * beta) * running / np.maximum(beta * beta * total + k, _EPS)
            edges.append(
                EdgeEstimate(band.hi, band.side, _estimate(p, alpha), _estimate(r, alpha), _estimate(fb, alpha))
            )
        result = LineEstimates(
            beta=beta,
            precision=precision,
            recall=recall,
            fbeta=fbeta,
            positives_above=_estimate(tp_above, alpha),
            positives_below=_estimate(tp_below, alpha),
            tail_positives=float(tail),
            tail_from_model=from_model,
            at_edges=tuple(edges),
            labelled=len(self.labels),
        )
        self._cache[:] = [key, result]
        return result

    def expected_shrink(self, b: int) -> float:
        """How much narrower the F-beta range is expected to get from one more round in band *b* (above the line).

        Pre-posterior: over the round's possible counts of right picks, each
        weighted by its beta-binomial predictive under the band's posterior,
        the F-beta range's width is recomputed with the band's draws
        refreshed and the others kept; the shrink is the current width less
        that expectation.  Zero for a band with nothing left to draw.
        """
        return self._expected_shrink(b, self._base_draws())

    def _base_draws(self) -> np.ndarray:
        """The joint draws :meth:`estimates` rests on (cached until the next label), for the look-ahead to refresh one band of."""
        if not self._counts or self._counts[0] != self._version:
            self._counts[:] = [self._version, self._draw_all(_rng(TEST_DRAW_SEED), self.budgets.draws)]
        return self._counts[1]

    def estimate_at(self, count: int, beta: float | None = None) -> EdgeEstimate:
        """What the line would ship if it kept the top *count*, from the same draws (the verdict's **Lean the Threshold**).

        Exact at a band edge, where it is :attr:`LineEstimates.at_edges`'
        entry.  Inside a band the band's labelled picks count where their
        ranks fall, and its unlabelled positives are split by the class
        model's posterior mass on either side of *count* (#4540): a band's
        top is richer than its bottom, and splitting it by size read a
        shallower preset's precision as much as 0.19 low.  With no model, or
        a band the model counts empty, the split is by size.  *beta*
        defaults to the test's.  A count past the corpus keeps all of it.
        """
        beta = self.beta if beta is None else float(beta)
        k = max(0, min(int(count), self.size))
        counts = self._base_draws()
        n, alpha = counts.shape[1], self.budgets.alpha
        total = counts.sum(axis=0)
        running = np.zeros(n)
        for band in self._bands:
            if band.hi <= k:
                running = running + counts[band.index]
            elif band.lo < k:
                running = running + self._partial_band(band, k, counts[band.index])
        kf = float(k)
        p = running / kf if kf > 0 else np.zeros(n)
        r = np.where(total > 0, running / np.maximum(total, _EPS), 0.0)
        fb = (1.0 + beta * beta) * running / np.maximum(beta * beta * total + kf, _EPS)
        side = ABOVE if k <= self.line_count else BELOW
        return EdgeEstimate(k, side, _estimate(p, alpha), _estimate(r, alpha), _estimate(fb, alpha))

    def _partial_band(self, band: Band, k: int, band_counts: np.ndarray) -> np.ndarray:
        """The draws of band *band*'s positives that rank above *k*: its picks there, plus its share of the rest.

        *band_counts* are the band's drawn positives (its right picks plus its
        unlabelled items at a drawn share).  The right picks ranked above *k*
        count as themselves; the unlabelled positives are split by the
        model's posterior mass on the unlabelled items above *k*, or by their
        number with no model or no mass, and held to what each side can hold:
        no more above *k* than it has unlabelled items, no fewer than the
        rest of the band below *k* cannot take.
        """
        right_above = right_all = 0
        for cid, ok in self.labels.items():
            if self.pick_band.get(cid) != band.index or not ok:
                continue
            right_all += 1
            if self._rank[cid] < k:
                right_above += 1
        unlabelled = band_counts - right_all
        ranks = np.arange(band.lo, band.hi)
        free = np.array([self.ranking_ids[r] not in self.labels for r in ranks])
        upper = free & (ranks < k)
        weights = self.posteriors[band.lo : band.hi] if self.posteriors is not None else None
        if weights is not None and float(weights[free].sum()) > _EPS:
            share = float(weights[upper].sum()) / float(weights[free].sum())
        else:
            n_free = int(free.sum())
            share = int(upper.sum()) / n_free if n_free else 0.0
        n_upper, n_lower = int(upper.sum()), int(free.sum()) - int(upper.sum())
        above = np.clip(unlabelled * share, np.maximum(unlabelled - n_lower, 0.0), np.minimum(unlabelled, n_upper))
        return right_above + above

    def _expected_shrink(self, b: int, base: np.ndarray) -> float:
        size, labelled, right = self.band_counts(b)
        m = min(self.budgets.picks_per_round, size - labelled)
        if m <= 0:
            return 0.0
        n, alpha, beta = self.budgets.draws, self.budgets.alpha, self.beta
        current = self._summarise(base, beta, alpha)[2].width
        # The round's predictive under the band's prior: the pooled share's mean at the band's weight (#4539).
        pooled_lab, pooled_right = self._pooled_counts()
        p_mean = (pooled_right + JEFFREYS) / (pooled_lab + 2.0 * JEFFREYS)
        w = self._pooled_weight(b)
        a = right + MODEL_FLOOR + w * p_mean
        bb = labelled - right + MODEL_FLOOR + w * (1.0 - p_mean)
        expected = 0.0
        for r in range(m + 1):
            weight = math.comb(m, r) * _beta_fn(a + r, bb + m - r) / _beta_fn(a, bb)
            if weight <= 0.0:
                continue
            counts = base.copy()
            counts[b] = self._band_draws(b, _rng(TEST_DRAW_SEED + 7919 * (b + 1) + r), n, (m, r))
            expected += weight * self._summarise(counts, beta, alpha)[2].width
        return current - expected

    # ---- the allocation rule

    def misses_walk(self) -> tuple[int | None, str | None]:
        """Where the walk below the line stands: ``(next band to audit, why it ended)``; one of the two is ``None``.

        The walk audits the first band under the line, then each next band.
        With a class model it goes on until the phase's pick budget stops it
        (:func:`line_phase`) or the bands run out (#4523).  Without one a
        band with no match in its picks and a posterior mass under
        :attr:`LineBudgets.dry_run_share` of the positives found above the
        line is a dry run that ends the walk; every band's mass is zero
        then, so any share above zero stops it at the first band with no
        match.  Past the last band the walk is exhausted; with nothing below
        the line it is exhausted before it starts.
        """
        below = self.below
        if not below:
            return None, STOP_EXHAUSTED
        last_audited = None
        for band in below:
            if self.audited(band.index):
                last_audited = band
        if last_audited is None:
            return below[0].index, None
        if self.posteriors is None:
            _, _, right = self.band_counts(last_audited.index)
            found_above = max(self.estimates().positives_above.point, 1.0)
            dry = right == 0 and self.band_mass(last_audited.index) < self.budgets.dry_run_share * found_above
            if dry:
                return None, STOP_DRY_RUN
        nxt = last_audited.index + 1
        if nxt >= len(self._bands):
            return None, STOP_EXHAUSTED
        return nxt, None

    def next_band(self) -> int | None:
        """The band the next round goes to, by the phase the test is in; ``None`` when the test is done.

        Above the line: every band once, from the band holding the line
        upward, then the band whose round shrinks the F-beta range most in
        expectation (ties to the band nearest the line).  Below the line:
        the walk's next band (:meth:`misses_walk`).
        """
        report = self.phase()
        if report.phase == PHASE_MATCHES:
            candidates = [b for b in reversed(self.above) if not self.exhausted(b.index)]
            for band in candidates:
                if not self.audited(band.index):
                    return band.index
            if not candidates:
                return None
            base = self._base_draws()
            best = max(candidates, key=lambda band: (self._expected_shrink(band.index, base), band.index))
            return best.index
        if report.phase == PHASE_MISSES:
            nxt, _ = self.misses_walk()
            return nxt
        return None

    # ---- the phase machine

    def phase(self) -> PhaseReport:
        """Where the test stands, derived from its state and budgets (:func:`line_phase`)."""
        return line_phase(self)

    # ---- the rounds

    def draw(self) -> tuple[int, ...]:
        """Deal the next round: fresh picks drawn uniformly from the band the allocation rule names.

        Fewer than a round when the band holds fewer unlabelled items (a
        census of it).  ``()`` once the test is done, or while a round is
        still pending.
        """
        if self.pending:
            return self.pending
        b = self.next_band()
        if b is None:
            return ()
        fresh = [cid for cid in self.band_ids(b) if cid not in self.labels]
        m = min(self.budgets.picks_per_round, len(fresh))
        if m <= 0:
            return ()
        chosen = self._generator.choice(len(fresh), size=m, replace=False)
        self.pending = tuple(fresh[int(i)] for i in chosen)
        self.band = b
        for cid in self.pending:
            self.pick_band[cid] = b
        return self.pending

    def record(self, votes: Mapping[int, bool]) -> bool:
        """Take the user's labels on the pending picks (``True`` = a match / Good).

        Returns ``True`` when the round completed.  A label on an item that
        was not dealt is refused; a partial round waits for the rest.
        """
        stray = [cid for cid in votes if cid not in self.pending]
        if stray:
            raise ValueError(f"not this round's picks: {stray}")
        for cid, right in votes.items():
            self.labels[int(cid)] = bool(right)
        self._invalidate()
        self.pending = tuple(cid for cid in self.pending if cid not in self.labels)
        if self.pending:
            return False
        self.band = None
        self.rounds += 1
        return True

    def unrecord(self, media_id: int) -> None:
        """Take back one label of the current round (the ↓ key): the pick goes back to pending."""
        cid = int(media_id)
        if cid not in self.labels or self.band is None or self.pick_band.get(cid) != self.band:
            raise ValueError(f"not a label of this round: {media_id}")
        del self.labels[cid]
        self._invalidate()
        self.pending = tuple(i for i in self.band_ids(self.band) if i in self.pick_band and i not in self.labels)

    # ---- the wire shape

    def as_dict(self) -> dict[str, Any]:
        """The test as a client sees it: the phase, the round, the bands and their picks, and the estimates."""
        report = self.phase()
        bands = []
        for band in self._bands:
            size, labelled, right = self.band_counts(band.index)
            bands.append(
                {
                    **band.as_dict(),
                    "labelled": labelled,
                    "right": right,
                    "range": self.band_range(band.index).as_dict() if labelled else None,
                }
            )
        return {
            "phase": report.phase,
            "report": report.as_dict(),
            "beta": self.beta,
            "line_count": self.line_count,
            "size": self.size,
            "round": self.rounds + (1 if self.pending else 0),
            "picks_per_round": self.budgets.picks_per_round,
            "band": None if self.band is None else self._bands[self.band].as_dict(),
            "picks": list(self.pending),
            "labelled": len(self.labels),
            "bands": bands,
            "estimates": None if self.nothing_to_test else self.estimates().as_dict(),
            "budgets": self.budgets.as_dict(),
            "kept_at": self.kept_at,
            "class_model": self.posteriors is not None,
        }


def _beta_fn(a: float, b: float) -> float:
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


def line_phase(test: LineTest) -> PhaseReport:
    """The phase a test is in and why each finished phase ended, from its state and budgets alone.

    ``nothing`` when the line keeps fewer items than one round.  Otherwise
    the matches phase runs until every band above the line is exhausted, its
    precision range is at or under its target width once every band above the
    line has had a round (#4539), or its picks reach the matches budget; then the misses
    phase, until the walk below the line ends (exhausted, or, with no class
    model, a dry run), its recall range is under its target after at least
    one round below the line (with no class model only), or its picks reach
    the misses budget; then ``done``.  With a class model the recall range's
    width never stops the walk: it is narrow from the start, because the
    unreached tail is the model's point, and it holds only once the walk has
    corrected that point band by band (#4523).  Width is read on the ranges
    as they stand, so a phase's verdict is a function of the picks and never
    of the order they were taken in; when more than one reason holds, the one
    named is the first in that order (a censused line is ``exhausted``, not
    ``width``, though its range has no width at all).
    """
    budgets = test.budgets
    if test.nothing_to_test:
        return PhaseReport(PHASE_NOTHING, None, None, None, None, 0, 0)
    est = test.estimates()
    picks_above, picks_below = test.picks_on(ABOVE), test.picks_on(BELOW)
    matches_stop: str | None = None
    if all(test.exhausted(b.index) for b in test.above):
        matches_stop = STOP_EXHAUSTED
    elif (
        # Every band audited first: the pooled prior (#4539) can narrow the
        # range on two bands' picks, but a band no pick has seen is a guess.
        all(test.audited(b.index) or test.exhausted(b.index) for b in test.above)
        and est.precision.width <= budgets.matches_width + _EPS
    ):
        matches_stop = STOP_WIDTH
    elif picks_above >= budgets.matches_picks:
        matches_stop = STOP_BUDGET
    if matches_stop is None:
        return PhaseReport(PHASE_MATCHES, None, None, est.precision.width, est.recall.width, picks_above, picks_below)
    misses_stop: str | None = None
    _, walk_end = test.misses_walk()
    if walk_end is not None:
        misses_stop = walk_end
    elif test.posteriors is None and picks_below > 0 and est.recall.width <= budgets.misses_width + _EPS:
        misses_stop = STOP_WIDTH
    elif picks_below >= budgets.misses_picks:
        misses_stop = STOP_BUDGET
    phase = PHASE_DONE if misses_stop is not None else PHASE_MISSES
    return PhaseReport(
        phase, matches_stop, misses_stop, est.precision.width, est.recall.width, picks_above, picks_below
    )
