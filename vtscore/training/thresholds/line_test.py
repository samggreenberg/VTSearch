"""The test sample: what Test mode measures about a detector's line, and how (#4527).

Test mode (``vtscore/docs/packages/training.md``) asks one question of a detector on a
corpus it never trained on: *if this line went to AutoFind, what share of what
it ships would be right, and what share of the real matches would it ship?*
The answer is the line's precision and recall on that corpus, each as a
likely range, and F-beta at the user's balance as the headline, all from
uniform picks within rank bands: every number here is a function of those
picks and of nothing the model chose (#4256: model-chosen votes broke 83% of
the old estimator's promises).  This module is the sample, its estimators,
the rule that says where the next round of picks goes, and the rule that
says when a phase is done.  It is pure statistics over ids, ranks and labels,
Flask-clean, and holds no vector; the app's routes and the eval harness both
call it, so the two cannot disagree.

**The sample** (:class:`LineTest`) is a frozen ranking and a line: the top
``line_count`` items are what the line keeps.  Both sides of the line are cut
into bands - the spot check's doubling bands from the top for the matches
(:func:`~vtscore.training.thresholds.check_schedule.band_edges`: 8, 8, 16, 32,
... to the line) and the same doubling from the line downward for the misses
- and a round of picks is drawn uniformly without replacement from one band
(a census when the band holds no more than a round).  Which band each pick
came from is recorded, because the estimators weight by band.

**The estimators** are joint Monte Carlo ranges from per-band posteriors,
drawn once per state and shared by every number (:meth:`LineTest.estimates`):

* Each band's share right has a Beta posterior, and the band's unlabelled
  items are drawn binomially at that share, so a censused band is exact and
  a band with no picks is as uncertain as its prior says.  Above the line a
  band's prior is the *pooled* share of the picks in its neighbouring bands
  (:data:`POOL_RADIUS`, Jeffreys on the pool) at :data:`POOLED_WEIGHT`
  picks' worth (#4539, #4560): independent Jeffreys priors read each empty
  band of a big sparse line as about 8% right, which put the range above the
  truth every time (#4523), and one pool over every band above the line
  mixed a rich top with a sparse tail, so that auditing the top lifted the
  tail and leaving it unaudited sank the top.
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
than its target or at its pick budget, either only once every band above
the line has had a round (#4539, #4560), or when its bands are exhausted; the
misses phase at its pick budget or when its bands are exhausted, and,
without a class model, on its width target or a dry run.  A line that keeps
fewer items than one round is ``nothing``: nothing to test.  The targets and
budgets are parameters whose defaults are the values #4523 priced
(``docs/experiments/2026-10-05-line-test-4523/REPORT.md``).

**Dealing ahead** (:meth:`LineTest.deal_ahead`, #5116) is how the app keeps
a round boundary from stalling: once a round has one pick left, both rounds
its vote could deal are drawn now, one per outcome, each on a copy of the
test at the generator's current state.  :meth:`LineTest.draw` then installs
the one for the label actually cast, which is the round it would have drawn
itself, pick for pick, so the band design and a seeded run's picks are as
they were; the other is discarded unseen.

Test votes never train the detector: they are recorded with
:data:`TEST_PROVENANCE` so a later merge can tell them from a check's.

**The vocabulary** - the constants, :class:`LineBudgets`, the bands, the
estimates, :class:`PhaseReport` and :class:`RoundAhead` - lives in
:mod:`~vtscore.training.thresholds.line_test_types`, which never reads the
sample; every name there is re-exported here.  The pool's knobs
(:data:`POOLED_WEIGHT`, :data:`POOL_RADIUS`, :data:`FIRST_PASS_BEFORE_BUDGET`
and :func:`pooled_weight`) stay here, where :class:`LineTest` and
:func:`line_phase` look them up, so a study that rebinds one
(``scripts/experiments/calibration/price_pooled_taper_4560.py``) patches
this module.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vtscore.training.thresholds.line_test_types import (
    _EPS,
    ABOVE,
    BELOW,
    DEFAULT_BUDGETS,
    FOUND_WORDS,
    JEFFREYS,
    MODEL_FLOOR,
    PHASE_DONE,
    PHASE_MATCHES,
    PHASE_MISSES,
    PHASE_NOTHING,
    PHASE_SCORE,
    PHASES,
    SIDES,
    STOP_BUDGET,
    STOP_DRY_RUN,
    STOP_EXHAUSTED,
    STOP_REASONS,
    STOP_WIDTH,
    TEST_ALPHA,
    TEST_DRAW_SEED,
    TEST_DRAWS,
    TEST_PROVENANCE,
    Band,
    EdgeEstimate,
    Estimate,
    LineBudgets,
    LineEstimates,
    PhaseReport,
    RoundAhead,
    _estimate,
    found_words,
    line_bands,
)
from vtscore.training.thresholds.check_schedule import (
    BETA_MAX,
    BETA_MIN,
    CHECK_MIN_PICKS,
)
from vtscore.training.thresholds.ranges import LikelyRange, likely_range

__all__ = [
    "ABOVE",
    "BELOW",
    "DEFAULT_BUDGETS",
    "FIRST_PASS_BEFORE_BUDGET",
    "FOUND_WORDS",
    "JEFFREYS",
    "MODEL_FLOOR",
    "PHASES",
    "PHASE_DONE",
    "PHASE_MATCHES",
    "PHASE_MISSES",
    "PHASE_NOTHING",
    "PHASE_SCORE",
    "POOLED_WEIGHT",
    "POOL_RADIUS",
    "SIDES",
    "STOP_BUDGET",
    "STOP_DRY_RUN",
    "STOP_EXHAUSTED",
    "STOP_REASONS",
    "STOP_WIDTH",
    "TEST_ALPHA",
    "TEST_DRAWS",
    "TEST_DRAW_SEED",
    "TEST_PROVENANCE",
    "Band",
    "EdgeEstimate",
    "Estimate",
    "LineBudgets",
    "LineEstimates",
    "LineTest",
    "PhaseReport",
    "RoundAhead",
    "found_words",
    "line_bands",
    "line_phase",
    "pooled_weight",
]

# ------------------------------------------------------------------ the pool


#: What the pooled share above the line is worth to one band, in picks (#4539).
#: Each band's prior is drawn from the pooled posterior of every pick above the
#: line at this weight, so one round of a band's own picks weighs as much as the
#: pool and a rich band still reads richer; but empty bands no longer read as 8%
#: each (the Jeffreys mean of 0 of 5) summed over a thousand items, which is how
#: a sparse line's precision range came to sit above its truth (#4523).
POOLED_WEIGHT = float(CHECK_MIN_PICKS)


#: How far the pool above the line reaches, in bands (#4560): ``None`` pools
#: every band above the line (#4539); ``r`` pools only the bands within *r*
#: of the band being drawn (itself excluded), falling back to the whole side
#: when none of them has a pick.  One rate pooled across bands from 80% right
#: at the top to 1% near a sparse line lifts the deep bands once the top ones
#: are audited, and drags the top ones down when they are not.  1, with
#: :data:`FIRST_PASS_BEFORE_BUDGET`, is what #4560 priced
#: (``docs/experiments/2026-10-06-pooled-taper-4560/REPORT.md``).
POOL_RADIUS: int | None = 1

#: Whether the matches phase's budget waits until every band above the line
#: has had a round (#4560).  Without it the top bands of a big line are often
#: never audited, and a preset inside them reads at the pool's mean.
FIRST_PASS_BEFORE_BUDGET: bool = True


def pooled_weight(depth: int, n_above: int) -> float:
    """What the pooled share is worth, in picks, to the band *depth* bands up from the line (0 holds the line).

    *n_above* is how many bands sit above the line.  The one place a band's
    pull toward the pool is decided, so a study can price another taper by
    replacing it (#4560).
    """
    del depth, n_above
    return POOLED_WEIGHT


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
    :meth:`phase` read the state.  :meth:`deal_ahead` draws the next round
    early, for both outcomes of a round's last pick, for :meth:`draw` to
    install.
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
    #: What :meth:`deal_ahead` dealt: ``[RoundAhead, version, generator state,
    #: {label: the copy that drew it}]``, or empty.
    _ahead: list[Any] = field(default_factory=list, repr=False, compare=False)

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

    def _pooled_counts(
        self, extra: dict[int, tuple[int, int]] | None = None, near: int | None = None
    ) -> tuple[int, int]:
        """``(picks, right)`` over the bands above the line that pool for band *near*, *extra*'s picks included.

        Every band above the line when *near* is ``None`` or :data:`POOL_RADIUS`
        is; else the bands within the radius of *near*, *near* itself
        excluded, or every band when none of those has a pick.
        """
        pool = list(self.above)
        if near is not None and POOL_RADIUS is not None:
            local = [b for b in pool if b.index != near and abs(b.index - near) <= POOL_RADIUS]
            if self._pick_total(local, extra) > 0:
                pool = local
        labelled = right = 0
        for band in pool:
            _, n_lab, n_right = self.band_counts(band.index)
            if extra and band.index in extra:
                n_lab += extra[band.index][0]
                n_right += extra[band.index][1]
            labelled += n_lab
            right += n_right
        return labelled, right

    def _pick_total(self, bands: Sequence[Band], extra: dict[int, tuple[int, int]] | None) -> int:
        return sum(
            self.band_counts(b.index)[1] + (extra[b.index][0] if extra and b.index in extra else 0) for b in bands
        )

    def _pooled_weight(self, b: int) -> float:
        """Band *b*'s pull toward the pool (:func:`pooled_weight`), counted in bands up from the line."""
        above = self.above
        return pooled_weight(len(above) - 1 - b, len(above))

    def _pooled_share(
        self,
        rng: np.random.Generator,
        n: int,
        extra: dict[int, tuple[int, int]] | None = None,
        near: int | None = None,
    ) -> np.ndarray:
        """*n* draws of the pooled share right above the line: Jeffreys' prior on the pool's picks."""
        labelled, right = self._pooled_counts(extra, near)
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
                pooled = self._pooled_share(rng, n, None if extra is None else {b: extra}, near=b)
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
        # One pooled draw shared by every band keeps the draws joint; a local
        # pool (POOL_RADIUS) is drawn per band instead.
        pooled = self._pooled_share(rng, n, extra) if POOL_RADIUS is None else None
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
        pooled_lab, pooled_right = self._pooled_counts(near=b)
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
        still pending.  When :meth:`deal_ahead` already drew this round for
        the labels as they stand, it is installed rather than drawn again:
        the same picks from the same generator state, so the same round.
        """
        if self.pending:
            return self.pending
        ahead = self._take_ahead()
        if ahead is not None:
            self._install(ahead)
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
        self._ahead.clear()
        self.pending = tuple(i for i in self.band_ids(self.band) if i in self.pick_band and i not in self.labels)

    # ---- dealing ahead

    def deal_ahead(self) -> RoundAhead | None:
        """Draw the next round now, for both ways the round's one pending pick can be voted (#5116).

        Without it the next round is drawn only once the last vote lands,
        because :meth:`next_band` reads the estimates and they depend on
        that label, so a client meets every round boundary with nothing to
        fetch ahead.  Each outcome is dealt on a copy of the test with that
        label recorded and the generator copied at its current state, by
        :meth:`record` and :meth:`draw` themselves; :meth:`draw` installs
        the copy whose labels match the ones actually recorded, so the round
        used is the uniform draw from the band the rule names for the
        realized labels, the very picks the generator would have dealt, and
        the other is thrown away.

        ``None``, holding nothing, unless exactly one pick is pending and the
        test is running.  Computed once per state; :attr:`ahead` reads it
        back, and an :meth:`unrecord` drops it.
        """
        if len(self.pending) != 1 or self.phase().done:
            self._ahead.clear()
            return None
        held = self.ahead
        if held is not None:
            return held
        (cid,) = self.pending
        state = self._generator.bit_generator.state
        forks: dict[bool, LineTest] = {}
        for right in (True, False):
            fork = self._fork()
            fork.record({cid: right})
            fork.draw()
            forks[right] = fork
        ahead = RoundAhead(cid, forks[True].pending, forks[False].pending)
        self._ahead[:] = [ahead, self._version, state, forks]
        return ahead

    @property
    def ahead(self) -> RoundAhead | None:
        """What :meth:`deal_ahead` dealt, while its pick is still the one pending and no label has changed; else ``None``."""
        if not self._ahead:
            return None
        ahead, version, _, _ = self._ahead
        if version != self._version or self.pending != (ahead.id,):
            return None
        return ahead

    def _fork(self) -> "LineTest":
        """A copy to deal ahead on: its own labels, picks and caches, and the generator copied at its state."""
        return dataclasses.replace(
            self,
            labels=dict(self.labels),
            pick_band=dict(self.pick_band),
            _generator=copy.deepcopy(self._generator),
            _cache=[],
            _counts=[],
            _ahead=[],
        )

    def _take_ahead(self) -> "LineTest | None":
        """The copy :meth:`deal_ahead` dealt for the labels as recorded, if the generator has not moved since; spent either way."""
        if not self._ahead:
            return None
        ahead, _, state, forks = self._ahead
        self._ahead.clear()
        if ahead.id not in self.labels or self._generator.bit_generator.state != state:
            return None
        fork = forks[self.labels[ahead.id]]
        return fork if fork.labels == self.labels else None

    def _install(self, fork: "LineTest") -> None:
        """Take *fork*'s round as this one's, with the generator it drew from and the draws it read the state on."""
        self.pending, self.band = fork.pending, fork.band
        if fork.band is not None:
            for cid in fork.pending:
                self.pick_band[cid] = fork.band
        self._generator = fork._generator
        # Its labels are these, so its draws (keyed on its own version) are this state's.
        if fork._counts and fork._counts[0] == fork._version:
            self._counts[:] = [self._version, fork._counts[1]]
        if fork._cache and fork._cache[0][0] == fork._version:
            self._cache[:] = [(self._version, fork._cache[0][1]), fork._cache[1]]

    # ---- the wire shape

    def as_dict(self) -> dict[str, Any]:
        """The test as a client sees it: the phase, the round, the bands and their picks, the estimates, and the round dealt ahead."""
        report = self.phase()
        ahead = self.ahead
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
            "ahead": None if ahead is None else ahead.as_dict(),
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
    # Every band above the line has had a round: before that, a band no pick
    # has seen is read at the pool's mean, the line's average, and the
    # presets Lean the Threshold offers inside it read low (#4539, #4560).
    first_pass = all(test.audited(b.index) or test.exhausted(b.index) for b in test.above)
    budget_open = first_pass or not FIRST_PASS_BEFORE_BUDGET
    if all(test.exhausted(b.index) for b in test.above):
        matches_stop = STOP_EXHAUSTED
    elif first_pass and est.precision.width <= budgets.matches_width + _EPS:
        matches_stop = STOP_WIDTH
    elif budget_open and picks_above >= budgets.matches_picks:
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
