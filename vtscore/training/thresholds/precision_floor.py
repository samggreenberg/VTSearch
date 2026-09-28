"""Precision-floor cut: return as much as possible while at least *X* of it is right.

The owner's ruling on #4223 replaces the Inclusion knob's objective with one a
person can state: *"I'm willing to look at X%-positive returns"*.  The cut then
returns the largest set whose **estimated** precision is at least *X* (#4224).
Nothing measures precision above a cut live - there are no labels there - so
the whole problem is the estimate, and this module is the one #4220 measured to
keep its promise (``docs/experiments/2026-09-28-precision-frames-4220/REPORT.md``):

1. **Fold-rank evidence.**  Each calibration fold's *held-out* vote scores are
   honest (that fold's model never trained on them) but live on that fold
   model's scale.  Each is mapped to its percentile in its own fold's haystack,
   which puts every fold on one coordinate - the same rank transfer the shipped
   fused cut uses.
2. **A posterior in that coordinate.**  ``P(positive | percentile)`` is fitted
   on those votes (logistic by default).  Autopilot picks items by score, never
   by label, so a posterior fitted on score-picked votes is unbiased despite the
   biased sample; only a class-conditional route would need the prior.
3. **Applied to the corpus** through the final model's pool percentiles.  The
   estimated precision of the top *k* is the running mean of the posterior over
   the corpus sorted by score.
4. **A lower bound, not a point estimate.**  A cut at the point estimate breaks
   its promise about half the time by construction; the curve is read at a low
   percentile of bootstrap refits of the votes instead.
5. **Label shift.**  When the corpus is not the pool the votes came from, the
   posterior is re-weighted to the corpus's own prior by Saerens EM on its
   unlabelled scores (``em=True``).
6. **A gate.**  Below :data:`MIN_CALIBRATION_POSITIVES` positives among the
   calibration votes the estimate breaks most of its promises (83% at X = 50%
   on COCO Better's natural pool), so it makes none.

That yields the three states #4224's control needs
(:class:`PrecisionFloorStatus`): a promise, "no cut reaches X on this corpus",
and "not enough evidence yet".

**Knobs #4221 is still pricing** are parameters rather than constants baked
into the body, so a closed-loop arm can vary them without a second copy:
the transfer ``coordinate`` (``"tail"``, ``-log(1 - percentile)``, is the lead
for positives that sit above the pool's 99.5th percentile, where a logistic in
the raw percentile cannot resolve the tail), the bound level
(``lower_percentile``), and the refit count.  The defaults are what #4220
measured.

The curve is fitted once and cut at any floor (:class:`PrecisionFloorCurve`),
the way a :class:`~vtscore.training.thresholds.FoldAnchoredCut` is fitted once
and cut at any inclusion; :class:`PrecisionFloorEstimate` holds one detector's
inputs and fits the curve the first time a floor is asked for.  Which line a
detector draws - the floor's, or the Inclusion knob's when no floor is set, and
the Inclusion 0 cut when the floor promises nothing - is
:func:`reporting_line`, shared by the app and the eval harness's default arm so
the two cannot disagree about it.

Pure numpy + scikit-learn; nothing here reads a detector context.  Wiring it to
a live detector (which fold orderings, which haystacks, which votes may serve
as evidence) is the caller's job: see
:func:`vtscore.detectors.training._fused_threshold`.
"""

from __future__ import annotations

import enum
import threading
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vtscore.training.thresholds.gmm import gmm_fit_array, scored_ordering
from vtscore.utils.scores import scored_only

#: Positives among the calibration folds' held-out votes below which no promise
#: is made.  #4220: gated here, the fold-rank lower bound with EM breaks 6% of
#: its X = 50% promises on the natural pool, 1% on a 5% pool and 3% under label
#: shift; a gate of 5 still breaks 18% on the natural pool.
MIN_CALIBRATION_POSITIVES = 10

#: Bootstrap refits of the votes behind the lower bound (#4220 used 30).
PRECISION_BOOTSTRAP_REFITS = 30

#: Percentile of the bootstrap refits read as the lower bound (#4220 used the
#: 10th).  Lower is safer and more timid; #4221 is sweeping it.
PRECISION_LOWER_PERCENTILE = 10.0

#: Seed of the bootstrap when the caller passes none, so one set of votes always
#: yields one cut - a line that moved between two identical requests would read
#: as a bug.
PRECISION_BOOTSTRAP_SEED = 4220

#: Fewest bootstrap refits that must yield a fit before their percentile is
#: trusted; below it the bound is not formed (resamples that draw one class
#: fit nothing).
MIN_BOOTSTRAP_FITS = 5

#: The transfer coordinates :func:`precision_floor_cut` accepts.
PRECISION_COORDINATES = ("percentile", "tail")

#: The posterior fits :func:`precision_floor_cut` accepts.
PRECISION_FITS = ("logistic", "isotonic")


#: The inclusion a floor's line falls back to when it promises nothing (owner,
#: 2026-09-28; #4247).  A floor that cannot be met never empties the results:
#: the line stays where Inclusion 0 would draw it, labelled as unpromised.
PRECISION_FLOOR_FALLBACK_INCLUSION = 0

#: A 1-D run of scores or labels: a list, or the numpy array a caller already holds.
ScoreArray = Sequence[float] | np.ndarray


class PrecisionFloorStatus(str, enum.Enum):
    """What a precision floor can say about a corpus."""

    #: At least *X* of the returned set is estimated right, at the lower bound.
    PROMISED = "promised"
    #: Enough evidence, but no cut's estimated precision reaches *X*.
    UNREACHABLE = "unreachable"
    #: Too few calibration positives (or none with a Bad beside them) to promise anything.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class PrecisionFloorCut:
    """The outcome of cutting a corpus at precision floor :attr:`floor`.

    :attr:`threshold` is ``None`` unless :attr:`status` is
    :attr:`~PrecisionFloorStatus.PROMISED`; a caller that must still draw a line
    decides what "no cut" means (#4224 leaves that open).  Items scoring
    ``>= threshold`` are the returned set, so :attr:`n_returned` can exceed the
    rank the estimate was read at when scores tie at the boundary.

    :attr:`estimated_precision` is the lower-bound estimate at the cut when a
    promise is made, and the best lower bound any cut reaches when it is not
    (so an unreachable floor can say what *is* reachable); ``None`` when there
    was no estimate at all.
    """

    status: PrecisionFloorStatus
    floor: float
    threshold: float | None
    n_returned: int
    estimated_precision: float | None
    calibration_positives: int


#: How a caller whose ``None`` already means "the shipped default" spells *no
#: floor* - the eval harness's arm knob, where ``None`` is the default arm.
NO_PRECISION_FLOOR = "off"


def resolve_min_precision(min_precision: float | str | None = None) -> float | None:
    """The floor an eval arm cuts at: ``None`` is the app's default, ``"off"`` is no floor.

    The three-state contract :func:`~vtscore.training.thresholds.resolve_exclusion_floor`
    follows.  ``None`` - what the harness's default arm passes - resolves to
    :data:`~vtscore.training.thresholds.DEFAULT_MIN_PRECISION`, the value an
    unset user setting resolves to in the app, so the default arm cuts where a
    live detector does.  :data:`NO_PRECISION_FLOOR` is the Inclusion-knob arm
    (``min_precision=None`` in the app's own vocabulary), and a number pins a
    floor, validated to ``(0, 1]``.
    """
    if min_precision is None:
        from vtscore.config.runtime import DEFAULT_MIN_PRECISION  # noqa: PLC0415

        return DEFAULT_MIN_PRECISION
    if min_precision == NO_PRECISION_FLOOR:
        return None
    if isinstance(min_precision, str) or isinstance(min_precision, bool):
        raise ValueError(
            f"min_precision must be a number in (0, 1], None or {NO_PRECISION_FLOOR!r}; got {min_precision!r}"
        )
    floor = float(min_precision)
    _check_floor(floor)
    return floor


def _check_floor(floor: float) -> None:
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"precision floor must be in (0, 1], got {floor!r}")


def percentile_in(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fraction of *ref_sorted* strictly below each ``x``: the rank transfer's coordinate."""
    return np.searchsorted(ref_sorted, x, side="left") / max(len(ref_sorted), 1)


def _to_coordinate(pct: np.ndarray, coordinate: str, n_ref: int) -> np.ndarray:
    if coordinate == "percentile":
        return pct
    # "tail": -log(1 - p), with p capped one reference step below 1 so an item
    # above the whole reference sample stays finite.  Stretches the top of the
    # ranking, where a positive at the 99.9th percentile and one at the 99.5th
    # are otherwise 0.004 apart.
    cap = 1.0 - 1.0 / (n_ref + 1)
    return -np.log1p(-np.minimum(pct, cap))


def fold_rank_evidence(
    fold_orderings: Sequence[tuple[ScoreArray, ScoreArray]],
    fold_haystacks: Sequence[ScoreArray],
) -> tuple[np.ndarray, np.ndarray]:
    """``(percentiles, labels)`` of every fold's held-out votes, each ranked in its own fold's haystack.

    *fold_orderings[k]* is fold *k*'s held-out ``(scores, labels)`` - the
    orderings the calibration folds already cache - and *fold_haystacks[k]* is
    that fold model's scores over the pool.  The two must be aligned fold for
    fold.  A fold with an empty haystack contributes nothing.
    """
    if len(fold_orderings) != len(fold_haystacks):
        raise ValueError(f"{len(fold_orderings)} fold orderings but {len(fold_haystacks)} fold haystacks")
    coords: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for (scores, labs), hay in zip(fold_orderings, fold_haystacks, strict=True):
        hay_sorted = np.sort(np.asarray(hay, dtype=np.float64))
        if hay_sorted.size == 0 or len(scores) == 0:
            continue
        coords.append(percentile_in(hay_sorted, np.asarray(scores, dtype=np.float64)))
        labels.append(np.asarray(labs, dtype=np.float64))
    if not coords:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(coords), np.concatenate(labels)


def fit_posterior(x: np.ndarray, y: np.ndarray, fit: str = "logistic") -> Callable[[np.ndarray], np.ndarray] | None:
    """``P(y = 1 | x)`` as a callable, or ``None`` when the labels carry no contrast.

    ``"logistic"`` is fitted on the standardized coordinate and effectively
    unpenalized (``C = 1e4``), as #4220 fitted it; ``"isotonic"`` is the
    monotone step fit, clipped to ``[0, 1]`` outside the data.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or y.min() == y.max():
        return None
    if fit == "logistic":
        from sklearn.exceptions import ConvergenceWarning  # noqa: PLC0415
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        mu, sd = float(x.mean()), float(x.std()) or 1.0
        with warnings.catch_warnings():
            # Near-separable votes are the *good* case here; the fit still
            # returns a usable (steep) posterior when lbfgs stops on its cap.
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = LogisticRegression(C=1e4, max_iter=2000).fit(((x - mu) / sd)[:, None], y)
        return lambda v: model.predict_proba(((np.asarray(v, dtype=np.float64) - mu) / sd)[:, None])[:, 1]
    if fit == "isotonic":
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)
        return lambda v: iso.predict(np.asarray(v, dtype=np.float64))
    raise ValueError(f"unknown posterior fit {fit!r}; expected one of {PRECISION_FITS}")


def em_prior_shift(posterior: np.ndarray, prior_fitted: float, iters: int = 200, tol: float = 1e-7) -> np.ndarray:
    """Re-weight *posterior* to the prior of the corpus it was evaluated on (Saerens-Latinne-Decaestecker EM).

    *prior_fitted* is the prior the posterior was fitted under: the voted pool's,
    read as the mean posterior over that pool.  Needs no labels.  On the pool
    itself the fixed point is immediate and the posterior comes back unchanged.
    """
    pi_fit = float(np.clip(prior_fitted, 1e-6, 1 - 1e-6))
    p = np.clip(np.asarray(posterior, dtype=np.float64), 1e-9, 1 - 1e-9)
    pi = pi_fit
    for _ in range(iters):
        a = (pi / pi_fit) * p
        b = ((1 - pi) / (1 - pi_fit)) * (1 - p)
        new = float((a / (a + b)).mean())
        if abs(new - pi) < tol:
            break
        pi = new
    a = (pi / pi_fit) * p
    b = ((1 - pi) / (1 - pi_fit)) * (1 - p)
    return a / (a + b)


def _check_options(fit: str, coordinate: str) -> None:
    if coordinate not in PRECISION_COORDINATES:
        raise ValueError(f"unknown coordinate {coordinate!r}; expected one of {PRECISION_COORDINATES}")
    if fit not in PRECISION_FITS:
        raise ValueError(f"unknown posterior fit {fit!r}; expected one of {PRECISION_FITS}")


def precision_lower_bound_curve(
    corpus_scores: ScoreArray,
    pool_scores: ScoreArray,
    fold_orderings: Sequence[tuple[ScoreArray, ScoreArray]],
    fold_haystacks: Sequence[ScoreArray],
    *,
    fit: str = "logistic",
    coordinate: str = "percentile",
    em: bool = True,
    lower_percentile: float = PRECISION_LOWER_PERCENTILE,
    n_boot: int = PRECISION_BOOTSTRAP_REFITS,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(scores, bound)``: the corpus sorted by score, and the lower-bound precision of each top *k*.

    ``bound[i]`` is the estimated precision of the ``i + 1`` highest-scoring
    corpus items, read at *lower_percentile* of *n_boot* bootstrap refits of the
    fold-rank evidence (module docstring).  This is the whole curve a
    precision-floor cut reads one point off, and what a chart of "how right
    would the top *k* be" plots.  Ungated: :func:`precision_floor_cut` applies
    :data:`MIN_CALIBRATION_POSITIVES`.  ``None`` when no bound can be formed -
    no evidence, an empty corpus or pool, or fewer than
    :data:`MIN_BOOTSTRAP_FITS` resamples that held both classes.

    *corpus_scores* are the final model's scores over the items the cut decides;
    *pool_scores* its scores over the pool the votes were drawn from (the same
    array when the corpus *is* that pool).  *fold_orderings* and
    *fold_haystacks* are the calibration folds' held-out ``(scores, labels)``
    and those fold models' scores over the pool, aligned fold for fold (see
    :func:`fold_rank_evidence`).

    The cost is *n_boot* posterior fits on the votes (tens of points) and as
    many passes over the corpus, plus an EM loop per pass when *em* - linear in
    the corpus, so a caller with a very large corpus should hand it a uniform
    sample and read the threshold off that.
    """
    _check_options(fit, coordinate)
    gen = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(PRECISION_BOOTSTRAP_SEED if rng is None else rng)
    )
    x_pct, y = fold_rank_evidence(fold_orderings, fold_haystacks)
    corpus = np.asarray(corpus_scores, dtype=np.float64)
    pool_sorted = np.sort(np.asarray(pool_scores, dtype=np.float64))
    if y.size < 2 or y.min() == y.max() or corpus.size == 0 or pool_sorted.size == 0:
        return None

    # The tail coordinate's cap needs a reference size; the pool's is the finer
    # one for every item the cut decides.
    x = _to_coordinate(x_pct, coordinate, pool_sorted.size)
    corpus_sorted = np.sort(corpus)[::-1]
    corpus_coord = _to_coordinate(percentile_in(pool_sorted, corpus_sorted), coordinate, pool_sorted.size)
    pool_coord = _to_coordinate(percentile_in(pool_sorted, pool_sorted), coordinate, pool_sorted.size)
    k = np.arange(1, corpus_sorted.size + 1, dtype=np.float64)

    boots: list[np.ndarray] = []
    for _ in range(n_boot):
        idx = gen.integers(0, len(x), len(x))
        post = fit_posterior(x[idx], y[idx], fit)
        if post is None:
            continue
        p = post(corpus_coord)
        if em:
            p = em_prior_shift(p, float(np.mean(post(pool_coord))))
        boots.append(np.cumsum(p) / k)
    if len(boots) < MIN_BOOTSTRAP_FITS:
        return None
    return corpus_sorted, np.percentile(np.stack(boots), lower_percentile, axis=0)


@dataclass(frozen=True, eq=False)
class PrecisionFloorCurve:
    """The lower-bound precision of every top *k* of one corpus: fitted once, cut at any floor.

    Fitting it is the whole cost of a precision floor - *n_boot* posterior
    refits and as many passes over the corpus - while :meth:`cut` is one scan of
    :attr:`bound`.  A caller that keeps the curve can therefore move the floor
    without refitting anything, the way an Inclusion slide re-cuts a
    :class:`~vtscore.training.thresholds.FoldAnchoredCut`.

    :attr:`scores` is the corpus sorted descending and :attr:`bound` the
    lower-bound precision of each top ``i + 1``
    (:func:`precision_lower_bound_curve`).  Both are empty when no bound was
    formed - the evidence held fewer than the gate's positives, or too few
    resamples fitted - and every floor then reads
    :attr:`~PrecisionFloorStatus.INSUFFICIENT_EVIDENCE`.
    :attr:`calibration_positives` counts the positives among the evidence
    whether or not the gate opened.
    """

    scores: np.ndarray
    bound: np.ndarray
    calibration_positives: int

    @property
    def formed(self) -> bool:
        """Whether a bound was formed, so a floor can be promised or found unreachable."""
        return self.bound.size > 0

    def cut(self, floor: float) -> PrecisionFloorCut:
        """The largest top *k* whose lower bound clears *floor*, or which state says why none does."""
        _check_floor(floor)
        n_pos = self.calibration_positives
        if not self.formed:
            return PrecisionFloorCut(PrecisionFloorStatus.INSUFFICIENT_EVIDENCE, floor, None, 0, None, n_pos)
        ok = np.flatnonzero(self.bound >= floor)
        if ok.size == 0:
            return PrecisionFloorCut(PrecisionFloorStatus.UNREACHABLE, floor, None, 0, float(self.bound.max()), n_pos)
        i = int(ok.max())
        threshold = float(self.scores[i])
        return PrecisionFloorCut(
            PrecisionFloorStatus.PROMISED,
            floor,
            threshold,
            int(np.count_nonzero(self.scores >= threshold)),
            float(self.bound[i]),
            n_pos,
        )


_EMPTY = np.zeros(0)


def fit_precision_floor_curve(
    corpus_scores: ScoreArray,
    pool_scores: ScoreArray,
    fold_orderings: Sequence[tuple[ScoreArray, ScoreArray]],
    fold_haystacks: Sequence[ScoreArray],
    *,
    fit: str = "logistic",
    coordinate: str = "percentile",
    em: bool = True,
    lower_percentile: float = PRECISION_LOWER_PERCENTILE,
    n_boot: int = PRECISION_BOOTSTRAP_REFITS,
    min_positives: int = MIN_CALIBRATION_POSITIVES,
    rng: np.random.Generator | int | None = None,
) -> PrecisionFloorCurve:
    """Gate on *min_positives* positives among the fold-rank evidence, then fit the lower-bound curve.

    The arguments are :func:`precision_lower_bound_curve`'s, plus the gate.
    Below the gate nothing is fitted - that is the common case today (#4220),
    and it costs one pass over the evidence rather than *n_boot* refits.
    """
    _check_options(fit, coordinate)
    n_pos = int(fold_rank_evidence(fold_orderings, fold_haystacks)[1].sum())
    curve = (
        precision_lower_bound_curve(
            corpus_scores,
            pool_scores,
            fold_orderings,
            fold_haystacks,
            fit=fit,
            coordinate=coordinate,
            em=em,
            lower_percentile=lower_percentile,
            n_boot=n_boot,
            rng=rng,
        )
        if n_pos >= min_positives
        else None
    )
    if curve is None:
        return PrecisionFloorCurve(_EMPTY, _EMPTY, n_pos)
    return PrecisionFloorCurve(curve[0], curve[1], n_pos)


def precision_floor_cut(
    floor: float,
    corpus_scores: ScoreArray,
    pool_scores: ScoreArray,
    fold_orderings: Sequence[tuple[ScoreArray, ScoreArray]],
    fold_haystacks: Sequence[ScoreArray],
    *,
    fit: str = "logistic",
    coordinate: str = "percentile",
    em: bool = True,
    lower_percentile: float = PRECISION_LOWER_PERCENTILE,
    n_boot: int = PRECISION_BOOTSTRAP_REFITS,
    min_positives: int = MIN_CALIBRATION_POSITIVES,
    rng: np.random.Generator | int | None = None,
) -> PrecisionFloorCut:
    """Cut *corpus_scores* where the most items still clear precision *floor*, or say why not.

    Gates on *min_positives* positives among the fold-rank evidence, then cuts
    at the largest top *k* whose :func:`precision_lower_bound_curve` value is at
    least *floor*.  The arguments are that function's; the result is one of the
    three :class:`PrecisionFloorStatus` states.  One call is one
    :func:`fit_precision_floor_curve` and one :meth:`PrecisionFloorCurve.cut`;
    a caller cutting one corpus at several floors should keep the curve.
    """
    _check_floor(floor)
    return fit_precision_floor_curve(
        corpus_scores,
        pool_scores,
        fold_orderings,
        fold_haystacks,
        fit=fit,
        coordinate=coordinate,
        em=em,
        lower_percentile=lower_percentile,
        n_boot=n_boot,
        min_positives=min_positives,
        rng=rng,
    ).cut(floor)


def eligible_fold_orderings(
    fold_orderings: Sequence[tuple[ScoreArray, ScoreArray]],
    holdout_rows: Sequence[Sequence[int]],
    eligible_rows: Sequence[bool] | None,
) -> list[tuple[list[float], list[float]]]:
    """Each fold's held-out ``(scores, labels)``, kept only where the vote behind it may calibrate a promise.

    *holdout_rows* is what the calibration's ``holdout_sink`` received
    (:func:`~vtscore.training.thresholds.compute_fold_orderings`): per fold,
    the training row behind each held-out score.  *eligible_rows* is
    indexed by training row.  The folds keep their order and their count, so the
    result stays aligned with the fold haystacks it will be ranked against.

    ``None`` for *eligible_rows* keeps every held-out vote - the caller has no
    provenance to filter on.  A fold whose held-out rows are missing or do not
    line up with its ordering contributes **nothing**: a vote that cannot be
    traced back cannot be shown to have been drawn fairly, and an empty fold
    only makes the promise more timid.
    """
    if eligible_rows is None:
        return [([float(v) for v in sc], [float(v) for v in lb]) for sc, lb in fold_orderings]
    out: list[tuple[list[float], list[float]]] = []
    for k, (sc, lb) in enumerate(fold_orderings):
        rows = holdout_rows[k] if k < len(holdout_rows) else None
        if rows is None or len(rows) != len(sc) or len(rows) != len(lb):
            out.append(([], []))
            continue
        keep = [j for j, r in enumerate(rows) if 0 <= r < len(eligible_rows) and eligible_rows[r]]
        out.append(([float(sc[j]) for j in keep], [float(lb[j]) for j in keep]))
    return out


class PrecisionFloorEstimate:
    """One detector's precision-floor inputs, with the curve fitted the first time a floor is asked for.

    Built beside the fold-anchored cut on every retrain, from the same
    populations: *corpus_scores* are the final model's scores over what the cut
    decides, *fold_orderings* the calibration folds' held-out votes (already cut
    down to the ones that may serve as evidence, :func:`eligible_fold_orderings`)
    and *fold_haystacks* those fold models' scores over the pool.  *pool_scores*
    is the reference the corpus is ranked against, and defaults to the corpus.
    The app passes the final model's scores over the whole haystack, voted
    items included, while its corpus is the unvoted remainder: that is the
    configuration #4220 measured, and #4221 found the estimator unsafe without
    it (the voted positives at the top of the reference are a conservative
    offset the estimate depends on).

    Nothing is fitted in the constructor.  Most detectors never reach the gate,
    and one whose owner cleared the floor never needs the curve, so the
    *n_boot* refits are paid by :meth:`curve` - once, under a lock, and then
    kept.  Moving the floor after that is :meth:`PrecisionFloorCurve.cut`.

    Every score array goes through :func:`~vtscore.utils.scores.scored_only`
    (and each held-out vote through
    :func:`~vtscore.training.thresholds.scored_ordering`), so an item the head
    could not score is absent rather than an observation a unit below the
    sigmoid range.  Above :data:`~vtscore.training.thresholds._GMM_MAX_SAMPLES`
    scores the curve is read off the same seeded uniform sample the
    fold-anchored cut fits on (:func:`~vtscore.training.thresholds.gmm_fit_array`):
    the fit is linear in the corpus per refit.  :meth:`cut` and
    :meth:`count_at` count against the whole corpus, so ``n_returned`` is a
    real count however large the dataset.

    The estimator's knobs (#4221's coordinate and bound level among them) are
    constructor arguments with :func:`precision_floor_cut`'s defaults.
    """

    def __init__(
        self,
        corpus_scores: ScoreArray,
        fold_orderings: Sequence[tuple[ScoreArray, ScoreArray]],
        fold_haystacks: Sequence[ScoreArray],
        *,
        pool_scores: ScoreArray | None = None,
        **knobs: Any,
    ) -> None:
        self._corpus_full = scored_only(np.asarray(corpus_scores, dtype=np.float64))
        self._corpus = gmm_fit_array(self._corpus_full)
        self._pool = (
            self._corpus
            if pool_scores is None
            else gmm_fit_array(scored_only(np.asarray(pool_scores, dtype=np.float64)))
        )
        self._fold_orderings = [scored_ordering((list(sc), list(lb))) for sc, lb in fold_orderings]
        self._fold_haystacks = [gmm_fit_array(scored_only(np.asarray(hay, dtype=np.float64))) for hay in fold_haystacks]
        if len(self._fold_orderings) != len(self._fold_haystacks):
            raise ValueError(
                f"{len(self._fold_orderings)} fold orderings but {len(self._fold_haystacks)} fold haystacks"
            )
        self._knobs = knobs
        _check_options(knobs.get("fit", "logistic"), knobs.get("coordinate", "percentile"))
        self._curve: PrecisionFloorCurve | None = None
        self._lock = threading.Lock()

    @property
    def calibration_positives(self) -> int:
        """Positives among the evidence, counted as the gate counts them (no fit needed)."""
        if self._curve is not None:
            return self._curve.calibration_positives
        return int(fold_rank_evidence(self._fold_orderings, self._fold_haystacks)[1].sum())

    @property
    def corpus_size(self) -> int:
        """How many scored items the cut decides."""
        return int(self._corpus_full.size)

    def curve(self) -> PrecisionFloorCurve:
        """The fitted curve, fitted on the first call and kept."""
        with self._lock:
            if self._curve is None:
                self._curve = fit_precision_floor_curve(
                    self._corpus, self._pool, self._fold_orderings, self._fold_haystacks, **self._knobs
                )
            return self._curve

    def count_at(self, threshold: float) -> int:
        """How many corpus items score at or above *threshold*: what a line there returns."""
        return int(np.count_nonzero(self._corpus_full >= threshold))

    def cut(self, floor: float) -> PrecisionFloorCut:
        """:meth:`PrecisionFloorCurve.cut` on :meth:`curve`, with ``n_returned`` counted on the whole corpus."""
        verdict = self.curve().cut(floor)
        if verdict.threshold is None:
            return verdict
        return PrecisionFloorCut(
            verdict.status,
            verdict.floor,
            verdict.threshold,
            self.count_at(verdict.threshold),
            verdict.estimated_precision,
            verdict.calibration_positives,
        )


@dataclass(frozen=True)
class ReportingLine:
    """Where a detector's reporting cut sits under its operating point, and why.

    :attr:`threshold` is ``None`` when there is no fold-anchored cut to read a
    line off (no usable folds, or a degenerate fit) - the caller keeps its own
    inclusion-blind fallback there, as it always has.  :attr:`inclusion` is the
    inclusion the line sits at, which Autopilot's acquisition offsets from:
    the knob's value when Inclusion governs, the fallback's when a floor
    promises nothing, and ``None`` when a promised floor put the line somewhere
    no inclusion named - :func:`line_inclusion` derives it from the line then.
    :attr:`floor` is the floor's verdict, ``None`` when no floor is set.
    """

    threshold: float | None
    inclusion: float | None
    floor: PrecisionFloorCut | None


def unpromised(floor: float, calibration_positives: int = 0) -> PrecisionFloorCut:
    """The verdict of a floor with no estimate behind it: not enough evidence yet."""
    return PrecisionFloorCut(PrecisionFloorStatus.INSUFFICIENT_EVIDENCE, floor, None, 0, None, calibration_positives)


def reporting_line(
    cut: Any,
    estimate: PrecisionFloorEstimate | None,
    *,
    inclusion_value: float,
    min_precision: float | None,
) -> ReportingLine:
    """The reporting cut at an operating point: a precision floor, or an inclusion when no floor is set.

    **The one definition of which line a detector draws**, called by the app's
    retrain (:func:`vtscore.detectors.training._fused_threshold`), its no-refit
    re-cut (:func:`vtscore.state.core.recut_detector_threshold`) and the eval
    harness's default arm, so the three cannot drift apart.

    * *min_precision* ``None``: the Inclusion knob governs, and the line is
      ``cut.threshold_at(inclusion_value)`` - exactly the pre-floor behaviour.
    * A floor that is **promised**: the floor's own threshold.  *inclusion_value*
      is ignored - a set floor wins over the knob (owner, 2026-09-28).
    * A floor that promises nothing (``unreachable`` or
      ``insufficient_evidence``, or no *estimate* at all): the line falls back
      to :data:`PRECISION_FLOOR_FALLBACK_INCLUSION` and the verdict says why
      (#4247).

    *cut* is the fitted :class:`~vtscore.training.thresholds.FoldAnchoredCut`,
    or ``None`` when training fitted none.
    """

    def _at(inclusion: float) -> float | None:
        if cut is None:
            return None
        value = float(cut.threshold_at(inclusion))
        return value if np.isfinite(value) else None

    if min_precision is None:
        return ReportingLine(_at(inclusion_value), float(inclusion_value), None)
    verdict = estimate.cut(min_precision) if estimate is not None else unpromised(min_precision)
    if verdict.status is PrecisionFloorStatus.PROMISED and verdict.threshold is not None:
        return ReportingLine(verdict.threshold, None, verdict)
    fallback = float(PRECISION_FLOOR_FALLBACK_INCLUSION)
    return ReportingLine(_at(fallback), fallback, verdict)


def line_inclusion(line: ReportingLine, cut: Any) -> float | None:
    """The inclusion *line* sits at: the one it was drawn at, or the strictest that reproduces it.

    Autopilot's acquisition cut sits a fixed number of inclusion steps stricter
    than the line (:func:`~vtscore.training.thresholds.acquisition_inclusion`),
    and an offset needs an origin.  A line a promised floor drew was not set in
    inclusion units, so its origin is recovered through
    :meth:`~vtscore.training.thresholds.FoldAnchoredCut.inclusion_for_threshold`.
    ``None`` when there is nothing to recover it from.
    """
    if line.inclusion is not None:
        return line.inclusion
    if cut is None or line.threshold is None:
        return None
    return cut.inclusion_for_threshold(line.threshold)
