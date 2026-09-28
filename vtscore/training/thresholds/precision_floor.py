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
   on those votes (logistic by default).  A posterior fitted on a biased sample
   is still unbiased **if the sample was chosen by the score it is fitted on** -
   the learned model's own sort, which is how Autopilot's learned phases pick;
   only a class-conditional route would need the prior.  **Votes chosen by any
   other ranker break this.**  The text-sort opening picks by the typed query's
   score, which carries label information the model's score lacks, so its
   positives are the easy ones and the fitted curve comes out optimistic.  #4222
   measured it: with the opening's Good round raised to 20, the gate below opened
   for 60% of COCO Better cells by vote 50, and 51-71% of the promises it then
   made broke (6% on today's 3-Good opening).  So the fold orderings a caller
   passes should hold only votes the learned model's sort surfaced; today's
   short opening is small enough to get away with it, a longer one is not
   (#4245).
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
   on COCO Better's default 0.44% pool), so it makes none.  The gate counts
   evidence; it cannot tell whether that evidence was drawn fairly (item 2).

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

Pure numpy + scikit-learn; nothing here reads a detector context.  Wiring it to
a live detector (which fold orderings, which haystacks) is the caller's job.
"""

from __future__ import annotations

import enum
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

#: Positives among the calibration folds' held-out votes below which no promise
#: is made.  #4220: gated here, the fold-rank lower bound with EM breaks 6% of
#: its X = 50% promises on COCO Better's default 0.44% pool, 1% on a 5% pool and
#: 3% under label shift; a gate of 5 still breaks 18% on the 0.44% pool.
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
    three :class:`PrecisionFloorStatus` states.
    """
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"precision floor must be in (0, 1], got {floor!r}")
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
        return PrecisionFloorCut(PrecisionFloorStatus.INSUFFICIENT_EVIDENCE, floor, None, 0, None, n_pos)
    scores, bound = curve
    ok = np.flatnonzero(bound >= floor)
    if ok.size == 0:
        return PrecisionFloorCut(PrecisionFloorStatus.UNREACHABLE, floor, None, 0, float(bound.max()), n_pos)
    i = int(ok.max())
    threshold = float(scores[i])
    return PrecisionFloorCut(
        PrecisionFloorStatus.PROMISED,
        floor,
        threshold,
        int(np.count_nonzero(scores >= threshold)),
        float(bound[i]),
        n_pos,
    )
