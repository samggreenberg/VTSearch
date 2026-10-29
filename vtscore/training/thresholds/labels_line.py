"""The line drawn from the labelset alone (#4452).

The owner's ruling (2026-10-02): an exported labelset must be enough to run
Find later, on another corpus and possibly another embedder, and Find on a new
corpus must return what training on the labels implies - never a count
re-drawn on the corpus it scores (the old line kept the top 26-128 of a
200-image corpus with no positives in it).  So the line under a balance is
derived from two things only:

* **What the labels say about the head's scores.**  The calibration folds hold
  every vote out once and score it with a head that never saw it
  (:class:`~vtscore.training.thresholds.conformal.CalibrationFolds`).  Their
  held-out scores of the Goods and of the Bads are modelled as two normals of
  one spread on the logit scale (:class:`ClassScoreModel`) - one spread, so
  the posterior is monotone in the score and the line is a single cut.  Any
  embedder re-derives this from the same labels; the same embedder and labels
  re-derive the same model, since every fit is seeded.
* **How common the target is in the corpus being decided, and what its
  negatives look like.**  Estimated on that corpus (:func:`fit_corpus`): its
  unvoted scores are a mixture of the labels' Good component (fixed) and a
  normal for the corpus's own bulk of negatives, fitted with their share; its
  Good votes count as they are.  Train estimates it
  on the dataset it trains on and Find on the dataset it searches; the owner's
  working assumption (2026-10-02) is that a Find corpus has the properties of
  the Train corpus, so the two estimates - and the two lines - agree, and when
  they do not (a Find corpus with no positives) the estimate falls with it.
  Only the labels travel: the labelset is the evidence, never a number
  derived from it, and more labels from any dataset improve the model.

The threshold is the cut at which the expected F-beta of the kept set peaks
(:func:`corpus_cut`): each item's chance of being a positive from a 3-part
fit (the labels' Good and Bad components and the corpus's bulk), the total
positives from the 2-part estimate, the returned size counted on the corpus
itself.  No count is
drawn on any corpus once the class model stands (while it is thin, the head's
count line serves; see below): a corpus with no positives returns at most its own
few high-scoring negatives.

**Known weakness.**  The votes are not a random sample: active learning picks
near the line and from the top, so the Bads' held-out scores sit higher than
the true negatives' and the negative tail is extrapolated by the normal.  The
pricing in ``docs/experiments`` measures what that costs on the objective.

**While the labels are few** (#4786), the class model has almost nothing to fit -
at the hand-over to the trained head a session holds four Bads - and its cut goes
wrong both ways: at beta 4 the line kept a median of 5 of a FHIBE person's faces at
the first trained click, one face in a fifth of sessions and thousands in the
weakly separated ones.  Until the class model holds :data:`HEAD_COUNT_GOODS`
held-out Goods, however many Bads (#4798), the line at every
balance keeps instead what stands out in the head's own scores, counted by the rule
that draws the Goods' centroid's line (:func:`head_count_cut`): the line Test gave
until the hand-over, now drawn on the head.  With two to five Goods the Good
component cannot place the corpus's positives: past 32 Bads the 3-part chances
collapse while the 2-part total stays several times the positives, and a beta-4 cut
ran to the corpus's end on faces (2.5% of clicks, up to 704 kept for 4 withheld)
in a way no fix to the total could tell from a hard target's sound deep cut.  That count is ``beta ** 0.708`` times
what stands out up to a balance of 1, so the line keeps more as the balance leans to recall.  Above 1 it
grows as ``beta ** 0.189`` (#5039): 1.3x what stands out at beta 4, not 2.67x, since a person's few faces
stand above a light-tailed bulk and 2.67x kept two to three other faces each.  A weak head's positives
sit deeper than what stands out, and the labels say when: where a held-out Good ranks below the count
at beta 1, the line keeps at least as deep as the head's excess over a Gaussian bulk is expected to
score best at the balance (:func:`~vtscore.training.thresholds.head_count._excess_depth`, from
:data:`HEAD_COUNT_RECALL_Z` bulk spreads up).
The scores alone cannot tell such a tail from a bulk heavier than a Gaussian, which a fifth to a half
of face sessions have.  Like the centroid's it is a count on the corpus being decided, not a cut the
labels imply; the class model takes over once it has the labels to stand on.  The Goods half
matters for a broad target: with many Goods its labels line is sound, while a count
of what stands out keeps several times the positives (COCO Better's Binary opening
lost F-beta there).

**When the Bads are a random sample** (#4490: a labelset imported from an
exhaustively labelled dataset; every label of the review's ceiling), one normal
fitted to them describes the negatives' bulk, and their heavy upper tail - where
the cut is decided - runs several times the normal's.  The line then read those
negatives as positives and cut deep.  So when there are enough Bads and they are
not over-represented at the top of the corpus (:func:`_bads_are_a_random_sample`),
the negatives are modelled by the Bads' own held-out scores instead
(:func:`_bads_shape_posteriors`).  A session's Bads, picked from the top, never
pass that test, and their line is unchanged.

**Where the pieces live** (#5426).  The line is four modules, none of which
calls one above it: :mod:`~vtscore.training.thresholds.class_model` (the
labels' class model), :mod:`~vtscore.training.thresholds.corpus_model` (the
corpus fits and the cut), :mod:`~vtscore.training.thresholds.head_count` (the
head's count line and Autopilot's cues), and this one, which draws the line
from them (:func:`fit_labels_line`, :class:`LabelsLine`).  Every public name of
the four is importable from here.  A knob is swapped where it is read:
``BAD_COMPONENT`` in ``corpus_model``, the ``HEAD_COUNT_*`` in ``head_count``,
and ``_BOUND_MARGIN`` and ``corpus_sigma_floor`` here, where :func:`_line_on`
reads them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vtscore.training.thresholds.class_model import (
    MIN_LOGIT_SIGMA,
    PREVALENCE_MAX,
    PREVALENCE_MIN,
    RANDOM_BADS_MIN,
    ClassScoreModel,
    _logit,
    _pooled_class_model,
    class_score_model,
)
from vtscore.training.thresholds.corpus_model import (
    BAD_COMPONENT,
    BAD_COMPONENTS,
    FOLD_SCALES,
    RELATIVE_SIGMA_FLOOR,
    CorpusNegatives,
    corpus_cut,
    corpus_expected_fbeta,
    corpus_posteriors,
    corpus_sigma_floor,
    estimate_positives,
    fit_corpus,
    fold_transfer_clipped,
    fold_transfer_shift,
    labels_line_threshold,
    parametric_expected_fbeta,
    target_precision_threshold,
    transfer_fold_orderings,
)
from vtscore.training.thresholds.head_count import (
    HEAD_COUNT_BADS,
    HEAD_COUNT_GOODS,
    HEAD_COUNT_PRECISION_EXPONENT,
    HEAD_COUNT_PRECISION_SCALE,
    HEAD_COUNT_PRECISION_TOP_Z,
    HEAD_COUNT_RECALL_EXPONENT,
    HEAD_COUNT_RECALL_Z,
    STOP_LEVEL,
    WEAK_CHECK_COOLDOWN,
    WEAK_CHECK_MIN_VOTES,
    WEAK_CHECK_OFF_MEDIA_TYPES,
    WEAK_CHECK_REPEAT,
    WEAK_SEPARATION_D,
    done_check_due,
    head_count_cut,
    head_count_serves,
    stop_level_met,
    stop_level_reading,
    weak_check_due,
    weak_check_prompts,
)

#: The top of the corpus the random-sample test reads, as a share of it, and how
#: over-represented the Bads may be there: a random sample puts this share of its
#: Bads there (enrichment 1; the review's ceilings sat at 0.65-1.48), while active
#: learning picks from the top (a session's median was 14x at click 150, and every
#: session that reached 100 Bads sat at 3.4x or more).  The test reads the top on
#: purpose: a median test let a spot check's mix of near-line and uniform Bads
#: through and collapsed those lines to one image.
RANDOM_BADS_TOP_SHARE = 0.05
RANDOM_BADS_MAX_ENRICHMENT = 2.0
#: The narrowest kernel the Bads' density takes, in logit units (identical Bads
#: would otherwise make a zero-width spike), and the points of its binned grid.
_KDE_MIN_BANDWIDTH = 0.01
_KDE_GRID_POINTS = 8001


@dataclass(frozen=True)
class LabelsLine:
    """What a retrain leaves for the line: the labels' class model and the corpus side it was cut on.

    *prevalence* is the positives' share of the whole corpus (its Good votes
    counted); *unvoted_share* their share among its unvoted items, and
    *unvoted_scores* those items' scores, best first - what the counted cut
    (:func:`corpus_cut`) reads.  With no scores (a line built by hand) the cut
    falls back to the parametric one.
    """

    model: ClassScoreModel
    prevalence: float
    negatives: CorpusNegatives | None = None
    unvoted_share: float | None = None
    unvoted_scores: np.ndarray | None = field(default=None, compare=False, repr=False)
    unvoted_posteriors: np.ndarray | None = field(default=None, compare=False, repr=False)
    #: The class model's spread as this corpus floors it (#4492); ``None`` on a line built by hand.
    spread: float | None = None
    #: Whether the negatives were modelled by the Bads' own scores, the labels' Bads
    #: being a random sample of this corpus (#4490), rather than by the corpus fit.
    bads_shape: bool = False
    #: The calibration folds' own model, floored on this corpus, when they ranked the
    #: held-out Goods no higher than the Bads (#4899).  *model* is then the in-sample
    #: stand-in, which draws the line as it always has; :attr:`separation` reads this.
    #: ``None`` when *model* is the folds' own, or the folds held nothing out.
    held_out: ClassScoreModel | None = None

    @property
    def separation(self) -> float:
        """How many spreads apart the labels' Good and Bad scores sit: d' = (Good mean - Bad mean) / spread.

        The spread is the one the line was cut with, the class model's own
        floored on this corpus, so the value is free of the logit scale.  Below
        :data:`WEAK_SEPARATION_D` the labels separate weakly (#4496).

        When the folds ranked the held-out Goods no higher than the Bads, it is
        their gap, at most 0, in their own spread floored on this corpus
        (:attr:`held_out`, #4899).  The in-sample stand-in that draws the line
        then was fitted on the head's scores of the labels it trained on, which
        it always separates: read from it, labels that do not separate at all
        read as separating strongly, and the weak check never came due.
        """
        if self.held_out is not None:
            return (self.held_out.mu_pos - self.held_out.mu_neg) / self.held_out.sigma
        spread = self.model.sigma if self.spread is None else self.spread
        return (self.model.mu_pos - self.model.mu_neg) / spread if spread > 0 else float("inf")

    def threshold(self, beta: float) -> float:
        if self.unvoted_scores is not None and self.unvoted_posteriors is not None and self.unvoted_share is not None:
            if head_count_serves(self.model):
                return head_count_cut(self.unvoted_scores, beta, held_out_goods=self.model.pos_logits)
            total = self.unvoted_share * self.unvoted_scores.size
            return corpus_cut(self.unvoted_posteriors, self.unvoted_scores, total, beta)
        return labels_line_threshold(self.model, self.prevalence, beta, self.negatives)

    def expected_fbeta(self, beta: float) -> float:
        """The expected F-*beta* of the best cut the class model sees on this corpus's unvoted items (#4878).

        What :func:`corpus_cut` maximises, read at its maximum: the class model's
        own estimate of how well the line's set does on the rest of the corpus.
        Read whichever line serves, so it is the class model's estimate even
        while the head's count line is the one drawn (:func:`head_count_serves`).
        A line built by hand, with no corpus counted, reads the parametric one.
        """
        if self.unvoted_scores is not None and self.unvoted_posteriors is not None and self.unvoted_share is not None:
            total = self.unvoted_share * self.unvoted_scores.size
            return corpus_expected_fbeta(self.unvoted_posteriors, total, beta)
        return self.expected_fbeta_fresh(beta)

    def expected_fbeta_fresh(self, beta: float) -> float:
        """The expected F-*beta* the class model gives its line on a fresh corpus like this one (#4878).

        The class model at the spread this corpus floored it to, at this
        corpus's prevalence and with its negatives' bulk
        (:func:`parametric_expected_fbeta`): a "Train on X, Find on Y" reading
        that counts no item, so it does not move as the session's votes take
        positives out of the corpus.
        """
        model = self.model if self.spread is None else replace(self.model, sigma=self.spread)
        return parametric_expected_fbeta(model, self.prevalence, beta, self.negatives)

    def on_corpus(
        self, scores: Any, ids: Iterable[int] | None = None, labels: Mapping[int, bool] | None = None
    ) -> "LabelsLine":
        """The same class model with the corpus side re-fitted on another corpus (a Find pass over a new dataset)."""
        line = _line_on(self.model, scores, ids, labels, held_out=self.held_out)
        return self if line is None else line


#: The smallest share of the labels' Good component a cut must hold for the
#: counted bound on the prevalence to read it (below it, ``R / S1`` is noise).
_BOUND_MIN_S1 = 0.05
#: The counted bound guards against a runaway estimate rather than replacing
#: it: the Goods a session found are its easiest positives, so ``S1`` reads
#: high and the bound reads low (0.24% for a 0.43% target on a clean corpus).
#: The mixture's estimate stands unless it exceeds the bound by this factor: 4 since #5040 (2 before), which reads
#: the prevalence at 0.87 to 0.93 of the truth early rather than half of it, while a corpus with no positives keeps no
#: more than it did (removing the bound kept 84 to 148 wrong images at beta 4 there).  The eval harness can set another
#: (``vtscore.eval.live_threshold_rules.labels_chances``).
_BOUND_MARGIN = 4.0


def _counted_share_bound(model: ClassScoreModel, unvoted: np.ndarray) -> float:
    """An upper bound on the positives' share among the unvoted items, from counts alone (#4452).

    A cut that returns ``R`` items holds at most ``R`` positives, and the
    labels say a share ``S1`` of all positives clear it, so the corpus holds
    at most ``R / S1`` positives - for every cut.  The tightest bound comes
    from cuts high enough that few negatives clear them, whatever the bulk's
    shape: it caps the mixture's estimate where a heavy upper tail of
    negatives would otherwise be read as positives.
    """
    from scipy.stats import norm  # noqa: PLC0415

    n = unvoted.size
    if n == 0:
        return PREVALENCE_MAX
    sd = np.sort(unvoted)[::-1]
    s1 = norm.sf((_logit(sd) - model.mu_pos) / model.sigma)
    ok = s1 >= _BOUND_MIN_S1
    if not ok.any():
        return PREVALENCE_MAX
    r = np.arange(1, n + 1, dtype=np.float64)
    return float(np.min(r[ok] / s1[ok]) / n)


def _bads_are_a_random_sample(model: ClassScoreModel, unvoted_logits: np.ndarray) -> bool:
    """Whether the labels' Bads look like a random sample of this corpus (#4490).

    At least :data:`RANDOM_BADS_MIN` of them, and at most
    :data:`RANDOM_BADS_MAX_ENRICHMENT` times their share in the corpus's top
    :data:`RANDOM_BADS_TOP_SHARE`.  Active learning picks Bads from the top, so a
    session's are many times over-represented there; a random sample is not.
    """
    neg = model.neg_logits
    if neg is None or neg.size < RANDOM_BADS_MIN or unvoted_logits.size == 0:
        return False
    cut = float(np.quantile(unvoted_logits, 1.0 - RANDOM_BADS_TOP_SHARE))
    enrichment = float((neg >= cut).mean()) / RANDOM_BADS_TOP_SHARE
    return enrichment <= RANDOM_BADS_MAX_ENRICHMENT


def _bads_density(neg_logits: np.ndarray):
    """A Gaussian kernel density of the Bads' logit scores (Silverman's bandwidth), binned on a fine grid."""
    from scipy.stats import norm  # noqa: PLC0415

    n = neg_logits.size
    sd = float(np.std(neg_logits, ddof=1)) if n > 1 else MIN_LOGIT_SIGMA
    iqr = float(np.subtract(*np.percentile(neg_logits, [75, 25]))) / 1.349 if n > 3 else sd
    h = max(0.9 * (min(sd, iqr) if iqr > 0 else sd) * n ** (-0.2), _KDE_MIN_BANDWIDTH)
    lo, hi = float(neg_logits[0]) - 12 * h, float(neg_logits[-1]) + 12 * h
    grid = np.linspace(lo, hi, _KDE_GRID_POINTS)
    step = grid[1] - grid[0]
    counts = np.bincount(np.clip(np.rint((neg_logits - lo) / step).astype(int), 0, grid.size - 1), minlength=grid.size)
    # Never wider than the grid: np.convolve's "same" output takes the longer input's length.
    half = min(int(np.ceil(12 * h / step)), (grid.size - 1) // 2)
    dens = np.convolve(counts.astype(np.float64), norm.pdf(np.arange(-half, half + 1) * step / h), mode="same")
    dens /= n * h

    def pdf(x: np.ndarray) -> np.ndarray:
        out = np.interp(x, grid, dens, left=0.0, right=0.0)
        far = (x < lo) | (x > hi)
        if far.any():
            edge = np.where(x[far] < lo, neg_logits[0], neg_logits[-1])
            out[far] = norm.pdf((x[far] - edge) / h) / (n * h)
        return out

    return pdf


def _bads_shape_posteriors(model: ClassScoreModel, unvoted_logits: np.ndarray, *, iterations: int = 500) -> np.ndarray:
    """Each unvoted item's chance of being a positive when the negatives are distributed as the Bads (#4490).

    Positives as the labels' Good component (*model*, floored on this corpus),
    negatives as the Bads' own kernel density; the positives' share fitted by EM
    on the corpus.  In the order of *unvoted_logits*.
    """
    from scipy.stats import norm  # noqa: PLC0415

    assert model.neg_logits is not None
    f1 = norm.pdf(unvoted_logits, model.mu_pos, model.sigma)
    f0 = _bads_density(model.neg_logits)(unvoted_logits)
    pi, r = 0.01, np.zeros_like(unvoted_logits)
    for _ in range(iterations):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nan_to_num(pi * f1 / (pi * f1 + (1.0 - pi) * f0), nan=0.0)
        new = min(max(float(r.mean()), PREVALENCE_MIN), PREVALENCE_MAX)
        done = abs(new - pi) < 1e-10
        pi = new
        if done:
            break
    return r


def _line_on(
    model: ClassScoreModel,
    scores: Any,
    ids: Iterable[int] | None,
    labels: Mapping[int, bool] | None,
    *,
    held_out: ClassScoreModel | None = None,
) -> "LabelsLine | None":
    a = np.asarray(scores, dtype=np.float64)
    keep = np.isfinite(a) & (a >= 0.0) & (a <= 1.0)
    n_items = int(keep.sum())
    if n_items == 0:
        return None
    id_list = list(ids) if ids is not None else list(range(a.size))
    voted = dict(labels or {})
    unvoted = np.array([s for i, s, k in zip(id_list, a, keep) if k and int(i) not in voted], dtype=np.float64)
    n_good = sum(1 for i, k in zip(id_list, keep) if k and voted.get(int(i)) is True)
    # The spread floor this corpus sets (#4492), on the class model and the bulk alike.
    floor = corpus_sigma_floor(unvoted)
    fitted = model.floored(floor)
    positives, negatives = fit_corpus(fitted, unvoted, n_good, floor=floor)
    order = np.argsort(-unvoted, kind="stable")
    bads_shape = _bads_are_a_random_sample(model, _logit(unvoted))
    if bads_shape:
        # #4490: the Bads are a random sample, so their own scores are the negatives'
        # shape - heavy upper tail and all - and the chances sum to the positives.
        post = _bads_shape_posteriors(fitted, _logit(unvoted))
        share = float(post.mean()) if unvoted.size else 0.0
    else:
        share = (positives - n_good) / unvoted.size if unvoted.size else 0.0
        share = min(share, _BOUND_MARGIN * _counted_share_bound(fitted, unvoted))
        post = corpus_posteriors(fitted, unvoted, floor=floor)
    positives = n_good + share * unvoted.size
    prevalence = min(max(positives / n_items, PREVALENCE_MIN), PREVALENCE_MAX)
    # The labels' model is kept unfloored, so a Find on another corpus floors it afresh.
    return LabelsLine(
        model,
        prevalence,
        negatives,
        float(share),
        unvoted[order].copy(),
        post[order].copy(),
        fitted.sigma,
        bads_shape,
        held_out=None if held_out is None else held_out.floored(floor),
    )


def corpus_fit(
    model: ClassScoreModel,
    scores: Any,
    ids: Iterable[int] | None = None,
    labels: Mapping[int, bool] | None = None,
) -> tuple[float, CorpusNegatives] | None:
    """``(prevalence, negatives)`` for a corpus: its Good votes plus the EM positives over its size, and its bulk.

    *labels* are the votes that sit in this corpus (``True`` = Good); a Find
    corpus usually holds none.  ``None`` for a corpus with no scorable item.
    """
    line = _line_on(model, scores, ids, labels)
    return None if line is None or line.negatives is None else (line.prevalence, line.negatives)


def corpus_prevalence(
    model: ClassScoreModel,
    scores: Any,
    ids: Iterable[int] | None = None,
    labels: Mapping[int, bool] | None = None,
    *,
    prior: float | None = None,
) -> float | None:
    """The prevalence :func:`corpus_fit` estimates for a corpus; ``None`` with no scorable item."""
    del prior
    fit = corpus_fit(model, scores, ids, labels)
    return None if fit is None else fit[0]


def _in_sample_ordering(
    scores: Any, ids: Iterable[int] | None, labels: Mapping[int, bool]
) -> tuple[list[float], list[float]]:
    """The head's scores of the labelled items in a corpus, as one ``(scores, labels)`` ordering."""
    a = np.asarray(scores, dtype=np.float64)
    id_list = list(ids) if ids is not None else list(range(a.size))
    out_s: list[float] = []
    out_y: list[float] = []
    for i, s in zip(id_list, a):
        y = labels.get(int(i))
        if y is not None:
            out_s.append(float(s))
            out_y.append(1.0 if y else 0.0)
    return out_s, out_y


def fit_labels_line(
    orderings: Sequence[tuple[Any, Any]] | None,
    corpus_scores: Any,
    corpus_ids: Iterable[int] | None = None,
    labels: Mapping[int, bool] | None = None,
) -> LabelsLine | None:
    """The labelset's line for a retrain over a corpus; ``None`` when the labels cannot support a class model.

    The class model comes from the folds' held-out scores of the labels; the
    prevalence from the corpus the retrain scored (:func:`corpus_prevalence`).

    **Too few labels to calibrate.**  The folds need two Goods to hold one out;
    with one (a small target early in a session) they fall back and hold
    nothing.  The class model needs only one: the spread is pooled with the
    Bads.  So the head's own scores of the labelled items in this corpus stand
    in - in-sample, so the Goods sit high and the line leans precise, which is
    the safe side.  Still the labels alone, never a count on the corpus: the
    fallback before this kept thousands of a corpus at its prevalence (#4452).

    **Folds that rank the wrong way.**  When the folds held Goods and Bads out
    but scored the Goods no higher than the Bads, the same stand-in draws the
    line (the owner's ruling, 2026-10-15), and the folds' own model rides on it
    as :attr:`LabelsLine.held_out`, so its :attr:`~LabelsLine.separation` says
    what they found: none (#4899).  The stand-in's own separation was read from
    the labels the head trained on, and labels that did not separate at all
    read as separating strongly.
    """
    folds = _pooled_class_model(orderings)
    model = folds if folds is not None and folds.mu_pos > folds.mu_neg else None
    wrong_way = folds if model is None else None
    if model is None and labels:
        model = class_score_model([_in_sample_ordering(corpus_scores, corpus_ids, labels)])
        if model is not None:
            # In-sample scores are not held out: the head saw these Bads, so they
            # cannot stand for the negatives' shape (#4490), nor its Goods for how deep
            # the positives reach (#5039).
            model = replace(model, neg_logits=None, pos_logits=None)
    if model is None:
        return None
    return _line_on(model, corpus_scores, corpus_ids, labels, held_out=wrong_way)


#: Every public name of the line's four modules, so an import through this one keeps resolving (#5426).
__all__ = [
    "HEAD_COUNT_BADS",
    "HEAD_COUNT_GOODS",
    "HEAD_COUNT_PRECISION_EXPONENT",
    "HEAD_COUNT_PRECISION_SCALE",
    "HEAD_COUNT_PRECISION_TOP_Z",
    "HEAD_COUNT_RECALL_EXPONENT",
    "HEAD_COUNT_RECALL_Z",
    "MIN_LOGIT_SIGMA",
    "RELATIVE_SIGMA_FLOOR",
    "RANDOM_BADS_MAX_ENRICHMENT",
    "RANDOM_BADS_MIN",
    "RANDOM_BADS_TOP_SHARE",
    "WEAK_CHECK_COOLDOWN",
    "WEAK_CHECK_MIN_VOTES",
    "WEAK_CHECK_REPEAT",
    "WEAK_SEPARATION_D",
    "WEAK_CHECK_OFF_MEDIA_TYPES",
    "STOP_LEVEL",
    "done_check_due",
    "weak_check_due",
    "weak_check_prompts",
    "stop_level_met",
    "stop_level_reading",
    "PREVALENCE_MAX",
    "PREVALENCE_MIN",
    "BAD_COMPONENT",
    "BAD_COMPONENTS",
    "ClassScoreModel",
    "CorpusNegatives",
    "LabelsLine",
    "class_score_model",
    "corpus_cut",
    "corpus_expected_fbeta",
    "corpus_fit",
    "corpus_posteriors",
    "corpus_prevalence",
    "corpus_sigma_floor",
    "estimate_positives",
    "fit_corpus",
    "fit_labels_line",
    "FOLD_SCALES",
    "fold_transfer_clipped",
    "fold_transfer_shift",
    "transfer_fold_orderings",
    "head_count_cut",
    "head_count_serves",
    "labels_line_threshold",
    "parametric_expected_fbeta",
    "target_precision_threshold",
]
