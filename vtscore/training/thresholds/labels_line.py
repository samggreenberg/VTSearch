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
drawn on any corpus: a corpus with no positives returns at most its own few
high-scoring negatives.

**Known weakness.**  The votes are not a random sample: active learning picks
near the line and from the top, so the Bads' held-out scores sit higher than
the true negatives' and the negative tail is extrapolated by the normal.  The
pricing in ``docs/experiments`` measures what that costs on the objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: Scores are clipped to ``[_CLIP, 1 - _CLIP]`` before the logit.
_CLIP = 1e-6
#: The smallest spread, in logit units, the class model accepts.  Folds that
#: separate perfectly give a near-zero pooled spread, and a normal that narrow
#: would put the line at an arbitrary point of the empty gap.
MIN_LOGIT_SIGMA = 0.25
#: The spread floor once a corpus is known (#4492): this share of the corpus's own
#: robust spread (1.4826 x the MAD of its logit scores), on the class model and on
#: the corpus's negative bulk alike.  An absolute floor is not scale-free: an early
#: head compresses every score (the withheld half's robust spread was ~0.09 at click
#: 5 on COCO Better, ~0.18 by 150), and a bulk held 2.5x wider than the data covers
#: the positives, so the corpus fit read none and the line kept one image for the
#: first ~15 clicks.  :data:`MIN_LOGIT_SIGMA` stays where no corpus is known (the
#: parametric fallback).
RELATIVE_SIGMA_FLOOR = 0.5
#: The smallest floor a corpus can set: a corpus of one score has no spread.
_MIN_CORPUS_FLOOR = 1e-3
#: How far past the class means, in spreads, the threshold search reaches.
_SEARCH_SPREADS = 8.0
#: Points on the threshold search's logit grid.
_SEARCH_POINTS = 4001
#: The prevalence range an estimate is held to: never a certainty either way.
PREVALENCE_MIN, PREVALENCE_MAX = 1e-6, 0.5


def _logit(scores: Any) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=np.float64), _CLIP, 1.0 - _CLIP)
    return np.log(s / (1.0 - s))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def _finite_unit(scores: Any) -> np.ndarray:
    """The scores that are real sigmoid outputs: finite and in ``[0, 1]`` (unscorable media carry a sentinel below)."""
    a = np.asarray(scores, dtype=np.float64)
    return a[np.isfinite(a) & (a >= 0.0) & (a <= 1.0)]


@dataclass(frozen=True)
class ClassScoreModel:
    """The head's held-out scores of the Goods and the Bads: two normals of one spread on the logit scale."""

    mu_pos: float
    mu_neg: float
    sigma: float
    n_pos: int
    n_neg: int
    #: The pooled spread before any floor; ``None`` on a model built by hand, whose
    #: *sigma* is then taken as given.  A corpus floors it afresh (:meth:`floored`).
    sigma_raw: float | None = None

    def floored(self, floor: float) -> "ClassScoreModel":
        """This model with its spread held to *floor* (a corpus's own, #4492) rather than the absolute one."""
        raw = self.sigma if self.sigma_raw is None else self.sigma_raw
        return replace(self, sigma=max(raw, floor))

    def _norm_sf(self, z: np.ndarray) -> np.ndarray:
        from scipy.stats import norm  # noqa: PLC0415

        return norm.sf(z)

    def survival(self, threshold: float) -> tuple[float, float]:
        """``(share of positives, share of negatives)`` scoring at or above *threshold*."""
        x = float(_logit([threshold])[0])
        s1, s0 = self._norm_sf(np.array([(x - self.mu_pos) / self.sigma, (x - self.mu_neg) / self.sigma]))
        return float(s1), float(s0)

    def posterior(self, scores: Any, prevalence: float) -> np.ndarray:
        """The chance each score is a positive, in a corpus at *prevalence*."""
        x = _logit(scores)
        llr = ((x - self.mu_neg) ** 2 - (x - self.mu_pos) ** 2) / (2.0 * self.sigma**2)
        pi = min(max(float(prevalence), PREVALENCE_MIN), 1.0 - PREVALENCE_MIN)
        z = llr + math.log(pi / (1.0 - pi))
        return 1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mu_pos": round(self.mu_pos, 6),
            "mu_neg": round(self.mu_neg, 6),
            "sigma": round(self.sigma, 6),
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "sigma_raw": None if self.sigma_raw is None else round(self.sigma_raw, 6),
        }


def class_score_model(orderings: Sequence[tuple[Any, Any]] | None) -> ClassScoreModel | None:
    """The class model the calibration folds' held-out scores imply; ``None`` when they cannot support one.

    *orderings* are :attr:`CalibrationFolds.orderings`: per fold, the held-out
    scores and their labels (``1.0`` Good).  Pooled over the folds.  ``None``
    with no held-out Good or no held-out Bad, or when the Goods do not score
    above the Bads on average (a head that ranks the wrong way has no line).
    """
    if not orderings:
        return None
    pos_parts: list[np.ndarray] = []
    neg_parts: list[np.ndarray] = []
    for scores, labels in orderings:
        # Any numeric dtype: the app's folds hold Python floats, the eval
        # harness's float32 arrays (an isinstance check on ``float`` silently
        # dropped every float32 score).
        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        n = min(s.size, y.size)
        s, y = s[:n], y[:n]
        ok = np.isfinite(s) & (s >= 0.0) & (s <= 1.0)
        pos_parts.append(s[ok & (y >= 0.5)])
        neg_parts.append(s[ok & (y < 0.5)])
    pos = np.concatenate(pos_parts) if pos_parts else np.empty(0)
    neg = np.concatenate(neg_parts) if neg_parts else np.empty(0)
    if pos.size == 0 or neg.size == 0:
        return None
    xp, xn = _logit(pos), _logit(neg)
    mu_pos, mu_neg = float(xp.mean()), float(xn.mean())
    if not mu_pos > mu_neg:
        return None
    dof = max(1, xp.size + xn.size - 2)
    var = (float(((xp - mu_pos) ** 2).sum()) + float(((xn - mu_neg) ** 2).sum())) / dof
    raw = math.sqrt(var)
    return ClassScoreModel(mu_pos, mu_neg, max(raw, MIN_LOGIT_SIGMA), int(xp.size), int(xn.size), raw)


def corpus_sigma_floor(scores: Any) -> float:
    """The spread floor a corpus sets (#4492): :data:`RELATIVE_SIGMA_FLOOR` x its robust spread on the logit scale."""
    x = _logit(_finite_unit(scores))
    if x.size == 0:
        return MIN_LOGIT_SIGMA
    robust = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    return max(RELATIVE_SIGMA_FLOOR * robust, _MIN_CORPUS_FLOOR)


@dataclass(frozen=True)
class CorpusNegatives:
    """The negatives of the corpus being decided: a normal on the logit scale, fitted there (#4452)."""

    mu: float
    sigma: float


def labels_line_threshold(
    model: ClassScoreModel, prevalence: float, beta: float, negatives: CorpusNegatives | None = None
) -> float:
    """The score at which the kept set's expected F-*beta* peaks, in a corpus at *prevalence*.

    With ``S1`` and ``S0`` the shares of positives and negatives at or above a
    cut, a corpus at prevalence ``p`` keeps ``p S1`` true and ``(1 - p) S0``
    false matches per item, and misses ``p (1 - S1)``, so
    ``F = (1 + b^2) p S1 / (p S1 + b^2 p + (1 - p) S0)``.  ``S1`` is the labels'
    Good component; ``S0`` the corpus's *negatives* when given (the fit
    :func:`fit_corpus` makes), else the labels' Bad component.  Searched on a
    fine logit grid; a tie keeps the higher cut.
    """
    p = min(max(float(prevalence), PREVALENCE_MIN), PREVALENCE_MAX)
    b2 = float(beta) * float(beta)
    mu0, s0 = (negatives.mu, negatives.sigma) if negatives is not None else (model.mu_neg, model.sigma)
    lo = min(mu0 - _SEARCH_SPREADS * s0, model.mu_pos - _SEARCH_SPREADS * model.sigma)
    hi = max(mu0 + _SEARCH_SPREADS * s0, model.mu_pos + _SEARCH_SPREADS * model.sigma)
    x = np.linspace(lo, hi, _SEARCH_POINTS)
    from scipy.stats import norm  # noqa: PLC0415

    s1 = norm.sf((x - model.mu_pos) / model.sigma)
    s0_ = norm.sf((x - mu0) / s0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (1.0 + b2) * p * s1 / (p * s1 + b2 * p + (1.0 - p) * s0_)
    f = np.nan_to_num(f, nan=0.0)
    best = float(f.max())
    i = int(np.flatnonzero(f >= best - 1e-12).max())
    return _sigmoid(float(x[i]))


def corpus_cut(posteriors_desc: np.ndarray, scores_desc: np.ndarray, total_positives: float, beta: float) -> float:
    """The cut maximising expected F-*beta* over what it would really return from a corpus (#4452).

    Items best first: the cut after the k-th returns ``R = k`` (counted),
    holds ``TP`` = the sum of its items' chances of being positive
    (:func:`corpus_posteriors`), out of *total_positives* in the corpus;
    ``F = (1 + b^2) TP / (b^2 P + R)``.  The chances come from the 3-part fit
    (honest precision near the top); the total from the 2-part estimate with
    its counted bound (honest about the positives deeper down, which the
    Goods a session found - its easiest - under-represent).  Each piece is
    the model that is right about it; priced offline on 900 saved heads it
    beat both alone at every preset and on a smaller Train pool.  A cut that
    returns nothing scores 0; with nothing worth keeping, a score above every
    item.
    """
    s = np.asarray(scores_desc, dtype=np.float64)
    n = s.size
    if n == 0:
        return 1.0 + 1e-9
    b2 = float(beta) * float(beta)
    tp = np.cumsum(np.asarray(posteriors_desc, dtype=np.float64))
    pos = max(float(total_positives), float(tp[-1]))
    r = np.arange(1, n + 1, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.nan_to_num((1.0 + b2) * tp / (b2 * pos + r), nan=0.0)
    k = int(np.argmax(f))
    if not f[k] > 0:
        return 1.0 + 1e-9
    # Between the last kept item and the first dropped one, not on an item: the
    # same image scored by another path (a size-band cohort, a re-scored Find
    # pass) can differ in the last bits and would flip sides of a cut placed on it.
    return float(s[k]) if k + 1 >= n else float((s[k] + s[k + 1]) / 2.0)


def fit_corpus(
    model: ClassScoreModel,
    unvoted_scores: Any,
    n_good: int,
    *,
    iterations: int = 500,
    tol: float = 1e-9,
    floor: float = MIN_LOGIT_SIGMA,
) -> tuple[float, CorpusNegatives]:
    """How many positives a corpus holds, and its negatives: ``(Good votes + EM positives, CorpusNegatives)``.

    The corpus's unvoted scores are a two-part mixture: positives distributed
    as the labels' Good component (fixed - what the labels know), negatives a
    normal fitted here jointly with their share (the corpus's own bulk).  The
    labels' Bads are not the negatives' distribution: active learning picks
    them near the line, far above the bulk, and modelling the bulk with them
    let the prevalence run away (34% for a 0.44% target on the first pricing
    cells).  Fitting the bulk on the corpus keeps a corpus with no positives
    at a share near zero.
    """
    from scipy.stats import norm  # noqa: PLC0415

    x = _logit(_finite_unit(unvoted_scores))
    if x.size == 0:
        return float(n_good), CorpusNegatives(model.mu_neg, model.sigma)
    mu0 = float(np.median(x))
    s0 = max(1.4826 * float(np.median(np.abs(x - mu0))), floor)
    pi = 0.01
    r = np.zeros_like(x)
    for _ in range(iterations):
        f1 = pi * norm.pdf(x, model.mu_pos, model.sigma)
        f0 = (1.0 - pi) * norm.pdf(x, mu0, s0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nan_to_num(f1 / (f1 + f0), nan=0.0)
        new_pi = min(max(float(r.mean()), PREVALENCE_MIN), PREVALENCE_MAX)
        w = 1.0 - r
        sw = float(w.sum())
        if sw <= 0:
            break
        new_mu0 = min(float((w * x).sum()) / sw, model.mu_pos)
        new_s0 = max(math.sqrt(float((w * (x - new_mu0) ** 2).sum()) / sw), floor)
        done = abs(new_pi - pi) < tol and abs(new_mu0 - mu0) < 1e-7 and abs(new_s0 - s0) < 1e-7
        pi, mu0, s0 = new_pi, new_mu0, new_s0
        if done:
            break
    return float(n_good) + float(r.sum()), CorpusNegatives(mu0, s0)


def corpus_posteriors(
    model: ClassScoreModel, unvoted_scores: Any, *, iterations: int = 300, floor: float = MIN_LOGIT_SIGMA
) -> np.ndarray:
    """Each unvoted item's chance of being a positive, under a 3-part fit of the corpus (#4452).

    The labels' Good component and the labels' Bad component keep their
    shapes (what the labels know: the Bads sit near the line, exactly where
    the cut is decided); a normal for the corpus's bulk below them is fitted
    with all three shares.  Modelling the negatives near the line by the
    Bads is what makes the precision near the top honest: with the bulk
    alone, the top of a ranking read as all positive and beta barely moved
    the cut.  In the order of *unvoted_scores*.
    """
    from scipy.stats import norm  # noqa: PLC0415

    x = _logit(_finite_unit(unvoted_scores))
    if x.size == 0:
        return np.zeros(0)
    mu0 = float(np.median(x))
    s0 = max(1.4826 * float(np.median(np.abs(x - mu0))), floor)
    w1, wb = 0.01, 0.05
    r1 = np.zeros_like(x)
    for _ in range(iterations):
        a1 = w1 * norm.pdf(x, model.mu_pos, model.sigma)
        ab = wb * norm.pdf(x, model.mu_neg, model.sigma)
        a0 = max(1.0 - w1 - wb, 1e-9) * norm.pdf(x, mu0, s0)
        tot = a1 + ab + a0
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = np.nan_to_num(a1 / tot, nan=0.0)
            rb = np.nan_to_num(ab / tot, nan=0.0)
        r0 = np.clip(1.0 - r1 - rb, 0.0, 1.0)  # rounding can push the remainder a hair below 0
        nw1 = min(max(float(r1.mean()), PREVALENCE_MIN), PREVALENCE_MAX)
        nwb = min(max(float(rb.mean()), PREVALENCE_MIN), 0.9)
        s_r0 = float(r0.sum())
        if s_r0 > 1e-12:
            mu0 = min(float((r0 * x).sum()) / s_r0, model.mu_neg)
            s0 = max(math.sqrt(max(float((r0 * (x - mu0) ** 2).sum()) / s_r0, 0.0)), floor)
        done = abs(nw1 - w1) < 1e-8 and abs(nwb - wb) < 1e-7
        w1, wb = nw1, nwb
        if done:
            break
    return r1


def estimate_positives(model: ClassScoreModel, unvoted_scores: Any, n_good: int, prior: float | None = None) -> float:
    """How many positives a corpus holds: its Good votes plus the EM estimate among its unvoted items (:func:`fit_corpus`)."""
    del prior  # the corpus fit starts from a fixed small share
    return fit_corpus(model, unvoted_scores, n_good)[0]


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

    def threshold(self, beta: float) -> float:
        if self.unvoted_scores is not None and self.unvoted_posteriors is not None and self.unvoted_share is not None:
            total = self.unvoted_share * self.unvoted_scores.size
            return corpus_cut(self.unvoted_posteriors, self.unvoted_scores, total, beta)
        return labels_line_threshold(self.model, self.prevalence, beta, self.negatives)

    def on_corpus(
        self, scores: Any, ids: Iterable[int] | None = None, labels: Mapping[int, bool] | None = None
    ) -> "LabelsLine":
        """The same class model with the corpus side re-fitted on another corpus (a Find pass over a new dataset)."""
        line = _line_on(self.model, scores, ids, labels)
        return self if line is None else line


#: The smallest share of the labels' Good component a cut must hold for the
#: counted bound on the prevalence to read it (below it, ``R / S1`` is noise).
_BOUND_MIN_S1 = 0.05
#: The counted bound guards against a runaway estimate rather than replacing
#: it: the Goods a session found are its easiest positives, so ``S1`` reads
#: high and the bound reads low (0.24% for a 0.43% target on a clean corpus).
#: The mixture's estimate stands unless it exceeds the bound by this factor.
_BOUND_MARGIN = 2.0


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


def _line_on(
    model: ClassScoreModel, scores: Any, ids: Iterable[int] | None, labels: Mapping[int, bool] | None
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
    share = (positives - n_good) / unvoted.size if unvoted.size else 0.0
    share = min(share, _BOUND_MARGIN * _counted_share_bound(fitted, unvoted))
    positives = n_good + share * unvoted.size
    prevalence = min(max(positives / n_items, PREVALENCE_MIN), PREVALENCE_MAX)
    order = np.argsort(-unvoted, kind="stable")
    post = corpus_posteriors(fitted, unvoted, floor=floor)
    # The labels' model is kept unfloored, so a Find on another corpus floors it afresh.
    return LabelsLine(model, prevalence, negatives, float(share), unvoted[order].copy(), post[order].copy())


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
    """
    model = class_score_model(orderings)
    if model is None and labels:
        model = class_score_model([_in_sample_ordering(corpus_scores, corpus_ids, labels)])
    if model is None:
        return None
    return _line_on(model, corpus_scores, corpus_ids, labels)


__all__ = [
    "MIN_LOGIT_SIGMA",
    "RELATIVE_SIGMA_FLOOR",
    "PREVALENCE_MAX",
    "PREVALENCE_MIN",
    "ClassScoreModel",
    "CorpusNegatives",
    "LabelsLine",
    "class_score_model",
    "corpus_cut",
    "corpus_fit",
    "corpus_posteriors",
    "corpus_prevalence",
    "corpus_sigma_floor",
    "estimate_positives",
    "fit_labels_line",
    "labels_line_threshold",
]
