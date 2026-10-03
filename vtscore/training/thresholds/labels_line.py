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
* **How common the target is in the corpus being decided.**  Estimated on
  that corpus with the same class model (:func:`estimate_positives`, an EM over
  its unvoted scores, its Good votes counted as they are).  Train estimates it
  on the dataset it trains on and Find on the dataset it searches; the owner's
  working assumption (2026-10-02) is that a Find corpus has the properties of
  the Train corpus, so the two estimates - and the two lines - agree, and when
  they do not (a Find corpus with no positives) the estimate falls with it.
  Only the labels travel: the labelset is the evidence, never a number
  derived from it, and more labels from any dataset improve the model.

The threshold is the score at which the expected F-beta of the kept set peaks
for a corpus at that prevalence (:func:`labels_line_threshold`).  No count is
drawn on any corpus: a corpus with no positives returns at most its own few
high-scoring negatives.

**Known weakness.**  The votes are not a random sample: active learning picks
near the line and from the top, so the Bads' held-out scores sit higher than
the true negatives' and the negative tail is extrapolated by the normal.  The
pricing in ``docs/experiments`` measures what that costs on the objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: Scores are clipped to ``[_CLIP, 1 - _CLIP]`` before the logit.
_CLIP = 1e-6
#: The smallest spread, in logit units, the class model accepts.  Folds that
#: separate perfectly give a near-zero pooled spread, and a normal that narrow
#: would put the line at an arbitrary point of the empty gap.
MIN_LOGIT_SIGMA = 0.25
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
        }


def class_score_model(orderings: Sequence[tuple[Sequence[float], Sequence[float]]] | None) -> ClassScoreModel | None:
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
    sigma = max(math.sqrt(var), MIN_LOGIT_SIGMA)
    return ClassScoreModel(mu_pos, mu_neg, sigma, int(xp.size), int(xn.size))


def labels_line_threshold(model: ClassScoreModel, prevalence: float, beta: float) -> float:
    """The score at which the kept set's expected F-*beta* peaks, in a corpus at *prevalence*.

    With ``S1`` and ``S0`` the shares of positives and negatives at or above a
    cut, a corpus at prevalence ``p`` keeps ``p S1`` true and ``(1 - p) S0``
    false matches per item, and misses ``p (1 - S1)``, so
    ``F = (1 + b^2) p S1 / (p S1 + b^2 p + (1 - p) S0)``.  Searched on a fine
    logit grid around the two class means; a tie keeps the higher cut.
    Nothing here reads the corpus the line will be applied to.
    """
    p = min(max(float(prevalence), PREVALENCE_MIN), PREVALENCE_MAX)
    b2 = float(beta) * float(beta)
    lo = min(model.mu_neg, model.mu_pos) - _SEARCH_SPREADS * model.sigma
    hi = max(model.mu_neg, model.mu_pos) + _SEARCH_SPREADS * model.sigma
    x = np.linspace(lo, hi, _SEARCH_POINTS)
    from scipy.stats import norm  # noqa: PLC0415

    s1 = norm.sf((x - model.mu_pos) / model.sigma)
    s0 = norm.sf((x - model.mu_neg) / model.sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (1.0 + b2) * p * s1 / (p * s1 + b2 * p + (1.0 - p) * s0)
    f = np.nan_to_num(f, nan=0.0)
    best = float(f.max())
    i = int(np.flatnonzero(f >= best - 1e-12).max())
    return _sigmoid(float(x[i]))


def estimate_positives(
    model: ClassScoreModel,
    unvoted_scores: Any,
    n_good: int,
    prior: float | None = None,
    *,
    iterations: int = 200,
    tol: float = 1e-8,
) -> float:
    """How many positives a corpus holds: its Good votes plus the EM estimate among its unvoted items.

    The EM (Saerens, Latinne & Decaestecker, 2002) finds the prevalence of
    the unvoted remainder that the class model's posteriors average to.  It
    starts from *prior* when there is one.
    """
    s = _finite_unit(unvoted_scores)
    if s.size == 0:
        return float(n_good)
    pi = min(max(float(prior) if prior is not None else 0.01, PREVALENCE_MIN), PREVALENCE_MAX)
    for _ in range(iterations):
        new = float(model.posterior(s, pi).mean())
        new = min(max(new, PREVALENCE_MIN), PREVALENCE_MAX)
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return float(n_good) + float(model.posterior(s, pi).sum())


@dataclass(frozen=True)
class LabelsLine:
    """What a retrain leaves for the line: the labels' class model and the prevalence estimated on a corpus."""

    model: ClassScoreModel
    prevalence: float

    def threshold(self, beta: float) -> float:
        return labels_line_threshold(self.model, self.prevalence, beta)

    def on_corpus(
        self, scores: Any, ids: Iterable[int] | None = None, labels: Mapping[int, bool] | None = None
    ) -> "LabelsLine":
        """The same class model with the prevalence re-estimated on another corpus (a Find pass over a new dataset)."""
        p = corpus_prevalence(self.model, scores, ids, labels, prior=self.prevalence)
        return LabelsLine(self.model, self.prevalence if p is None else p)


def corpus_prevalence(
    model: ClassScoreModel,
    scores: Any,
    ids: Iterable[int] | None = None,
    labels: Mapping[int, bool] | None = None,
    *,
    prior: float | None = None,
) -> float | None:
    """The prevalence *model* estimates for a corpus: its Good votes plus the EM among its unvoted items, over its size.

    *labels* are the votes that sit in this corpus (``True`` = Good); a Find
    corpus usually holds none.  ``None`` for a corpus with no scorable item.
    """
    a = np.asarray(scores, dtype=np.float64)
    keep = np.isfinite(a) & (a >= 0.0) & (a <= 1.0)
    n_items = int(keep.sum())
    if n_items == 0:
        return None
    id_list = list(ids) if ids is not None else list(range(a.size))
    voted = dict(labels or {})
    unvoted = np.array([s for i, s, k in zip(id_list, a, keep) if k and int(i) not in voted], dtype=np.float64)
    n_good = sum(1 for i, k in zip(id_list, keep) if k and voted.get(int(i)) is True)
    positives = estimate_positives(model, unvoted, n_good, prior=prior)
    return min(max(positives / n_items, PREVALENCE_MIN), PREVALENCE_MAX)


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
    orderings: Sequence[tuple[Sequence[float], Sequence[float]]] | None,
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
    p = corpus_prevalence(model, corpus_scores, corpus_ids, labels)
    if p is None:
        return None
    return LabelsLine(model, p)


__all__ = [
    "MIN_LOGIT_SIGMA",
    "PREVALENCE_MAX",
    "PREVALENCE_MIN",
    "ClassScoreModel",
    "LabelsLine",
    "class_score_model",
    "corpus_prevalence",
    "estimate_positives",
    "fit_labels_line",
    "labels_line_threshold",
]
