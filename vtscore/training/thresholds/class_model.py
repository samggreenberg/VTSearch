"""The labels line's class model (#4452): the head's held-out scores of the Goods and the Bads.

The calibration folds' held-out scores of the labels, modelled as two normals of
one spread on the logit scale (:class:`ClassScoreModel`, :func:`class_score_model`),
and the logit helpers and the prevalence range every layer of the line reads.  The
lowest of the line's four modules; :mod:`~vtscore.training.thresholds.labels_line`
tells the line's design.  Reads nothing above it; numpy and SciPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

#: Scores are clipped to ``[_CLIP, 1 - _CLIP]`` before the logit.
_CLIP = 1e-6
#: The smallest spread, in logit units, the class model accepts.  Folds that
#: separate perfectly give a near-zero pooled spread, and a normal that narrow
#: would put the line at an arbitrary point of the empty gap.
MIN_LOGIT_SIGMA = 0.25
#: The fewest Bads whose own scores model the negatives (#4490): a kernel density of a
#: random 30 cannot see the upper tail (it lost 0.09 F at beta 1/4 on the review's
#: ceiling), of 100 it gains.  A model keeps its Bads' scores only from this many.
RANDOM_BADS_MIN = 100
#: The prevalence range an estimate is held to: never a certainty either way.
PREVALENCE_MIN, PREVALENCE_MAX = 1e-6, 0.5
#: Whether the head's count line's reach (:data:`~vtscore.training.thresholds.head_count.HEAD_COUNT_RECALL_Z`)
#: skips a hard-view face's held-out Good (#5409): a hard view (Face Quality under the cut, #5372) that the
#: calibration folds hold out scores low, ranks below what stands out, and sends the line at a balance above 1
#: down the excess curve.  Four-photo sessions with one hard starting face returned about twice the faces at
#: beta 4 and halved their precision; skipping it raised F at 50 votes +0.11 / +0.06 (1024 / 640 px), back to
#: the level of leaving the face out of the session, with recall unchanged and every session with no hard Good
#: identical (``docs/experiments/2026-10-29-hard-good-line-5409``).  The face stays a Good everywhere else: the
#: fit, the class model's mean, spread and counts.  The folds tag the rows
#: (:class:`~vtscore.training.thresholds.fold_orderings.FoldOrdering`).  ``False`` is the line before #5409.
REACH_SKIPS_HARD_VIEWS: bool = True


def _logit(scores: Any) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=np.float64), _CLIP, 1.0 - _CLIP)
    return np.log(s / (1.0 - s))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def _finite_unit(scores: Any) -> np.ndarray:
    """The scores that are real sigmoid outputs: finite and in ``[0, 1]`` (unscorable media carry a sentinel below)."""
    a = np.asarray(scores, dtype=np.float64)
    return a[np.isfinite(a) & (a >= 0.0) & (a <= 1.0)]


#: The normal density's constant, computed as scipy's ``_norm_pdf_C`` is.
_SQRT_2PI = np.sqrt(2 * np.pi)


def _norm_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """The normal density at *x*: ``scipy.stats.norm.pdf(x, mu, sigma)`` bit for bit, without its per-call cost (#5072).

    scipy's own arithmetic in scipy's order - standardise, ``exp(-z**2 / 2) /
    sqrt(2 pi)``, then divide by *sigma* - so every value is the same float.
    scipy's argument handling around it was ~44% of a simulated session in the
    EM loops below, and the one case it decides is a spread that is not
    positive, which gets scipy's NaN here too (every caller floors its spread
    above 0).  Non-finite *x* agree without it: ``±inf`` gives 0, NaN gives NaN.
    Typed, unlike scipy's: pyright re-infers an unannotated call's type around
    a loop until it settles (#5070).
    """
    if not sigma > 0:
        return np.full(np.shape(x), np.nan)
    z = (x - mu) / sigma
    return np.exp(-(z**2) / 2.0) / _SQRT_2PI / sigma


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
    #: The Bads' held-out logit scores, sorted, when there are at least
    #: :data:`RANDOM_BADS_MIN` of them (#4490): what a corpus reads to tell a random
    #: sample of Bads from a selected one, and the shape it then gives the negatives.
    #: ``None`` with fewer, and on a model built by hand.  Not in :meth:`as_dict`.
    neg_logits: np.ndarray | None = field(default=None, compare=False, repr=False)
    #: The Bads' own spread of their held-out logit scores (``ddof=1``), not pooled with the Goods: what a 3-part fit
    #: that places the Bad component on the corpus pools them with
    #: (:data:`~vtscore.training.thresholds.corpus_model.BAD_COMPONENT`, #5040).  ``None`` with one Bad, and on a model
    #: built by hand.  Not in :meth:`as_dict`.
    neg_sd: float | None = field(default=None, compare=False, repr=False)
    #: The Goods' held-out logit scores, sorted: where the head's count line looks for a Good ranked
    #: below what stands out (:data:`~vtscore.training.thresholds.head_count.HEAD_COUNT_RECALL_Z`, #5039).
    #: A hard-view face's Good is left out (:data:`REACH_SKIPS_HARD_VIEWS`, #5409), though it counts in
    #: every other field.  ``None`` on a model built by hand, and on the in-sample stand-in, whose Goods
    #: the head trained on.  Not in :meth:`as_dict`.
    pos_logits: np.ndarray | None = field(default=None, compare=False, repr=False)

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
    model = _pooled_class_model(orderings)
    return model if model is not None and model.mu_pos > model.mu_neg else None


def _pooled_class_model(orderings: Sequence[tuple[Any, Any]] | None) -> ClassScoreModel | None:
    """:func:`class_score_model` whichever way the folds rank: ``None`` only with no held-out Good or Bad.

    A model whose Goods do not score above its Bads draws no line, but its gap is
    still what the folds say about the labels' separation (#4899).
    """
    if not orderings:
        return None
    pos_parts: list[np.ndarray] = []
    neg_parts: list[np.ndarray] = []
    reach_parts: list[np.ndarray] = []
    for ordering in orderings:
        scores, labels = ordering
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
        hard = getattr(ordering, "hard_views", None) if REACH_SKIPS_HARD_VIEWS else None
        clear = np.ones(n, dtype=bool) if hard is None else ~np.asarray(hard, dtype=bool)[:n]
        reach_parts.append(s[ok & (y >= 0.5) & clear])
    pos = np.concatenate(pos_parts) if pos_parts else np.empty(0)
    neg = np.concatenate(neg_parts) if neg_parts else np.empty(0)
    if pos.size == 0 or neg.size == 0:
        return None
    xp, xn = _logit(pos), _logit(neg)
    mu_pos, mu_neg = float(xp.mean()), float(xn.mean())
    dof = max(1, xp.size + xn.size - 2)
    var = (float(((xp - mu_pos) ** 2).sum()) + float(((xn - mu_neg) ** 2).sum())) / dof
    raw = math.sqrt(var)
    neg = np.sort(xn) if xn.size >= RANDOM_BADS_MIN else None
    neg_sd = float(np.std(xn, ddof=1)) if xn.size > 1 else None
    reach = _logit(np.concatenate(reach_parts))
    return ClassScoreModel(
        mu_pos, mu_neg, max(raw, MIN_LOGIT_SIGMA), int(xp.size), int(xn.size), raw, neg, neg_sd, np.sort(reach)
    )
