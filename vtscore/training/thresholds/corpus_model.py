"""The labels line's corpus side (#4452): what the corpus being decided holds, and the cut on it.

The corpus's unvoted scores fitted as the labels' Good component plus a normal
for its own negatives (:func:`fit_corpus`), and as a 3-part fit that gives each
item its chance of being a positive (:func:`corpus_posteriors`); the cut where
the kept set's expected F-beta peaks (:func:`corpus_cut`, or
:func:`labels_line_threshold` with no corpus counted); the spread floor a corpus
sets (#4492); and the folds' scale transfer the eval harness prices (#4910).
Reads :mod:`~vtscore.training.thresholds.class_model`;
:mod:`~vtscore.training.thresholds.labels_line` draws the line from these.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from vtscore.training.thresholds.class_model import (
    MIN_LOGIT_SIGMA,
    PREVALENCE_MAX,
    PREVALENCE_MIN,
    ClassScoreModel,
    _finite_unit,
    _logit,
    _norm_pdf,
    _sigmoid,
)

if TYPE_CHECKING:
    from vtscore.training.thresholds.labels_line import LabelsLine

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


#: How the calibration folds' held-out scores reach the class model (#4910).  ``"raw"`` pools
#: them as each fold model scored them: the shipped line, which mixes the fold models' scales
#: with the final model's, whose corpus scores the line is cut on.  ``"quantile"`` first carries
#: each fold's scores onto the final model's scale by their quantile in that fold model's own
#: haystack (:func:`~vtscore.training.thresholds.anchored.rank_transfer`, which saturates at the
#: haystack's ends); ``"quantile_tails"`` does the same inside the haystack's range and, past
#: either end, keeps the score's logit distance from that end.  Eval arms only: the app pools raw.
FOLD_SCALES: tuple[str, ...] = ("raw", "quantile", "quantile_tails")


def transfer_fold_orderings(
    orderings: Sequence[tuple[Any, Any]],
    fold_haystacks: Sequence[Any],
    final_haystack: Any,
    *,
    tails: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The folds' held-out scores carried onto the final model's scale, fold by fold (#4910).

    Fold *k*'s held-out score ``s`` becomes the final haystack's quantile at the
    share of fold *k*'s haystack scoring strictly below ``s``: what
    :func:`~vtscore.training.thresholds.anchored.rank_transfer` does to one cut,
    done to every held-out score.  *fold_haystacks[k]* must be fold *k*'s model's
    scores of the same population *final_haystack* holds the final model's
    scores of (both with the voted media dropped, or both with them kept), so a
    quantile means the same item in both.  Unscorable entries (outside
    ``[0, 1]``, non-finite) are dropped from the haystacks and left as they are
    in the orderings, which
    :func:`~vtscore.training.thresholds.class_model.class_score_model` drops anyway.

    The quantile saturates: a held-out Good above every item of its fold's
    haystack lands on the final haystack's top score, however far above it
    sat.  With *tails*, a score past either end of its fold's haystack keeps
    its logit distance from that end instead, measured from the final
    haystack's same end.

    A fold with no haystack (fewer haystacks than orderings) or an empty one,
    or an empty *final_haystack*, keeps its scores as they are.  Labels pass
    through unchanged, in order.
    """
    tgt = np.sort(_finite_unit(final_haystack))
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k, (scores, labels) in enumerate(orderings):
        s = np.asarray(scores, dtype=np.float64).ravel().copy()
        y = np.asarray(labels, dtype=np.float64).ravel()
        src = np.sort(_finite_unit(fold_haystacks[k])) if k < len(fold_haystacks) else np.empty(0)
        ok = np.isfinite(s) & (s >= 0.0) & (s <= 1.0)
        if src.size and tgt.size and ok.any():
            v = s[ok]
            q = np.searchsorted(src, v, side="left") / float(src.size)
            mapped = np.asarray(np.quantile(tgt, np.clip(q, 0.0, 1.0)), dtype=np.float64)
            if tails:
                x = _logit(v)
                for past, s_end, t_end in ((v > src[-1], src[-1], tgt[-1]), (v < src[0], src[0], tgt[0])):
                    if past.any():
                        shift = float(_logit([t_end])[0] - _logit([s_end])[0])
                        mapped[past] = 1.0 / (1.0 + np.exp(-(x[past] + shift)))
            s[ok] = mapped
        out.append((s, y))
    return out


def fold_transfer_shift(orderings: Sequence[tuple[Any, Any]], transferred: Sequence[tuple[Any, Any]]) -> np.ndarray:
    """``|logit(transferred) - logit(raw)|`` for every scorable held-out score, folds pooled (#4910's gate)."""
    parts: list[np.ndarray] = [np.empty(0)]
    for (raw, _), (moved, _) in zip(orderings, transferred, strict=True):
        a = np.asarray(raw, dtype=np.float64).ravel()
        b = np.asarray(moved, dtype=np.float64).ravel()
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
        ok = np.isfinite(a) & (a >= 0.0) & (a <= 1.0)
        parts.append(np.abs(_logit(b[ok]) - _logit(a[ok])))
    return np.concatenate(parts)


def fold_transfer_clipped(
    orderings: Sequence[tuple[Any, Any]], fold_haystacks: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray]:
    """``(side, good)`` per scorable held-out score, folds pooled (#4910's gate).

    *side* is ``1`` above the score's fold's whole haystack, ``-1`` below it and
    ``0`` inside: where :func:`transfer_fold_orderings` saturates without
    *tails*.  A fold with no haystack reads ``0`` throughout, its scores not
    moved.  *good* is whether the score is a Good's.
    """
    sides: list[np.ndarray] = [np.empty(0, dtype=np.int8)]
    goods: list[np.ndarray] = [np.empty(0, dtype=bool)]
    for k, (scores, labels) in enumerate(orderings):
        a = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        n = min(a.size, y.size)
        a, y = a[:n], y[:n]
        ok = np.isfinite(a) & (a >= 0.0) & (a <= 1.0)
        a, y = a[ok], y[ok]
        src = _finite_unit(fold_haystacks[k]) if k < len(fold_haystacks) else np.empty(0)
        side = np.zeros(a.size, dtype=np.int8)
        if src.size:
            side[a > src.max()] = 1
            side[a < src.min()] = -1
        sides.append(side)
        goods.append(y >= 0.5)
    return np.concatenate(sides), np.concatenate(goods)


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
    x, f = _parametric_fbeta(model, prevalence, beta, negatives)
    best = float(f.max())
    i = int(np.flatnonzero(f >= best - 1e-12).max())
    return _sigmoid(float(x[i]))


def _parametric_fbeta(
    model: ClassScoreModel, prevalence: float, beta: float, negatives: CorpusNegatives | None
) -> tuple[np.ndarray, np.ndarray]:
    """``(logit grid, expected F-beta of the cut at each point)`` for :func:`labels_line_threshold`."""
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
    return x, np.nan_to_num(f, nan=0.0)


def parametric_expected_fbeta(
    model: ClassScoreModel, prevalence: float, beta: float, negatives: CorpusNegatives | None = None
) -> float:
    """The expected F-*beta* at :func:`labels_line_threshold`'s cut: what the line would score on a fresh corpus.

    A corpus at *prevalence* whose positives score as the labels' Good component
    and whose negatives as *negatives* (else the Bad component).  Unlike
    :func:`corpus_expected_fbeta` it counts no item, so it does not fall as a
    session labels its corpus's positives away: it reads the class model alone.
    """
    return float(_parametric_fbeta(model, prevalence, beta, negatives)[1].max())


def _corpus_fbeta(posteriors_desc: np.ndarray, total_positives: float, beta: float) -> np.ndarray:
    """The expected F-*beta* of each cut of a corpus's items, best first, as :func:`corpus_cut` reads it."""
    b2 = float(beta) * float(beta)
    tp = np.cumsum(np.asarray(posteriors_desc, dtype=np.float64))
    if tp.size == 0:
        return tp
    pos = max(float(total_positives), float(tp[-1]))
    r = np.arange(1, tp.size + 1, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num((1.0 + b2) * tp / (b2 * pos + r), nan=0.0)


def corpus_expected_fbeta(posteriors_desc: np.ndarray, total_positives: float, beta: float) -> float:
    """The expected F-*beta* at :func:`corpus_cut`'s cut: what the line expects to return from this corpus.

    0 for a corpus with nothing worth keeping.  It counts the corpus's own
    unvoted items, so it describes the rest of the session's corpus and falls
    as the session labels that corpus's positives away, even with the model
    unchanged; :func:`parametric_expected_fbeta` is the fresh-corpus reading.
    """
    f = _corpus_fbeta(posteriors_desc, total_positives, beta)
    return float(f.max()) if f.size and f.max() > 0 else 0.0


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
    f = _corpus_fbeta(posteriors_desc, total_positives, beta)
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
    x = _logit(_finite_unit(unvoted_scores))
    if x.size == 0:
        return float(n_good), CorpusNegatives(model.mu_neg, model.sigma)
    mu0 = float(np.median(x))
    s0 = max(1.4826 * float(np.median(np.abs(x - mu0))), floor)
    pi = 0.01
    r = np.zeros_like(x)
    for _ in range(iterations):
        f1 = pi * _norm_pdf(x, model.mu_pos, model.sigma)
        f0 = (1.0 - pi) * _norm_pdf(x, mu0, s0)
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


#: Where the 3-part fit places the labels' Bad component (#5040).  ``"corpus_mean"``, the app's since #5040: its mean
#: fitted on the corpus being decided, the held-out Bads counted as labelled points of it (each M-step pools them with
#: the corpus's soft members of the component), its spread the class model's; so it sits where the corpus's own
#: negatives near the line are rather than where the Bads acquisition picked are.  ``"corpus"`` fits its spread too
#: (on Region Photo that overrated the positives near the line, 0.87 against a true 0.37); ``"labels"``, the line
#: before #5040, pins it at the held-out Bads' mean.  Priced on #4937's heads and end to end: the mean alone, with
#: :data:`~vtscore.training.thresholds.labels_line._BOUND_MARGIN` at 4, gained +0.023 / +0.016 / +0.005 mean F-beta
#: over clicks 0-200 at beta 1/4 / 1 / 4 (``docs/experiments/2026-10-17-chances-5040/REPORT.md``).  The eval harness
#: can set another (``vtscore.eval.live_threshold_rules.labels_chances``), here where :func:`corpus_posteriors` reads it.
BAD_COMPONENTS: tuple[str, ...] = ("labels", "corpus", "corpus_mean")
BAD_COMPONENT = "corpus_mean"


def corpus_posteriors(
    model: ClassScoreModel,
    unvoted_scores: Any,
    *,
    iterations: int = 300,
    floor: float = MIN_LOGIT_SIGMA,
    bad: str | None = None,
) -> np.ndarray:
    """Each unvoted item's chance of being a positive, under a 3-part fit of the corpus (#4452).

    The labels' Good component keeps its shape, and the labels' Bad component
    stands for the negatives near the line (the Bads sit there, exactly where
    the cut is decided), placed by *bad* (below); a normal for the corpus's
    bulk below them is fitted with all three shares.  Modelling the negatives near the line by the
    Bads is what makes the precision near the top honest: with the bulk
    alone, the top of a ranking read as all positive and beta barely moved
    the cut.  In the order of *unvoted_scores*.

    *bad* (default :data:`BAD_COMPONENT`, ``"corpus_mean"``) fits the Bad
    component's mean on the corpus, the held-out Bads counted as *n_neg*
    labelled points of it: each M-step's Bad mean pools them with the
    corpus's soft Bad members, and the bulk's mean is held below the Bad mean
    the step ends with.  ``"corpus"`` fits its spread the same way (pooling
    :attr:`ClassScoreModel.neg_sd`, floored at *floor*); ``"labels"`` keeps it
    at the held-out Bads' mean and the model's spread, the fit before #5040.
    """
    bad = BAD_COMPONENT if bad is None else bad
    if bad not in BAD_COMPONENTS:
        raise ValueError(f"bad must be one of {BAD_COMPONENTS}; got {bad!r}")
    x = _logit(_finite_unit(unvoted_scores))
    if x.size == 0:
        return np.zeros(0)
    mu0 = float(np.median(x))
    s0 = max(1.4826 * float(np.median(np.abs(x - mu0))), floor)
    w1, wb = 0.01, 0.05
    b_mu, b_sd = model.mu_neg, model.sigma
    fit_bad = bad in ("corpus", "corpus_mean")
    # The held-out Bads as labelled points: their count, mean and sum of squares about that mean.
    n_b = float(model.n_neg)
    m2_b = (n_b - 1.0) * model.neg_sd**2 if model.neg_sd is not None else 0.0
    r1 = np.zeros_like(x)
    for _ in range(iterations):
        a1 = w1 * _norm_pdf(x, model.mu_pos, model.sigma)
        ab = wb * _norm_pdf(x, b_mu, b_sd)
        a0 = max(1.0 - w1 - wb, 1e-9) * _norm_pdf(x, mu0, s0)
        tot = a1 + ab + a0
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = np.nan_to_num(a1 / tot, nan=0.0)
            rb = np.nan_to_num(ab / tot, nan=0.0)
        r0 = np.clip(1.0 - r1 - rb, 0.0, 1.0)  # rounding can push the remainder a hair below 0
        nw1 = min(max(float(r1.mean()), PREVALENCE_MIN), PREVALENCE_MAX)
        nwb = min(max(float(rb.mean()), PREVALENCE_MIN), 0.9)
        moved = 0.0
        if fit_bad:
            n = n_b + float(rb.sum())
            nm = (n_b * model.mu_neg + float((rb * x).sum())) / n
            if bad == "corpus":
                ss = m2_b + n_b * (model.mu_neg - nm) ** 2 + float((rb * (x - nm) ** 2).sum())
                ns = max(math.sqrt(ss / n), floor)
            else:
                ns = b_sd
            moved = max(abs(nm - b_mu), abs(ns - b_sd))
            b_mu, b_sd = nm, ns
        s_r0 = float(r0.sum())
        if s_r0 > 1e-12:
            mu0 = min(float((r0 * x).sum()) / s_r0, b_mu)
            s0 = max(math.sqrt(max(float((r0 * (x - mu0) ** 2).sum()) / s_r0, 0.0)), floor)
        done = abs(nw1 - w1) < 1e-8 and abs(nwb - wb) < 1e-7 and moved < 1e-7
        w1, wb = nw1, nwb
        if done:
            break
    return r1


def target_precision_threshold(line: "LabelsLine | None", p: float) -> float | None:
    """The score where *line*'s corpus posterior falls below *p*: Autopilot samples where picks are *p* positive (#3546).

    *line*'s unvoted scores are best first, and its ``unvoted_posteriors`` are
    each one's chance of being a positive under the 3-part corpus fit the line
    is cut on (:func:`corpus_posteriors`; computed here for a line built by
    hand).  The cut is the first (highest) score whose posterior is below *p*;
    if every item is at least *p*, the deepest score; if none is, the top one.
    It states acquisition in the line's own terms, so neither the Inclusion
    step's shortfall nor the fold-anchored scale's saturation can reach it.
    ``None`` with no line or no unvoted score.
    """
    if line is None or line.unvoted_scores is None:
        return None
    scores = np.asarray(line.unvoted_scores, dtype=np.float64)
    if scores.size == 0:
        return None
    kept = line.unvoted_posteriors
    post = np.asarray(kept, dtype=np.float64) if kept is not None and len(kept) == scores.size else None
    if post is None:
        post = corpus_posteriors(line.model, scores)
    below = np.flatnonzero(post < p)
    return float(scores[below[0] if below.size else scores.size - 1])


def estimate_positives(model: ClassScoreModel, unvoted_scores: Any, n_good: int, prior: float | None = None) -> float:
    """How many positives a corpus holds: its Good votes plus the EM estimate among its unvoted items (:func:`fit_corpus`)."""
    del prior  # the corpus fit starts from a fixed small share
    return fit_corpus(model, unvoted_scores, n_good)[0]
