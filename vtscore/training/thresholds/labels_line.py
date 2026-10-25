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
score best at the balance (:func:`_excess_depth`, from :data:`HEAD_COUNT_RECALL_Z` bulk spreads up).
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
#: Labels whose Good and Bad scores sit fewer spreads apart than this separate
#: weakly (:attr:`LabelsLine.separation`, #4466, #4496).  Below it a session
#: returns too much: over 200 withheld images in 37-77% of such sessions from
#: click 5 on, against 2-18% above it, and no cut rescues its F.
WEAK_SEPARATION_D = 1.5
#: The fewest votes before weak separation makes a spot check due (#4496): the
#: first few labels separate noisily, and a check needs a line to walk.
WEAK_CHECK_MIN_VOTES = 10
#: Votes after a check ends before the check before Done runs (:func:`done_check_due`, #4496): a check that ended
#: moments ago has already measured this line.  It was also the weak check's repeat until #4909.
WEAK_CHECK_COOLDOWN = 25
#: Votes after a check ends before weak separation makes another due; ``None``: never, so the weak check prompts at
#: most once a session, and only before any other check has ended (#4909, the owner's choice of 2026-10-15).  #4496
#: had chosen again after 25 votes, on an app where a check's picks were worth far more than Autopilot's.  At
#: today's head prompting once matched or beat it: +0.0006 (beta 1) and +0.0013 (beta 4) in mean F-beta over clicks
#: 0-200, level after the closing check, Done at the same click and F-beta, and 24-25 fewer of 200 clicks spent
#: checking (``docs/experiments/2026-10-15-weak-repeat-4909/REPORT.md``).
WEAK_CHECK_REPEAT: int | None = None
#: Media types whose sessions never get a prompted weak check (#4778, #4799).  A
#: face search's opening finds the person's few photos before the check could
#: start (a median of 4 Goods at vote 11), so a check's votes find nothing and
#: only pick which Bads get labelled, worse than acquisition does: without it
#: the face review's sessions gained +0.007 to +0.018 over votes 1-50, with the
#: objective after the end-of-run check and final AP unchanged.  On COCO the
#: check pays most where it starts with fewest Goods, so a Good count cannot
#: stand in for the media type.  Autopilot's check before Done stays (:func:`done_check_due`).
WEAK_CHECK_OFF_MEDIA_TYPES: frozenset[str] = frozenset({"face"})
#: The fewest Bads whose own scores model the negatives (#4490): a kernel density of a
#: random 30 cannot see the upper tail (it lost 0.09 F at beta 1/4 on the review's
#: ceiling), of 100 it gains.  A model keeps its Bads' scores only from this many.
RANDOM_BADS_MIN = 100
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
#: How far past the class means, in spreads, the threshold search reaches.
_SEARCH_SPREADS = 8.0
#: Points on the threshold search's logit grid.
_SEARCH_POINTS = 4001
#: The prevalence range an estimate is held to: never a certainty either way.
PREVALENCE_MIN, PREVALENCE_MAX = 1e-6, 0.5
#: The Bads half of the thin-model gate: while the class model holds fewer than this many Bads
#: (its held-out ones, about 0.6 of the Bads in hand) and fewer than :data:`HEAD_COUNT_GOODS`
#: Goods, the line at every balance keeps the head's count (:func:`head_count_cut`) rather than
#: the class model's cut.  Unbounded since #4798, so the Goods alone close the gate; #4786
#: shipped 32.  ``0`` is the line before #4786.
HEAD_COUNT_BADS: float = math.inf
#: The thin-model gate (#4786, #4798): with this many held-out Goods the class
#: model's Good component stands on its own, and a broad target's labels line keeps what its
#: many positives need where the head's count, a count of what stands out, keeps several
#: times too many.
HEAD_COUNT_GOODS = 6
#: How the head's count grows above a balance of 1 (#5039): ``beta ** HEAD_COUNT_RECALL_EXPONENT``
#: times what stands out, in place of #4603's ``beta ** 0.708`` (:func:`head_count_cut`): 1.3x at beta 4
#: (``log 1.3 / log 4``), what faces and Region Photo's heads want there.  ``None`` keeps #4603's, the
#: count before #5039.
HEAD_COUNT_RECALL_EXPONENT: float | None = 0.18925
#: Where the head's count line reaches down at a balance above 1 (#5039): when a held-out Good ranks
#: below the count of what stands out, the line keeps at least as deep as the head's excess curve's
#: expected F-beta peaks, read from this many bulk spreads up (:func:`_excess_depth`).  Closed loop at
#: beta 4 with the exponent above: Find's mean F-beta +0.0035 over a Binary Photo session's 200 clicks
#: and +0.015 to +0.017 over a face session's 150 votes; 3 spreads gave faces more (+0.022 to +0.023)
#: and Binary half (``docs/experiments/2026-10-18-head-count-5039``).  ``None`` never reaches, the line
#: before #5039.
HEAD_COUNT_RECALL_Z: float | None = 2.75
#: How the head's count scales at a balance of 1 and below (#5084): ``HEAD_COUNT_PRECISION_SCALE * beta **
#: HEAD_COUNT_PRECISION_EXPONENT`` times what stands out, in place of #4603's ``beta ** 0.708``
#: (:func:`head_count_cut`).  The count of what stands out reads the bulk's heavy upper tail as positives on a
#: weak head (Region Photo's keeps 2.6x its positives at beta 1), so a smaller multiplier keeps nearer the
#: positives.  0.71 beta (0.18x the count at beta 1/4 where #4603 kept 0.375x, 0.71x at beta 1): offline on the
#: head's count line Region Photo gains +0.007 to +0.084 F-beta at every prevalence, Binary Photo +0.007 to +0.037
#: at 0.44% and 0.1% and -0.014 / -0.004 at 1%, faces at most 0.008 either way; closed loop, Find's mean F-beta
#: over clicks 0-50 +0.0093 / +0.021 on Region Photo (beta 1 / 1/4) and +0.0030 / +0.0014 on Binary Photo
#: (``docs/experiments/2026-10-24-head-bulk-5084``).  ``None`` is #4603's piece: a scale of 1, an exponent of
#: 0.708.
HEAD_COUNT_PRECISION_SCALE: float | None = 0.71
HEAD_COUNT_PRECISION_EXPONENT: float | None = 1.0
#: The level Autopilot's Done also reads (#4878): Done once every quality light is green, or once
#: the labels line expects at least this F-beta at the session's balance over the corpus it was
#: cut on (:func:`stop_level_met`).  Priced on 2,160 sessions and A/B'd in the phase machine:
#: no change to the session, and the stop comes where the clicks pay
#: (``docs/experiments/2026-10-14-stop-signal-4878``).
STOP_LEVEL = 0.5


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
    #: that places the Bad component on the corpus pools them with (:data:`BAD_COMPONENT`, #5040).  ``None`` with one
    #: Bad, and on a model built by hand.  Not in :meth:`as_dict`.
    neg_sd: float | None = field(default=None, compare=False, repr=False)
    #: The Goods' held-out logit scores, sorted: where the head's count line looks for a Good ranked
    #: below what stands out (:data:`HEAD_COUNT_RECALL_Z`, #5039).  ``None`` on a model built by hand,
    #: and on the in-sample stand-in, whose Goods the head trained on.  Not in :meth:`as_dict`.
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
    dof = max(1, xp.size + xn.size - 2)
    var = (float(((xp - mu_pos) ** 2).sum()) + float(((xn - mu_neg) ** 2).sum())) / dof
    raw = math.sqrt(var)
    neg = np.sort(xn) if xn.size >= RANDOM_BADS_MIN else None
    neg_sd = float(np.std(xn, ddof=1)) if xn.size > 1 else None
    return ClassScoreModel(
        mu_pos, mu_neg, max(raw, MIN_LOGIT_SIGMA), int(xp.size), int(xn.size), raw, neg, neg_sd, np.sort(xp)
    )


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
    in the orderings, which :func:`class_score_model` drops anyway.

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
#: :data:`_BOUND_MARGIN` at 4, gained +0.023 / +0.016 / +0.005 mean F-beta over clicks 0-200 at beta 1/4 / 1 / 4
#: (``docs/experiments/2026-10-17-chances-5040/REPORT.md``).  The eval harness can set another
#: (``vtscore.eval.live_threshold_rules.labels_chances``).
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


def stop_level_reading(line: "LabelsLine | None", beta: float | None, level: float = STOP_LEVEL) -> dict[str, Any]:
    """What Autopilot's stop reads off the labels line (#4878): ``expected_fbeta``, ``level`` and ``met``.

    ``expected_fbeta`` is :meth:`LabelsLine.expected_fbeta` at *beta*, the
    line's own estimate of the F-beta of the set it keeps; ``None`` with no
    line, no balance, or while the head's count line serves
    (:func:`head_count_serves`), whose class model is too thin to read.  ``met``
    is whether it reaches *level*.
    """
    if line is None or beta is None or head_count_serves(line.model):
        return {"expected_fbeta": None, "level": float(level), "met": False}
    expected = float(line.expected_fbeta(float(beta)))
    return {"expected_fbeta": expected, "level": float(level), "met": expected >= level}


def stop_level_met(line: "LabelsLine | None", beta: float | None, level: float = STOP_LEVEL) -> bool:
    """Whether the labels line expects at least *level* F-beta at *beta* (:func:`stop_level_reading`)."""
    return bool(stop_level_reading(line, beta, level)["met"])


def head_count_serves(model: ClassScoreModel) -> bool:
    """Whether the line is the head's count line (#4786): a class model with too few Goods to cut on (#4798)."""
    return model.n_neg < HEAD_COUNT_BADS and model.n_pos < HEAD_COUNT_GOODS


def head_count_cut(scores_desc: Any, beta: float, held_out_goods: Any = None) -> float:
    """The cut that keeps what stands out in the head's own scores at *beta* (#4786).

    *scores_desc* are a corpus's unvoted scores, best first.  On their logits the
    rule that draws the Goods' centroid's line
    (:data:`~vtscore.detectors.centroid_head.CENTROID_LINE_RULE`: ``beta ** 0.708``
    times the larger of the Gaussian bulk's excess and the top's widest drop, never
    fewer than that drop) counts what stands out; the cut keeps exactly those items,
    placed between the last kept and the first dropped as :func:`corpus_cut` places
    its own.  An empty corpus keeps nothing.

    At a balance of 1 and below (#5084) the multiplier is
    ``HEAD_COUNT_PRECISION_SCALE * beta ** HEAD_COUNT_PRECISION_EXPONENT`` (0.71 beta) in
    place of ``beta ** 0.708`` (:func:`_scaled_count`); both at ``None`` is the rule's own.

    Above a balance of 1 (#5039) the count grows as ``beta **``
    :data:`HEAD_COUNT_RECALL_EXPONENT` instead, and where one of
    *held_out_goods* (the class model's held-out Goods, logits) ranks below the
    count at a balance of 1 - a positive the count of what stands out misses - the
    line keeps at least as deep as the head's excess curve's expected F-beta
    peaks (:func:`_excess_depth`, from :data:`HEAD_COUNT_RECALL_Z` bulk spreads
    up).  Either constant at ``None`` is that piece before #5039.
    """
    from vtscore.detectors.centroid_head import centroid_cut  # noqa: PLC0415
    from vtscore.training.thresholds.gmm import TEXT_SORT_COUNT_EXPONENT  # noqa: PLC0415

    s = np.asarray(scores_desc, dtype=np.float64)
    if s.size == 0:
        return 1.0 + 1e-9
    x = _logit(s)
    b = float(beta)
    above = b > 1.0
    # The count's own beta: the one whose #4603 multiplier is beta ** HEAD_COUNT_RECALL_EXPONENT.
    b_count = (
        b ** (HEAD_COUNT_RECALL_EXPONENT / TEXT_SORT_COUNT_EXPONENT)
        if above and HEAD_COUNT_RECALL_EXPONENT is not None
        else b
    )
    if not above and (HEAD_COUNT_PRECISION_SCALE is not None or HEAD_COUNT_PRECISION_EXPONENT is not None):
        k = _scaled_count(x, b)
    else:
        k = int(np.count_nonzero(x >= centroid_cut(x, beta=b_count)))
    goods = None if held_out_goods is None else np.asarray(held_out_goods, dtype=np.float64)
    if above and HEAD_COUNT_RECALL_Z is not None and goods is not None and goods.size:
        standouts = int(np.count_nonzero(x >= centroid_cut(x, beta=1.0)))
        if int(np.count_nonzero(x >= float(goods.min()))) > standouts:
            k = max(k, _excess_depth(x, b, float(HEAD_COUNT_RECALL_Z)))
    if k <= 0:
        return 1.0 + 1e-9
    return float(s[k - 1]) if k >= s.size else float((s[k - 1] + s[k]) / 2.0)


def _scaled_count(x: np.ndarray, beta: float) -> int:
    """How many of the logits *x* the head's count keeps at *beta* <= 1 under #5084's multiplier.

    :func:`~vtscore.detectors.centroid_head.centroid_cut`'s rule
    (:data:`~vtscore.detectors.centroid_head.CENTROID_LINE_RULE`) with the multiplier
    ``HEAD_COUNT_PRECISION_SCALE * beta ** HEAD_COUNT_PRECISION_EXPONENT`` in place of
    ``beta ** 0.708``: that many times the larger of #4603's count and the top's widest drop
    (from beta 1, the drop below a standout first, #4783), never fewer than that drop.  With
    too few scores to count, the guarded line the rule falls back to.
    """
    from vtscore.detectors.centroid_head import _top_gap_count, centroid_cut  # noqa: PLC0415
    from vtscore.training.thresholds.gmm import TEXT_SORT_COUNT_EXPONENT, _count_estimate  # noqa: PLC0415

    counted = _count_estimate(x)
    if counted is None:
        return int(np.count_nonzero(x >= centroid_cut(x, beta=beta)))
    scale = 1.0 if HEAD_COUNT_PRECISION_SCALE is None else float(HEAD_COUNT_PRECISION_SCALE)
    exponent = (
        TEXT_SORT_COUNT_EXPONENT if HEAD_COUNT_PRECISION_EXPONENT is None else float(HEAD_COUNT_PRECISION_EXPONENT)
    )
    gap = _top_gap_count(x, drop="skip25" if beta >= 1.0 else "widest")
    keep = min(max(1, round(scale * beta**exponent * max(counted, float(gap)))), x.size)
    srt = -np.sort(-x)
    return int(np.count_nonzero(x >= min(float(srt[keep - 1]), float(srt[gap - 1]))))


def _excess_depth(x_desc: np.ndarray, beta: float, z_min: float) -> int:
    """How many of the best logits *x_desc* keep the head's excess curve at its expected F-*beta* peak (#5039).

    The excess at the *k*-th best is *k* less what a Gaussian bulk puts above its
    score, the bulk's centre and spread the median and ``1.4826`` x the MAD of
    *x_desc*, as #4603's count reads them; the positives' total is the largest
    excess from *z_min* bulk spreads up.  The cut at *k* is expected to score
    ``(1 + beta**2) * excess_k / (beta**2 * total + k)``, the excess held to
    ``[0, total]``; the depth is where that peaks among the cuts at or above
    *z_min*.  ``0`` below #4603's fewest scores, with no spread, or with no score
    that high.
    """
    from scipy.special import ndtr  # noqa: PLC0415

    from vtscore.training.thresholds.gmm import TEXT_SORT_COUNT_MIN_SCORES  # noqa: PLC0415

    n = int(x_desc.size)
    if n < TEXT_SORT_COUNT_MIN_SCORES:
        return 0
    med = float(np.median(x_desc))
    sigma = 1.4826 * float(np.median(np.abs(x_desc - med)))
    if not sigma > 0.0:
        return 0
    z = (x_desc - med) / sigma
    up = z >= z_min
    if not up.any():
        return 0
    k = np.arange(1, n + 1, dtype=np.float64)
    excess = k - n * ndtr(-z)
    total = max(float(excess[up].max()), 1.0)
    b2 = beta * beta
    f = np.where(up, (1.0 + b2) * np.clip(excess, 0.0, total) / (b2 * total + k), -1.0)
    return int(np.argmax(f)) + 1


def weak_check_prompts(media_type: str | None) -> bool:
    """Whether weak separation may prompt a spot check on *media_type*'s detectors (#4799).

    False on :data:`WEAK_CHECK_OFF_MEDIA_TYPES` (face crops); true elsewhere,
    including an unknown type.  The app's :func:`vtscore.state.core.detector_check_prompt`
    and the eval harness's default ``spot_check`` both read it, so the default
    arm stays the app.
    """
    return media_type not in WEAK_CHECK_OFF_MEDIA_TYPES


def weak_check_due(
    separation: float | None,
    n_votes: int,
    votes_at_last_check: int | None,
    *,
    threshold: float = WEAK_SEPARATION_D,
    min_votes: int = WEAK_CHECK_MIN_VOTES,
    cooldown: int | None = WEAK_CHECK_REPEAT,
) -> bool:
    """Whether the labels separate weakly enough that a spot check is due (#4496, the owner's ruling of 2026-10-05).

    Due when the labels line's :attr:`LabelsLine.separation` is below
    *threshold*, at least *min_votes* votes have been cast, and either no
    check has ended yet or *cooldown* votes have been cast since the last one
    ended (*votes_at_last_check*, the vote count when it ended; ``None``
    before any).  ``cooldown=None``, the default (:data:`WEAK_CHECK_REPEAT`,
    #4909), makes a check due once at most: never after any check has ended.

    The app's Autopilot runs the check when it is due; the eval harness's
    default arm (``spot_check="weak"``) runs it where this says, so both read
    this rule.
    A weakly separated session's uniform picks within the line's bands are
    better training votes than the acquisition picks they replace: priced at
    equal clicks, +0.008 to +0.033 at click 20 and +0.023 to +0.049 at 50
    over the three presets
    (``docs/experiments/2026-10-05-weak-check-4496/REPORT.md``).
    """
    if separation is None or not separation < threshold or n_votes < min_votes:
        return False
    if votes_at_last_check is None:
        return True
    return cooldown is not None and n_votes - votes_at_last_check >= cooldown


def done_check_due(n_votes: int, votes_at_last_check: int | None, *, cooldown: int = WEAK_CHECK_COOLDOWN) -> bool:
    """Whether Autopilot's spot check before Done is due: no check has ended in the last *cooldown* votes.

    Autopilot checks the line once a run, the first time every quality light is
    green, and only then reports Done.  The check is skipped when one ended
    fewer than *cooldown* votes ago (*votes_at_last_check*, the vote count when
    it ended; ``None`` before any): a weak-separation check just run, or a
    session re-entered right after its own check, has already measured this
    line.  Unlike :func:`weak_check_due` it reads no separation and runs on
    every media type, face crops included.

    The app's Autopilot reads it (``done_check_due`` beside ``check_due``), and
    so does the eval harness's default arm.
    """
    return votes_at_last_check is None or n_votes - votes_at_last_check >= cooldown


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


__all__ = [
    "HEAD_COUNT_BADS",
    "HEAD_COUNT_GOODS",
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
    "done_check_due",
    "weak_check_due",
    "weak_check_prompts",
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
    "FOLD_SCALES",
    "fold_transfer_clipped",
    "fold_transfer_shift",
    "transfer_fold_orderings",
    "head_count_cut",
    "head_count_serves",
    "labels_line_threshold",
]
