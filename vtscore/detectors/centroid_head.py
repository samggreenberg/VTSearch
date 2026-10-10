"""The Goods' centroid as a detector head: what a labelset gives below the quota (#4643).

Under :mod:`~vtscore.detectors.label_quota`'s quota there are too few labels to
fit a head, and too few Bads for an SVM at all.  So the detector is the sort the
app already draws for several uploaded examples
(:func:`vtscore.training.query_sort.example_sort_from_paths`): the mean of the
L2-normalised Good vectors, every media ranked by its cosine to that centroid
(max-pooled over a patch media's rows, as every scorer here pools), and the line
the two-Gaussian midpoint of those cosines
(:func:`~vtscore.training.thresholds.calculate_gmm_threshold`).

**The line does not take the balance.**  It is the example sort's cut, the one
every non-text cosine sort draws.  The typed query's balance-aware rules
(#4603) were measured on typed queries only, and the trained head's labels line
needs held-out Bads the centroid does not have.  A balance change leaves it
where it is (:func:`~vtscore.state.core.recut_detector_threshold` finds nothing
to re-cut), and a Find on another corpus refits the midpoint there, as a cold
Find refits a trained head's line.

**It is a ``Linear(D, 1)``, so it travels where a head travels.**  The weight is
:data:`CENTROID_LOGIT_SCALE` times the unit centroid and the bias puts the
fitted cut at logit 0, so ``sigmoid(head(x))`` is ``0.5`` exactly on the line
and the threshold is :data:`CENTROID_THRESHOLD`.  Ranking it is ranking by
cosine; every scoring path, the region max-pool, the CLI's weight export and a
portable consumer that applies ``sigmoid(w.x + b) >= threshold`` take it as
they take the linear SVM.  :func:`is_centroid_head` tells the two apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn

#: Logits per unit of cosine.  Any positive value ranks the same and cuts at the
#: same media; this one spreads a typical cosine range (a few tenths either side
#: of the cut) over most of ``(0, 1)`` without saturating a float32 sigmoid,
#: whose top would otherwise tie the best matches at ``1.0``.
CENTROID_LOGIT_SCALE = 10.0

#: The score a centroid head puts on its own line: ``sigmoid(0)``.
CENTROID_THRESHOLD = 0.5

#: The attribute that marks a ``Linear(D, 1)`` as the Goods' centroid.
_MARK = "vtsearch_centroid_goods"


def goods_centroid(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """The unit mean of the L2-normalised *vectors*: every Good weighs the same.

    The same mean :func:`~vtscore.training.query_sort.example_sort_from_paths`
    takes of several uploaded examples, renormalised so a cosine to it is a
    cosine.  Raises :class:`ValueError` on no vectors.
    """
    if not len(vectors):
        raise ValueError("the Goods' centroid needs at least one Good")
    rows = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vectors])
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    rows = np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)
    mean = rows.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return (mean / norm if norm > 0 else mean).astype(np.float32)


def centroid_head(centroid: np.ndarray, cut: float, *, scale: float = CENTROID_LOGIT_SCALE) -> "nn.Sequential":
    """A ``Linear(D, 1)`` scoring ``scale * (cosine - cut)``, marked as a centroid head."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    c = np.asarray(centroid, dtype=np.float32).ravel()
    layer = nn.Linear(c.shape[0], 1)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(c * np.float32(scale)).unsqueeze(0))
        layer.bias.fill_(-float(scale) * float(cut))
    layer.requires_grad_(False)
    head = nn.Sequential(layer)
    setattr(head, _MARK, True)
    return head


def is_centroid_head(model: object) -> bool:
    """Whether *model* is a Goods' centroid head rather than a trained one."""
    return bool(getattr(model, _MARK, False))


#: The rules a centroid's line can be drawn by (#4732), each over the corpus cosines:
#:
#: * ``midpoint`` - the two-Gaussian midpoint, the example sort's line since before #4643;
#: * ``text`` - the typed query's display line at the balance
#:   (:func:`~vtscore.training.thresholds.text_sort_cuts`: the count line at beta 1 or
#:   below, #4603, else the guarded line);
#: * ``guarded`` - the guarded line alone, which takes no balance (#3826);
#: * ``count`` - the count line at every balance, ``beta ** 0.708`` times the bulk's excess
#:   (#4603), the guarded line where the count has nothing to measure.
CENTROID_LINE_RULES = ("midpoint", "text", "guarded", "count")

#: The rule the app draws a centroid's line by.
CENTROID_LINE_RULE = "midpoint"


def centroid_cut(
    cosines: "Sequence[float] | np.ndarray", *, rule: str | None = None, beta: float | None = None
) -> float:
    """The line through the centroid's cosines, drawn by *rule* (default :data:`CENTROID_LINE_RULE`) at *beta*.

    *beta* is the balance (F-beta's beta); the ``midpoint`` and ``guarded`` rules
    ignore it, and ``text`` and ``count`` keep the guarded line without one.
    """
    from vtscore.training.thresholds import (  # noqa: PLC0415
        _count_line,
        calculate_gmm_threshold,
        guarded_text_sort_threshold,
        text_sort_cuts,
    )

    chosen = CENTROID_LINE_RULE if rule is None else rule
    if chosen not in CENTROID_LINE_RULES:
        raise ValueError(f"unknown centroid line rule {chosen!r}; expected one of {CENTROID_LINE_RULES}")
    scores = [float(c) for c in cosines]
    if chosen == "midpoint":
        return float(calculate_gmm_threshold(scores))
    if chosen == "text":
        return float(text_sort_cuts(scores, beta=beta).threshold)
    if chosen == "count" and beta is not None:
        counted = _count_line(np.asarray(scores, dtype=np.float64), float(beta))
        if counted is not None:
            return float(counted)
    return float(guarded_text_sort_threshold(scores)[0])


def _gap_centre(cosines: np.ndarray, line: float) -> float:
    """*line* moved to the middle of the gap it falls in, keeping the same media (``cosine >= line``).

    The count line sits exactly on a media's cosine (#4603's ``_count_line``), and a
    float32 head computes ``scale * (cosine - cut)`` with rounding either side of 0, so
    the media on the line could fall out of the set.  The gap's middle keeps exactly
    the media the line keeps, with the widest margin either side.  A line above or
    below every cosine is returned as it is.
    """
    s = np.sort(np.asarray(cosines, dtype=np.float64))
    i = int(np.searchsorted(s, line, side="left"))
    if 0 < i < s.size:
        return float((s[i - 1] + s[i]) / 2.0)
    return float(line)


@dataclass(frozen=True)
class CentroidLine:
    """A fitted centroid head's line, kept so a balance change can redraw it without a re-score (#4732).

    *cosines* are the corpus cosines the line was drawn on, in the order it was
    drawn on them (the mixture fit subsamples by position); *cut* is the
    cosine the head's bias puts at logit 0; *rule* is the rule that drew it.  The
    head is never rebuilt: a new line is a new threshold on its scores,
    ``sigmoid(scale * (new_cut - cut))``, which keeps exactly the media whose
    cosine clears *new_cut*.
    """

    cosines: np.ndarray
    cut: float
    rule: str
    scale: float = CENTROID_LOGIT_SCALE

    def _threshold_for(self, cut: float) -> float:
        return float(1.0 / (1.0 + np.exp(-self.scale * (cut - self.cut))))

    def threshold_at(self, beta: float | None) -> float:
        """The head's threshold for the line *rule* draws at *beta*, on the same corpus."""
        return self._threshold_for(_gap_centre(self.cosines, centroid_cut(self.cosines, rule=self.rule, beta=beta)))

    def acquisition_threshold(self) -> float:
        """The head's threshold at the cosines' midpoint: where Hard picks sample, whatever the line (#4136)."""
        return self._threshold_for(_gap_centre(self.cosines, centroid_cut(self.cosines, rule="midpoint")))


def fit_centroid(
    goods: Sequence[np.ndarray],
    score: Callable[["nn.Sequential"], Sequence[float]],
    *,
    rule: str | None = None,
    beta: float | None = None,
) -> tuple["nn.Sequential", float, CentroidLine]:
    """The Goods' centroid head, its threshold, and its :class:`CentroidLine`, cut on the corpus *score* scores.

    *score* runs a head over the corpus the line decides and returns one score
    per media, in whatever geometry the caller scores in - the app's
    :func:`~vtscore.detectors.training.score_rows_with_model` over
    :func:`~vtscore.detectors.training.scoring_rows_for_snap`'s rows, or the
    eval harness's test-half scorer.  It is called once, with a probe head at
    unit scale and no cut, whose ``sigmoid(cosine)`` is inverted back to the
    max-pooled cosine; the line is drawn on those, so the cut is the one
    :func:`~vtscore.training.query_sort.cosine_sort_active` draws for the same
    centroid on the same corpus.  Media the scorer could not score (the
    non-finite sentinel, outside ``(0, 1)``) are left out of the fit.
    *rule* and *beta* are :func:`centroid_cut`'s; the cut is centred in the gap
    the line falls in (:func:`_gap_centre`), so the head keeps exactly what the
    line keeps.
    """
    chosen = CENTROID_LINE_RULE if rule is None else rule
    centroid = goods_centroid(goods)
    probe = centroid_head(centroid, 0.0, scale=1.0)
    s = np.asarray(score(probe), dtype=np.float64)
    s = s[(s > 0.0) & (s < 1.0)]
    cosines = np.log(s) - np.log1p(-s)
    cut = _gap_centre(cosines, centroid_cut(cosines, rule=chosen, beta=beta))
    return centroid_head(centroid, cut), CENTROID_THRESHOLD, CentroidLine(cosines, cut, chosen)


def fit_centroid_head(
    goods: Sequence[np.ndarray],
    score: Callable[["nn.Sequential"], Sequence[float]],
    *,
    rule: str | None = None,
    beta: float | None = None,
) -> tuple["nn.Sequential", float]:
    """The Goods' centroid head and its threshold: :func:`fit_centroid` without the line."""
    head, threshold, _line = fit_centroid(goods, score, rule=rule, beta=beta)
    return head, threshold
