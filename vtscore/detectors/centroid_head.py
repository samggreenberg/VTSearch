"""The Goods' centroid as a detector head: what a labelset gives below the quota (#4643).

Under :mod:`~vtscore.detectors.label_quota`'s quota there are too few labels to
fit a head, and too few Bads for an SVM at all.  So the detector is the sort the
app already draws for several uploaded examples
(:func:`vtscore.training.query_sort.example_sort_from_paths`): the mean of the
L2-normalised Good vectors, every media ranked by its cosine to that centroid
(max-pooled over a patch media's rows, as every scorer here pools), and the line
:func:`centroid_cut` draws over those cosines.

**The line is a count of what stands out, at the balance (#4732, #4759).**  It
keeps ``beta ** 0.708`` times the larger of two counts: the media standing out of
the cosines' bulk (#4603's count) and the media above the widest drop between
neighbours in the top 11.  It never keeps fewer than that drop's count, whatever the
balance; the guarded line stays where there is nothing to count.  The midpoint it
replaced (#4732) split a rare target's negatives: on FHIBE's face crops the centroid
ranked a person's withheld photos first of 5,439, and the midpoint kept about 2,300.
The typed query's count alone (#4732's line) kept one of a person's ~4, because
their photos sit 3 to 4 bulk sigmas up, among the bulk's own top; a lower bar or a
mirrored bulk only subtracts more.  The drop just below the person's last photo is
what separates them.  Priced on the FHIBE cells and the State of the App's Binary
opening (``docs/experiments/2026-10-10-centroid-count-4759``).  The trained head's
labels line needs held-out Bads the centroid does not have.  A balance change
redraws the line on the cosines it was drawn on (:class:`CentroidLine`,
:func:`~vtscore.state.core.recut_detector_threshold`), Autopilot's Hard select
still samples at the cosines' midpoint (:meth:`CentroidLine.acquisition_threshold`),
and a Find on another corpus redraws the line there, as a cold Find refits a
trained head's line.

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
#: * ``midpoint`` - the two-Gaussian midpoint, the centroid's and the example sort's line until #4732;
#: * ``text`` - the typed query's display line at the balance
#:   (:func:`~vtscore.training.thresholds.text_sort_cuts`: the count line at beta 1 or
#:   below, #4603, else the guarded line);
#: * ``guarded`` - the guarded line alone, which takes no balance (#3826);
#: * ``count`` - the count line at every balance, ``beta ** 0.708`` times the bulk's excess
#:   (#4603), the guarded line where the count has nothing to measure.  The app's from #4732
#:   to #4759.
#:
#: and #4759's candidates, each ``beta ** 0.708`` times another count of what stands out
#: (:data:`CENTROID_COUNTS`), the guarded line where it has nothing to measure:
#:
#: * ``count_z3`` / ``count_z3.5`` - the Gaussian bulk's excess above a lower bar;
#: * ``mirror_z3.5`` - the excess above the bar over the count as far below the median;
#: * ``gap`` - the cosines above the widest drop between neighbours in the top
#:   :data:`CENTROID_GAP_TOP` + 1;
#: * ``count_gap`` - the larger of ``count``'s and ``gap``'s;
#: * ``gap_floor`` - ``count``'s line, lowered to the widest drop when that keeps more: the gap
#:   as a floor on the set at every balance, not a count scaled by it;
#: * ``count_gap_floor`` - ``count_gap``'s line, lowered to the widest drop when that keeps more.
#:   It keeps exactly what ``gap_floor`` keeps at beta 1 or below and what ``count_gap`` keeps
#:   from beta 1 up, since ``beta ** 0.708`` is at least 1 there.  The app's (#4759).
#:
#: and #4783's, ``count_gap_floor`` with another drop (:data:`CENTROID_DROPS`) where the widest
#: one stops after a standout first cosine:
#:
#: * ``count_gap_floor_skip25`` / ``_skip50`` - when the widest drop is the first, the widest one
#:   below the second cosine, if it is at least 1/4 (1/2) as wide;
#: * ``count_gap_floor_rank5`` - the drop that is widest once each is weighted by its rank
#:   (``i * d_i``, which evens out a bulk tail's spacings), over the top 6;
#:
#: and one rule gated on the balance (:data:`CENTROID_GATED`):
#:
#: * ``count_gap_floor_skip25_b1`` - ``count_gap_floor`` below beta 1, ``count_gap_floor_skip25``
#:   from beta 1 up.  At a precision balance a lone standout is the set; from beta 1 up the
#:   line looks past it.  The app's (#4783).
CENTROID_COUNTS = (
    "count",
    "count_z3",
    "count_z3.5",
    "mirror_z3.5",
    "gap",
    "count_gap",
    "gap_floor",
    "count_gap_floor",
    "count_gap_floor_skip25",
    "count_gap_floor_skip50",
    "count_gap_floor_rank5",
)
#: The drop each gap-counting rule reads (:func:`_top_gap_count`); ``widest`` unless named here.
CENTROID_DROPS = {
    "count_gap_floor_skip25": "skip25",
    "count_gap_floor_skip50": "skip50",
    "count_gap_floor_rank5": "rank5",
}
#: The counts whose line never sits above their drop (:func:`centroid_cut`).
_GAP_FLOORED = ("gap_floor", "count_gap_floor", *CENTROID_DROPS)
#: Rules gated on the balance: ``rule -> (below beta 1, from beta 1 up)`` (#4783).
CENTROID_GATED = {"count_gap_floor_skip25_b1": ("count_gap_floor", "count_gap_floor_skip25")}
CENTROID_LINE_RULES = ("midpoint", "text", "guarded", *CENTROID_COUNTS, *CENTROID_GATED)

#: The rule the app draws a centroid's line by: the larger count, floored at the widest drop (#4759),
#: and from beta 1 up at the drop below a standout first cosine (#4783).
CENTROID_LINE_RULE = "count_gap_floor_skip25_b1"

#: How deep ``gap`` looks for the widest drop: the top this many + 1 cosines (#4759).
CENTROID_GAP_TOP = 10


def _top_gap_count(cosines: np.ndarray, top: int = CENTROID_GAP_TOP, drop: str = "widest") -> int:
    """How many of the top ``top + 1`` *cosines* sit above *drop* between neighbours.

    ``widest`` is the widest drop (#4759).  ``skip25`` / ``skip50`` take, when the widest is the
    first, the widest drop below the second cosine if it is at least 1/4 (1/2) as wide (#4783).
    ``rank5`` takes the widest of the first five drops once each is weighted by its rank.
    """
    head = -np.sort(-cosines)[: top + 1]
    if head.size < 2:
        return int(head.size)
    d = head[:-1] - head[1:]
    widest = int(np.argmax(d))
    if drop == "widest":
        return widest + 1
    if drop in ("skip25", "skip50"):
        if widest != 0 or d.size < 2:
            return widest + 1
        below = int(np.argmax(d[1:])) + 1
        return below + 1 if d[below] >= (0.25 if drop == "skip25" else 0.5) * d[0] else 1
    if drop == "rank5":
        first = d[:5]
        return int(np.argmax(first * np.arange(1, first.size + 1))) + 1
    raise ValueError(f"unknown drop {drop!r}")


def _centroid_count(cosines: np.ndarray, rule: str) -> float | None:
    """*rule*'s count of the cosines that stand out (:data:`CENTROID_COUNTS`), or ``None`` with nothing to measure.

    ``gap_floor`` counts as ``count`` does and ``count_gap_floor`` (and #4783's variants, with
    their own drop) as ``count_gap``; :func:`centroid_cut` lowers their lines to the gap.
    """
    from vtscore.training.thresholds import _count_estimate  # noqa: PLC0415

    if rule == "count_z3":
        return _count_estimate(cosines, z=3.0)
    if rule == "count_z3.5":
        return _count_estimate(cosines, z=3.5)
    if rule == "mirror_z3.5":
        return _count_estimate(cosines, z=3.5, bulk="mirror")
    counted = _count_estimate(cosines)
    if rule in ("count", "gap_floor") or counted is None:
        return counted
    gap = float(_top_gap_count(cosines, drop=CENTROID_DROPS.get(rule, "widest")))
    return gap if rule == "gap" else max(gap, counted)


def centroid_cut(
    cosines: "Sequence[float] | np.ndarray", *, rule: str | None = None, beta: float | None = None
) -> float:
    """The line through the centroid's cosines, drawn by *rule* (default :data:`CENTROID_LINE_RULE`) at *beta*.

    *beta* is the balance (F-beta's beta); the ``midpoint`` and ``guarded`` rules
    ignore it, and ``text`` and the counts keep the guarded line without one.
    """
    from vtscore.training.thresholds import (  # noqa: PLC0415
        _count_keep_line,
        calculate_gmm_threshold,
        guarded_text_sort_threshold,
        text_sort_cuts,
    )

    chosen = CENTROID_LINE_RULE if rule is None else rule
    if chosen not in CENTROID_LINE_RULES:
        raise ValueError(f"unknown centroid line rule {chosen!r}; expected one of {CENTROID_LINE_RULES}")
    if chosen in CENTROID_GATED:
        below, above = CENTROID_GATED[chosen]
        chosen = above if beta is not None and beta >= 1.0 else below
    scores = [float(c) for c in cosines]
    if chosen == "midpoint":
        return float(calculate_gmm_threshold(scores))
    if chosen == "text":
        return float(text_sort_cuts(scores, beta=beta).threshold)
    if chosen in CENTROID_COUNTS and beta is not None:
        arr = np.asarray(scores, dtype=np.float64)
        counted = _centroid_count(arr, chosen)
        if counted is not None:
            line = _count_keep_line(arr, float(beta), counted)
            if chosen in _GAP_FLOORED:
                drop = CENTROID_DROPS.get(chosen, "widest")
                line = min(line, float(-np.sort(-arr)[_top_gap_count(arr, drop=drop) - 1]))
            return line
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
    line_mask: "Sequence[bool] | np.ndarray | None" = None,
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

    *line_mask*, one flag per score, keeps only the flagged media in the line's
    count (#4779): the app passes the unvoted ones, since the Goods it was built
    from sit at the top of their own ranking and would otherwise be counted as
    what stands out.  The head still scores every media.
    """
    chosen = CENTROID_LINE_RULE if rule is None else rule
    centroid = goods_centroid(goods)
    probe = centroid_head(centroid, 0.0, scale=1.0)
    s = np.asarray(score(probe), dtype=np.float64)
    if line_mask is not None:
        mask = np.asarray(line_mask, dtype=bool)
        if mask.shape != s.shape:
            raise ValueError(f"line_mask has {mask.size} flags for {s.size} scores")
        s = s[mask]
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
    line_mask: "Sequence[bool] | np.ndarray | None" = None,
) -> tuple["nn.Sequential", float]:
    """The Goods' centroid head and its threshold: :func:`fit_centroid` without the line."""
    head, threshold, _line = fit_centroid(goods, score, rule=rule, beta=beta, line_mask=line_mask)
    return head, threshold
