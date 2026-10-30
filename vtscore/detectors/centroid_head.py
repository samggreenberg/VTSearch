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

import re
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


def goods_centroid(vectors: Sequence[np.ndarray], weights: "Sequence[float] | np.ndarray | None" = None) -> np.ndarray:
    """The unit mean of the L2-normalised *vectors*: every Good weighs the same, or its *weights* entry.

    The same mean :func:`~vtscore.training.query_sort.example_sort_from_paths`
    takes of several uploaded examples, renormalised so a cosine to it is a
    cosine.  *weights* (one per vector) is the trained head's weighting of a
    hard-view face (:func:`vtscore.training.svm_prior.hard_view_good_weights`, #5413).
    Raises :class:`ValueError` on no vectors.
    """
    if not len(vectors):
        raise ValueError("the Goods' centroid needs at least one Good")
    rows = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vectors])
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    rows = np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)
    mean = (
        rows.mean(axis=0)
        if weights is None
        else np.average(rows, axis=0, weights=np.asarray(weights, dtype=np.float64))
    )
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
#:
#: and #5325's, the app's rule with the bulk's excess read above a higher bar (:data:`CENTROID_BARS`):
#: on imagekit's face cosines the bulk's upper tail is far heavier than a Gaussian's (6-8 of a
#: person's impostors above 4 spreads, where a Gaussian bulk puts 0.3), so ``count`` reads them as
#: the person:
#:
#: * ``count_gap_floor_z<bar>`` / ``count_gap_floor_skip25_z<bar>`` - those rules' count above
#:   ``<bar>`` spreads; ``count_gap_floor_skip25_b1_z<bar>`` gates them as the app's rule does.
CENTROID_BAR_ZS = (5.0, 5.5, 6.0)
CENTROID_BARS = {f"{rule}_z{z:g}": z for z in CENTROID_BAR_ZS for rule in ("count_gap_floor", "count_gap_floor_skip25")}
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
    *CENTROID_BARS,
)
#: The drop each gap-counting rule reads (:func:`_top_gap_count`); ``widest`` unless named here.
CENTROID_DROPS = {
    "count_gap_floor_skip25": "skip25",
    "count_gap_floor_skip50": "skip50",
    "count_gap_floor_rank5": "rank5",
    **{rule: "skip25" for rule in CENTROID_BARS if rule.startswith("count_gap_floor_skip25_")},
}
#: The counts whose line never sits above their drop (:func:`centroid_cut`).
_GAP_FLOORED = ("gap_floor", "count_gap_floor", *CENTROID_DROPS, *CENTROID_BARS)
#: Rules gated on the balance: ``rule -> (below beta 1, from beta 1 up)`` (#4783, #5325).
CENTROID_GATED = {
    "count_gap_floor_skip25_b1": ("count_gap_floor", "count_gap_floor_skip25"),
    **{
        f"count_gap_floor_skip25_b1_z{z:g}": (f"count_gap_floor_z{z:g}", f"count_gap_floor_skip25_z{z:g}")
        for z in CENTROID_BAR_ZS
    },
}
#: #5478's face line (:func:`face_ball_cut`): the ball from beta 1 up once there are two Goods, else
#: :data:`FACE_BALL_BELOW`.
FACE_BALL_RULE = "face_ball"
CENTROID_LINE_RULES = ("midpoint", "text", "guarded", *CENTROID_COUNTS, *CENTROID_GATED, FACE_BALL_RULE)

#: #5478's ball, a measurement rule written with its numbers rather than listed: ``ball<floor>`` keeps
#: every cosine at or above ``<floor>`` whatever the corpus, and ``ball<floor>_s<slope>_c<centre>`` moves
#: the floor by ``slope * (spread - centre)``, the spread being how close each Good sits to the centroid of
#: the others (:func:`goods_spread`).  On FHIBE's buffalo_s faces the centroid of a person's four starting
#: photos and one floor per balance scored level with the 150-vote session
#: (``docs/experiments/2026-10-30-face-balls-5474``).  The balance takes no part: a ball priced at another
#: beta is another floor.
_BALL = re.compile(r"ball(?P<floor>-?\d+(?:\.\d+)?)(?:_s(?P<slope>-?\d+(?:\.\d+)?)_c(?P<centre>-?\d+(?:\.\d+)?))?")


def ball_rule(rule: str) -> tuple[float, float, float] | None:
    """``(floor, slope, centre)`` of a #5478 ball rule (slope 0 without a spread term), or ``None`` for any other."""
    m = _BALL.fullmatch(rule)
    if m is None:
        return None
    return float(m["floor"]), float(m["slope"] or 0.0), float(m["centre"] or 0.0)


def is_centroid_line_rule(rule: str) -> bool:
    """Whether *rule* draws a centroid line: one of :data:`CENTROID_LINE_RULES`, or a ball (:func:`ball_rule`)."""
    return rule in CENTROID_LINE_RULES or ball_rule(rule) is not None


def goods_spread(vectors: Sequence[np.ndarray]) -> float | None:
    """The mean cosine of each Good to the unit centroid of the others, or ``None`` under two Goods (#5474)."""
    if len(vectors) < 2:
        return None
    rows = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vectors])
    rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    rest = (rows.sum(axis=0)[None, :] - rows) / (len(rows) - 1)
    rest /= np.maximum(np.linalg.norm(rest, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(rows * rest, axis=1)))


#: The face crops' ball (#5478): the Goods' centroid keeps every face whose cosine to it is at least a floor,
#: moved by :data:`FACE_BALL_SLOPE` times how far the Goods' spread (:func:`goods_spread`) sits from
#: :data:`FACE_BALL_CENTRE`: a person whose own photos are spread out gets a lower floor.  The floors are per
#: balance (beta 1 and 4; between them, and past 4, read off a line in log beta, held at beta 1 below it), fitted
#: on FHIBE's buffalo_s faces (#5474's ``adaptive`` rule over both crop sizes and two to every Good).  Replayed on
#: the shipped sessions (6,743, the review's 562 people), Find's set from beta 1 up with two Goods or more
#: gained +0.010 to +0.012 F-beta over votes 1-150 at beta 1 from four photos and +0.016 to +0.019 at beta 4,
#: less for she/her people than for he/him at beta 1 (+0.005 against +0.015).  At beta 1/4 the ball lost 0.007 to
#: 0.012, and from one photo it is a poor centre, so both keep :data:`FACE_BALL_BELOW`
#: (``docs/experiments/2026-10-30-face-ball-line-5478``).  The floors are cosines in buffalo_s's space: another
#: face pack needs its own.
FACE_BALL_FLOORS: dict[float, float] = {1.0: 0.4425, 4.0: 0.385}
FACE_BALL_SLOPE = 0.25
FACE_BALL_CENTRE = 0.6371
#: The face crops' line where the ball does not hold: #5325's count read above z 5.5.
FACE_BALL_BELOW = "count_gap_floor_skip25_b1_z5.5"


def face_ball_holds(beta: float | None, n_goods: int) -> bool:
    """Whether the face crops' line is the ball (#5478): a balance of 1 or more, and two Goods or more."""
    return beta is not None and beta >= 1.0 and n_goods >= 2


def face_ball_floor(beta: float, spread: float) -> float:
    """The ball's cosine floor at *beta* for Goods of *spread* (:data:`FACE_BALL_FLOORS`, moved by the spread)."""
    lo, hi = min(FACE_BALL_FLOORS), max(FACE_BALL_FLOORS)
    x = (np.log(max(beta, lo)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    floor = FACE_BALL_FLOORS[lo] + x * (FACE_BALL_FLOORS[hi] - FACE_BALL_FLOORS[lo])
    return float(floor + FACE_BALL_SLOPE * (spread - FACE_BALL_CENTRE))


def needs_spread(rule: str | None) -> bool:
    """Whether *rule*'s line reads the Goods' spread: a ball, or the face crops' line."""
    return rule is not None and (rule == FACE_BALL_RULE or ball_rule(rule) is not None)


#: The rule the app draws a centroid's line by: the larger count, floored at the widest drop (#4759),
#: and from beta 1 up at the drop below a standout first cosine (#4783).
CENTROID_LINE_RULE = "count_gap_floor_skip25_b1"
#: The rule a media type's centroid line is drawn by where it is not :data:`CENTROID_LINE_RULE` (#5325).
#: Face crops count what stands out at z 5.5 instead of 4: ArcFace's cosines carry about 6 look-alikes per
#: session above z 4, where a Gaussian bulk puts 0.3, and the count read them as the person (one photo,
#: beta 1: 6.4 faces kept for 3.8 at click 0, F 0.73; at z 5.5, 3.5 and 0.85).  Shared, the same bar cut
#: COCO Better's click 0 by 0.06-0.15, so it is the face crops' alone
#: (``docs/experiments/2026-10-27-face-lines-5325/REPORT.md``).
#: From #5478 the face crops' line is the ball from beta 1 up once there are two Goods (:data:`FACE_BALL_RULE`),
#: and that count below it.
CENTROID_LINE_RULE_BY_MEDIA_TYPE: dict[str, str] = {"face": FACE_BALL_RULE}


def centroid_line_rule(media_type: str | None) -> str:
    """The rule the app draws *media_type*'s centroid line by: its own (:data:`CENTROID_LINE_RULE_BY_MEDIA_TYPE`),
    else :data:`CENTROID_LINE_RULE`, including for an unknown type.  The app's example sort and centroid head and
    the eval harness's default centroid line all read it, so the default arm stays the app."""
    return CENTROID_LINE_RULE_BY_MEDIA_TYPE.get(media_type or "", CENTROID_LINE_RULE)


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
    their own drop, and #5325's, with their own bar) as ``count_gap``; :func:`centroid_cut` lowers
    their lines to the gap.
    """
    from vtscore.training.thresholds import _count_estimate  # noqa: PLC0415

    if rule == "count_z3":
        return _count_estimate(cosines, z=3.0)
    if rule == "count_z3.5":
        return _count_estimate(cosines, z=3.5)
    if rule == "mirror_z3.5":
        return _count_estimate(cosines, z=3.5, bulk="mirror")
    counted = _count_estimate(cosines, z=CENTROID_BARS.get(rule))
    if rule in ("count", "gap_floor") or counted is None:
        return counted
    gap = float(_top_gap_count(cosines, drop=CENTROID_DROPS.get(rule, "widest")))
    return gap if rule == "gap" else max(gap, counted)


def _ball_cut(rule: str, beta: float | None, spread: float | None) -> float | None:
    """A ball's floor (#5478): :func:`ball_rule`'s, or the face crops' where it holds; ``None`` for a line drawn
    over the cosines (every other rule, and the face crops' below beta 1 or with one Good)."""
    if rule == FACE_BALL_RULE:
        return face_ball_floor(float(beta), spread) if spread is not None and face_ball_holds(beta, 2) else None
    ball = ball_rule(rule)
    if ball is None:
        return None
    floor, slope, centre = ball
    return floor + (slope * (spread - centre) if spread is not None else 0.0)


def centroid_cut(
    cosines: "Sequence[float] | np.ndarray",
    *,
    rule: str | None = None,
    beta: float | None = None,
    spread: float | None = None,
) -> float:
    """The line through the centroid's cosines, drawn by *rule* (default :data:`CENTROID_LINE_RULE`) at *beta*.

    *beta* is the balance (F-beta's beta); the ``midpoint`` and ``guarded`` rules
    ignore it, and ``text`` and the counts keep the guarded line without one.
    A ball (:func:`ball_rule`) is its floor, moved by *spread* (:func:`goods_spread`)
    when the rule has a slope; without a spread (one Good) it is the floor.  The face
    crops' line (:data:`FACE_BALL_RULE`) is the ball (:func:`face_ball_floor`) from beta 1
    up with a spread, else :data:`FACE_BALL_BELOW` over the cosines.
    """
    from vtscore.training.thresholds import (  # noqa: PLC0415
        _count_keep_line,
        calculate_gmm_threshold,
        guarded_text_sort_threshold,
        text_sort_cuts,
    )

    chosen = CENTROID_LINE_RULE if rule is None else rule
    ball = _ball_cut(chosen, beta, spread)
    if ball is not None:
        return ball
    if chosen == FACE_BALL_RULE:
        chosen = FACE_BALL_BELOW
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
    #: The Goods' spread a ball's floor moves by (:func:`goods_spread`); ``None`` for every other rule.
    spread: float | None = None

    def _threshold_for(self, cut: float) -> float:
        return float(1.0 / (1.0 + np.exp(-self.scale * (cut - self.cut))))

    def threshold_at(self, beta: float | None) -> float:
        """The head's threshold for the line *rule* draws at *beta*, on the same corpus."""
        line = centroid_cut(self.cosines, rule=self.rule, beta=beta, spread=self.spread)
        return self._threshold_for(_gap_centre(self.cosines, line))

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
    spread = goods_spread(goods) if needs_spread(chosen) else None
    cut = _gap_centre(cosines, centroid_cut(cosines, rule=chosen, beta=beta, spread=spread))
    return centroid_head(centroid, cut), CENTROID_THRESHOLD, CentroidLine(cosines, cut, chosen, spread=spread)


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
