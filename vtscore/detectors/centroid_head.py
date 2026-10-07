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


def centroid_cut(cosines: Sequence[float]) -> float:
    """The line through the centroid's cosines: the two-Gaussian midpoint every cosine sort draws."""
    from vtscore.training.thresholds import calculate_gmm_threshold  # noqa: PLC0415

    return float(calculate_gmm_threshold([float(c) for c in cosines]))


def fit_centroid_head(
    goods: Sequence[np.ndarray],
    score: Callable[["nn.Sequential"], Sequence[float]],
) -> tuple["nn.Sequential", float]:
    """The Goods' centroid head and its threshold, cut on the corpus *score* scores.

    *score* runs a head over the corpus the line decides and returns one score
    per media, in whatever geometry the caller scores in - the app's
    :func:`~vtscore.detectors.training.score_rows_with_model` over
    :func:`~vtscore.detectors.training.scoring_rows_for_snap`'s rows, or the
    eval harness's test-half scorer.  It is called once, with a probe head at
    unit scale and no cut, whose ``sigmoid(cosine)`` is inverted back to the
    max-pooled cosine; the midpoint is fitted on those, so the cut is the one
    :func:`~vtscore.training.query_sort.cosine_sort_active` draws for the same
    centroid on the same corpus.  Media the scorer could not score (the
    non-finite sentinel, outside ``(0, 1)``) are left out of the fit.
    """
    centroid = goods_centroid(goods)
    probe = centroid_head(centroid, 0.0, scale=1.0)
    s = np.asarray(score(probe), dtype=np.float64)
    s = s[(s > 0.0) & (s < 1.0)]
    cosines = np.log(s) - np.log1p(-s)
    return centroid_head(centroid, centroid_cut(cosines)), CENTROID_THRESHOLD
