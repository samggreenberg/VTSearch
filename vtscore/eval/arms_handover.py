"""#4776's hand-over evidence: past the label quota, what a rule choosing between the centroid and the head would read.

Below the label quota (#4643) Test gives the Goods' centroid; at the quota it hands
over to the trained head.  On FHIBE's face crops the first head ranks a person
worse than the centroid it replaces (AP -0.17 to -0.22, the face review #4762), so
the hand-over wants a rule.  Neither detector picks: Autopilot's picks come from
the learned sort, which trains the head whatever Test gives
(:func:`~vtscore.detectors.labelset_training.labelset_train_and_score` with
``label_quota=False``).  A rule therefore changes only what Test shows, and one
session prices every rule once each click past the quota also records:

* the centroid's Test, as a row tagged :data:`HANDOVER_CENTROID_TAG`
  (``voting_iterations._handover_centroid_rows``), scored exactly as the
  centroid's rows under the quota are;
* the evidence a rule may read, from the labels and the corpus alone, as the
  app could (:class:`HandoverJudge`, one :data:`HANDOVER_COLUMNS` row per click
  in the ``__handover.csv`` side frame).

**The evidence is held-out Goods' ranks.**  Each Good is held out in turn, and its
rank among the unvoted pool is read under the centroid of the other Goods and
under the head fitted without it (the shipped linear SVM's fit,
:func:`~vtscore.training.svm.train_svm` as
:func:`~vtscore.training.svm.fit_linear_svm_head` calls it).  A Find is judged by
the top of its ranking, and the labels alone cannot show the top: on a face, both
detectors separate a handful of Goods from a handful of Bads perfectly.  The pool
can.  It holds the corpus's negatives (and the few positives not yet found, which
cost both detectors alike).  With one Good there is nothing to hold out, and the
ranks are empty.

**The diagnostic: a head that starts from the centroid.**  The issue's second
candidate.  ``prior_ap_s<S>`` is the withheld half's AP under the linear SVM fitted
to every label with one extra feature, ``S`` times the vector's dot with the unit
Goods' centroid.  The L2 penalty then costs ``1/S**2`` as much along the centroid
as along any other direction, so the head starts at the centroid and leaves it as
the labels pay for.  Ranking only: such a head has no line of its own yet.

Single-vector runs only: the region path max-pools rows a held-out Good would
need its own geometry for.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.label_curve import _average_precision
from vtscore.eval.labels import media_is_positive

#: The ``gmm_variant`` tag of the centroid's Test past the quota.
HANDOVER_CENTROID_TAG = "handover:centroid"

#: The centroid-prior heads' feature scales (the diagnostic above).
PRIOR_SCALES: tuple[float, ...] = (1.0, 3.0, 10.0)

#: The side frame's columns, one row per click Test gives the trained head.
HANDOVER_COLUMNS = [
    "seed",
    "dataset",
    "category",
    "calibration_seed",
    "style",
    "t",
    "phase",
    "n_good",
    "n_bad",
    "n_pool",
    # Space-separated, one per held-out Good in vote order: 1 + the unvoted media scored above it.
    "centroid_ranks",
    "head_ranks",
    *[f"prior_ap_s{s:g}" for s in PRIOR_SCALES],
]


def _svm(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """The shipped head's hyperplane on *X*, *y*: :func:`~vtscore.training.svm.fit_linear_svm_head`'s fit."""
    from vtscore import config  # noqa: PLC0415
    from vtscore.training.svm import train_svm  # noqa: PLC0415

    clf = train_svm(
        X,
        y,
        kernel="linear",
        C=config.SVM_HEAD_C,
        calibration="decision_sigmoid",
        inclusion_value=0,
        seed=42,
        standardize=False,
        backend="sklearn",
    )
    return np.asarray(clf.base.coef_, dtype=np.float64).ravel(), float(np.asarray(clf.base.intercept_).ravel()[0])


def _svm_primal(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """:func:`_svm`'s objective solved in the primal, for the centroid-prior heads.

    The scaled centroid feature makes the dual coordinate descent ``train_svm``
    picks (``dual="auto"`` with fewer labels than dimensions) crawl to its
    iteration cap; the primal Newton solver reaches the same optimum in a few steps.
    """
    from sklearn.svm import LinearSVC  # noqa: PLC0415

    from vtscore import config  # noqa: PLC0415

    clf = LinearSVC(C=config.SVM_HEAD_C, class_weight="balanced", dual=False, max_iter=5000, random_state=42)
    clf.fit(X, y)
    return np.asarray(clf.coef_, dtype=np.float64).ravel(), float(np.asarray(clf.intercept_).ravel()[0])


def _unit_mean(rows: np.ndarray) -> np.ndarray:
    """:func:`~vtscore.detectors.centroid_head.goods_centroid` on a stacked array."""
    from vtscore.detectors.centroid_head import goods_centroid  # noqa: PLC0415

    return np.asarray(goods_centroid(list(rows)), dtype=np.float64)


def held_out_rank(pool_scores: np.ndarray, score: float) -> int:
    """1 + the number of *pool_scores* strictly above *score*: where a held-out Good would sit."""
    return 1 + int(np.count_nonzero(pool_scores > score))


class HandoverJudge:
    """One run's hand-over evidence, a :data:`HANDOVER_COLUMNS` row per click (see the module docstring).

    *sim_ids* is the corpus the session votes in (its unvoted part is the pool),
    *test_ids* the withheld half the diagnostic's AP is read on.  The Good vectors
    are *good_vec*'s, the ones the head and the centroid are fitted on.
    """

    def __init__(
        self,
        clips_dict: dict[int, dict[str, Any]],
        sim_ids: Iterable[int],
        test_ids: Sequence[int],
        target_category: str,
        good_vec: Callable[[dict[str, Any]], np.ndarray],
    ) -> None:
        self._clips = clips_dict
        self._good_vec = good_vec
        ids = sorted(sim_ids)
        self._row = {cid: i for i, cid in enumerate(ids)}
        self._X_sim = np.stack([np.asarray(media_embedding(clips_dict[c]), dtype=np.float32) for c in ids])
        test = list(test_ids)
        self._X_test = (
            np.stack([np.asarray(media_embedding(clips_dict[c]), dtype=np.float32) for c in test]) if test else None
        )
        self._y_test = np.array([media_is_positive(clips_dict[c], target_category) for c in test], dtype=np.int8)

    def row(
        self,
        good_votes: Iterable[int],
        bad_votes: Iterable[int],
        pool: Iterable[int],
    ) -> dict[str, Any]:
        """The evidence at this click: held-out Goods' ranks under each detector, and the prior heads' AP."""
        goods, bads = list(good_votes), list(bad_votes)
        X_good = np.stack([np.asarray(self._good_vec(self._clips[g]), dtype=np.float64) for g in goods])
        X_bad = (
            np.stack([np.asarray(media_embedding(self._clips[b]), dtype=np.float64) for b in bads])
            if bads
            else np.empty((0, X_good.shape[1]))
        )
        unvoted = [self._row[c] for c in pool if c in self._row]
        X_pool = self._X_sim[unvoted].astype(np.float64)
        c_ranks: list[int] = []
        h_ranks: list[int] = []
        if len(goods) >= 2 and len(bads) >= 1:
            for i, g in enumerate(goods):
                rest = np.delete(X_good, i, axis=0)
                x = np.asarray(media_embedding(self._clips[g]), dtype=np.float64)
                centroid = _unit_mean(rest)
                c_ranks.append(held_out_rank(X_pool @ centroid, float(x @ centroid)))
                w, b = _svm(
                    np.vstack([rest, X_bad]),
                    np.concatenate([np.ones(len(rest)), np.zeros(len(X_bad))]),
                )
                h_ranks.append(held_out_rank(X_pool @ w + b, float(x @ w + b)))
        out: dict[str, Any] = {
            "n_good": len(goods),
            "n_bad": len(bads),
            "n_pool": len(unvoted),
            "centroid_ranks": " ".join(str(r) for r in c_ranks),
            "head_ranks": " ".join(str(r) for r in h_ranks),
        }
        out.update(self._prior_aps(X_good, X_bad))
        return out

    def _prior_aps(self, X_good: np.ndarray, X_bad: np.ndarray) -> dict[str, float]:
        """The withheld half's AP under each centroid-prior head (NaN with no Bad or no withheld positive)."""
        keys = [f"prior_ap_s{s:g}" for s in PRIOR_SCALES]
        if self._X_test is None or not len(X_bad) or not self._y_test.any():
            return dict.fromkeys(keys, float("nan"))
        centroid = _unit_mean(X_good)
        X = np.vstack([X_good, X_bad])
        y = np.concatenate([np.ones(len(X_good)), np.zeros(len(X_bad))])
        out: dict[str, float] = {}
        for key, scale in zip(keys, PRIOR_SCALES, strict=True):
            w, _b = _svm_primal(np.hstack([X, scale * (X @ centroid)[:, None]]), y)
            direction = w[:-1] + scale * w[-1] * centroid
            out[key] = round(float(_average_precision(self._X_test.astype(np.float64) @ direction, self._y_test)), 6)
        return out
