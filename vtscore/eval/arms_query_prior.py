"""#4384's typed-query prior: the head started from the typed query as well as from the Goods.

On a typed-query detector the first trained heads rank the corpus worse than
the query they replace.  #4736 holds the query's set on screen until 30 labels
because the head's set only passes it around vote 28 (vote 45 at beta 4), and
the head started from the Goods' centroid (#4785) did not close the gap.  The
query is the better prior: three Goods off the top of its own sort say less
about the class than the text that found them.

Since #4859 the app ships it: a typed-query detector on its query's own
single-vector space fits every head within
:func:`vtscore.training.svm_prior.typed_query_prior`, and so does the harness on
a typed-query run given the query's vector.  Two pieces here:

* :func:`query_prior_scale`, the arm.  It holds
  :data:`~vtscore.training.svm_prior.QUERY_PRIOR_SCALE` at another scale for a
  run; ``0`` is the head before #4859 (the Goods' centroid alone).
  ``CALIB_QUERY_PRIOR=<scale>``, unset the app's 10.
* :class:`QueryPriorShadow`, the screen.  At every click a trained head is
  drawn, it refits a few candidate heads on the same votes and records where
  the withheld half's positives sit in each one's ranking (one
  :data:`QUERY_PRIOR_COLUMNS` row in the ``__queryprior.csv`` side frame).  AP
  and the F-beta of any top *K* follow from the ranks.  The candidates:

  - ``head``: the centroid's feature alone, at
    :data:`~vtscore.training.svm_prior.CENTROID_PRIOR_SCALE` (the app's head
    before #4859; the run's own head when the prior is off);
  - ``query``: the typed query's sort;
  - ``q<S>``: the SVM with the query's feature at scale ``S`` and no centroid;
  - ``cq<S>``: the centroid plus the query's feature at scale ``S`` (``cq10`` is
    the app's head since #4859);
  - ``blend<A>``: the centroid-only head's score and the query's, each divided by
    its spread over the voting half, mixed with weight ``A`` on the query
    (#4384's second arm, a fixed blend).

  Recorded only: the session is the app's.  ``CALIB_QUERY_PRIOR_SHADOW=1``.

The query's feature spreads about a third as widely over SigLIP's images as the
centroid's does (0.022 against 0.061 on COCO Better), so a scale buys less
there than the same scale on the centroid.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.labels import media_is_positive

#: The query's feature scales beside the centroid (``cq<S>``).
QUERY_PRIOR_SCALES: tuple[float, ...] = (3.0, 10.0, 30.0, 100.0)

#: The query's feature scales alone (``q<S>``).
QUERY_ONLY_SCALES: tuple[float, ...] = (10.0, 30.0)

#: The query's weight in the fixed blend (``blend<A>``).
BLEND_WEIGHTS: tuple[float, ...] = (0.25, 0.5, 0.75)

#: Every candidate the shadow ranks the withheld half by, in column order.
VARIANTS: tuple[str, ...] = (
    "head",
    "query",
    *[f"q{s:g}" for s in QUERY_ONLY_SCALES],
    *[f"cq{s:g}" for s in QUERY_PRIOR_SCALES],
    *[f"blend{a:g}" for a in BLEND_WEIGHTS],
)

#: The side frame's columns, one row per click a trained head is drawn.
QUERY_PRIOR_COLUMNS = [
    "seed",
    "dataset",
    "category",
    "calibration_seed",
    "style",
    "t",
    "phase",
    "n_good",
    "n_bad",
    "n_test",
    "n_test_pos",
    # Space-separated 0-based ranks of the withheld positives (score descending, ties by id).
    *[f"ranks_{v}" for v in VARIANTS],
]


@contextmanager
def query_prior_scale(scale: float | None) -> Iterator[None]:
    """Hold :data:`~vtscore.training.svm_prior.QUERY_PRIOR_SCALE` at *scale* while the block runs.

    ``None`` changes nothing (the app's); ``0`` turns the typed query's prior off,
    the head before #4859.  Process-wide, as
    :func:`~vtscore.eval.live_threshold_rules.head_count_bads` is, which suits a
    task that runs one cell, and undone on the way out.
    """
    if scale is None:
        yield
        return
    if not np.isfinite(scale) or scale < 0:
        raise ValueError(f"query_prior_scale must be a number >= 0; got {scale!r}")
    from vtscore.training import svm_prior  # noqa: PLC0415

    shipped = svm_prior.QUERY_PRIOR_SCALE
    svm_prior.QUERY_PRIOR_SCALE = float(scale)
    try:
        yield
    finally:
        svm_prior.QUERY_PRIOR_SCALE = shipped


def positive_ranks(ids: Sequence[int], scores: np.ndarray, positive: set[int]) -> str:
    """The 0-based ranks of *positive* in *ids* by *scores*, descending, ties by id.

    A :class:`~vtscore.training.thresholds.LineRanking`'s order, on raw margins:
    that class keeps only sigmoid-range scores.
    """
    id_arr = np.asarray(list(ids), dtype=np.int64)
    order = np.lexsort((id_arr, -np.asarray(scores, dtype=np.float64)))
    return " ".join(str(r) for r, cid in enumerate(id_arr[order]) if int(cid) in positive)


class QueryPriorShadow:
    """One run's candidate heads, a :data:`QUERY_PRIOR_COLUMNS` row per click (see the module docstring).

    *sim_ids* is the voting half (the blend's spreads are read there), *test_ids*
    the withheld half the ranks are read on, *query_vector* the typed query's
    embedding in the media's space.  The Good vectors are *good_vec*'s, the ones
    the head is fitted on.
    """

    def __init__(
        self,
        clips_dict: dict[int, dict[str, Any]],
        sim_ids: Iterable[int],
        test_ids: Sequence[int],
        target_category: str,
        query_vector: np.ndarray,
        good_vec: Callable[[dict[str, Any]], np.ndarray],
    ) -> None:
        self._clips = clips_dict
        self._good_vec = good_vec
        q = np.asarray(query_vector, dtype=np.float64).ravel()
        self._q = q / np.linalg.norm(q)
        self._X_sim = np.stack([np.asarray(media_embedding(clips_dict[c]), dtype=np.float64) for c in sorted(sim_ids)])
        self._test_ids = [int(c) for c in test_ids]
        self._X_test = np.stack([np.asarray(media_embedding(clips_dict[c]), dtype=np.float64) for c in self._test_ids])
        self._positive = {c for c in self._test_ids if media_is_positive(clips_dict[c], target_category)}
        self._q_sd = float(np.std(self._X_sim @ self._q)) or 1.0
        self._query_ranks = positive_ranks(self._test_ids, self._X_test @ self._q, self._positive)

    def row(self, good_votes: Iterable[int], bad_votes: Iterable[int]) -> dict[str, Any]:
        """The withheld positives' ranks under every candidate fitted on these votes (empty without both classes)."""
        from vtscore.training.svm_prior import centroid_prior_direction  # noqa: PLC0415

        goods, bads = list(good_votes), list(bad_votes)
        out: dict[str, Any] = {
            "n_good": len(goods),
            "n_bad": len(bads),
            "n_test": len(self._test_ids),
            "n_test_pos": len(self._positive),
            **{f"ranks_{v}": "" for v in VARIANTS},
            "ranks_query": self._query_ranks,
        }
        if not goods or not bads:
            return out
        X = np.vstack(
            [np.asarray(self._good_vec(self._clips[g]), dtype=np.float64) for g in goods]
            + [np.asarray(media_embedding(self._clips[b]), dtype=np.float64) for b in bads]
        )
        y = np.r_[np.ones(len(goods)), np.zeros(len(bads))]
        # An explicit zero-scale query keeps a run under the arm from leaking it into the shipped fit.
        off = (self._q, 0.0)
        heads: dict[str, np.ndarray] = {"head": centroid_prior_direction(X, y, query=off)[0]}
        for s in QUERY_ONLY_SCALES:
            heads[f"q{s:g}"] = centroid_prior_direction(X, y, scale=0.0, query=(self._q, s))[0]
        for s in QUERY_PRIOR_SCALES:
            heads[f"cq{s:g}"] = centroid_prior_direction(X, y, query=(self._q, s))[0]
        for name, w in heads.items():
            out[f"ranks_{name}"] = positive_ranks(self._test_ids, self._X_test @ w, self._positive)
        w = heads["head"]
        h_sd = float(np.std(self._X_sim @ w)) or 1.0
        for a in BLEND_WEIGHTS:
            blended = (1.0 - a) * (self._X_test @ w) / h_sd + a * (self._X_test @ self._q) / self._q_sd
            out[f"ranks_blend{a:g}"] = positive_ranks(self._test_ids, blended, self._positive)
        return out
