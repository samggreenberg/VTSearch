"""#4816's set overlap: where the typed query's set and the trained head's disagree, at every trained click.

A typed-query session keeps the typed query's set on screen until 30 votes on a
single-vector space (#4736), then the head's.  A fixed count captures about a third
of what a per-run choice could, and the judges #4736 could read off the sessions
on disk never beat it: a judge that scores both sets on every vote reads mostly
Hard picks, which sit at the head's own line, where its calls are coin flips.

The judge #4816 prices reads only the votes where the two sets **disagree**.  Over
the voting collection, the head's set ``D`` (its score at or above the line the
step drew there) and the query's ``Q`` (its cosine at or above the display line)
cut the unvoted media into four parts: both, ``D`` only, ``Q`` only and neither.
The votes that fall in each part estimate its share of positives, and the parts'
sizes turn those shares into each set's expected F-beta.  So every such click
records, in the ``__overlap.csv`` side frame (one :data:`OVERLAP_COLUMNS` row):

* the parts' sizes among the unvoted media (``n_d``, ``n_q``, ``n_dq``, of
  ``n_pool``), what a judge reads;
* the positives in each part, from the truth: never a judge's input, but the
  oracle an estimate is checked against;
* the head's class model's expected positives in each part (the labels line's
  posteriors summed), so a judge can ask how far each part's votes sit from what
  the head itself expects there, which is how much the picks' position matters;
* the voted Goods and Bads in each part under this click's sets (in sample: the
  head was fitted to them).

and every pick records, in the pick log, where it sat when it was picked - in
``D`` and ``Q`` at the previous click, its rank by the head's score in the pool
and within its own part, and the head's posterior there
(:data:`PICK_OVERLAP_COLUMNS`, NaN with no trained head yet or the frame off).

The query's display line is drawn as the app draws it over a collection, and as
the text baseline the hold was priced on draws it (``text_baseline.py``'s
``text_line_cut_<b>``): :func:`~vtscore.training.thresholds.gmm.text_sort_threshold`
at the run's beta over every media of the cell.  So ``Q`` here is the set whose
F-beta #4736's pricing reads.

Recorded only: nothing here picks, trains or moves a line, so the session is the
app's.  ``CALIB_SET_OVERLAP=1`` turns it on.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

#: The parts of the unvoted media, in column order.
PARTS = ("dq", "d_only", "q_only", "none")

#: The side frame's columns, one row per click a trained head scored the pool.
OVERLAP_COLUMNS = [
    "seed",
    "dataset",
    "category",
    "calibration_seed",
    "style",
    "t",
    "phase",
    "n_good",
    "n_bad",
    "beta",
    # The head's line over the voting collection (the step's threshold) and the query's display cut (a cosine).
    "head_threshold",
    "query_cut",
    # Among the unvoted media: how many, and how many in D, in Q and in both.
    "n_pool",
    "n_d",
    "n_q",
    "n_dq",
    # The truth: positives in each part of the unvoted media (an oracle, never a judge's input).
    *[f"pos_{p}" for p in PARTS],
    # The head's class model's expected positives in each part (NaN with no class model this click).
    *[f"post_{p}" for p in PARTS],
    # The class model's positives among the unvoted media (the line's share x their count).
    "est_positives",
    # The voted Goods and Bads in each part under this click's sets (in sample).
    *[f"good_{p}" for p in PARTS],
    *[f"bad_{p}" for p in PARTS],
]

#: The pick log's columns this frame fills, at the previous click's sets.
PICK_OVERLAP_COLUMNS: tuple[str, ...] = (
    "picked_in_head",
    "picked_in_query",
    # 0-based: the unvoted media the head scored above the pick, in the pool and in the pick's own part.
    "picked_head_rank",
    "picked_part_rank",
    "picked_head_posterior",
)

_NAN = float("nan")


def pick_overlap_blank() -> dict[str, float]:
    """The pick-log columns of a pick with no sets to read (the frame off, or no trained head yet)."""
    return dict.fromkeys(PICK_OVERLAP_COLUMNS, _NAN)


def _part_index(in_d: np.ndarray, in_q: np.ndarray) -> np.ndarray:
    """Each item's part, as its index in :data:`PARTS`."""
    return np.where(in_d & in_q, 0, np.where(in_d, 1, np.where(in_q, 2, 3)))


def _posteriors(line: Any, scores: np.ndarray) -> np.ndarray | None:
    """The labels line's posterior at each of *scores*, read off its unvoted media; ``None`` with no class model."""
    if line is None or line.unvoted_scores is None or line.unvoted_posteriors is None:
        return None
    s = np.asarray(line.unvoted_scores, dtype=np.float64)
    p = np.asarray(line.unvoted_posteriors, dtype=np.float64)
    if s.size == 0:
        return None
    order = np.argsort(s, kind="stable")
    return np.interp(scores, s[order], p[order])


class SetOverlap:
    """One typed-query run's overlap of the two sets (see the module docstring).

    *query_scores* are the typed query's cosines (every media of the cell), *beta*
    the run's balance, which draws the query's display line; *labels* each voting
    media's truth (1.0 a positive).
    """

    def __init__(self, query_scores: Mapping[int, float], beta: float, labels: Mapping[int, float]) -> None:
        from vtscore.training.thresholds.gmm import text_sort_threshold  # noqa: PLC0415

        self._labels = labels
        self.query_cut = float(text_sort_threshold([float(s) for s in query_scores.values()], beta=float(beta)))
        self._in_q = frozenset(int(c) for c, s in query_scores.items() if float(s) >= self.query_cut)
        # The last click's sets over the pool, which the next picks are read against.
        self._ids: dict[int, int] = {}
        self._scores = np.empty(0)
        self._part = np.empty(0, dtype=np.int64)
        self._post: np.ndarray | None = None
        self._threshold = _NAN

    def in_query(self, cid: int) -> bool:
        return int(cid) in self._in_q

    def click(
        self,
        pool_scores: Mapping[int, float],
        threshold: float,
        line: Any,
        good_votes: Iterable[int],
        bad_votes: Iterable[int],
        voted_scores: Mapping[int, float] | None = None,
    ) -> dict[str, Any]:
        """This click's row (all but the identifying columns), and the sets the next picks are read against.

        *pool_scores* are the head's scores of the unvoted media, *threshold* the
        line it drew over the voting collection and *line* that line's labels line
        (``None`` with no class model).  *voted_scores*, the head's scores of the
        voted media, fill the in-sample columns (NaN without them).
        """
        ids = np.fromiter(pool_scores.keys(), dtype=np.int64, count=len(pool_scores))
        scores = np.fromiter(pool_scores.values(), dtype=np.float64, count=len(pool_scores))
        in_d = scores >= threshold
        in_q = np.fromiter((int(c) in self._in_q for c in ids), dtype=bool, count=ids.size)
        part = _part_index(in_d, in_q)
        y = np.array([float(self._labels.get(int(c), 0.0)) for c in ids], dtype=np.float64)
        post = _posteriors(line, scores)
        self._ids = {int(c): i for i, c in enumerate(ids)}
        self._scores, self._part, self._post, self._threshold = scores, part, post, float(threshold)
        out: dict[str, Any] = {
            "head_threshold": float(threshold),
            "query_cut": self.query_cut,
            "n_pool": int(ids.size),
            "n_d": int(in_d.sum()),
            "n_q": int(in_q.sum()),
            "n_dq": int((in_d & in_q).sum()),
        }
        for k, name in enumerate(PARTS):
            at = part == k
            out[f"pos_{name}"] = int(y[at].sum())
            out[f"post_{name}"] = float(post[at].sum()) if post is not None else _NAN
        out["est_positives"] = (
            float(line.unvoted_share) * int(line.unvoted_scores.size)
            if line is not None and line.unvoted_share is not None and line.unvoted_scores is not None
            else _NAN
        )
        for kind, votes in (("good", good_votes), ("bad", bad_votes)):
            counts = dict.fromkeys(PARTS, 0)
            ok = voted_scores is not None
            for cid in votes:
                if voted_scores is None or cid not in voted_scores:
                    ok = False
                    break
                k = int(_part_index(np.array([voted_scores[cid] >= threshold]), np.array([self.in_query(cid)]))[0])
                counts[PARTS[k]] += 1
            for name in PARTS:
                out[f"{kind}_{name}"] = counts[name] if ok else _NAN
        return out

    def pick(self, cid: int) -> dict[str, float]:
        """Where *cid* sat at the last click: its sets, its ranks by the head's score, the head's posterior."""
        i = self._ids.get(int(cid))
        if i is None:
            return {**pick_overlap_blank(), "picked_in_query": float(self.in_query(cid))}
        s = self._scores[i]
        above = self._scores > s
        part = self._part[i]
        return {
            "picked_in_head": float(s >= self._threshold),
            "picked_in_query": float(self.in_query(cid)),
            "picked_head_rank": float(np.count_nonzero(above)),
            "picked_part_rank": float(np.count_nonzero(above & (self._part == part))),
            "picked_head_posterior": float(self._post[i]) if self._post is not None else _NAN,
        }


__all__ = ["OVERLAP_COLUMNS", "PARTS", "PICK_OVERLAP_COLUMNS", "SetOverlap", "pick_overlap_blank"]
