"""A precision floor's line, read off where the positives sit in a ranking (#4357).

A floor *P*'s line keeps a **set**: the top *K* of the ranking, where *K* is the
floor's unchecked starting candidate (the top 128 at 10%, 64 at 25%, 32 at 50%
and above; #4272; ``analyze_line_estimate_4383.floor_schedule``).  So how good the line is,
and how good the best cut on the same ranking could have been, depends on the
positives' ranks and on nothing else.  That is what a rank frame records
(``task_NNNN__rankframes.csv``, ``vtscore.eval.voting_columns.RANK_FRAME_COLUMNS``)
and what ``text_baseline.py`` builds for the typed query, so the click-0 sort
and the clicked detector are read by one definition.

Numpy only; *ranks* are 0-based, best first, in the order
:class:`~vtscore.training.thresholds.LineRanking` sorts (score descending, ties
by id).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: The floors the app offers, left to right along its control
#: (the floor presets the app offered, #4298).
FLOORS: tuple[float, ...] = (0.1, 0.5, 0.9)


def floor_tag(floor: float) -> str:
    """``p10`` for 10%: the column suffix a floor's metrics carry (P, the precision floor)."""
    return f"p{round(floor * 100):d}"


def parse_ranks(text: object) -> np.ndarray:
    """A rank frame's space-separated ranks as an int array (empty for a blank or NaN cell)."""
    if not isinstance(text, str) or not text.strip():
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.asarray(text.split(), dtype=np.int64))


def ranks_from_scores(ids: Sequence[int], scores: Sequence[float], labels: Sequence[float]) -> np.ndarray:
    """The positives' ranks in *scores* sorted as a line ranks them: descending, ties by id.

    ``LineRanking``'s order without its sigmoid-range mask: that mask drops
    anything outside ``[0, 1]`` as unscorable, and a text sort's cosine
    similarities are legitimately negative.  Only non-finite scores are left out.
    """
    id_arr = np.asarray(list(ids), dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if not (id_arr.shape == s.shape == y.shape):
        raise ValueError("ids, scores and labels must align")
    keep = np.isfinite(s)
    order = np.lexsort((id_arr[keep], -s[keep]))
    return np.flatnonzero(y[keep][order] >= 0.5).astype(np.int64)


def kept_count(floor: float, n: int) -> int:
    """How many items the line at *floor* keeps on a corpus of *n*: the unchecked candidate.

    ``32 * 2**max(0, floor(log2(0.5 / P)))``, the precision floor's schedule
    (``analyze_line_estimate_4383.floor_schedule``), capped at *n*.
    """
    import math  # noqa: PLC0415

    return int(min(32 * 2 ** max(0, math.floor(math.log2(0.5 / floor) + 1e-9)), n))


def top_k_right(ranks: np.ndarray, k: int) -> int:
    """How many of the top *k* are positives."""
    return int(np.count_nonzero(ranks < k))


def oracle_recall(ranks: np.ndarray, n_pos: int, floor: float) -> float:
    """The best recall any top-*k* cut of this ranking reaches while at least *floor* of it is right.

    A cut's precision peaks just after a positive, so only those cuts are
    candidates; recall only grows with *k*, so the answer is the deepest one
    that still clears *floor*.  0 when none does.
    """
    if n_pos <= 0 or ranks.size == 0:
        return float("nan") if n_pos <= 0 else 0.0
    hits = np.arange(1, ranks.size + 1)
    ok = np.flatnonzero(hits / (ranks + 1) >= floor - 1e-12)
    return float(hits[ok[-1]] / n_pos) if ok.size else 0.0


def oracle_f1(ranks: np.ndarray, n_pos: int) -> float:
    """The best F1 any top-*k* cut of this ranking reaches, whatever its precision.

    F1 at a cut of *k* is ``2 * right / (k + n_pos)``, and between two positives
    it only falls, so the candidates are the cuts just after each positive.
    Floor-independent: it is the ranking's own ceiling on the line's F1.
    """
    if n_pos <= 0:
        return float("nan")
    if ranks.size == 0:
        return 0.0
    hits = np.arange(1, ranks.size + 1)
    return float(np.max(2.0 * hits / (ranks + 1 + n_pos)))


def average_precision(ranks: np.ndarray, n_pos: int) -> float:
    """Average precision of the ranking (sklearn's, when no two scores tie)."""
    if n_pos <= 0:
        return float("nan")
    return float(np.sum(np.arange(1, ranks.size + 1) / (ranks + 1)) / n_pos)


def frame_k(frame: dict, floor: float) -> int | None:
    """How many the shipped line keeps on this frame's test half at *floor*, when the frame recorded it.

    ``test_line_k_p50`` and its siblings (#4389: the smaller of the schedule's
    count and the vote-anchored mixture's). ``None`` - read the schedule's
    count - for a frame without the column (a run before it), a skyline, or
    the text sort, which no session drew a line on.
    """
    v = frame.get(f"test_line_k_{floor_tag(floor)}")
    try:
        k = int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return k if k >= 1 else None


BETAS: tuple[float, ...] = (0.5, 1.0, 2.0)


def beta_tag(beta: float) -> str:
    """``b05`` / ``b1`` / ``b2``: the column suffix a balance's metrics carry (``vtscore.eval.voting_columns.beta_tag``)."""
    return "b" + (f"{beta:g}".replace(".", "") if beta < 1 else f"{beta:g}")


def fbeta_of(right: int, k: int, n_pos: int, beta: float) -> float:
    denominator = beta * beta * n_pos + k
    return (1.0 + beta * beta) * right / denominator if denominator > 0 else 0.0


def oracle_fbeta(ranks: np.ndarray, n_pos: int, beta: float) -> float:
    """The best F-beta any cut of the ranking reaches: checked just after each positive, where it can only peak."""
    if n_pos <= 0 or ranks.size == 0:
        return float("nan")
    tp = np.arange(1, ranks.size + 1)
    k = ranks + 1
    return float(np.max((1.0 + beta * beta) * tp / (beta * beta * n_pos + k)))


def frame_beta_k(frame: dict, beta: float) -> int | None:
    """How many the shipped balance line keeps on this frame's test half at *beta* (``test_line_k_b1``, ...), or ``None``."""
    v = frame.get(f"test_line_k_{beta_tag(beta)}")
    try:
        k = int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return k if k >= 1 else None


def balance_metrics(ranks: np.ndarray, n: int, n_pos: int, beta: float, k: int | None) -> dict[str, float]:
    """The returned set at a balance (#4413): its F-beta over the best F-beta of any cut, with precision and recall.

    *k* is the balance line's count the frame recorded (:func:`frame_beta_k`);
    ``None`` reads the balance's cap (the floor schedule's 32 at beta <= 1, 128
    above), which is what a run before the column, the text sort and the
    ceiling keep.
    """
    nan = float("nan")
    if n <= 0:
        return {"k": 0, "precision": nan, "recall": nan, "fbeta": nan, "oracle_fbeta": nan, "fb_share": nan}
    if k is None:
        k = 32 if beta <= 1.0 else 128
    k = int(max(1, min(k, n)))
    right = top_k_right(ranks, k)
    best = oracle_fbeta(ranks, n_pos, beta)
    f = fbeta_of(right, k, n_pos, beta) if n_pos > 0 else nan
    return {
        "k": k,
        "precision": right / k,
        "recall": right / n_pos if n_pos > 0 else nan,
        "fbeta": f,
        "oracle_fbeta": best,
        "fb_share": f / best if (n_pos > 0 and best > 0) else nan,
    }


def line_metrics(ranks: np.ndarray, n: int, n_pos: int, floor: float, k: int | None = None) -> dict[str, float]:
    """The line at *floor* on this ranking, as the report reads it.

    * ``k`` - how many the line keeps: *k* when the frame recorded the shipped
      line's count (:func:`frame_k`), else the schedule's (:func:`kept_count`);
    * ``precision`` - the share of them that is right;
    * ``shortfall`` - ``max(0, floor - precision)``, how far short of the promise;
    * ``meets`` - 1 when ``precision >= floor``;
    * ``recall`` - the share of the corpus's positives the line keeps;
    * ``oracle_recall`` - the best recall a cut of the same ranking gets at
      precision >= *floor* (:func:`oracle_recall`);
    * ``f1`` - F1 of the kept set, the returned set's own balance of precision
      and recall (owner, 2026-09-30: AP is all ranking, F1 is the line);
    * ``oracle_f1`` - the best F1 any cut of the same ranking reaches
      (:func:`oracle_f1`), whatever the floor.
    """
    nan = float("nan")
    if n <= 0:
        return {
            "k": 0,
            "precision": nan,
            "shortfall": nan,
            "meets": nan,
            "recall": nan,
            "oracle_recall": nan,
            "f1": nan,
            "oracle_f1": nan,
        }
    k = kept_count(floor, n) if k is None else int(max(1, min(k, n)))
    right = top_k_right(ranks, k)
    precision = right / k if k else nan
    return {
        "k": k,
        "precision": precision,
        "shortfall": max(0.0, floor - precision),
        "meets": float(precision >= floor - 1e-12),
        "recall": right / n_pos if n_pos > 0 else nan,
        "oracle_recall": oracle_recall(ranks, n_pos, floor),
        "f1": 2.0 * right / (k + n_pos) if n_pos > 0 else nan,
        "oracle_f1": oracle_f1(ranks, n_pos),
    }
