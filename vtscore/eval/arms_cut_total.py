"""#4798's candidate cuts: what the labels line's class-model cut keeps when its positives total is held.

At beta 4 the labels line sometimes keeps a whole corpus.  ``corpus_cut`` scores each
cut ``k`` (the corpus's unvoted items, best first) as ``F = (1 + b^2) TP_k / (b^2 P + k)``,
with ``TP_k`` the sum of the 3-part chances of the kept items and
``P = max(the 2-part total, the sum of every chance)``.  The two counts of the corpus's
positives can disagree by orders of magnitude, and ``P`` takes the larger.  Then ``F``
is small at every cut, the cut's marginal bar (an item stays while ``(1 + b^2) p > F``)
falls with it, and at beta 4 the cut runs to the corpus's end.  Either count can be the
large one: diffuse chances under a thin class model (#4786's report: a 2-part total of
0.0, chances summing to hundreds, 3,925 faces kept), or a 2-part total that runs away on
a weakly separated broad target (562 positives estimated, 50 withheld, chances summing
to 6).

Each candidate below reads only what the line holds - the unvoted scores best first,
their chances, the 2-part total - so a winner moves into
:meth:`~vtscore.training.thresholds.labels_line.LabelsLine.threshold` unchanged:

* ``class`` - today's cut (:func:`~vtscore.training.thresholds.labels_line.corpus_cut`);
* ``ptotal`` - ``P`` is the 2-part total, never less than the kept set's own chances
  (``P_k = max(total, TP_k)``): the sum of every chance no longer sets it;
* ``pchances`` - ``P`` is the 3-part fit's own total, the sum of its chances: one model
  for both the chances and the count;
* ``pmin`` - ``P`` is the smaller of the two counts, never less than the kept set's
  chances (``P_k = max(min(total, sum), TP_k)``);
* ``capsum`` - the chances scaled down, together, until their sum is at most the
  counted bound the 2-part total already respects (twice the bound, as there);
* ``captotal`` - the chances scaled down until their sum is at most the 2-part total;
* ``ncap`` - today's cut, keeping at most ``beta ** 0.708`` times the 2-part total
  (the count line's own multiple, #4603), never fewer than one item.

Recorded by the line counts (``__linecounts.csv``, ``k_cut_<name>``), so every
candidate is priced exactly on one session set.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: The candidates, in the order the side frame records them.
CUT_VARIANTS: tuple[str, ...] = ("class", "ptotal", "pchances", "pmin", "capsum", "captotal", "ncap")


def _parts(line: Any) -> tuple[np.ndarray, np.ndarray, float] | None:
    """``(scores best first, their chances, the 2-part total)`` of *line*'s corpus side; ``None`` without one."""
    if line is None or line.unvoted_scores is None or line.unvoted_posteriors is None or line.unvoted_share is None:
        return None
    s = np.asarray(line.unvoted_scores, dtype=np.float64)
    p = np.asarray(line.unvoted_posteriors, dtype=np.float64)
    if s.size == 0 or p.size != s.size:
        return None
    return s, p, float(line.unvoted_share) * s.size


def _cut_at(scores_desc: np.ndarray, k: int) -> float:
    """The threshold keeping exactly the top *k* (between the last kept and the first dropped)."""
    if k <= 0:
        return 1.0 + 1e-9
    if k >= scores_desc.size:
        return float(scores_desc[-1])
    return float((scores_desc[k - 1] + scores_desc[k]) / 2.0)


def _ptotal_cut(post: np.ndarray, s: np.ndarray, total: float, beta: float) -> float:
    """The F-beta argmax with ``P_k = max(total, TP_k)``: *total* the positives count, never below the kept set's chances."""
    b2 = float(beta) * float(beta)
    tp = np.cumsum(post)
    r = np.arange(1, s.size + 1, dtype=np.float64)
    pk = np.maximum(float(total), tp)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.nan_to_num((1.0 + b2) * tp / (b2 * pk + r), nan=0.0)
    k = int(np.argmax(f))
    if not f[k] > 0:
        return 1.0 + 1e-9
    return _cut_at(s, k + 1)


def variant_threshold(name: str, line: Any, beta: float) -> float | None:
    """The threshold candidate *name* draws on *line*'s corpus at *beta*; ``None`` with no corpus side."""
    from vtscore.training.thresholds.labels_line import _counted_share_bound, corpus_cut  # noqa: PLC0415

    parts = _parts(line)
    if parts is None:
        return None
    s, p, total = parts
    if name == "class":
        return corpus_cut(p, s, total, beta)
    if name == "ptotal":
        return _ptotal_cut(p, s, total, beta)
    if name == "pchances":
        return _ptotal_cut(p, s, float(p.sum()), beta)
    if name == "pmin":
        return _ptotal_cut(p, s, min(total, float(p.sum())), beta)
    if name in ("capsum", "captotal"):
        if name == "capsum":
            fitted = line.model.floored(line.spread) if line.spread is not None else line.model
            cap = 2.0 * _counted_share_bound(fitted, s) * s.size
        else:
            cap = total
        total_p = float(p.sum())
        scaled = p * (min(1.0, cap / total_p) if total_p > 0 else 0.0)
        return corpus_cut(scaled, s, total, beta)
    if name == "ncap":
        thr = corpus_cut(p, s, total, beta)
        k = int(np.count_nonzero(s >= thr))
        cap_k = max(1, math.ceil(float(beta) ** 0.708 * total))
        return thr if k <= cap_k else _cut_at(s, cap_k)
    raise ValueError(f"unknown cut variant {name!r}; expected one of {CUT_VARIANTS}")


def variant_counts(line: Any, scores: Any, beta: float) -> dict[str, int]:
    """What each candidate keeps of *scores* (the corpus *line* was fitted on): ``{name: count}``, -1 without a line."""
    s = np.asarray(scores, dtype=np.float64)
    out: dict[str, int] = {}
    for name in CUT_VARIANTS:
        thr = variant_threshold(name, line, beta)
        out[name] = -1 if thr is None else int(np.count_nonzero(s >= thr))
    return out


def chances_sum(line: Any) -> float:
    """The sum of every 3-part chance on *line*'s corpus (NaN without one): what sets ``P`` when it is diffuse."""
    parts = _parts(line)
    return float("nan") if parts is None else float(parts[1].sum())


__all__ = ["CUT_VARIANTS", "chances_sum", "variant_counts", "variant_threshold"]
