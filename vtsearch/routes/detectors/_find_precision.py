"""The Find Stats precision curve: how right the returned set is against how much is returned.

One reading of one ranking, sampled at a few dozen return counts *k* (the top
*k* of the Find corpus by frozen score): **verified precision**, of the items
in the top *k* the user checked by hand, the share they kept Good.  Measured,
not estimated, but only over what was checked: the boundary walk surfaces items
near the line, so it is not a random sample of the top *k*.

The curve carries no model-based estimate.  The #4220 estimator's lower-bound
curve is not drawn: it keeps its promise only through an in-sample offset in
its reference pool, and against a consistent pool it breaks 83-91% of its
X = 50% promises (#4221, #4256), so it cannot back the "at least" a chart
would be read as (#4360).  The one range the chart shows is the spot check's,
for the set the line keeps (the ``floor`` state).
Pure read; nothing is cached or persisted.
"""

from __future__ import annotations

import numpy as np

#: Return counts the curve is sampled at, log-spaced from 1 to the corpus size.
PRECISION_CURVE_POINTS = 40


def curve_counts(n_scored: int, extra: int | None = None, n_points: int = PRECISION_CURVE_POINTS) -> list[int]:
    """Distinct return counts in ``[1, n_scored]``, log-spaced, plus *extra* when it is in range.

    Log spacing keeps the top of the ranking, where a precision curve moves,
    from collapsing into one point on a large corpus.  *extra* is the current
    cut's count, so the curve has a point exactly at the marker.
    """
    if n_scored <= 0:
        return []
    counts = {int(round(k)) for k in np.geomspace(1, n_scored, num=min(n_scored, n_points))}
    if extra is not None and 1 <= extra <= n_scored:
        counts.add(extra)
    return sorted(counts)


def verified_precision_at(
    ranked_ids: list[int], checked_good: dict[int, bool], counts: list[int]
) -> list[tuple[int, int, float | None]]:
    """``(checked, kept_good, precision)`` of the top *k* of *ranked_ids*, for each *k* in *counts*.

    *checked_good* maps each item the user verified to whether its adopted label
    is Good.  Precision is ``None`` where no item in the top *k* was checked.
    """
    checked = np.cumsum([1 if cid in checked_good else 0 for cid in ranked_ids])
    good = np.cumsum([1 if checked_good.get(cid) else 0 for cid in ranked_ids])
    out: list[tuple[int, int, float | None]] = []
    for k in counts:
        n_checked, n_good = int(checked[k - 1]), int(good[k - 1])
        out.append((n_checked, n_good, n_good / n_checked if n_checked else None))
    return out
