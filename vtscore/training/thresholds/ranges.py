"""The likely range: how much of a set is right, bounded from the check's uniform picks inside it (#4483).

:func:`likely_range` bounds a set of known size exactly (the hypergeometric
tails, :func:`hypergeometric_lower` / :func:`hypergeometric_upper`), at the
tail :func:`range_tail` gives; :class:`LikelyRange` is the record it returns.
The one-sided Clopper-Pearson bounds it replaced stay beside it.  Reads
:data:`~vtscore.training.thresholds.check_schedule.CHECK_ALPHA` and no other
sibling; SciPy is imported on first use.

Split out of :mod:`~vtscore.training.thresholds.spot_check` (#5425), which
re-exports every name here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vtscore.training.thresholds.check_schedule import CHECK_ALPHA


def range_tail(rounds: int, alpha: float = CHECK_ALPHA) -> float:
    """Each tail of a band's bound when *rounds* bands make up the set: ``alpha``, whatever the bands (#4483).

    #4272 split *alpha* over the bands (the union bound) because, under the
    precision floor, the same labels decided a pass and drew its range: a plain
    90% range sat above the truth 11% of the time after a first pass.  The
    check has been advisory since #4452, and with exact band bounds and no
    split the range held the truth in at least 98.9% of 3,568 real checks and
    of simulated checks at clicks 25-100, every preset, for 0.01-0.08 less
    width (``docs/experiments/2026-10-15-check-range-4483/REPORT.md``).
    """
    del rounds
    return alpha


# ---------------------------------------------------------------- the bound


def clopper_pearson_lower(right: int, labelled: int, level: float) -> float:
    """One-sided Clopper-Pearson lower bound on a proportion at confidence ``1 - level``."""
    if labelled <= 0 or right <= 0:
        return 0.0
    from scipy.stats import beta  # noqa: PLC0415

    return float(beta.ppf(level, right, labelled - right + 1))


def clopper_pearson_upper(right: int, labelled: int, level: float) -> float:
    """One-sided Clopper-Pearson upper bound on a proportion at confidence ``1 - level``."""
    if labelled <= 0 or right >= labelled:
        return 1.0
    from scipy.stats import beta  # noqa: PLC0415

    return float(beta.ppf(1.0 - level, right + 1, labelled - right))


def hypergeometric_lower(right: int, labelled: int, size: int, level: float) -> int:
    """The fewest right items a set of *size* can hold, at confidence ``1 - level``, when *right* of *labelled* uniform picks were.

    The least K under which drawing *right* or more right ones has
    probability above *level* (the exact one-sided bound for sampling without
    replacement from a set of known size).  It is never below *right*.
    """
    if labelled <= 0 or right <= 0:
        return 0
    from scipy.stats import hypergeom  # noqa: PLC0415

    k = np.arange(right, size - (labelled - right) + 1)
    return int(k[np.argmax(hypergeom.sf(right - 1, size, k, labelled) > level)])


def hypergeometric_upper(right: int, labelled: int, size: int, level: float) -> int:
    """The most right items a set of *size* can hold, at confidence ``1 - level``, when *right* of *labelled* uniform picks were.

    The greatest K under which drawing *right* or fewer right ones has
    probability above *level*.  It is never above ``size - (labelled - right)``.
    """
    if labelled <= 0:
        return size
    if right >= labelled:
        return size
    from scipy.stats import hypergeom  # noqa: PLC0415

    k = np.arange(right, size - (labelled - right) + 1)
    return int(k[np.flatnonzero(hypergeom.cdf(right, size, k, labelled) > level)[-1]])


@dataclass(frozen=True)
class LikelyRange:
    """How much of a set is likely right, from the check's labels inside it.

    ``lo`` / ``hi`` bound the set's precision; ``labelled`` is how many of the
    set's items the check labelled and ``right`` how many of those were right.
    """

    lo: float
    hi: float
    labelled: int
    right: int

    def as_dict(self) -> dict[str, Any]:
        return {"lo": round(self.lo, 4), "hi": round(self.hi, 4), "labelled": self.labelled, "right": self.right}


def likely_range(right: int, labelled: int, candidate: int, tail: float) -> LikelyRange:
    """The range shown for a set of *candidate* items whose *labelled* uniform labels hold *right*.

    Exact (``right / candidate``) for a census; otherwise the exact bound for a
    set of known size (:func:`hypergeometric_lower` / :func:`hypergeometric_upper`,
    each tail at *tail*), 0-100% with no labels.  #4483 replaced the
    Clopper-Pearson interval, which bounds an endless list: a band of 8 with 4
    of 5 right holds 4 to 7 right items, which it read as 34-99% and this reads
    as 50-88%.  The reference is ``hg_bounds`` in
    ``scripts/experiments/state_of_app/check_range_4483.py``.
    """
    if candidate > 0 and labelled >= candidate:
        exact = right / candidate
        return LikelyRange(exact, exact, labelled, right)
    if candidate <= 0 or labelled <= 0:
        return LikelyRange(0.0, 1.0, max(0, labelled), max(0, right))
    lo = hypergeometric_lower(right, labelled, candidate, tail)
    hi = hypergeometric_upper(right, labelled, candidate, tail)
    return LikelyRange(lo / candidate, hi / candidate, labelled, right)
