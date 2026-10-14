"""#4864's split rounding: each class's calibration holdout rounded to nearest, not dithered.

The calibration folds hold out a share of each class
(:func:`vtscore.training.thresholds.conformal._split_train_counts`).  The app
sizes that share over the whole labelset first and then per class, and rounds
both counts stochastically (:func:`~vtscore.training.thresholds.conformal._dithered_count`,
seeded from the labelset, #3286).  So at a given count of Goods, how many of
them the folds hold out is a coin flip, drawn again at every vote.  The labels
line stands once the class model holds
:data:`~vtscore.training.thresholds.labels_line.HEAD_COUNT_GOODS` held-out Goods,
so over a band of Goods counts (6 to 11 on a single-vector space) the line a
trained head draws is that coin's.

:func:`split_rounding` ``("nearest")`` is the arm.  Each class holds out
``round(calibration_fraction * count)``, halves rounding up, kept between 1 and
``count - 1``, with no whole-labelset stage and no coin.  The held-out Goods are
then a fixed, non-decreasing function of the Goods alone, and the labels line
starts at one count of Goods for good: 9 on a single-vector space (0.3 held
out), 5 on a patch space (0.5), at the app's two folds.  What it may cost is
#3286's ripple: sizes that step on a fixed cadence step together across the
eval's runs.  ``CALIB_SPLIT_ROUNDING=nearest``; unset (``dither``) is the app.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

#: The rules :func:`split_rounding` knows.  ``dither`` is the app's.
SPLIT_ROUNDINGS: tuple[str, ...] = ("dither", "nearest")


def nearest_holdout(class_total: int, calibration_fraction: float) -> int:
    """How many of a class of *class_total* votes the ``nearest`` rule holds out of each fold.

    ``calibration_fraction`` of them, halves rounding up, kept between 1 and
    ``class_total - 1`` so both sides of the split keep a vote of the class.  A
    class of fewer than two votes has no split; the fold code refuses it before
    it trains.
    """
    held = math.floor(class_total * calibration_fraction + 0.5)
    return max(1, min(class_total - 1, held))


def _nearest_train_counts(
    n: int, n_pos: int, n_neg: int, calibration_fraction: float, dither: "np.random.RandomState"
) -> tuple[int, int] | None:
    """:func:`~vtscore.training.thresholds.conformal._split_train_counts` under the ``nearest`` rule.

    *n* and *dither* are the shipped rule's inputs and are not read: each class
    is sized on its own count, with nothing drawn.
    """
    del n, dither
    return n_pos - nearest_holdout(n_pos, calibration_fraction), n_neg - nearest_holdout(n_neg, calibration_fraction)


@contextmanager
def split_rounding(rule: str) -> Iterator[None]:
    """Size the calibration folds' split by *rule* while the block runs (#4864).

    ``nearest`` swaps the app's split sizing for :func:`_nearest_train_counts`.
    The row-wise and grouped folds both size through that one function, so the
    swap reaches both paths and every fold count.
    ``dither`` changes nothing (the app's).  Process-wide, as
    :func:`~vtscore.eval.live_threshold_rules.head_count_bads` is, which suits a
    task that runs one arm, and undone on the way out.
    """
    if rule not in SPLIT_ROUNDINGS:
        raise ValueError(f"unknown split_rounding {rule!r}; expected one of {', '.join(SPLIT_ROUNDINGS)}")
    if rule == "dither":
        yield
        return
    from vtscore.training.thresholds import conformal  # noqa: PLC0415

    shipped = conformal._split_train_counts
    conformal._split_train_counts = _nearest_train_counts
    try:
        yield
    finally:
        conformal._split_train_counts = shipped
