"""#4864's split rounding: the calibration split before #4864 (dithered), as an eval arm.

The calibration folds hold out a share of each class
(:func:`vtscore.training.thresholds.conformal._split_train_counts`).  Since
#4864 the app rounds each class's share to the nearest whole vote
(:func:`~vtscore.training.thresholds.conformal.nearest_holdout`), with no coin,
so the labels line takes over at one count of Goods and stays.  Before it, the
app sized the share over the whole labelset first and then per class, and
rounded both counts stochastically, with a coin seeded from the labelset
(#3286, #3288).  Over a band of Goods counts (6 to 11 on a single-vector space)
the line a trained head drew was then that coin's, drawn again at every vote.

:func:`split_rounding` ``("dither")`` is that rule, kept so the A/B can be run
again on a later commit (``CALIB_SPLIT_ROUNDING=dither``).  Unset (``nearest``)
is the app.  #4864 measured the trade
(``docs/experiments/2026-10-14-split-rounding-4864/REPORT.md``): the dither
decoheres the eval's runs completely, where ``nearest`` leaves the Bads'
holdout a small shared cadence on a single-vector space; ``nearest`` switches a
run's line once, where the dither switched it twice a run on average.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from vtscore.training.thresholds.conformal import nearest_holdout

#: The rules :func:`split_rounding` knows.  ``nearest`` is the app's.
SPLIT_ROUNDINGS: tuple[str, ...] = ("nearest", "dither")

# Rows sampled from the training matrix when seeding the dither (see
# :func:`_split_dither_rng`).  Small on purpose: this only has to separate two
# labelsets of equal size, not summarise them.
_DITHER_SAMPLE_ROWS = 32

__all__ = ["SPLIT_ROUNDINGS", "nearest_holdout", "split_rounding"]


def _split_dither_rng(X_np: np.ndarray, y_np: np.ndarray) -> np.random.RandomState:
    """The dither's RNG, seeded from the labelset.

    Not a constant seed: a draw from a constant is the same number on every
    call and would replace one deterministic function of the vote count with
    another, which is the failure the dither existed to fix (issue #3286).
    Seeding from a digest of the training vectors and their labels gives a draw
    that is stable for a given labelset (so a threshold stays a pure function of
    the votes, and the calibration cache stays valid) while differing between
    two labelsets that merely happen to be the same size.

    The digest is taken over the labels in full plus a **strided sample** of
    the training rows: bounded cost on a flooded patch labelset (at most
    :data:`_DITHER_SAMPLE_ROWS` rows), and sensitive to the whole labelset,
    which a fixed prefix would not be (the rows are laid out Good-then-Bad and
    the earliest votes never move).  Only slicing and ``tobytes`` are involved,
    so the digest is byte-exact across machines, as a reduction over the
    embeddings would not be (#3166).
    """
    stride = max(1, len(X_np) // _DITHER_SAMPLE_ROWS)
    h = hashlib.blake2b(np.ascontiguousarray(X_np[::stride], dtype=np.float32).tobytes(), digest_size=8)
    h.update(np.ascontiguousarray(y_np, dtype=np.float32).tobytes())
    return np.random.RandomState(int.from_bytes(h.digest()[:4], "little"))


def _dithered_count(exact: float, rng: np.random.RandomState) -> int:
    """Round *exact* to an integer, breaking a fractional part at random.

    Stochastic rounding: ``P(round up) = frac(exact)``, so the count is
    unbiased instead of being pinned to whichever side ``round`` picks.  A whole
    number is returned unchanged and draws nothing.  At a split of 0.5 it fires
    on odd vote counts only; at 0.3 almost every count has a fractional part.

    Why it was there (#3286): ``round`` is round-half-to-even, so at a 50/50
    split the odd vote joined Train at ``n % 4 == 1`` and Calibrate at
    ``n % 4 == 3``.  The eval casts one vote per step, so that seesaw was
    phase-locked across every run and survived averaging as a 4-vote ripple on
    the learning curves.  Randomising the rounding decoheres the runs.
    """
    low = math.floor(exact)
    frac = exact - low
    if frac <= 0.0:
        return int(low)
    return int(low) + (1 if rng.random_sample() < frac else 0)


def _dithered_counts(
    n: int, n_pos: int, n_neg: int, calibration_fraction: float, rng: np.random.RandomState
) -> tuple[int, int] | None:
    """The dithered split's train counts per fold, drawing from *rng*; ``None`` when too few would train.

    The calibration share is sized over all *n* votes, then each class trains
    its proportional share of the rest, both counts dithered, each class
    keeping at least one vote on either side.
    """
    n_cal = max(1, _dithered_count(n * calibration_fraction, rng))
    n_train = n - n_cal
    if n_train < 2 or n_cal < 1:
        return None

    def _per_class_n_train(class_total: int) -> int:
        target = _dithered_count(class_total * n_train / n, rng)
        return max(1, min(class_total - 1, target))

    return _per_class_n_train(n_pos), _per_class_n_train(n_neg)


def _dithered_train_counts(
    n: int, n_pos: int, n_neg: int, calibration_fraction: float, labelset: tuple[Any, Any] | None
) -> tuple[int, int] | None:
    """:func:`~vtscore.training.thresholds.conformal._split_train_counts` under the dither: the rule before #4864.

    The coin is seeded from *labelset* (:func:`_split_dither_rng`), as the app
    seeded it.
    """
    if labelset is None:
        raise ValueError("the dithered split seeds its coin from the labelset; none was passed")
    X_np, y_np = labelset
    return _dithered_counts(n, n_pos, n_neg, calibration_fraction, _split_dither_rng(X_np, y_np))


@contextmanager
def split_rounding(rule: str) -> Iterator[None]:
    """Size the calibration folds' split by *rule* while the block runs (#4864).

    ``dither`` swaps the app's split sizing for :func:`_dithered_train_counts`.
    The row-wise and grouped folds both size through that one function, so the
    swap reaches both paths and every fold count.  ``nearest`` changes nothing
    (the app's).  Process-wide, as
    :func:`~vtscore.eval.live_threshold_rules.head_count_bads` is, which suits a
    task that runs one arm, and undone on the way out.
    """
    if rule not in SPLIT_ROUNDINGS:
        raise ValueError(f"unknown split_rounding {rule!r}; expected one of {', '.join(SPLIT_ROUNDINGS)}")
    if rule == "nearest":
        yield
        return
    from vtscore.training.thresholds import conformal  # noqa: PLC0415

    shipped = conformal._split_train_counts
    conformal._split_train_counts = _dithered_train_counts
    try:
        yield
    finally:
        conformal._split_train_counts = shipped
