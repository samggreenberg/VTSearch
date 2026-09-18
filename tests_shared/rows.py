"""Comparing two harness runs' rows, for the determinism tests in both tiers.

Several tests assert that a knob is a no-op by running the simulation twice and
comparing the emitted rows.  Two things make that comparison subtler than
``a == b``, and both used to be re-solved per test file:

* **Wall clocks differ between identical runs.**  `drop_timing` removes them,
  reading the canonical set from :data:`vtscore.eval.voting_columns.TIMING_COLUMNS`
  so a new timing column joins one tuple rather than three private copies.
* **`nan != nan`.**  Rows legitimately carry NaN for a quantity a rule declined
  to measure - a threshold on a step with no fit, a stopping-rule margin before
  the rule has enough history (issue #3560) - and two runs that agree perfectly
  both carry it.  Plain ``==`` on the row dicts reports those identical runs as
  different, which is a false failure that says nothing about determinism.

`same_rows` is therefore the comparison these tests want: identical keys,
identical values, and NaN matching NaN.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from vtscore.eval.voting_columns import TIMING_COLUMNS


def drop_timing(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """*rows* with the wall-clock columns removed."""
    return [{k: v for k, v in r.items() if k not in TIMING_COLUMNS} for r in rows]


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


def same_rows(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> bool:
    """Whether two runs' rows agree, treating NaN as equal to NaN.

    Timing columns are **not** dropped here - pass the rows through
    :func:`drop_timing` first when the two runs are separate wall clocks.  A
    comparison that silently ignored them would also hide a run that stopped
    emitting one.
    """
    if len(a) != len(b):
        return False
    return all(ra.keys() == rb.keys() and all(_same(ra[k], rb[k]) for k in ra) for ra, rb in zip(a, b, strict=True))


def assert_same_rows(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> None:
    """:func:`same_rows`, but failing with the first differing cell named."""
    assert len(a) == len(b), f"row count differs: {len(a)} vs {len(b)}"
    for i, (ra, rb) in enumerate(zip(a, b, strict=True)):
        assert ra.keys() == rb.keys(), f"row {i}: columns differ: {set(ra) ^ set(rb)}"
        for k in ra:
            assert _same(ra[k], rb[k]), f"row {i}, column {k!r}: {ra[k]!r} != {rb[k]!r}"
