"""Which metric an analyzer decides on, and the objective's columns on frames written before them (#4584).

The app draws its line at an F-beta balance (#4413) and every review since
#4436 headlines the objective: F-beta of the withheld half above the threshold
the app holds (#4427).  A cell frame carries that number as ``fbeta``, at the
row's own ``beta``, with ``fbeta_b025`` / ``_b1`` / ``_b4`` beside it
(:func:`vtscore.eval.calibration_metrics.fbeta_metrics`).  ``cost`` stays in
the frame as a diagnostic: the runner prices it at ``INCLUSION = 0`` whatever
beta drew the line, so on a balance run it measures a preference the app no
longer holds.

So the rule every analyzer here follows is one line: **a frame that carries a
beta is decided on the objective; one that does not (the Inclusion arm,
``CALIB_BETA=off``, or a Cost-era run) is decided on cost.**

Frames written before the columns existed still carry ``precision``,
``recall`` and ``beta``; :func:`with_objective` fills the columns from those
through :func:`~vtscore.eval.calibration_metrics.fbeta_from_rates`, so a
balance-era cell already on ``/expscratch`` reads exactly as a new one would.

Imports ``vtscore`` lazily: a stage calls ``common.setup_env()`` before
anything under ``vtscore`` is imported.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

#: The objective's column, at the row's own beta.
OBJECTIVE = "fbeta"
#: The decision metric of a frame with no balance.
COST = "cost"


def preset_columns() -> tuple[str, ...]:
    """``fbeta_b025`` / ``fbeta_b1`` / ``fbeta_b4``: the objective at each of the app's presets."""
    from vtscore.eval.voting_columns import FBETA_COLUMNS  # noqa: PLC0415

    return tuple(c for c in FBETA_COLUMNS if c != OBJECTIVE)


def frame_betas(df: pd.DataFrame) -> list[float]:
    """The finite balances a frame's rows were drawn at, sorted; empty with no ``beta`` column or no balance."""
    if "beta" not in df.columns:
        return []
    b = pd.to_numeric(df["beta"], errors="coerce")
    return sorted(float(x) for x in b[np.isfinite(b)].unique())


def carries_beta(df: pd.DataFrame) -> bool:
    """Whether any row of *df* had its line drawn at a balance."""
    return bool(frame_betas(df))


def primary_metric(df: pd.DataFrame) -> str:
    """The metric a decision on *df* reads: the objective when the frame carries a beta, else cost."""
    return OBJECTIVE if carries_beta(df) else COST


def lower_is_better(metric: str) -> bool:
    """The direction of *metric*, from the one table the viewer reads it from.

    The preset columns share ``fbeta``'s direction.  A metric the table does
    not know (``regret``, ``degenerate``, ``oracle_cost``) is a cost-like
    quantity in every analyzer here, so lower is better.
    """
    from vtscore.eval.calibration_metrics import DETECTION_METRICS  # noqa: PLC0415

    key = OBJECTIVE if metric.startswith(f"{OBJECTIVE}_") else metric
    entry = DETECTION_METRICS.get(key)
    return True if entry is None else bool(entry[1])


def with_objective(df: pd.DataFrame) -> pd.DataFrame:
    """*df* with the objective's columns, filled from its rates where the frame predates them.

    A value already written is left as written: the runner computed it from
    the counts, which the 6-dp rates can only approximate.  A NaN in a column
    that exists is filled like a missing column, because a frame concatenated
    from cells written before and after the columns existed (a resumed study,
    an A/B whose control ran earlier) carries the column with the older rows
    empty, and an empty objective reads as a loss, not a refusal.  Without
    ``precision`` and ``recall`` there is nothing to fill from, and *df* comes
    back unchanged.  A row with no balance gets a NaN ``fbeta`` and its preset
    columns, as a new row would.
    """
    if df.empty or not {"precision", "recall"} <= set(df.columns):
        return df
    from vtscore.eval.calibration_metrics import fbeta_from_rates  # noqa: PLC0415
    from vtscore.eval.voting_columns import RANK_FRAME_BETAS, beta_tag  # noqa: PLC0415

    want = {OBJECTIVE: None, **{f"{OBJECTIVE}_{beta_tag(b)}": b for b in RANK_FRAME_BETAS}}
    missing = [c for c in want if c not in df.columns or df[c].isna().any()]
    if not missing:
        return df
    out = df.copy()
    p = pd.to_numeric(out["precision"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(out["recall"], errors="coerce").to_numpy(dtype=float)
    row_beta = (
        pd.to_numeric(out["beta"], errors="coerce").to_numpy(dtype=float)
        if "beta" in out.columns
        else np.full(len(out), np.nan)
    )
    for col in missing:
        beta = want[col]
        filled = fbeta_from_rates(p, r, row_beta if beta is None else beta)
        if col in out.columns:
            written = pd.to_numeric(out[col], errors="coerce")
            out[col] = written.where(written.notna(), filled)
        else:
            out[col] = filled
    return out


def _num(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def row_objective(row: Mapping[str, Any]) -> float:
    """The objective of one row read with :mod:`csv` (a dict of strings), for the stdlib analyzers.

    Its ``fbeta`` where the runner wrote one, else the same number from its
    ``precision``, ``recall`` and ``beta`` (:func:`with_objective`'s rule).
    """
    v = _num(row, OBJECTIVE)
    if v == v:
        return v
    from vtscore.eval.calibration_metrics import fbeta_from_rates  # noqa: PLC0415

    return float(fbeta_from_rates(_num(row, "precision"), _num(row, "recall"), _num(row, "beta")))


def rows_carry_beta(rows: Iterable[Mapping[str, Any]]) -> bool:
    """:func:`carries_beta` for rows read with :mod:`csv`."""
    return any(np.isfinite(_num(r, "beta")) for r in rows)
