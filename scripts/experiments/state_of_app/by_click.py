"""The objective at every click, filled (#4599): what the user has in hand at each click, per preset.

Photo Finish and Patch Notes drew the objective as straight segments between five checkpoints
(25, 50, 100, 150 and after the check), so a session's first clicks read as a steady climb from the
typed query. The owner (2026-10-07): "shouldn't the FBeta stay flat for the first few clicks, since
we're measuring and collecting before we're learning?" It does: the app stays on the text sort through
Autopilot's whole opening and shows a detector only from the Hard phase on (#4605), a median ~40 clicks
in. (A first version switched at the opening's own first detector, around click 4, and drew a dip there
that no user sees.)

`curves.csv` (analyze.py) records each run's objective (``thr_fbeta``, at the run's own beta) at every
click once the app shows the run's detector (``app_trained``), and nothing before. This fills the gap the
way the user meets it:

* before that, the run scores the **typed query's** own set (the text sort at its own line in the app,
  `_text_app_line` off the text baseline), which is what the user sees;
* after it, a click with no value (a spot-check step) carries the run's last value forward.

The mean over every run of the review at each click is the curve (#4631): a run that never trained a
detector is never shown one, so it scores its typed query at every click (owner, 2026-10-07: what its
session shows, over the viewer's empty set). Click 0 is the typed query, so the curve starts on the notch.

    python by_click.py --run 0.25=<b025>/analysis-binary --run 1=<b1>/... --run 4=<b4>/... \\
        --baseline <text_baseline.csv> --embedder siglip --out <dir>/objective_by_click.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import _text_app_line  # noqa: E402

RUN = ["category", "seed"]


def typed_query(baseline: pd.DataFrame, embedder: str, beta: float, metric: str = "fbeta") -> pd.Series:
    """Each run's typed-query *metric* (``fbeta``, ``precision`` or ``recall``) at *beta*, keyed by (category,
    seed): the text sort at its own line."""
    base = baseline[baseline["embedder"] == embedder]
    out = {}
    for r in base.to_dict("records"):
        text = {**r, "text_gmm_precision": r["text_precision"], "text_gmm_recall": r["text_recall"],
                "text_gmm_fpr": r["text_fpr"]}  # fmt: skip
        out[(r["category"], int(r["seed"]))] = _text_app_line(text, beta)[metric]
    return pd.Series(out, dtype=float)


def filled_curve(curves: pd.DataFrame, runs: pd.DataFrame, text: pd.Series, col: str = "thr_fbeta") -> pd.DataFrame:
    """``t, fbeta, runs, with_detector``: the mean over *runs* (every run of the review) of the filled *col*.

    A run scores *text* (its typed query) at every click before the app shows its detector (#4605): its
    ``shown_from`` in ``cells.csv``, never for a run that never trained; without that column, its first
    ``thr_fbeta``. After that, a missing click carries the last value forward. *col* is ``thr_fbeta`` or the
    ``thr_precision`` / ``thr_recall`` behind it (the right panel's path); the column is named ``fbeta``
    whichever it is.
    """
    c = curves.merge(runs[RUN], on=RUN)

    # unstack, not pivot_table(dropna=False): the latter rebuilds every category x seed pair, runs no review
    # attempted included; unstack keeps the all-empty early clicks.
    def wide_of(name: str) -> pd.DataFrame:
        w = c.drop_duplicates(RUN + ["t"]).set_index(RUN + ["t"])[name].unstack("t")
        return w.reindex(columns=range(int(c["t"].min()), int(c["t"].max()) + 1))

    wide = wide_of(col)
    if "shown_from" in runs:
        first = runs.set_index(RUN)["shown_from"].astype(float).reindex(wide.index).to_numpy()
        started = pd.DataFrame(
            wide.columns.to_numpy()[None, :] >= first[:, None], index=wide.index, columns=wide.columns
        )
    else:
        started = wide_of("thr_fbeta").notna().cummax(axis=1)
    carried = wide.ffill(axis=1)
    tq = pd.Series([text.get((cat, int(seed)), np.nan) for cat, seed in wide.index], index=wide.index)
    filled = carried.where(started, other=tq.to_numpy()[:, None] * np.ones(wide.shape))
    return pd.DataFrame(
        {
            "t": wide.columns.astype(int),
            "fbeta": filled.mean(axis=0).to_numpy(),
            "runs": filled.notna().sum(axis=0).to_numpy(),
            "with_detector": started.sum(axis=0).to_numpy(),
        }
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="beta=analysis dir (curves.csv, cells.csv)")
    ap.add_argument("--baseline", required=True, help="the text baseline the typed query is scored from")
    ap.add_argument("--embedder", required=True, help="the baseline rows the review's arm opens on")
    ap.add_argument(
        "--dataset", default=None, help="keep one dataset's baseline rows (a face review's sizes share a path, #4762)"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    baseline = pd.read_csv(args.baseline)
    if args.dataset:
        # The baseline is keyed on (category, seed) below; two datasets of one path share both.
        baseline = baseline[baseline["dataset"] == args.dataset]
    rows = []
    for spec in args.run:
        beta_s, d = spec.split("=", 1)
        beta = float(beta_s)
        cells = pd.read_csv(Path(d) / "cells.csv")  # every run, the never-trained included (#4631)
        curves = pd.read_csv(Path(d) / "curves.csv")
        curve = filled_curve(curves, cells, typed_query(baseline, args.embedder, beta))
        # The precision and recall behind it, where the analysis carries them (#4605): the right panel's path.
        for m in ("precision", "recall"):
            if f"thr_{m}" in curves:
                tq = typed_query(baseline, args.embedder, beta, m)
                curve[m] = filled_curve(curves, cells, tq, f"thr_{m}")["fbeta"].to_numpy()
        rows.append(curve.assign(beta=f"{beta:g}"))
    out = pd.concat(rows)
    out = out[[c for c in ("beta", "t", "fbeta", "precision", "recall", "runs", "with_detector") if c in out]]
    out.to_csv(args.out, index=False)
    print(f"by_click -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
