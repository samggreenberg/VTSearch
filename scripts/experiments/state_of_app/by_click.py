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

The mean over a review's trained runs at each click is the curve; never-trained runs are left out, as
in the path table (`perp.py`). Click 0 is the typed query, so the curve starts on the notch.

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


def typed_query(baseline: pd.DataFrame, embedder: str, beta: float) -> pd.Series:
    """Each run's typed-query F-beta at *beta*, keyed by (category, seed): the text sort at its own line."""
    base = baseline[baseline["embedder"] == embedder]
    out = {}
    for r in base.to_dict("records"):
        text = {**r, "text_gmm_precision": r["text_precision"], "text_gmm_recall": r["text_recall"],
                "text_gmm_fpr": r["text_fpr"]}  # fmt: skip
        out[(r["category"], int(r["seed"]))] = _text_app_line(text, beta)["fbeta"]
    return pd.Series(out, dtype=float)


def filled_curve(curves: pd.DataFrame, trained: pd.DataFrame, text: pd.Series) -> pd.DataFrame:
    """``t, fbeta, runs, with_detector``: the mean over *trained* runs of the filled objective at each click.

    A run scores *text* (its typed query) at every click before its first ``thr_fbeta``; after that, a
    missing click carries the last value forward.
    """
    c = curves.merge(trained[RUN], on=RUN)
    # unstack, not pivot_table(dropna=False): the latter rebuilds every category x seed pair and so brings
    # back the never-trained runs this was meant to leave out; unstack keeps the all-empty early clicks.
    wide = c.drop_duplicates(RUN + ["t"]).set_index(RUN + ["t"])["thr_fbeta"].unstack("t")
    wide = wide.reindex(columns=range(int(c["t"].min()), int(c["t"].max()) + 1))
    has = wide.notna()
    started = has.cummax(axis=1)
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
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    baseline = pd.read_csv(args.baseline)
    rows = []
    for spec in args.run:
        beta_s, d = spec.split("=", 1)
        beta = float(beta_s)
        cells = pd.read_csv(Path(d) / "cells.csv")
        trained = cells[~cells["never_trained"].astype(bool)]
        curves = pd.read_csv(Path(d) / "curves.csv")
        curve = filled_curve(curves, trained, typed_query(baseline, args.embedder, beta))
        rows.append(curve.assign(beta=f"{beta:g}"))
    out = pd.concat(rows)[["beta", "t", "fbeta", "runs", "with_detector"]]
    out.to_csv(args.out, index=False)
    print(f"by_click -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
