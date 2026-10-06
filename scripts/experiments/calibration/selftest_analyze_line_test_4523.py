#!/usr/bin/env python3
"""Planted-answer checks for ``analyze_line_test_4523.py`` (#4523).

Run it directly (``python selftest_analyze_line_test_4523.py``); it exits
non-zero on the first failed check.  What it pins: the thinner keeps every
positive and lands the prevalence; the summary's coverage, stop shares and
cost columns reproduce hand-computed answers on planted rows; the pick rule
takes the cheapest eligible grid point, ties to the wider target, and names
the best coverage when nothing is eligible; and a planted snapshot replays
through the cell path to one row per grid point and Test seed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_line_test_4523 as a  # noqa: E402

from vtscore.training.thresholds.labels_line import ClassScoreModel  # noqa: E402


def check(cond: bool, what: str) -> None:
    if not cond:
        raise SystemExit(f"FAILED: {what}")
    print(f"ok  {what}")


def _rows(world: str, beta: float, width: float, budget: int, held: list[int], picks: list[int], stops: list[str]):
    out = []
    for i, (h, p, s) in enumerate(zip(held, picks, stops)):
        out.append(
            {
                "world": world,
                "beta": beta,
                "width": width,
                "budget": budget,
                "walk": a.DEFAULT_WALK,
                "category": f"c{i}",
                "seed": 0,
                "phase": "done",
                "line_count": 40,
                "n_test_pos": 50,
                "picks_total": p,
                "picks_above": p - 5,
                "picks_below": 5,
                "rounds": p // 5,
                "matches_stop": s,
                "misses_stop": "dry_run",
                "n_edges": 10,
                "found_match": 1,
                "tail_from_model": 1,
                "tail_positives_model": 1.0,
                "tail_positives_true": 2,
                "positives_below_point": 3.0,
                "positives_below_true": 2,
                "verdict": "ship",
                "oracle_verdict": "ship",
                "verdict_match": 1,
                "fbeta_at_best_edge_est_true": 0.5,
                "fbeta_true": 0.6,
                "fbeta_best_cut_true": 0.7,
                **{
                    f"{m}_{k}": v
                    for m in ("precision", "recall", "fbeta")
                    for k, v in (("held", h), ("lo", 0.4), ("hi", 0.6), ("point", 0.5), ("true", 0.5))
                },
                **{f"edges_{m}_held": 9 for m in ("precision", "recall", "fbeta")},
            }
        )
    return out


def main() -> int:
    # The thinner.
    labels = np.zeros(1000)
    labels[:20] = 1
    keep = a.thin_mask(labels, 0.05, np.random.default_rng(0))
    check(bool(keep[:20].all()), "thin_mask keeps every positive")
    check(int(keep.sum()) == 400 and abs(20 / keep.sum() - 0.05) < 1e-9, "thin_mask lands the prevalence (20 of 400)")
    check(
        bool(a.thin_mask(labels, 0.001, np.random.default_rng(0)).all()),
        "thin_mask keeps everything when already sparser",
    )

    # The summary on planted rows: 4 sessions, 3 held, picks 20/25/30/35, stops width x3 + budget.
    rows = pd.DataFrame(
        _rows("w", 1.0, 0.2, 40, [1, 1, 1, 0], [20, 25, 30, 35], ["width", "width", "width", "budget"])
        + _rows("w", 1.0, 0.3, 40, [1, 1, 0, 0], [10, 10, 10, 10], ["width"] * 4)
        + _rows("w", 1.0, 0.15, 80, [1, 1, 1, 1], [60, 60, 60, 60], ["width"] * 4)
    )
    s = a.summarise(rows)
    r = s[(s["width"] == 0.2) & (s["budget"] == 40)].iloc[0]
    check(abs(r["precision_cov"] - 0.75) < 1e-9, "summary coverage = 3/4")
    check(abs(r["picks_mean"] - 27.5) < 1e-9 and r["picks_p90"] > 30, "summary picks mean 27.5")
    check(abs(r["matches_width"] - 0.75) < 1e-9 and abs(r["matches_budget"] - 0.25) < 1e-9, "summary stop shares")
    check(abs(r["tail_err"] - (-1.0)) < 1e-9 and abs(r["below_abs_err"] - 1.0) < 1e-9, "summary tail and below errors")
    check(abs(r["edges_precision_cov"] - 0.9) < 1e-9, "summary edge coverage 36/40")

    # The pick rule: 0.15/80 covers 100% at 60 picks, 0.2/40 covers 75% at 27.5, 0.3/40 covers 50% at 10.
    p = a.pick(s, bar=0.93).iloc[0]
    check(p["width"] == 0.15 and p["budget"] == 80 and p["eligible"] == 1, "pick takes the only eligible point")
    p = a.pick(s, bar=0.7).iloc[0]
    check(p["width"] == 0.2 and p["budget"] == 40, "pick takes the cheapest eligible point")
    p = a.pick(s, bar=1.01).iloc[0]
    check(p["eligible"] == 0 and p["width"] == 0.15, "pick names the best coverage when nothing is eligible")

    # The cell path on a planted snapshot: one row per grid point x Test seed, the harness row reproduced.
    rng = np.random.default_rng(1)
    n = 600
    y = np.zeros(n, dtype=np.int8)
    y[rng.choice(n, 30, replace=False)] = 1
    scores = np.clip(rng.normal(0.3, 0.1, n) + 0.35 * y, 0.01, 0.99).astype(np.float32)
    model = ClassScoreModel(mu_pos=1.0, mu_neg=-1.0, sigma=0.5, n_pos=10, n_neg=20)
    meta = {"t": 50, "phase": "hard", "train_threshold": 0.6, "beta": 1.0, "model": model.as_dict()}
    with tempfile.TemporaryDirectory() as td:
        cells = Path(td) / "cells"
        cells.mkdir()
        np.savez_compressed(
            cells / "task_0000__testscores.npz",
            **{
                "siglip/whole_image/last/scores": scores,
                "siglip/whole_image/last/labels": y,
                "siglip/whole_image/last/meta": np.array(json.dumps(meta)),
            },
        )
        snap = {**meta, "scores": scores, "labels": y}
        harness = {
            "seed": 3,
            "dataset": "d",
            "category": "cat@small",
            "calibration_seed": "",
            "style": "whole_image",
            **a.row_from_snapshot(snap, seed=3),
        }
        pd.DataFrame([harness]).to_csv(cells / "task_0000__linetest.csv", index=False)
        rows, chk = a.cell_rows(
            (
                "w",
                str(cells / "task_0000__testscores.npz"),
                cells / "task_0000__linetest.csv",
                (0.2, 0.3),
                (20, 40),
                2,
                None,
                a.parse_variants("0.05:5:0.25,0:5:0.001"),
            )
        )
        check(len(rows) == (2 * 2 + 1) * 2, "cell_rows: one row per grid point (plus each other walk) x Test seed")
        check(
            sum(1 for r in rows if r["walk"] != a.DEFAULT_WALK) == 2
            and all(r["width"] == 0.2 and r["budget"] == 40 for r in rows if r["walk"] != a.DEFAULT_WALK),
            "cell_rows: a walk variant runs at the plan's width and budget",
        )
        check(chk["replayed"] and chk["same"], f"cell_rows: the harness row replays identically ({chk['diff']})")
        check(
            all(r["category"] == "cat@small" and r["seed"] == 3 and r["band"] == "small" for r in rows),
            "cell_rows: identity from the harness row",
        )
        thinned, _ = a.cell_rows(
            (
                "w",
                str(cells / "task_0000__testscores.npz"),
                cells / "task_0000__linetest.csv",
                (0.2,),
                (40,),
                1,
                0.2,
                [],
            )
        )
        check(
            thinned[0]["n_test"] == 150 and thinned[0]["n_test_pos"] == 30,
            "cell_rows: a thinned world's corpus is thinned before the line",
        )
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
