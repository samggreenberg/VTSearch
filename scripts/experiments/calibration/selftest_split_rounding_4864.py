#!/usr/bin/env python3
"""Planted-answer self-test for #4864's step 0 (``split_cadence_4864``) and analyzer (``analyze_split_rounding_4864``).

* The ripple detector finds a planted sinusoid that every run shares, and does
  not find one whose phase differs run to run (the null it is meant to keep).
* Step 0's replay: the ``round`` reference phase-locks every run that votes one
  vote a step (its residual's shared amplitude dwarfs the null), the ``nearest``
  rule draws the labels line from exactly 9 Goods at a 0.3 split and 5 at 0.5,
  and the dither's chance of it rises across a band.
* The switch counter counts changes between the count line and the labels line,
  skips the fallback cut's clicks, and counts inside the band alone.

    python selftest_split_rounding_4864.py
"""

from __future__ import annotations

import sys

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_split_rounding_4864 as A  # noqa: E402
import split_cadence_4864 as S  # noqa: E402

FAILED: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILED.append(what)


def planted_curves(rng: np.random.Generator, amp: float, shared: bool) -> pd.DataFrame:
    """60 categories x 5 seeds of a rising curve plus noise, with a 0.3-cycle sinusoid of *amp*."""
    t = np.arange(0, A.HORIZON + 1)
    rows, idx = [], []
    for c in range(60):
        for s in range(5):
            phase = 0.0 if shared else rng.uniform(0, 2 * np.pi)
            y = 0.5 * (1 - np.exp(-t / 40)) + rng.normal(0, 0.03, len(t)) + amp * np.cos(2 * np.pi * 0.3 * t + phase)
            rows.append(y)
            idx.append((f"c{c}@small", s))
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=A.RUN), columns=t)


def test_ripple() -> None:
    rng = np.random.default_rng(1)
    locked, _ = A.ripple(planted_curves(rng, 0.004, shared=True), [0.3, 0.1], rng)
    t03 = locked["tests"]["0.3"]
    check(
        t03["x_noise"] > 4 and t03["p"] < 0.01,
        f"a shared 0.3-cycle ripple is found ({t03['x_noise']:.1f}x, p={t03['p']:.3g})",
    )
    check(
        abs(t03["amplitude"] - 0.004) < 0.001, f"its amplitude is recovered ({t03['amplitude']:.4f} of 0.004 planted)"
    )
    check(t03["ci_lo"] < 0.004 < t03["ci_hi"], "the planted amplitude sits in the bootstrap interval")
    check(locked["tests"]["0.1"]["x_noise"] < 3, "nothing is found where nothing was planted (0.1)")
    loose, _ = A.ripple(planted_curves(rng, 0.004, shared=False), [0.3], rng)
    check(
        loose["tests"]["0.3"]["p"] > 0.01,
        f"a run-by-run phase is not a shared ripple (p={loose['tests']['0.3']['p']:.3g})",
    )


def test_replay() -> None:
    # Runs that vote one vote a step, with Goods arriving at run-specific votes.
    rng = np.random.default_rng(2)
    rows = []
    for c in range(40):
        for s in range(3):
            first = int(rng.integers(5, 40))
            g = 2
            for t in range(first, A.HORIZON + 1):
                if rng.random() < 0.15:
                    g += 1
                rows.append(
                    {"category": f"c{c}", "seed": s, "style": "whole_image", "t": t, "n_good": g, "n_bad": t - g}
                )
    table = S.replay(pd.DataFrame(rows), 0.3)
    srng = np.random.default_rng(3)
    rnd, _ = S.cadence(table, "round:res_total", [0.3], srng)
    near, _ = S.cadence(table, "nearest:res_good", [0.3], srng)
    check(
        rnd["tests"]["0.3"]["x_noise"] > 5,
        f"round() phase-locks the runs ({rnd['tests']['0.3']['x_noise']:.1f}x noise)",
    )
    check(
        near["tests"]["0.3"]["amplitude"] < rnd["tests"]["0.3"]["amplitude"] / 5,
        "nearest's held-out Goods do not lock as round() does",
    )
    pl = S.p_labels_by_goods(table).set_index("n_good")
    check(
        pl.loc[8, "nearest"] == 0 and pl.loc[9, "nearest"] == 1,
        "nearest draws the labels line from exactly 9 Goods at 0.3",
    )
    check(0 < pl.loc[8, "dither"] < 1, f"the dither draws either line at 8 Goods (P = {pl.loc[8, 'dither']:.2f})")
    check(
        S.labels_band(table)[0] <= 8 <= S.labels_band(table)[1],
        f"the dither's band holds 8 Goods ({S.labels_band(table)})",
    )
    half = S.replay(
        pd.DataFrame(
            [
                {"category": "c", "seed": 0, "style": "max_patch", "t": g + 20, "n_good": g, "n_bad": 20}
                for g in range(2, 12)
            ]
        ),
        0.5,
    )
    first = half[(S.FOLDS * half["nearest:held_good"]) >= S.HEAD_COUNT_GOODS]["n_good"].min()
    check(first == 5, f"at a 0.5 split nearest draws the labels line from 5 Goods (got {first})")


def test_switches() -> None:
    lc = pd.DataFrame(
        {
            "category": ["a"] * 7,
            "seed": [0] * 7,
            "t": range(1, 8),
            "n_good": [6, 7, 8, 8, 9, 12, 13],
            "n_bad": [10] * 7,
            "model_n_pos": [4, 6, np.nan, 4, 6, 6, 4],
            "model_n_neg": [6] * 7,
        }
    )
    sw = A.switches(lc, (7, 11)).iloc[0]
    # Without the NaN click the kinds run count, labels, count, labels, labels, count: 4 switches.
    check(sw["switches"] == 4, f"switches over the run skip the fallback click ({sw['switches']} of 4)")
    # Inside 7-11 Goods: labels (7), count (8), labels (9): 2.
    check(sw["band_switches"] == 2, f"switches inside the band ({sw['band_switches']} of 2)")
    frame = pd.DataFrame(
        {
            "category": ["a"] * 7,
            "seed": [0] * 7,
            "t": range(1, 8),
            "phase": ["hard"] * 7,
            "app_trained": [0, 0, 1, 1, 1, 1, 1],
        }
    )
    # The in-sample fallback's clicks hold every vote: 7 Goods pass the gate though the folds held out 2 a fold.
    fb = pd.DataFrame(
        {
            "category": ["b"] * 5,
            "seed": [0] * 5,
            "t": range(30, 35),
            "n_good": [7, 7, 7, 8, 9],
            "n_bad": [23, 24, 25, 25, 25],
            "model_n_pos": [4, 7, 4, 4, 6],
            "model_n_neg": [14, 24, 16, 16, 16],
        }
    )
    fs = A.switches(fb, (7, 11)).iloc[0]
    check(fs["in_sample_clicks"] == 1, "the in-sample fallback's click is tagged")
    check(
        fs["switches"] == 3 and fs["held_out_switches"] == 1,
        f"its switches are told apart from the split's ({fs['switches']} all, {fs['held_out_switches']} held-out)",
    )
    shown = A.shown_only(lc, frame)
    check(list(shown["t"]) == [3, 4, 5, 6, 7], "only the clicks with a trained detector on screen are kept")


def main() -> int:
    test_ripple()
    test_replay()
    test_switches()
    print(f"\n{len(FAILED)} failed" if FAILED else "\nall planted answers recovered")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
