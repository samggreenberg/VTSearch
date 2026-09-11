#!/usr/bin/env python
"""Planted-answer check on `analyze_calseed_3796.py`, run BEFORE the array.

This study's output is a set of *spreads*, and a spread is the one statistic
that looks entirely plausible when it is computed on the wrong axis.  An sd
taken over the 150 steps of one trajectory, or over the draws AND the seeds at
once, would come back a well-formed number several times the real one — the
`rule_inefficiency` failure of #2897 in a different costume.  So the analyzer is
run here on fabricated cells whose spread is known by construction.

Six things are planted, each a different way to be wrong:

1. **A known across-draw sd**, and a *different* known across-seed sd, so a
   decomposition that pools the two axes cannot pass.
2. **A shrinking spread**: the draw effect is scaled down with vote count, so
   the spread-vs-votes curve has to fall — deliverable 3, and the direction the
   mechanics predict.
3. **A per-geometry difference**: the region geometry is built three times as
   noisy as the binary ones, so a report quoting one pooled number for the
   study is visibly wrong.
4. **A pin at the median**: draw 42's effect is the median of the set, so its
   percentile must come out at 0.5.  (A shifted-pin variant is then checked to
   confirm the percentile can move at all — a statistic pinned to 0.5 by a bug
   would pass the first check on its own.)
5. **A pick log beside the cells**, agreeing until a planted click and diverging
   after it, so the divergence onset is a number to recover rather than a shape
   to eyeball — and so a side frame leaking into the metric frame shows up as a
   changed cell count.
6. **A single-draw run**, which is a default run and not this study, and must be
   refused rather than analysed into a table of zeros.

Run: `python selftest_analyze_calseed_3796.py`
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PIN = 42
#: 42 plus nine others.  The draw EFFECTS below are what is planted; the labels
#: are arbitrary, exactly as they are in the real grid.
DRAWS = (PIN, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
GEOMETRIES = (
    ("siglip", "whole_image"),
    ("siglip+dinov3_patch", "whole_image"),
    ("siglip+dinov3_patch", "max_patch"),
)
CATEGORIES = [f"cls{i}" for i in range(4)]
SEEDS = (0, 1, 2, 3)
STEPS = 150

#: The planted per-draw offsets, in cost units, before the per-geometry gain and
#: the vote-count decay.  ELEVEN draws, not ten: the pin's percentile is a
#: mid-rank, so 42 is the exact median only when the other draws split evenly
#: either side of it, and an even grid cannot do that (five below and four above
#: puts an unbiased pin at 0.55, which would then have to be written into the
#: test as an expectation rather than as the answer).
DRAW_EFFECT = {d: e for d, e in zip(DRAWS, (0.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0))}
#: The planted per-cell-seed offsets.  Deliberately WIDER than the draw effects
#: on the binary geometries and NARROWER on the region one, so `share_draw` has
#: to come out on opposite sides of 0.5 for the two.
SEED_EFFECT = {s: e for s, e in zip(SEEDS, (-0.6, -0.2, 0.2, 0.6))}

#: Per-geometry gain on the draw effect.  Region is three times as noisy.
GEOM_GAIN = {"whole_image": 0.01, "max_patch": 0.03}
#: Per-geometry gain on the SEED effect.  Flat, so the two shares differ only
#: because the draw half moved.
SEED_GAIN = 0.02


#: The draw effect decays with votes: at t=150 it is half what it was at t=1.
def decay(t: int) -> float:
    return 1.0 - 0.5 * (t - 1) / (STEPS - 1)


#: The click after which the draws' pick logs disagree.
DIVERGE_AT = 17


def planted_cost(style: str, draw: int, seed: int, t: int, rng: np.random.Generator) -> float:
    return (
        0.30 + GEOM_GAIN[style] * DRAW_EFFECT[draw] * decay(t) + SEED_GAIN * SEED_EFFECT[seed] + rng.normal(0.0, 1e-5)
    )


def build(cells: Path, draws=DRAWS, pin_effect_shift: float = 0.0) -> None:
    """Write one `task_*.csv` (+ pick log) per (geometry, class, seed, draw)."""
    rng = np.random.default_rng(11)
    cells.mkdir(parents=True, exist_ok=True)
    idx = 0
    for draw in draws:
        for emb, style in GEOMETRIES:
            for cat in CATEGORIES:
                for seed in SEEDS:
                    rows = []
                    picks = []
                    for t in range(1, STEPS + 1):
                        cost = planted_cost(style, draw, seed, t, rng)
                        if draw == PIN:
                            cost += pin_effect_shift
                        rows.append(
                            {
                                "seed": seed,
                                "dataset": "vg_scale_any",
                                "category": cat,
                                "style": style,
                                "t": t,
                                "n_good": t // 2,
                                "n_bad": t - t // 2,
                                "gmm_variant": "",
                                "schedule": "",
                                "pool_variant": "",
                                "threshold": 0.5 + 0.05 * DRAW_EFFECT[draw],
                                "cost": cost,
                                # The RANKING metrics do not move with the draw
                                # at all here, which is the null for "did the
                                # trajectories diverge, or only the cut?".
                                "auroc": 0.88,
                                "oracle_cost": 0.25,
                                "regret": cost - 0.25,
                                "f1": 1.0 - cost,
                                "average_precision": 1.0 - cost,
                                "embedder": emb,
                                "seed_mode": "text",
                                "seed_query": cat,
                                "seed_embedder": "siglip",
                                "calibration_seed": draw,
                                "calibration_fraction": 0.5,
                            }
                        )
                        # Every draw picks the same media until DIVERGE_AT, then
                        # each picks its own.  A cell's pick log holds ONE draw,
                        # as a real one does; the disagreement is across files.
                        picks.append(
                            {
                                "seed": seed,
                                "dataset": "vg_scale_any",
                                "category": cat,
                                "style": style,
                                "embedder": emb,
                                "calibration_seed": draw,
                                "t": t,
                                "picked_id": t if t < DIVERGE_AT else t * 1000 + draw,
                            }
                        )
                    pd.DataFrame(rows).to_csv(cells / f"task_{idx:04d}.csv", index=False)
                    pd.DataFrame(picks).to_csv(cells / f"task_{idx:04d}__picks.csv", index=False)
                    idx += 1


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="calseed-selftest-"))
    try:
        results = tmp / "run" / "results"
        out = tmp / "out"
        build(results / "cells")

        import analyze_calseed_3796 as A

        if A.main(["--results", str(results), "--out", str(out), "--no-figures"]) != 0:
            raise SystemExit("analyze_calseed_3796 failed on the fabricated cells")

        failures: list[str] = []

        spread = pd.read_csv(out / "draw_spread.csv")
        cost = spread[spread["metric"] == "cost"]

        # (5b) the pick log stayed out of the metric frame: one block per
        # (geometry, class, seed), no more.
        n_blocks = cost[["geometry", "category", "seed"]].drop_duplicates().shape[0]
        expected_blocks = len(GEOMETRIES) * len(CATEGORIES) * len(SEEDS)
        if n_blocks != expected_blocks:
            failures.append(f"{n_blocks} blocks != {expected_blocks} - a side frame leaked into the main frame")

        # (1) the across-draw sd is the planted one.  Within a band the decay is
        # not constant, so the expectation is the band-mean decay times the sd
        # of the effects, which is what the analyzer's band means also see.
        eff_sd = float(np.std(list(DRAW_EFFECT.values()), ddof=1))
        for (geom, band), g in cost.groupby(["geometry", "band"]):
            style = geom.split("/")[-1]
            lo, hi = next((lo, hi) for name, lo, hi in A.BANDS if name == band)
            want = GEOM_GAIN[style] * eff_sd * float(np.mean([decay(t) for t in range(lo, hi + 1)]))
            got = float(g["sd"].median())
            if abs(got - want) > max(2e-4, 0.05 * want):
                failures.append(f"{geom}/{band}: across-draw sd {got:.5f} != planted {want:.5f}")

        # (3) the region geometry is three times as noisy as the binary ones, in
        # every band.  A pooled number would sit between them and match neither.
        for band in cost["band"].unique():
            b = cost[cost["band"] == band]
            reg = float(b[b["mode"] == "region"]["sd"].median())
            bina = float(b[b["mode"] == "binary"]["sd"].median())
            if not 2.5 < reg / bina < 3.5:
                failures.append(f"{band}: region/binary sd ratio {reg / bina:.2f} != the planted 3")

        # (2) and it SHRINKS with votes, in every geometry.
        by_step = pd.read_csv(out / "spread_by_step.csv")
        for geom, g in by_step.groupby("geometry"):
            early = float(g[g["t"] == 20]["median"].iloc[0])
            deep = float(g[g["t"] == 150]["median"].iloc[0])
            if not deep < early * 0.75:
                failures.append(f"{geom}: spread did not fall with votes ({early:.5f} -> {deep:.5f})")

        # (1b) the decomposition separates the two axes.  Binary: the seed
        # effect is wider, so the draw owns a MINORITY of the variance.  Region:
        # the draw effect is three times bigger, so it owns a majority.
        vc = pd.read_csv(out / "variance_components.csv")
        vcc = vc[vc["metric"] == "cost"]
        for mode, want_side in (("binary", "lt"), ("region", "gt")):
            share = float(vcc[vcc["mode"] == mode]["share_draw"].median())
            if want_side == "lt" and not share < 0.45:
                failures.append(f"{mode}: share_draw {share:.3f} should be well under 0.5")
            if want_side == "gt" and not share > 0.55:
                failures.append(f"{mode}: share_draw {share:.3f} should be well over 0.5")
        # ... and it recovers the planted SEED sd, which is the half a pooled sd
        # would silently fold into the draw term.
        want_seed = SEED_GAIN * float(np.std(list(SEED_EFFECT.values()), ddof=1))
        got_seed = float(vcc["sd_seed"].median())
        if abs(got_seed - want_seed) > max(5e-4, 0.1 * want_seed):
            failures.append(f"sd_seed {got_seed:.5f} != planted {want_seed:.5f}")

        # (4) the pin sits at the median of its own block, by construction.
        pct = cost["pin_percentile"].dropna()
        if abs(float(pct.mean()) - 0.5) > 0.02:
            failures.append(f"pin percentile {float(pct.mean()):.3f} != 0.5 with 42 planted at the median")

        # (5) the divergence onset is the planted click.
        onset = pd.read_csv(out / "divergence_onset.csv")
        got = sorted(onset["first_divergent_click"].dropna().unique())
        if got != [float(DIVERGE_AT)]:
            failures.append(f"first divergent click {got} != [{float(DIVERGE_AT)}]")
        if bool(onset["never_diverged"].any()):
            failures.append("a block reported as never diverging, but every block's picks were built to diverge")

        # The null on the ranking metrics: nothing in the fabricated data moves
        # auroc with the draw, so its spread must be zero.  This is what says a
        # non-zero auroc spread in the real run is the closed loop and not the
        # analyzer pooling two things.
        auroc = spread[spread["metric"] == "auroc"]["sd"].abs().max()
        if not (pd.isna(auroc) or auroc < 1e-9):
            failures.append(f"auroc spread {auroc:.3g} should be exactly zero on data where the ranking is fixed")

        # (4b) the percentile is not pinned to 0.5 by a bug: shift the pin below
        # every other draw and it must go to the bottom.
        shifted = tmp / "shifted" / "results"
        build(shifted / "cells", pin_effect_shift=-1.0)
        if A.main(["--results", str(shifted), "--out", str(tmp / "out_shift"), "--no-figures"]) != 0:
            raise SystemExit("analyze_calseed_3796 failed on the shifted-pin cells")
        sp2 = pd.read_csv(tmp / "out_shift" / "draw_spread.csv")
        pct2 = sp2[sp2["metric"] == "cost"]["pin_percentile"].dropna()
        if float(pct2.max()) > 0.1:
            failures.append(f"a pin planted below every draw came back at percentile {float(pct2.max()):.3f}")

        # (6) a single-draw run is a DEFAULT run and must be refused.
        one = tmp / "one" / "results"
        build(one / "cells", draws=(PIN,))
        try:
            A.main(["--results", str(one), "--out", str(tmp / "out_one"), "--no-figures"])
        except SystemExit:
            pass  # what it is supposed to do
        else:
            failures.append("a one-draw (default) run was analysed instead of refused")

        if failures:
            for x in failures:
                print(f"FAIL: {x}")
            return 1
        print(
            "selftest_analyze_calseed_3796: OK (draw sd, seed sd, share, shrink, geometry ratio, "
            "pin percentile, divergence onset, picks, default-run guard)"
        )
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
