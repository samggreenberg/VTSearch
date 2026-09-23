"""Paired bench analysis for #3159, broken out by dataset and scale band.

    python analyze_grid_3159.py --root <bench root> --out <dir>

``analyze_bench_precision.py`` (#3143) gives the pooled headline and its
verdict against the 0.005 margin.  This adds what that one cannot: the same
paired statistic **per dataset and per scale band**, because the issue's
concern is specifically the sub-patch band, and a pooled null can hide one
band exactly the way #3143's pooled row hid a per-embedder split.

It also writes the measurement files the report's ``figures.py`` draws from, so
the figures rebuild without the GRID.

Method, unchanged from #3143: pair on (dataset, embedder, category, seed, style,
t); collapse each cell to its mean over the window; SE over cells.  Two windows:
the whole trajectory and the deep regime (t >= 100).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_bench_precision import CELL_KEY, PAIR_KEY  # noqa: E402

sys.path.insert(0, str(HERE.parent / "calibration"))
from _cells_io import _base_rows  # noqa: E402
from _cells_paths import main_frame_files  # noqa: E402

METRICS = ["cost", "average_precision", "fpr", "fnr", "n_good"]
MARGIN = 0.005


def load_arm(root: Path, arm: str) -> pd.DataFrame:
    """The main frames, only the columns this analysis reads.

    A cell's CSV carries ~100 columns and, since safe thresholds went on by
    default (#3400), ~33 rows per step: the production row plus counterfactual
    cut variants.  Only the production row (``_base_rows``) is paired - pairing
    on ``t`` alone would cross-join the variants.
    """
    want = set(PAIR_KEY) | set(METRICS) | {"gmm_variant", "schedule", "pool_variant", "threshold_provenance"}
    frames = [
        _base_rows(pd.read_csv(f, usecols=lambda c: c in want))
        for f in main_frame_files(root / arm / "results" / "cells")
    ]
    return pd.concat(frames, ignore_index=True)


def band_table(root: Path) -> dict[tuple[str, str], str]:
    """(dataset, category) -> scale band, from the arm's own prepare_info.json."""
    info = json.loads((root / "fp16" / "results" / "prepare_info.json").read_text())
    out: dict[tuple[str, str], str] = {}
    for ds, per_emb in info["datasets"].items():
        for _emb, rec in per_emb.items():
            for band, b in (rec.get("category_selection", {}).get("bands") or {}).items():
                for cat in b.get("selected", []):
                    out[(ds, cat)] = band
    return out


def paired(merged: pd.DataFrame, metric: str, t_from: int) -> pd.DataFrame:
    rc, ac = f"{metric}_ref", f"{metric}_arm"
    sub = merged[(merged["t"] >= t_from)].dropna(subset=[rc, ac])
    sub = sub.assign(_d=sub[ac] - sub[rc])
    g = sub.groupby(CELL_KEY)
    return pd.DataFrame({"ref": g[rc].mean(), "arm": g[ac].mean(), "diff": g["_d"].mean()}).reset_index()


def summarise(per_cell: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    groups = per_cell.groupby(by) if by else [((), per_cell)]
    for key, g in groups:
        n = len(g)
        mean = float(g["diff"].mean())
        se = float(g["diff"].std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=False))
        rec.update(
            {
                "n_cells": n,
                "fp16": float(g["ref"].mean()),
                "fp32": float(g["arm"].mean()),
                "diff_fp32_minus_fp16": mean,
                "se": se,
                "resolved_below_margin": bool(np.isfinite(se) and abs(mean) + 2 * se < MARGIN),
                "resolvable_from_zero": bool(np.isfinite(se) and abs(mean) > 2 * se),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deep-from", type=int, default=100)
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = load_arm(root, "fp16")
    arm = load_arm(root, "fp32")
    merged = ref.merge(arm, on=PAIR_KEY, suffixes=("_ref", "_arm"))
    bands = band_table(root)
    merged["band"] = [bands.get((d, c), "unbanded") for d, c in zip(merged["dataset"], merged["category"], strict=True)]
    n_cells = merged.groupby(CELL_KEY).ngroups
    print(f"paired cells: {n_cells} (fp16 {ref.groupby(CELL_KEY).ngroups}, fp32 {arm.groupby(CELL_KEY).ngroups})")

    # How often was the whole trajectory literally the same?  A cell whose every
    # step has an identical cost and n_good made the same decisions throughout.
    step_same = (merged["cost_ref"] == merged["cost_arm"]) & (merged["n_good_ref"] == merged["n_good_arm"])
    same = step_same.groupby([merged[k] for k in CELL_KEY]).all()
    first_div = merged.assign(
        diverged=(merged["cost_ref"] != merged["cost_arm"]) | (merged["n_good_ref"] != merged["n_good_arm"])
    )
    first_t = first_div[first_div["diverged"]].groupby(CELL_KEY)["t"].min()
    ident = {
        "cells": int(n_cells),
        "cells_identical_throughout": int(same.sum()),
        "share_identical": float(same.mean()),
        "first_divergence_t_median": float(first_t.median()) if len(first_t) else None,
        "first_divergence_t_quartiles": [float(x) for x in first_t.quantile([0.25, 0.75])] if len(first_t) else None,
        "steps_cost_bit_identical": float((merged["cost_ref"] == merged["cost_arm"]).mean()),
    }
    print(json.dumps(ident, indent=2))
    (out / "identity.json").write_text(json.dumps(ident, indent=2) + "\n")
    first_t.reset_index().rename(columns={"t": "first_divergence_t"}).to_csv(out / "first_divergence.csv", index=False)

    tables = []
    for metric in METRICS:
        for window, t_from in (("all", 0), ("deep", args.deep_from)):
            pc = paired(merged, metric, t_from)
            pc["band"] = [bands.get((d, c), "unbanded") for d, c in zip(pc["dataset"], pc["category"], strict=True)]
            pc.assign(metric=metric, window=window).to_csv(out / f"cells_{metric}_{window}.csv", index=False)
            for by in ([], ["dataset"], ["band"], ["dataset", "band"]):
                s = summarise(pc, by)
                s["metric"], s["window"] = metric, window
                s["split"] = "+".join(by) or "pooled"
                tables.append(s)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "paired_by_split.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 500, "display.max_columns", 20):
        for metric in ("cost", "average_precision"):
            print(f"\n=== {metric} ===")
            print(
                table[table["metric"] == metric][
                    [
                        "window",
                        "split",
                        "dataset",
                        "band",
                        "n_cells",
                        "fp16",
                        "fp32",
                        "diff_fp32_minus_fp16",
                        "se",
                        "resolved_below_margin",
                        "resolvable_from_zero",
                    ]
                ].to_string(index=False)
            )

    # Mean trajectory per arm and the mean paired difference per t, per dataset
    # and band: the averaged figure.  Per-cell traces come from cells_*.csv.
    traj = (
        merged.groupby(["dataset", "band", "t"])
        .agg(
            cost_fp16=("cost_ref", "mean"),
            cost_fp32=("cost_arm", "mean"),
            ap_fp16=("average_precision_ref", "mean"),
            ap_fp32=("average_precision_arm", "mean"),
            n=("cost_ref", "size"),
        )
        .reset_index()
    )
    traj.to_csv(out / "trajectory.csv", index=False)
    per_step = merged[
        [*CELL_KEY, "band", "t", "cost_ref", "cost_arm", "average_precision_ref", "average_precision_arm"]
    ]
    # Every 5th vote at 4 decimals: the per-cell figure needs the shape of each
    # trace, not every step, and the full frame (1.5 MB) is over the repo's
    # large-file limit.
    per_step[per_step["t"] % 5 == 0].round(4).to_csv(out / "per_step.csv.gz", index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
