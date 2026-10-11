#!/usr/bin/env python
"""Price counts tuned for the Goods' centroid (#4759) against the shipped count line, from tagged rows.

#4732 drew the centroid's line by the typed query's count (#4603): ``beta ** 0.708`` times the media
above ``median + 4 sigma`` beyond a Gaussian bulk's share. On FHIBE's face crops that keeps one of a
person's ~3 withheld photos. The runs here carry ``CALIB_CENTROID_LINE_VARIANTS`` rows for that line
(``count``), for each candidate in ``CENTROID_LINE_RULES`` and for the midpoint, each at beta 1/4, 1
and 4, on the same sessions. The centroid never picks and the opening reads no balance, so two lines
differ only on the centroid's clicks and only in what they keep.

Each run is read by ``analyze_centroid_line_4732.read_cell`` (every run in every average, #4631; a run
that found no Good scores 0 under every line). That reader gives each rule's F-beta curve minus the
midpoint's, so a candidate's difference to ``count`` is the difference of two such differences,
paired per run. Session means are over votes 1-10, 1-50 and 1-150; a run cut at H votes is exact over
1-150 when its centroid tier ended inside H (``tier_past_h``), because the curves agree from there.

Writes under ``--out``:

- ``cells.csv``: one row per run x rule x beta (FHIBE rows name people: keep owner-only);
- ``summary.csv``: per bench x rule x beta, the tier's size and quality and the paired difference to
  the reference with its SE (SE clusters on the category, the person on FHIBE);
- ``curves.csv``: per vote, each rule's mean difference to the reference at each beta (``d_ref``), and
  F-beta, precision and recall at the run's beta;
- ``summary.md``.

Usage::

    python analyze_centroid_count_4759.py --run fhibe-k1=<run dir> --run coco-b1=<run dir> --out <dir> \\
        --variants "$CALIB_CENTROID_LINE_VARIANTS" [--reference count] [--horizon 40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_centroid_line_4732 import SPANS, _mean_se, load_run  # noqa: E402

KEY = ["run", "dataset", "category", "seed", "beta"]


def paired(cells: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Each rule's per-run difference to *reference*, over every span: ``dd_<span>``."""
    ref = cells[cells["rule"] == reference].set_index(KEY)
    out = cells.set_index(KEY)
    for span in SPANS:
        out[f"dd_{span}"] = out[f"d_fbeta_{span}"] - ref[f"d_fbeta_{span}"].reindex(out.index)
    return out.reset_index()


def summary_table(cells: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for (run, ds, rule, beta), h in cells.groupby(["run", "dataset", "rule", "beta"], sort=False):
        cl = h["category"]
        rec = {
            "run": run,
            "dataset": ds,
            "rule": rule,
            "beta": beta,
            "runs": len(h),
            "no_good": int(h["no_good"].sum()),
            "tier_clicks_mean": h["n_tier"].mean(),
            "tier_kept_median": h["tier_n_median"].median(),
            "tier_precision": h["tier_precision"].mean(),
            "tier_recall": h["tier_recall"].mean(),
            "tier_fbeta": h["tier_fbeta"].mean(),
            "tier_past_h": h["tier_past_h"].mean(),
        }
        for span in SPANS:
            m, se, n = _mean_se(h[f"dd_{span}"], cl)
            rec[f"dd_{span}"], rec[f"dd_{span}_se"], rec["clusters"] = m, se, n
        if (h["beta"] == h["run_beta"]).all():
            rec["fbeta_1-10"] = h["fbeta_1-10"].mean()
            rec[f"fbeta_1-{horizon}"] = h[f"fbeta_1-{horizon}"].mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(summary: pd.DataFrame, reference: str, horizon: int) -> str:
    lines = [f"# #4759: counts for the centroid, paired against `{reference}` (H = {horizon} votes)", ""]
    lines.append(
        f"dF is the rule's mean F-beta minus `{reference}`'s over the votes named, paired per run, at the "
        "row's beta; SE clusters on the category (the person on FHIBE). Kept, precision, recall and F-beta "
        "are on the centroid's own clicks."
    )
    lines.append("")
    for (run, ds), g in summary.groupby(["run", "dataset"], sort=False):
        lines.append(f"## {run} / {ds}")
        lines.append("")
        lines.append(
            "| rule | beta | runs | tier clicks | kept (median) | precision | recall | tier F-beta "
            "| dF 1-10 | dF 1-50 | dF 1-150 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in g.sort_values(["beta", "rule"]).iterrows():
            lines.append(
                f"| {r['rule']} | {r['beta']:g} | {r['runs']} | {r['tier_clicks_mean']:.1f} | {r['tier_kept_median']:.0f} "
                f"| {r['tier_precision']:.2f} | {r['tier_recall']:.2f} | {r['tier_fbeta']:.3f} | "
                + " | ".join(f"{r[f'dd_{s}']:+.3f} ± {r[f'dd_{s}_se']:.3f}" for s in SPANS)
                + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="label=<run dir>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", required=True, help="the runs' CALIB_CENTROID_LINE_VARIANTS (with midpoint@*)")
    ap.add_argument("--reference", default="count")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run-beta", type=float, default=1.0, help="the runs' balance, for the runs that found no Good")
    args = ap.parse_args()
    variants = [(v.split("@")[0], float(v.split("@")[1])) for v in args.variants.split(",") if v]
    if not {"midpoint", args.reference} <= {r for r, _ in variants}:
        raise SystemExit("the variants must include midpoint (the reader's anchor) and the reference")
    args.out.mkdir(parents=True, exist_ok=True)
    frames, curves = [], {}
    for spec in args.run:
        label, _, run = spec.partition("=")
        df, cv = load_run(label, Path(run), args.horizon, args.workers, variants, args.run_beta)
        frames.append(df)
        curves.update(cv)
    cells = paired(pd.concat(frames, ignore_index=True), args.reference)
    cells.to_csv(args.out / "cells.csv", index=False)
    summary = summary_table(cells, args.horizon)
    summary.to_csv(args.out / "summary.csv", index=False)
    rows = []
    for (label, ds, rule, beta), c in curves.items():
        ref = curves.get((label, ds, args.reference, beta), {})
        for m, vs in c.items():
            mean = np.mean(np.stack(vs), axis=0)
            if m == "d_fbeta":
                if "d_fbeta" not in ref:
                    continue
                mean = mean - np.mean(np.stack(ref["d_fbeta"]), axis=0)
                m = "d_ref"
            rows += [
                {"run": label, "dataset": ds, "rule": rule, "beta": beta, "metric": m, "t": t + 1, "value": float(v)}
                for t, v in enumerate(mean)
            ]
    pd.DataFrame(rows).to_csv(args.out / "curves.csv", index=False)
    (args.out / "summary.md").write_text(summarize(summary, args.reference, args.horizon))
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
