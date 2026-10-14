#!/usr/bin/env python
"""EmbeddingGemma 2 against SigLIP on COCO Better's Binary path, session for session (#4837).

``eg2_4837.sh`` runs both embedders in one grid, so every EmbeddingGemma 2 session pairs with the SigLIP
session of the same class, size band and seed: the same split, the same withheld half. Each arm opens on its
own typed query and learns in its own space. The arm's effect is the mean paired difference, with its SE
over classes (49 on COCO Better; a class's bands and seeds are not independent draws of anything).

Per preset (beta 1/4, 1, 4), for each arm and the difference:

* the owner's single number, the **session mean**: each run's objective (F-beta of the withheld set at the
  app's line, at the run's own beta) averaged over clicks 1-150, filled as ``by_click.py`` fills it (the typed
  query's set until the app shows a detector, then the last value carried); also over clicks 1-50;
* click 0, the typed query alone: its objective at its own line and its AP (the ranking);
* AP at 25 and 50 clicks and at the end, the objective after the check, and the full-label ceiling's AP
  (what the space can do with every label: the embedder's own limit);
* the runs that never trained a detector.

    python eg2_compare_4837.py --run 0.25=<b025 dir> --run 1=<b1 dir> --run 4=<b4 dir> --out <dir>

Each run dir holds ``text_baseline.csv`` and ``analysis-{binary,eg2}/{cells,curves}.csv``
(``eg2_4837.sh analyze``). Writes ``compare.csv`` (per preset), ``by_band.csv``, ``by_class.csv``,
``objective_by_click.csv`` and ``figures/objective_by_click.png``; prints the tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from by_click import RUN, filled_runs, typed_query  # noqa: E402

ARMS = {"siglip": "binary", "embeddinggemma2": "eg2"}
LABEL = {"siglip": "SigLIP", "embeddinggemma2": "EmbeddingGemma 2"}
COLS = ("text_thr_fbeta", "text_ap", "ap_25", "ap_50", "final_ap", "thr_fbeta_final", "ceiling_ap")
HORIZON = 150


def arm_runs(run_dir: Path, embedder: str, beta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(runs, filled)``: one arm's per-run numbers and its filled objective at every click."""
    d = run_dir / f"analysis-{ARMS[embedder]}"
    cells = pd.read_csv(d / "cells.csv")
    curves = pd.read_csv(d / "curves.csv")
    base = pd.read_csv(run_dir / "text_baseline.csv")
    filled, _ = filled_runs(curves, cells, typed_query(base, embedder, beta))
    out = cells.set_index(RUN)[[c for c in (*COLS, "never_trained", "class", "band") if c in cells]].copy()
    for lo, hi in ((1, 50), (1, HORIZON)):
        clicks = [t for t in filled.columns if lo <= int(t) <= hi]
        out[f"session_{lo}_{hi}"] = filled[clicks].mean(axis=1).reindex(out.index)
    return out, filled


def paired(sig: pd.DataFrame, eg2: pd.DataFrame, cols: list[str]) -> dict:
    """Arm means, the mean paired difference and its SE over classes."""
    j = sig.join(eg2, lsuffix="_s", rsuffix="_e", how="inner")
    row: dict = {"runs": len(j), "unpaired": len(sig) + len(eg2) - 2 * len(j)}
    cls = j["class_s"]
    for col in cols:
        s, e = j[f"{col}_s"], j[f"{col}_e"]
        diff = (e - s).groupby(cls).mean().dropna()
        row[f"{col}_siglip"] = float(s.mean())
        row[f"{col}_eg2"] = float(e.mean())
        row[f"d_{col}"] = float((e - s).mean())
        row[f"se_{col}"] = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    if "never_trained_s" in j:
        row["never_trained_siglip"] = int(j["never_trained_s"].astype(bool).sum())
        row["never_trained_eg2"] = int(j["never_trained_e"].astype(bool).sum())
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="beta=run dir")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["session_1_150", "session_1_50", *COLS]
    rows, bands, classes, curves = [], [], [], []
    for spec in args.run:
        beta_s, d = spec.split("=", 1)
        beta = float(beta_s)
        sig, fs = arm_runs(Path(d), "siglip", beta)
        eg2, fe = arm_runs(Path(d), "embeddinggemma2", beta)
        present = [c for c in cols if c in sig and c in eg2]
        rows.append({"beta": beta} | paired(sig, eg2, present))
        for band in sorted(sig["band"].dropna().unique()):
            bands.append(
                {"beta": beta, "band": band} | paired(sig[sig["band"] == band], eg2[eg2["band"] == band], present)
            )
        for cls in sorted(sig["class"].dropna().unique()):
            s, e = sig[sig["class"] == cls], eg2[eg2["class"] == cls]
            j = s.join(e, lsuffix="_s", rsuffix="_e", how="inner")
            classes.append(
                {"beta": beta, "class": cls, "runs": len(j)}
                | {f"{c}_siglip": float(j[f"{c}_s"].mean()) for c in ("session_1_150", "text_ap", "final_ap")}
                | {f"{c}_eg2": float(j[f"{c}_e"].mean()) for c in ("session_1_150", "text_ap", "final_ap")}
                | {
                    f"d_{c}": float((j[f"{c}_e"] - j[f"{c}_s"]).mean())
                    for c in ("session_1_150", "text_ap", "final_ap", "ceiling_ap")
                }
            )
        common = fs.index.intersection(fe.index)
        for t in [c for c in fs.columns if 0 <= int(c) <= HORIZON]:
            curves.append(
                {
                    "beta": beta,
                    "t": int(t),
                    "siglip": float(fs.loc[common, t].mean()),
                    "eg2": float(fe.loc[common, t].mean()),
                }
            )
    comp = pd.DataFrame(rows)
    comp.to_csv(args.out / "compare.csv", index=False)
    pd.DataFrame(bands).to_csv(args.out / "by_band.csv", index=False)
    by_class = pd.DataFrame(classes)
    by_class.to_csv(args.out / "by_class.csv", index=False)
    curve = pd.DataFrame(curves)
    curve.to_csv(args.out / "objective_by_click.csv", index=False)

    pd.set_option("display.width", 200)
    show = ["beta", "runs", "unpaired"]
    for c in ("session_1_150", "session_1_50", "text_thr_fbeta", "text_ap", "final_ap", "ceiling_ap"):
        show += [f"{c}_siglip", f"{c}_eg2", f"d_{c}", f"se_{c}"]
    print(comp[[c for c in show if c in comp]].round(4).T.to_string())
    if not by_class.empty:
        print(by_class[by_class["beta"] == 1.0].sort_values("d_session_1_150").round(3).to_string(index=False))

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    betas = sorted(curve["beta"].unique())
    fig, axes = plt.subplots(1, len(betas), figsize=(5.2 * len(betas), 4), sharey=True, squeeze=False)
    for ax, beta in zip(axes[0], betas, strict=True):
        c = curve[curve["beta"] == beta]
        ax.plot(c["t"], c["siglip"], color="#52514e", label=LABEL["siglip"])
        ax.plot(c["t"], c["eg2"], color="#2a78d6", label=LABEL["embeddinggemma2"])
        ax.set_title(f"beta {beta:g}")
        ax.set_xlabel("click")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("objective (withheld F-beta at the line)")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    (args.out / "figures").mkdir(exist_ok=True)
    fig.savefig(args.out / "figures" / "objective_by_click.png", dpi=130)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
