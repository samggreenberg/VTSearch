#!/usr/bin/env python
"""Aggregate ``run_summary_power_5568.py`` into the #5568 tables and figure.

    python analyze_summary_power_5568.py --results DIR --out DIR [--figure PATH]

Per (embedder, arm), over 3 seeds:

- **null**: the held-out in-domain p-values' KS from uniform (median over the
  5 datasets x 3 seeds) and their share below alpha (mean), and the false-alarm
  rate of the ``shifted`` verdict on the 15 own-holdout pairs;
- **whole-dataset pairs** (#3348's): detection on the different-source pairs
  and separation = detection - false alarm, as #3348 defines them;
- **partial shift** (30 % of the query from another dataset): detection on the
  different-source pairs, and how much of the full report's signal the arm
  keeps -- *capture*, the arm's excess over alpha as a share of the full
  report's, and *recall*, the share of the items the full report flags that
  the arm also flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_node_models_3348 import SOURCES

ALPHA = 0.05
FULL = "full"
#: A pair enters the capture ratio only when the full report's excess is at
#: least this, so the ratio is not a quotient of two noise terms.
MIN_EXCESS = 0.02
ARM_ORDER = ["full", "sum32", "sum64", "sum128", "sum256", "sum512", "rand128", "radius128"]


def load(results: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    pit = pd.concat([pd.read_csv(f) for f in sorted(results.glob("pit_*.csv"))], ignore_index=True)
    shift = pd.concat([pd.read_csv(f) for f in sorted(results.glob("shift_*.csv"))], ignore_index=True)
    timing = [json.loads(f.read_text()) for f in sorted(results.glob("timing_*.json"))]
    return pit, shift, timing


def measures(pit: pd.DataFrame, shift: pd.DataFrame) -> pd.DataFrame:
    p = pit[(pit["scope"] == "holdout")]
    null = (
        p.pivot_table(index=["embedder", "family"], columns="statistic", values="value", aggfunc=list)
        .apply(lambda r: pd.Series({"ks": np.median(r["ks_uniform"]), "rate05": np.mean(r["frac_below_0.05"])}), axis=1)
        .reset_index()
        .rename(columns={"family": "variant"})
    )
    s = shift.copy()
    s["kind"] = np.where(
        s["is_self"],
        "self",
        np.where(s["build_dataset"].map(SOURCES) == s["query_dataset"].map(SOURCES), "same", "diff"),
    )
    # The full report's fraction on the same pair, for the capture ratio.
    key = ["embedder", "seed", "build_dataset", "query_dataset", "mix"]
    full = s[s["variant"] == FULL][[*key, "frac_atypical"]].rename(columns={"frac_atypical": "frac_full"})
    s = s.merge(full, on=key)
    s["capture"] = np.where(
        s["frac_full"] - ALPHA >= MIN_EXCESS, (s["frac_atypical"] - ALPHA) / (s["frac_full"] - ALPHA), np.nan
    )
    g = ["embedder", "variant"]
    whole = s[s["mix"] == 0]
    out = null.set_index(g)
    out["false_alarm"] = whole[whole["kind"] == "self"].groupby(g)["shifted"].mean()
    out["detect"] = whole[whole["kind"] == "diff"].groupby(g)["shifted"].mean()
    out["same_fire"] = whole[whole["kind"] == "same"].groupby(g)["shifted"].mean()
    out["separation"] = out["detect"] - out["false_alarm"]
    out["capture_whole"] = whole[whole["kind"] == "diff"].groupby(g)["capture"].mean()
    part = s[(s["mix"] > 0) & (s["kind"] == "diff")]
    out["detect_mix"] = part.groupby(g)["shifted"].mean()
    out["frac_mix"] = part.groupby(g)["frac_atypical"].mean()
    out["capture_mix"] = part.groupby(g)["capture"].mean()
    out["recall_mix"] = part.groupby(g)["recall_of_full"].mean()
    out["n_mix_pairs"] = part.groupby(g)["shifted"].size()
    out["n_capture_pairs"] = part.groupby(g)["capture"].count()
    out = out.reset_index()
    out["variant"] = pd.Categorical(out["variant"], ARM_ORDER, ordered=True)
    return out.sort_values(["embedder", "variant"]).reset_index(drop=True)


def overall(table: pd.DataFrame) -> pd.DataFrame:
    """Each arm averaged over embedders, with its worst embedder beside it."""
    cols = ["ks", "rate05", "false_alarm", "separation", "detect_mix", "capture_mix", "recall_mix"]
    g = table.groupby("variant", observed=True)
    o = g[cols].mean()
    o["min_separation"] = g["separation"].min()
    o["min_capture_mix"] = g["capture_mix"].min()
    o["max_ks"] = g["ks"].max()
    return o.reset_index()


def fmt_md(table: pd.DataFrame, ov: pd.DataFrame) -> str:
    def md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    parts = ["# #5568 summary power: aggregates", "", "## Over embedders", "", md(ov), ""]
    for e in sorted(table["embedder"].unique()):
        parts += [f"## {e}", "", md(table[table["embedder"] == e].drop(columns=["embedder"])), ""]
    return "\n".join(parts)


def figure(table: pd.DataFrame, path: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.ticker  # noqa: PLC0415

    ns = [32, 64, 128, 256, 512]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for e in sorted(table["embedder"].unique()):
        t = table[table["embedder"] == e].set_index("variant")
        axes[0].plot(ns, [t.loc[f"sum{n}", "capture_mix"] for n in ns], marker="o", label=e)
        axes[1].plot(ns, [t.loc[f"sum{n}", "rate05"] for n in ns], marker="o", label=e)
    axes[0].axhline(1.0, color="grey", lw=0.8, ls="--")
    axes[0].set(xscale="log", xticks=ns, xticklabels=ns, xlabel="anchors", ylim=(0, 1.1))
    axes[0].set_ylabel("share of the full report's excess kept\n(30 % shift, different source)")
    axes[1].axhspan(0.03, 0.075, color="0.9")
    axes[1].axhline(ALPHA, color="grey", lw=0.8, ls="--")
    axes[1].set(xscale="log", xticks=ns, xticklabels=ns, xlabel="anchors", ylabel="in-domain holdout share below 0.05")
    axes[1].set_ylim(0, 0.1)
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.axvline(128, color="0.6", lw=0.8)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(ns, [str(n) for n in ns])
    fig.savefig(path, dpi=110)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure")
    args = ap.parse_args(list(argv) if argv is not None else None)
    pit, shift, timing = load(Path(args.results))
    table = measures(pit, shift)
    ov = overall(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table.csv", index=False)
    ov.to_csv(out / "overall.csv", index=False)
    (out / "tables.md").write_text(fmt_md(table, ov))
    q = pd.DataFrame([{"embedder": t["embedder"], "seed": t["seed"], **t["query_seconds"]} for t in timing])
    q.to_csv(out / "query_seconds.csv", index=False)
    if args.figure:
        figure(table, Path(args.figure))
    print(ov.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
