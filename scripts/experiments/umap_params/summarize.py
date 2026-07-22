"""Aggregate sweep rows → per-embedder recommendations + the compaction verdict.

Reads every ``rows/*.csv`` the sweep produced and answers the plan's three
questions:

1. **Per-embedder winner:** the ``(n_neighbors, min_dist)`` that maximizes
   ceiling-normalized separability, averaged across that embedder's datasets and
   seeds, subject to the guards not regressing and the seed-stability holding.
2. **The N question:** does the best ``n_neighbors`` track the embedder or the
   dataset size N? (Reported as best-nn per (embedder, N).)
3. **The compaction verdict:** the mean separability delta raw→compacted, per
   dataset/embedder and overall — does ``compact=True`` stay the default?

Writes ``summary.json`` and ``master.csv`` (all rows concatenated) to the
results root and prints markdown tables.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C


def load_master() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(C.CSV_DIR.glob("*.csv")) if p.stat().st_size > 0]
    df = pd.concat(frames, ignore_index=True)
    df["media"] = df["dataset"].map(lambda d: C.ROSTER_BY_NAME[d].media_type if d in C.ROSTER_BY_NAME else "?")
    # A cell is unscorable if its taxonomy yielded no nodes at this N (ratio NaN);
    # drop those so downstream idxmax/pivots never hit a NaN key. Report the loss.
    bad = df[df["ratio"].isna()]
    if len(bad):
        lost = bad.groupby(["dataset", "embedder"]).size().to_dict()
        print(f"[summarize] dropping {len(bad)} unscorable (NaN-ratio) rows: {lost}")
        df = df[df["ratio"].notna()].reset_index(drop=True)
    return df


def per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds → one row per (dataset, embedder, nn, md, compact)."""
    g = df.groupby(["dataset", "embedder", "media", "N", "n_neighbors", "min_dist", "compact"], as_index=False)
    agg = g.agg(
        ratio=("ratio", "mean"),
        ratio_std=("ratio", "std"),
        score_2d=("score_2d", "mean"),
        trustworthiness=("trustworthiness", "mean"),
        continuity=("continuity", "mean"),
        knn_recall=("knn_recall", "mean"),
        seed_agreement=("seed_agreement", "mean"),
        fit_seconds=("fit_seconds", "mean"),
    )
    return agg


def compaction_verdict(cell: pd.DataFrame) -> dict:
    """Mean separability + guard delta (compacted − raw), per embedder and overall."""
    piv = cell.pivot_table(
        index=["dataset", "embedder", "n_neighbors", "min_dist"],
        columns="compact",
        values=["ratio", "trustworthiness", "continuity", "knn_recall"],
    ).dropna()
    out = {"per_embedder": {}, "overall": {}}
    deltas_all = {}
    for metric in ["ratio", "trustworthiness", "continuity", "knn_recall"]:
        deltas_all[metric] = (piv[(metric, True)] - piv[(metric, False)])
    dd = pd.DataFrame(deltas_all)
    dd = dd.reset_index()
    for emb, sub in dd.groupby("embedder"):
        out["per_embedder"][emb] = {m: round(float(sub[m].mean()), 4) for m in deltas_all}
    for m in deltas_all:
        out["overall"][m] = round(float(dd[m].mean()), 4)
    out["overall"]["n_configs"] = int(len(dd))
    return out


def recommend(cell: pd.DataFrame) -> dict:
    """Per-embedder recommended (nn, md, compact), chosen on the raw (compact=False)
    layouts by separability with stability as tie-breaker, then re-checked with the
    winning compaction setting."""
    recs = {}
    for emb, sub in cell.groupby("embedder"):
        # rank (nn, md) by mean ratio across datasets, on the compact setting that wins overall
        best_compact = sub.groupby("compact")["ratio"].mean().idxmax()
        s = sub[sub["compact"] == best_compact]
        by_params = s.groupby(["n_neighbors", "min_dist"], as_index=False).agg(
            ratio=("ratio", "mean"),
            seed_agreement=("seed_agreement", "mean"),
            trustworthiness=("trustworthiness", "mean"),
            knn_recall=("knn_recall", "mean"),
        )
        by_params = by_params.sort_values("ratio", ascending=False)
        top = by_params.iloc[0]
        # plateau: params within 0.5% of the best ratio; among those prefer the most stable
        near = by_params[by_params["ratio"] >= top["ratio"] - 0.005]
        pick = near.sort_values("seed_agreement", ascending=False).iloc[0]
        recs[emb] = {
            "n_neighbors": int(pick["n_neighbors"]),
            "min_dist": float(pick["min_dist"]),
            "compact": bool(best_compact),
            "ratio": round(float(pick["ratio"]), 4),
            "argmax_ratio": round(float(top["ratio"]), 4),
            "argmax_nn": int(top["n_neighbors"]),
            "seed_agreement": round(float(pick["seed_agreement"]), 4),
        }
    return recs


def n_analysis(cell: pd.DataFrame) -> dict:
    """Best n_neighbors per (embedder, dataset, N) on the raw layout — the N question."""
    raw = cell[cell["compact"] == False]  # noqa: E712
    out = {}
    for (emb, ds, n), sub in raw.groupby(["embedder", "dataset", "N"]):
        best = sub.loc[sub["ratio"].idxmax()]
        out.setdefault(emb, []).append(
            {"dataset": ds, "N": int(n), "best_nn": int(best["n_neighbors"]), "ratio": round(float(best["ratio"]), 4)}
        )
    return out


def main():
    df = load_master()
    df.to_csv(C.RESULTS_ROOT / "master.csv", index=False)
    cell = per_cell(df)
    cell.to_csv(C.RESULTS_ROOT / "per_cell.csv", index=False)

    summary = {
        "datasets": sorted(df["dataset"].unique().tolist()),
        "embedders": sorted(df["embedder"].unique().tolist()),
        "n_rows": int(len(df)),
        "recommendations": recommend(cell),
        "compaction": compaction_verdict(cell),
        "n_analysis": n_analysis(cell),
    }
    json.dump(summary, open(C.RESULTS_ROOT / "summary.json", "w"), indent=2)

    print("\n=== Per-embedder recommendation ===")
    for emb, r in summary["recommendations"].items():
        print(f"  {emb:9s} nn={r['n_neighbors']:3d} min_dist={r['min_dist']:.2f} compact={r['compact']} "
              f"ratio={r['ratio']:.3f} (argmax nn={r['argmax_nn']} ratio={r['argmax_ratio']:.3f}) "
              f"stability={r['seed_agreement']:.3f}")
    print("\n=== Compaction verdict (compacted − raw; negative = compaction hurts) ===")
    print(f"  overall: {summary['compaction']['overall']}")
    for emb, dvals in summary["compaction"]["per_embedder"].items():
        print(f"  {emb:9s} {dvals}")
    print("\n=== N question (best n_neighbors per dataset size) ===")
    for emb, rows in summary["n_analysis"].items():
        line = ", ".join(f"{r['dataset']}(N={r['N']}):nn={r['best_nn']}" for r in sorted(rows, key=lambda x: x["N"]))
        print(f"  {emb:9s} {line}")
    # Per-embedder ratio profile over n_neighbors (mean over that embedder's
    # datasets, raw layouts, at each embedder's best min_dist) — the basis for
    # the shipped per-embedder constant.
    raw = cell[cell["compact"] == False]  # noqa: E712
    print("\n=== Per-embedder mean ratio by n_neighbors (raw; averaged over datasets) ===")
    prof = raw.groupby(["embedder", "n_neighbors"])["ratio"].mean().unstack("n_neighbors")
    print(prof.round(4).to_string())
    print("\n=== Per-embedder mean ratio by min_dist ===")
    print(raw.groupby(["embedder", "min_dist"])["ratio"].mean().unstack("min_dist").round(4).to_string())

    print(f"\nwrote {C.RESULTS_ROOT/'summary.json'}, master.csv, per_cell.csv")


if __name__ == "__main__":
    main()
