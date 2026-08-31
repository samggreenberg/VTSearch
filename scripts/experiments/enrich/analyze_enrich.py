#!/usr/bin/env python
"""Analyse the description-enrichment cells and evaluate the #3127 rule.

Reads every cell CSV written by ``run_enrich.py``, pairs the arms, and applies
the decision rule pre-registered in
``docs/experiments/2026-08-31-enrich-descriptions-3127/PLAN.md`` -- mechanically,
from module constants, so the verdict is not a paragraph written around the
numbers it is describing.

Everything the report quotes comes out of here as a CSV, and every figure is
drawn from those same CSVs.

    python analyze_enrich.py --results /expscratch/$USER/enrich-3127/results \\
        --out docs/experiments/2026-08-31-enrich-descriptions-3127
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- the pre-registered rule, as constants -------------------------------
MARGIN_SE = 2.0
"""How many clustered standard errors a paired difference must clear to count."""

DEFAULT_EMBEDDERS = {"audio": "clap_general", "image": "siglip", "text": "e5", "video": "xclip"}
"""The media-type defaults the rule is about.  `clap` is a control, not a default."""

SIZE_SUFFIXES = ("_s", "_m", "_l", "_a")


def corpus_of(dataset: str) -> str:
    """`esc50_m` -> `esc50`: the three size slices are one corpus, sliced.

    They share categories and queries, so a category's three slices are repeated
    measures of one question, not three independent draws -- which is what the
    clustering below is for.
    """
    for suffix in SIZE_SUFFIXES:
        if dataset.endswith(suffix):
            return dataset[: -len(suffix)]
    return dataset


def cluster_se(values: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-robust standard error of the mean (CR1).

    Categories inside one corpus are correlated -- the same query against the
    same haystack, sliced three ways -- so the naive `sd/sqrt(n)` understates.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2:
        return float("nan")
    resid = values - values.mean()
    groups = pd.Series(resid).groupby(pd.Series(clusters)).sum().to_numpy()
    g = groups.size
    if g < 2:
        return float("nan")
    var = (groups**2).sum() / n**2
    var *= g / (g - 1)  # CR1 small-cluster correction
    return float(np.sqrt(var))


def naive_se(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(values.size))


def load_cells(results: Path) -> pd.DataFrame:
    paths = sorted(results.glob("*.csv"))
    if not paths:
        sys.exit(f"no cell CSVs under {results}")
    frames, empty = [], []
    for p in paths:
        if p.stat().st_size == 0:
            empty.append(p.name)
            continue
        frames.append(pd.read_csv(p))
    if empty:
        # Count what you dropped, out loud: a silently excluded cell is how a
        # disk incident becomes a wrong verdict.
        print(f"WARNING: {len(empty)} zero-byte cell(s) skipped: {', '.join(empty)}", file=sys.stderr)
    df = pd.concat(frames, ignore_index=True)
    df["corpus"] = df["dataset"].map(corpus_of)
    print(
        f"loaded {len(paths) - len(empty)} cells, {len(df)} rows, "
        f"{df['dataset'].nunique()} datasets, {df['embedder'].nunique()} embedders"
    )
    return df


def paired(df: pd.DataFrame, arm: str, base: str = "plain", metric: str = "ap") -> pd.DataFrame:
    """One row per (embedder, dataset, category): base, arm, and their difference."""
    keys = ["embedder", "media_type", "corpus", "dataset", "category", "query"]
    a = df[df["arm"] == base].set_index(keys)[metric]
    b = df[df["arm"] == arm].set_index(keys)[metric]
    joined = pd.concat({"base": a, "arm": b}, axis=1).dropna().reset_index()
    joined["delta"] = joined["arm"] - joined["base"]
    # The clustering unit: one category of one corpus, however many size slices
    # of that corpus it appears in.
    joined["cluster"] = joined["corpus"] + "|" + joined["category"]
    return joined


def summarise_by(pair: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """`summarise` over each group of *keys*, as a flat frame.

    Written as an explicit loop rather than ``groupby.apply``: the grouping
    columns a summary needs (`cluster`, `dataset`) are exactly the ones
    ``include_groups=False`` takes away.
    """
    rows = []
    for vals, g in pair.groupby(keys, sort=True):
        rec = dict(zip(keys, vals if isinstance(vals, tuple) else (vals,)))
        rec.update(summarise(g))
        rows.append(rec)
    return pd.DataFrame(rows)


def summarise(group: pd.DataFrame) -> dict:
    d = group["delta"].to_numpy()
    clusters = group["cluster"].to_numpy()
    se = cluster_se(d, clusters)
    return {
        "n_pairs": int(d.size),
        "n_clusters": int(pd.unique(clusters).size),
        "n_datasets": int(group["dataset"].nunique()),
        "map_plain": float(group["base"].mean()),
        "map_arm": float(group["arm"].mean()),
        "delta": float(d.mean()),
        "se_cluster": se,
        "se_naive": naive_se(d),
        "t": float(d.mean() / se) if se and np.isfinite(se) and se > 0 else float("nan"),
        "helped": int((d > 0).sum()),
        "hurt": int((d < 0).sum()),
    }


def planted_answer_check(df: pd.DataFrame) -> pd.DataFrame:
    """The `{text}` wrapper is an identity: its arm must reproduce `plain`.

    If it does not, the harness moved, not the setting -- so this is checked
    before any verdict is read off the same rows.  The identity sits at a
    *different index per embedder* (`w2` for `clap_general` and `e5`, `w3` for
    `siglip`), so the check is per (embedder, arm), never per arm: pooling the
    arms compares "an image of X" against a plain query and calls the harness
    broken.
    """
    rows = []
    identity = df[df["wrapper"] == "{text}"][["embedder", "arm"]].drop_duplicates()
    for _, r in identity.iterrows():
        sub = df[df["embedder"] == r["embedder"]]
        pair = paired(sub, r["arm"])
        worst = float(pair["delta"].abs().max()) if len(pair) else float("nan")
        rows.append(
            {
                "embedder": r["embedder"],
                "arm": r["arm"],
                "n": len(pair),
                "max_abs_delta": worst,
                "ok": bool(len(pair)) and worst < 1e-9,
            }
        )
    return pd.DataFrame(rows)


def write_examples(pair: pd.DataFrame, tables: Path) -> None:
    """The categories that actually moved, in both directions, with their queries.

    An error rate a reader cannot see behind is a number they have to take on
    faith, so both tails are written out verbatim: overall, and again *within*
    each default -- a global top-10 is all audio, whose per-category swings are
    an order of magnitude wider than anyone else's, which is itself a finding
    rather than a reason to hide the rest.
    """
    cols = ["media_type", "embedder", "dataset", "category", "query", "base", "arm", "delta"]
    names = {"base": "ap_plain", "arm": "ap_enriched"}
    ordered = pair.sort_values("delta")
    pd.concat([ordered.head(10), ordered.tail(10)])[cols].rename(columns=names).to_csv(
        tables / "examples.csv", index=False
    )

    tails = []
    for (mt, emb), g in pair.groupby(["media_type", "embedder"]):
        if DEFAULT_EMBEDDERS.get(mt) != emb:
            continue
        g = g.sort_values("delta")
        tails.append(pd.concat([g.head(5), g.tail(5)]))
    if tails:
        pd.concat(tails)[cols].rename(columns=names).to_csv(tables / "examples_by_media_type.csv", index=False)


def mechanism_tables(pair: pd.DataFrame, per_wrapper: pd.DataFrame, tables: Path) -> None:
    """Two cheap reads on *why* a wrapper helps or hurts, from rows already here.

    Both are descriptive, and neither is a controlled test -- four wrappers per
    embedder is four points. They are written out so the report's mechanism
    section quotes measurements rather than a story.
    """
    # (a) does the damage scale with how much generic prefix the wrapper adds?
    if len(per_wrapper):
        w = per_wrapper.copy()
        w["wrapper_extra_words"] = w["wrapper"].map(lambda t: len(t.replace("{text}", "").split()))
        w[["embedder", "media_type", "arm", "wrapper", "wrapper_extra_words", "delta", "se_cluster"]].sort_values(
            ["embedder", "wrapper_extra_words"]
        ).to_csv(tables / "mechanism_wrapper_length.csv", index=False)

    # (b) does it scale with how short the typed query is -- i.e. how large a
    # fraction of the embedded string the wrapper becomes?
    rows = []
    q = pair.copy()
    q["query_words"] = q["query"].map(lambda t: len(str(t).split()))
    for (mt, emb), g in q.groupby(["media_type", "embedder"]):
        by_cat = g.groupby("cluster").agg(delta=("delta", "mean"), query_words=("query_words", "first"))
        if by_cat["query_words"].nunique() < 2:
            continue
        rows.append(
            {
                "media_type": mt,
                "embedder": emb,
                "n_categories": len(by_cat),
                "pearson_r": float(by_cat["delta"].corr(by_cat["query_words"])),
                "spearman_r": float(by_cat["delta"].corr(by_cat["query_words"], method="spearman")),
                "mean_query_words": float(by_cat["query_words"].mean()),
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(tables / "mechanism_query_length.csv", index=False)


def probe_table(probe_dir: Path, tables: Path) -> pd.DataFrame | None:
    """Fold in `probe_text_prompt.py`: what does the query/passage gap cost?

    Same datasets, same metric, same pairing as everything else, so the number
    can be put beside the enrichment difference without an apology.
    """
    paths = sorted(probe_dir.glob("probe_prompt__*.csv")) if probe_dir.is_dir() else []
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["media_type"] = "text"
    df["corpus"] = df["dataset"].map(corpus_of)
    pair = paired(df, "passage", base="query")
    out = summarise_by(pair, ["embedder"])
    out.to_csv(tables / "probe_query_vs_passage.csv", index=False)
    return out


def verdict(per_mt: pd.DataFrame, pooled: dict) -> dict:
    """Apply the pre-registered rule to the four media-type defaults."""
    defaults = per_mt[per_mt.apply(lambda r: DEFAULT_EMBEDDERS.get(r["media_type"]) == r["embedder"], axis=1)]
    harmed = defaults[defaults["delta"] < -MARGIN_SE * defaults["se_cluster"]]
    helped = defaults[defaults["delta"] > MARGIN_SE * defaults["se_cluster"]]
    pooled_resolved = pooled["delta"] > MARGIN_SE * pooled["se"]
    if len(harmed):
        call = "DO_NOT_FLIP__HARMS_A_DEFAULT"
    elif len(helped) and pooled_resolved:
        call = "FLIP_TO_TRUE"
    else:
        call = "DO_NOT_FLIP__UNRESOLVED"
    return {
        "verdict": call,
        "margin_se": MARGIN_SE,
        "n_defaults": int(len(defaults)),
        "harmed": sorted(harmed["media_type"].tolist()),
        "helped": sorted(helped["media_type"].tolist()),
        "pooled_delta": pooled["delta"],
        "pooled_se": pooled["se"],
        "pooled_resolved": bool(pooled_resolved),
    }


def wrapper_arms(df: pd.DataFrame, tables: Path, metric: str) -> pd.DataFrame:
    """Every individual wrapper against `plain`, per embedder.

    "Enrichment helps" and "one of its five templates helps" are different
    findings with different fixes, and the docs note this issue asks for has to
    say which.
    """
    rows = []
    for arm in sorted(a for a in df["arm"].unique() if a.startswith("w")):
        pair = paired(df, arm, metric=metric)
        for (emb, mt), g in pair.groupby(["embedder", "media_type"]):
            rec = summarise(g)
            rec.update(
                {
                    "arm": arm,
                    "embedder": emb,
                    "media_type": mt,
                    "wrapper": df[(df["arm"] == arm) & (df["embedder"] == emb)]["wrapper"].iloc[0],
                }
            )
            rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out = out[
            [
                "embedder",
                "media_type",
                "arm",
                "wrapper",
                "map_plain",
                "map_arm",
                "delta",
                "se_cluster",
                "se_naive",
                "n_pairs",
                "n_clusters",
                "helped",
                "hurt",
            ]
        ].sort_values(["media_type", "embedder", "arm"])
        out.to_csv(tables / "per_wrapper.csv", index=False)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="study directory (tables/ and figures/ land here)")
    ap.add_argument("--metric", default="ap", help="metric column to pair on (default: ap)")
    ap.add_argument(
        "--probe",
        type=Path,
        default=None,
        help="probe results dir (default: <results>/../results_probe)",
    )
    ap.add_argument("--no-figures", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    tables = args.out / "tables"
    figures = args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    df = load_cells(args.results)

    # --- 0. the planted answer, before anything is read off these rows ---
    planted = planted_answer_check(df)
    planted.to_csv(tables / "planted_answer.csv", index=False)
    print("\n=== planted answer: identity wrapper vs plain ===")
    print(planted.to_string(index=False))
    if len(planted) and not planted["ok"].all():
        print("PLANTED ANSWER FAILED -- the harness moved, not the setting", file=sys.stderr)

    pair = paired(df, "enriched", metric=args.metric)
    pair.to_csv(tables / "paired_categories.csv", index=False)

    # --- 1. per dataset ---
    per_ds = summarise_by(pair, ["embedder", "media_type", "corpus", "dataset"]).sort_values(
        ["media_type", "embedder", "dataset"]
    )
    n_media = df.groupby(["embedder", "dataset"], as_index=False)["n_media"].first()
    per_ds = per_ds.merge(n_media, on=["embedder", "dataset"], how="left")
    per_ds.to_csv(tables / "per_dataset.csv", index=False)

    # --- 2. per media type (the unit the rule reads) ---
    per_mt = summarise_by(pair, ["media_type", "embedder"]).sort_values(["media_type", "embedder"])
    per_mt.to_csv(tables / "per_media_type.csv", index=False)

    # --- 3. the equal-weighted pool over the four defaults ---
    defaults = per_mt[per_mt.apply(lambda r: DEFAULT_EMBEDDERS.get(r["media_type"]) == r["embedder"], axis=1)]
    pooled = {
        "delta": float(defaults["delta"].mean()),
        "se": float(np.sqrt((defaults["se_cluster"] ** 2).sum()) / len(defaults)) if len(defaults) else float("nan"),
        "n_media_types": int(len(defaults)),
    }

    # --- 3b. the secondary metric, on the same pairing ---
    # P@10 is what a user sees on the first screen; AP is what the study decides
    # on.  Reported so a win that only lives deep in the ranking cannot pass for
    # a win the user would notice.
    for metric in ("p10", "p5"):
        alt = paired(df, "enriched", metric=metric)
        summarise_by(alt, ["media_type", "embedder"]).sort_values(["media_type", "embedder"]).to_csv(
            tables / f"per_media_type_{metric}.csv", index=False
        )

    # --- 4. per-wrapper diagnostic ---
    per_wrapper = wrapper_arms(df, tables, args.metric)

    # --- 5. literal examples: the categories that actually moved ---
    write_examples(pair, tables)

    # --- 6. cost: what the extra encoder passes cost per query ---
    cost = (
        df.groupby(["embedder", "dataset", "arm"], as_index=False)["arm_seconds"]
        .first()
        .pivot_table(index=["embedder", "dataset"], columns="arm", values="arm_seconds")
        .reset_index()
    )
    n_q = df.groupby(["embedder", "dataset"])["category"].nunique().rename("n_queries").reset_index()
    cost = cost.merge(n_q, on=["embedder", "dataset"], how="left")
    if {"plain", "enriched"}.issubset(cost.columns):
        cost["extra_s_per_query"] = (cost["enriched"] - cost["plain"]) / cost["n_queries"]
    cost.to_csv(tables / "arm_cost.csv", index=False)

    mechanism_tables(pair, per_wrapper, tables)
    probe = probe_table(args.probe or args.results.parent / "results_probe", tables)
    if probe is not None:
        print("\n=== probe: ranking with the passage encoder instead of the query encoder ===")
        print(probe[["embedder", "n_pairs", "map_plain", "map_arm", "delta", "se_cluster"]].to_string(index=False))

    call = verdict(per_mt, pooled)
    summary = {
        "pooled": pooled,
        **call,
        "planted_answer_ok": bool(planted["ok"].all()) if len(planted) else None,
        "cells": int(df.groupby(["embedder", "dataset"]).ngroups),
        "paired_observations": int(len(pair)),
    }
    # Trailing newline: the file is committed, and end-of-file-fixer rewrites
    # (and so fails) any commit that carries one without.
    (args.out / "verdict.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== per media type (enriched - plain, paired on AP) ===")
    show = per_mt.copy()
    for col in ("map_plain", "map_arm", "delta", "se_cluster", "se_naive"):
        show[col] = show[col].map(lambda v: f"{v:+.4f}" if col == "delta" else f"{v:.4f}")
    print(
        show[
            [
                "media_type",
                "embedder",
                "n_pairs",
                "n_clusters",
                "map_plain",
                "map_arm",
                "delta",
                "se_cluster",
                "se_naive",
                "helped",
                "hurt",
            ]
        ].to_string(index=False)
    )
    print(
        f"\npooled over {pooled['n_media_types']} defaults (equal weight): "
        f"{pooled['delta']:+.4f} +/- {pooled['se']:.4f}"
    )
    print(f"VERDICT: {call['verdict']}  (harmed={call['harmed']} helped={call['helped']})")

    if not args.no_figures:
        from figures_enrich import make_figures  # noqa: PLC0415

        made = make_figures(pair, per_ds, per_mt, per_wrapper, figures)
        print("\nfigures:")
        for name in made:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
