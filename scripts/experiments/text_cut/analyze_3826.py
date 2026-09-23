#!/usr/bin/env python
"""Turn the #3826 gate frames into the tables the report is read off.

    python analyze_3826.py --parts <analysis>/parts --out <analysis>

Every cut in the gate is a threshold on the same score array, so **any two
admitted sets of one sort are nested**: the number of verdicts that differ
between two lines is exactly ``|n_adm_a - n_adm_b|``.  That is what lets every
comparison below - rule against rule, rule against its own perturbation - be
read off admitted counts with no per-media join.

Tables written to ``<out>/tables/``:

``reproduce.csv``    the issue's own numbers, re-derived on #3585's corpus
``stability.csv``    per rule: optimiser flips and bootstrap flips
``quality.csv``      per rule: cost (FPR+FNR), excess over the oracle, P/R/F1, admitted fraction
``paired.csv``       per rule: cost and F1 paired against ``gmm_shipped``, SE clustered on (dataset, category)
``by_dataset.csv``   the headline rules' cost / F1 / bootstrap flip per dataset
``lodo.csv``         each family's constant chosen on three datasets, scored on the fourth
``shipped_shape.csv`` what the shipped mixture actually does per dataset (admitted fraction, Ashman's D)
``null_check.csv``   the tail rules' bulk model against the negatives it claims to describe
``autopilot.csv``    positive rate just below each line - what Autopilot's Bad phase would vote on
``examples.csv``     literal sorts behind the headline failures
``timing.csv``       median seconds per rule on the sort as captured
``summary.json``     the headline numbers the report quotes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HEADLINE = [
    "gmm_shipped",
    "gmm_priorfree",
    "gmm_converged",
    "gmm_multistart",
    "otsu",
    "gmm_guarded_z3",
    "gmm_guarded_fdr0.2",
    "quantile0.02",
    "quantile0.05",
    "tail_z2.5",
    "tail_z3",
    "tail_fdr0.2",
    "tail_fdr0.3",
]
SORT_KEYS = ["dataset", "embedder", "category"]


def load(parts: Path) -> "tuple[pd.DataFrame, pd.DataFrame]":
    new, old = [], []
    for f in sorted(parts.glob("*_cuts.csv")):
        df = pd.read_csv(f, low_memory=False)
        (old if f.name.startswith("old_") else new).append(df)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return cat(new), cat(old)


def with_flips(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's full-frame admitted count and the nested-set flip."""
    keys = ["capture", *SORT_KEYS, "rule"]
    full = df[df["frame"] == "full"][keys + ["n_adm"]].rename(columns={"n_adm": "full_adm"})
    out = df.merge(full, on=keys, how="left", validate="many_to_one")
    out["flip"] = (out["n_adm"] - out["full_adm"]).abs() / out["n"]
    return out


def reproduce(old: pd.DataFrame) -> pd.DataFrame:
    """The issue's numbers on the #3585 corpus: re-initialise sklearn, re-tolerance the native fit."""
    rows = []
    o = with_flips(old)
    for cap, g in o.groupby("capture"):
        text = g[~g["embedder"].isin(["dinov3_patch"])]
        kpp = text[(text["rule"] == "ref_sklearn") & (text["frame"] == "opt:kmeanspp")]
        full = text[text["frame"] == "full"].pivot_table(index=SORT_KEYS, columns="rule", values="n_adm")
        n = text[text["frame"] == "full"].groupby(SORT_KEYS)["n"].first()
        d_param = (full["ref_param1e-8"] - full["gmm_shipped"]).abs() / n
        d_tol5 = text[(text["rule"] == "gmm_shipped") & (text["frame"] == "opt:tol1e-5")]
        rinit = text[(text["rule"] == "gmm_shipped") & text["frame"].str.startswith("opt:rinit")]
        rows.append(
            {
                "capture": cap,
                "sorts": int(kpp.shape[0]),
                "kmeanspp_mean_flip": kpp["flip"].mean(),
                "kmeanspp_mean_medias": (kpp["flip"] * kpp["n"]).mean(),
                "kmeanspp_max_flip": kpp["flip"].max(),
                "param1e-8_vs_shipped_mean": d_param.mean(),
                "param1e-8_vs_shipped_max": d_param.max(),
                "shipped_tol1e-5_mean": d_tol5["flip"].mean(),
                "shipped_tol1e-5_max": d_tol5["flip"].max(),
                "shipped_rinit_mean": rinit["flip"].mean(),
                "shipped_rinit_max": rinit["flip"].max(),
                "shipped_median_adm_frac": (full["gmm_shipped"] / n).median(),
            }
        )
    return pd.DataFrame(rows)


def stability(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule, g in d.groupby("rule"):
        opt = g[g["frame"].str.startswith("opt:")]
        boot = g[g["frame"].str.startswith("boot:")]
        per_sort_boot = boot.groupby(SORT_KEYS)["flip"].mean()
        per_sort_opt = opt.groupby(SORT_KEYS)["flip"].mean() if len(opt) else pd.Series(dtype=float)
        per_sort_jac = boot.groupby(SORT_KEYS)["jaccard_dist"].mean()
        rinit = opt[opt["frame"].str.startswith("opt:rinit")]
        tol = opt[opt["frame"].str.startswith("opt:tol")]
        rows.append(
            {
                "rule": rule,
                "family": g["family"].iloc[0],
                "sorts": int(per_sort_boot.size),
                "opt_mean_flip": per_sort_opt.mean() if len(opt) else np.nan,
                "opt_p90_flip": per_sort_opt.quantile(0.9) if len(opt) else np.nan,
                "opt_share_sorts_moved_1pct": (per_sort_opt > 0.01).mean() if len(opt) else np.nan,
                "rinit_mean_flip": rinit["flip"].mean() if len(rinit) else np.nan,
                "rinit_max_flip": rinit["flip"].max() if len(rinit) else np.nan,
                "tol_mean_flip": tol["flip"].mean() if len(tol) else np.nan,
                "tol_max_flip": tol["flip"].max() if len(tol) else np.nan,
                "boot_mean_flip": per_sort_boot.mean(),
                "boot_p90_flip": per_sort_boot.quantile(0.9),
                "boot_mean_jaccard": per_sort_jac.mean(),
                "boot_p90_jaccard": per_sort_jac.quantile(0.9),
            }
        )
    return pd.DataFrame(rows).sort_values("boot_mean_flip")


def quality_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Full-frame rows with rates, plus the per-sort oracle cost joined on."""
    f = d[d["frame"] == "full"].copy()
    f["neg"] = f["n"] - f["n_pos"]
    f["tpr"] = f["tp"] / f["n_pos"]
    f["fpr"] = f["fp"] / f["neg"]
    f["cost"] = f["fpr"] + (1 - f["tpr"])
    f["precision"] = np.where(f["n_adm"] > 0, f["tp"] / f["n_adm"].clip(lower=1), np.nan)
    f["recall"] = f["tpr"]
    f["f1"] = 2 * f["tp"] / (f["n_adm"] + f["n_pos"])
    f["adm_frac"] = f["n_adm"] / f["n"]
    f["prevalence"] = f["n_pos"] / f["n"]
    f["adm_over_pos"] = f["n_adm"] / f["n_pos"]
    orc = f[f["rule"] == "oracle_cost"][SORT_KEYS + ["cost"]].rename(columns={"cost": "oracle_cost"})
    orf = f[f["rule"] == "oracle_f1"][SORT_KEYS + ["f1"]].rename(columns={"f1": "oracle_f1"})
    f = f.merge(orc, on=SORT_KEYS, how="left", validate="many_to_one").merge(
        orf, on=SORT_KEYS, how="left", validate="many_to_one"
    )
    f["excess_cost"] = f["cost"] - f["oracle_cost"]
    f["f1_gap"] = f["oracle_f1"] - f["f1"]
    return f


def quality(q: pd.DataFrame) -> pd.DataFrame:
    agg = q.groupby("rule").agg(
        family=("family", "first"),
        sorts=("cost", "size"),
        cost=("cost", "mean"),
        excess_cost=("excess_cost", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        f1=("f1", "mean"),
        f1_gap=("f1_gap", "mean"),
        median_adm_frac=("adm_frac", "median"),
        median_adm_over_pos=("adm_over_pos", "median"),
        share_empty=("n_adm", lambda s: float((s == 0).mean())),
    )
    return agg.reset_index().sort_values("cost")


def _clustered_se(diff: pd.Series, clusters: pd.Series) -> float:
    """SE of a mean whose rows are correlated within (dataset, category) - four embedders see one query."""
    g = diff.groupby(clusters.values)
    sums = g.sum()
    n = diff.size
    k = sums.size
    if k < 2:
        return float("nan")
    resid = sums - g.size() * diff.mean()
    return float(np.sqrt(k / (k - 1) * (resid**2).sum()) / n)


def paired(q: pd.DataFrame, base: str = "gmm_shipped") -> pd.DataFrame:
    b = q[q["rule"] == base][SORT_KEYS + ["cost", "f1"]].rename(columns={"cost": "b_cost", "f1": "b_f1"})
    j = q[~q["rule"].isin([base])].merge(b, on=SORT_KEYS, how="inner", validate="many_to_one")
    rows = []
    for rule, g in j.groupby("rule"):
        cl = g["dataset"] + "|" + g["category"]
        dc = g["cost"] - g["b_cost"]
        df1 = g["f1"] - g["b_f1"]
        rows.append(
            {
                "rule": rule,
                "sorts": len(g),
                "d_cost": dc.mean(),
                "se_cost": _clustered_se(dc, cl),
                "d_f1": df1.mean(),
                "se_f1": _clustered_se(df1, cl),
                "wins_cost": float((dc < 0).mean()),
                "wins_f1": float((df1 > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("d_cost")


def by_dataset(q: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    boot = d[d["frame"].str.startswith("boot:")].groupby(["dataset", "rule"])["flip"].mean().rename("boot_flip")
    qq = q.groupby(["dataset", "rule"]).agg(
        cost=("cost", "mean"),
        excess_cost=("excess_cost", "mean"),
        f1=("f1", "mean"),
        median_adm_frac=("adm_frac", "median"),
        sorts=("cost", "size"),
    )
    out = qq.join(boot).reset_index()
    return out[out["rule"].isin(HEADLINE + ["oracle_cost", "oracle_f1"])]


def lodo(q: pd.DataFrame) -> pd.DataFrame:
    """Pick each family's constant on three datasets by mean cost (and by F1), score it on the fourth."""
    rows = []
    for fam_prefix in ("quantile", "tail_z", "tail_fdr"):
        members = sorted(r for r in q["rule"].unique() if r.startswith(fam_prefix))
        sub = q[q["rule"].isin(members)]
        for metric, better in (("cost", "min"), ("f1", "max")):
            for held in sorted(q["dataset"].unique()):
                train = sub[sub["dataset"] != held].groupby("rule")[metric].mean()
                pick = train.idxmin() if better == "min" else train.idxmax()
                test = sub[(sub["dataset"] == held) & (sub["rule"] == pick)]
                best_in = sub[sub["dataset"] == held].groupby("rule")[metric].mean()
                rows.append(
                    {
                        "family": fam_prefix,
                        "criterion": metric,
                        "held_out": held,
                        "picked": pick,
                        "held_out_value": test[metric].mean(),
                        "held_out_best_member": best_in.idxmin() if better == "min" else best_in.idxmax(),
                        "held_out_best_value": best_in.min() if better == "min" else best_in.max(),
                    }
                )
    return pd.DataFrame(rows)


def shipped_shape(q: pd.DataFrame) -> pd.DataFrame:
    s = q[q["rule"] == "gmm_shipped"]
    return (
        s.groupby("dataset")
        .agg(
            sorts=("n", "size"),
            median_n=("n", "median"),
            median_prevalence=("prevalence", "median"),
            median_adm_frac=("adm_frac", "median"),
            p90_adm_frac=("adm_frac", lambda x: x.quantile(0.9)),
            median_adm_over_pos=("adm_over_pos", "median"),
            share_separated=("ashman_d", lambda x: float((x >= 2).mean())),
            median_ashman_d=("ashman_d", "median"),
            cost=("cost", "mean"),
            f1=("f1", "mean"),
        )
        .reset_index()
    )


def null_check(q: pd.DataFrame) -> pd.DataFrame:
    t = q[q["family"] == "tail"].copy()
    t["ratio"] = t["fp"] / t["bulk_pred_fp"].replace(0, np.nan)
    return (
        t.groupby(["rule", "dataset"])
        .agg(pred_fp=("bulk_pred_fp", "median"), actual_fp=("fp", "median"), median_ratio=("ratio", "median"))
        .reset_index()
    )


def autopilot(q: pd.DataFrame, corpus: Path, rules: "list[str]", k: int = 8) -> pd.DataFrame:
    """Positive rate among the *k* items just below each line: Autopilot's Bad phase votes there."""
    rows = []
    for f in sorted(corpus.glob("*.npz")):
        z = np.load(f)
        keys = [k_[: -len("|scores")] for k_ in z.files if k_.endswith("|scores")]
        meta = json.loads(bytes(z["_meta"].tobytes()).decode("utf-8"))
        for key in keys:
            x = z[f"{key}|scores"]
            y = z[f"{key}|labels"].astype(bool)
            ds, emb = key.split("|")[:2]
            cat = meta["queries"][key]["category"]
            order = np.argsort(-x, kind="stable")
            xs, ys = x[order], y[order]
            sub = q[(q["dataset"] == ds) & (q["embedder"] == emb) & (q["category"] == cat) & q["rule"].isin(rules)]
            for _, r in sub.iterrows():
                below = np.nonzero(xs < float(r["cut"]))[0][:k]
                rows.append(
                    {
                        "rule": r["rule"],
                        "dataset": ds,
                        "pos_rate_below": float(ys[below].mean()) if below.size else np.nan,
                    }
                )
    a = pd.DataFrame(rows)
    return a.groupby(["rule", "dataset"])["pos_rate_below"].mean().unstack().reset_index()


def examples(q: pd.DataFrame) -> pd.DataFrame:
    """Literal sorts: shipped's worst admissions, and each tail rule's worst misses."""
    cols = SORT_KEYS + ["query", "n", "n_pos"]
    wide = (
        q[q["rule"].isin(HEADLINE + ["oracle_cost"])]
        .pivot_table(index=cols, columns="rule", values=["n_adm", "tp"])
        .reset_index()
    )
    wide.columns = ["|".join(c for c in col if c) if isinstance(col, tuple) else col for col in wide.columns]
    picks = []
    s = q[q["rule"] == "gmm_shipped"].copy()
    s["waste"] = s["fp"] / s["n"]
    picks += [("shipped admits most negatives", k) for k in s.nlargest(6, "waste")[SORT_KEYS].itertuples(index=False)]
    for rule in ("tail_fdr0.2", "tail_z3", "quantile0.05"):
        t = q[q["rule"] == rule]
        picks += [
            (f"{rule} misses most positives", k) for k in t.nsmallest(4, "recall")[SORT_KEYS].itertuples(index=False)
        ]
        picks += [
            (f"{rule} admits most negatives", k) for k in t.nsmallest(3, "precision")[SORT_KEYS].itertuples(index=False)
        ]
    out = []
    for why, (ds, emb, cat) in picks:
        m = wide[(wide["dataset"] == ds) & (wide["embedder"] == emb) & (wide["category"] == cat)]
        if len(m):
            out.append({"why": why, **m.iloc[0].to_dict()})
    return pd.DataFrame(out)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="")
    args = ap.parse_args(list(argv) if argv is not None else None)
    parts, out = Path(args.parts), Path(args.out)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    new, old = load(parts)
    # Count what was analysed against what is on disk - a dropped sort looks
    # exactly like one never captured (#3585's 29-of-62).
    n_sorts = new[new["frame"] == "full"].groupby(["capture", *SORT_KEYS]).ngroups
    summary: dict = {"sorts_analysed": n_sorts, "captures": int(new["capture"].nunique())}
    if args.corpus:
        on_disk = sum(
            sum(1 for k in np.load(f).files if k.endswith("|scores")) for f in Path(args.corpus).glob("*.npz")
        )
        summary["sorts_on_disk"] = on_disk
        if on_disk != n_sorts:
            raise SystemExit(f"analysed {n_sorts} sorts but {on_disk} are on disk")

    if len(old):
        rep = reproduce(old)
        rep.to_csv(tables / "reproduce.csv", index=False)
        summary["reproduce"] = rep.to_dict(orient="records")

    d = with_flips(new)
    st = stability(d[d["family"] != "oracle"])
    st.to_csv(tables / "stability.csv", index=False)
    q = quality_frame(d)
    ql = quality(q)
    ql.to_csv(tables / "quality.csv", index=False)
    pr = paired(q[q["family"] != "oracle"])
    pr.to_csv(tables / "paired.csv", index=False)
    by_dataset(q, d).to_csv(tables / "by_dataset.csv", index=False)
    lodo(q).to_csv(tables / "lodo.csv", index=False)
    shipped_shape(q).to_csv(tables / "shipped_shape.csv", index=False)
    null_check(q).to_csv(tables / "null_check.csv", index=False)
    examples(q).to_csv(tables / "examples.csv", index=False)
    t = d[(d["frame"] == "full") & d["seconds"].notna()].groupby("rule")["seconds"].median().rename("median_seconds")
    t.reset_index().to_csv(tables / "timing.csv", index=False)
    if args.corpus:
        autopilot(q, Path(args.corpus), HEADLINE + ["oracle_cost", "oracle_f1"]).to_csv(
            tables / "autopilot.csv", index=False
        )

    head = st.set_index("rule").join(ql.set_index("rule")[["cost", "excess_cost", "f1", "median_adm_frac"]])
    summary["headline"] = head.loc[[r for r in HEADLINE if r in head.index]].reset_index().to_dict(orient="records")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3g}".format):
        print(head.loc[[r for r in HEADLINE if r in head.index]])
        print(pr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
