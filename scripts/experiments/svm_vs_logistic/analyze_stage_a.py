"""#3197 Stage A analysis: the heads on fixed vote sets.

Every contrast is a paired difference on the SAME vote set (same env, category,
seed, condition, level, size), with a standard error clustered on
(env, category).  Differences are written ``a - b`` in the metric's own units;
for AP/AUROC positive means ``a`` ranks better, for oracle cost negative does.

Tables written (all CSV, all consumed by ``figures.py`` and the report):

* ``A_base.csv``           each arm vs the shipped ``svm`` and ``lr``, per env x size;
* ``A_bestC.csv``          each family at its best C, chosen on the OTHER seeds;
* ``A_mechanism.csv``      the gap ``svm - lr`` (and ``svm - lrconv_C1``) along
                           every manipulated axis: dilute k, noise r, misvote,
                           Bad:Good ratio, the input-scale arm;
* ``A_geometry.csv``       active-margin fraction, ||w||, direction cosines;
* ``A_control.csv``        the shuffled-label control;
* ``A_replay.csv``         the heads on Stage B's own Autopilot vote sets.

    python analyze_stage_a.py --controlled DIR --replay DIR --out DIR
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["env", "category", "seed", "condition", "level", "size"]
CLUSTER = ["env", "category"]
METRICS = ("test_ap", "test_auroc", "test_oracle_cost")


def load(d: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(d / "*.csv.gz")))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["env"] = (
        df["dataset"].str.replace("visual_genome_m", "vg").str.replace("caltech101_m", "caltech") + "/" + df["embedder"]
    )
    return df


def diff(sub: pd.DataFrame, a: str, b: str, m: str, by: list[str] | None = None) -> pd.DataFrame:
    """Paired ``a - b`` on *m*, summarised per *by* group with clustered SE."""
    p = sub.pivot_table(index=KEY, columns="arm", values=m)
    if a not in p or b not in p:
        return pd.DataFrame()
    x = (p[a] - p[b]).dropna().rename("d").reset_index()
    groups = by or []
    out = []
    for g, gx in x.groupby(groups) if groups else [((), x)]:
        c = gx.groupby(CLUSTER)["d"].mean()
        se = float(c.std(ddof=1) / np.sqrt(len(c))) if len(c) > 1 else float("nan")
        row = dict(zip(groups, g if isinstance(g, tuple) else (g,))) if groups else {}
        row.update(a=a, b=b, metric=m, mean=float(gx["d"].mean()), se=se, n=len(gx), n_clusters=len(c))
        out.append(row)
    return pd.DataFrame(out)


def best_c(df: pd.DataFrame, family: str, arms: list[str]) -> pd.DataFrame:
    """Per row, the family's score at the C that is best on the OTHER seeds of that env x size.

    Cross-fitted so the chosen C never saw the vote set it is scored on.
    """
    b = df[(df.condition == "base") & df.arm.isin(arms)]
    means = b.groupby(["env", "size", "seed", "arm"])["test_ap"].mean().reset_index()
    rows = []
    for (env, size, seed), _ in means.groupby(["env", "size", "seed"]):
        other = means[(means.env == env) & (means["size"] == size) & (means.seed != seed)]
        pick = other.groupby("arm")["test_ap"].mean().idxmax()
        sel = b[(b.env == env) & (b["size"] == size) & (b.seed == seed) & (b.arm == pick)].copy()
        sel["picked"] = pick
        sel["arm"] = family
        rows.append(sel)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controlled", required=True, type=Path)
    ap.add_argument("--replay", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = load(args.controlled)
    real = d[~d.env.str.endswith("@raw")]
    base = real[real.condition == "base"]

    # --- base: every arm against both shipped heads ---
    arms = sorted(set(base.arm))
    parts = []
    for ref in ("svm", "lr"):
        for a in arms:
            if a == ref:
                continue
            for m in METRICS:
                parts.append(diff(base, a, ref, m, by=["size"]))
                parts.append(diff(base, a, ref, m, by=["env", "size"]))
    pd.concat(parts, ignore_index=True).to_csv(args.out / "A_base.csv", index=False)

    # --- best-C per family, cross-fitted ---
    svm_fam = best_c(real, "svm_best", ["svm_C0.01", "svm_C0.1", "svm", "svm_C10", "svm_C100"])
    lr_fam = best_c(real, "lrconv_best", [f"lrconv_C{c}" for c in ("0.01", "0.1", "1", "10", "100")])
    both = pd.concat([base, svm_fam, lr_fam], ignore_index=True)
    bc = (
        [diff(both, "lrconv_best", "svm_best", m, by=["size"]) for m in METRICS]
        + [diff(both, "svm_best", "svm", m, by=["size"]) for m in METRICS]
        + [diff(both, "lrconv_best", "lr", m, by=["size"]) for m in METRICS]
    )
    picked = pd.concat([svm_fam, lr_fam]).groupby(["arm", "size", "picked"]).size().rename("n").reset_index()
    pd.concat(bc, ignore_index=True).to_csv(args.out / "A_bestC.csv", index=False)
    picked.to_csv(args.out / "A_bestC_picks.csv", index=False)

    # --- mechanism axes ---
    mech = []
    for cond, by in (
        ("dilute", ["size", "level"]),
        ("noise", ["size", "level"]),
        ("misvote", ["size", "level"]),
        ("ratio", ["level"]),
    ):
        sub = real[real.condition == cond]
        for a, b in (
            ("svm", "lr"),
            ("svm", "lrconv_C1"),
            ("lrconv_C1", "lr"),
            ("svm_hinge", "svm"),
            ("svm_C0.1", "svm"),
        ):
            for m in METRICS:
                r = diff(sub, a, b, m, by=by)
                if not r.empty:
                    r["condition"] = cond
                    mech.append(r)
    raw = d[(d.condition == "base") & d.env.str.endswith("@raw")]
    norm = d[(d.condition == "base") & (d.env == "coco_val/siglip")]
    for label, sub in (("scale_raw", raw), ("scale_unit", norm)):
        for a, b in (("svm", "lr"), ("svm", "lrconv_C1"), ("lrconv_C1", "lr")):
            for m in METRICS:
                r = diff(sub, a, b, m, by=["size"])
                if not r.empty:
                    r["condition"] = label
                    mech.append(r)
    pd.concat(mech, ignore_index=True).to_csv(args.out / "A_mechanism.csv", index=False)

    # absolute levels along each axis, for the figures
    lv = (
        real[real.condition.isin(["dilute", "noise", "misvote", "ratio"])]
        .groupby(["condition", "size", "level", "arm"])[list(METRICS)]
        .mean()
        .reset_index()
    )
    lv.to_csv(args.out / "A_levels.csv", index=False)
    base.groupby(["size", "arm"])[list(METRICS)].mean().reset_index().to_csv(
        args.out / "A_base_levels.csv", index=False
    )

    # --- geometry ---
    geo = (
        base.groupby(["size", "arm"])[["active_frac", "wnorm", "cos_svm", "cos_lrconv1", "cos_centroid", "train_auroc"]]
        .mean()
        .reset_index()
    )
    geo.to_csv(args.out / "A_geometry.csv", index=False)

    # --- control ---
    sh = real[real.condition == "shuffled"]
    ctl = sh.groupby(["level", "arm"]).agg(
        test_auroc=("test_auroc", "mean"),
        test_auroc_se=("test_auroc", "sem"),
        train_auroc=("train_auroc", "mean"),
        n=("test_auroc", "size"),
    )
    ctl.reset_index().replace({"level": {0: "NOISE", 1: "REAL"}}).to_csv(args.out / "A_control.csv", index=False)

    # --- replay ---
    if args.replay is not None and list(args.replay.glob("*.csv.gz")):
        r = load(args.replay)
        rep = []
        for cond in sorted(r.condition.unique()):
            sub = r[r.condition == cond]
            for a, b in (
                ("svm", "lr"),
                ("lrconv_C1", "lr"),
                ("lrconv_C1", "svm"),
                ("svm_C0.1", "svm"),
                ("lr_ep2000", "lr"),
                ("mlp", "svm"),
            ):
                for m in METRICS:
                    x = diff(sub, a, b, m, by=["level"])
                    if not x.empty:
                        x["condition"] = cond
                        rep.append(x)
        pd.concat(rep, ignore_index=True).to_csv(args.out / "A_replay.csv", index=False)
        r.groupby(["condition", "level", "arm"])[list(METRICS)].mean().reset_index().to_csv(
            args.out / "A_replay_levels.csv", index=False
        )
    print("wrote", sorted(p.name for p in args.out.glob("A_*.csv")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
