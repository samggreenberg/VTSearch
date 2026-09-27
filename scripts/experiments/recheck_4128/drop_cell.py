"""#4128: recompute each study's decision rule with the raw ``coco_val x siglip`` rows dropped.

``coco_val__siglip.pkl`` held un-normalised SigLIP vectors (norms 12-19) from
its build on 2026-08-04 until #4099's fix on 2026-09-23, and the harness read it
raw.  Every study below trained and scored that environment on a detector the
app never builds.  For each one this prints the study's own headline twice --
over every environment (it must reproduce the committed number) and with the
``coco_val x siglip`` rows dropped -- so the reader can see whether the verdict
rested on the cell.

Read-only.  Inputs are the committed ``agg/`` tables where a study kept them and
the archived results roots on /expscratch otherwise; a study whose results root
is gone was recomputed from its REPORT tables by hand and is not in here (see
the report's table).

    python scripts/experiments/recheck_4128/drop_cell.py [study ...]

Runs on a CPU node (``srun --partition=cpu --mem=8G``): the overview bench reads
~500 cell CSVs.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs" / "experiments"
G = Path(os.environ.get("RECHECK_ROOT", "/expscratch/sgreenberg"))

RAW_ARM = "coco_val/siglip/whole_image"


def is_raw(df: pd.DataFrame) -> pd.Series:
    return (df.dataset == "coco_val") & (df.embedder == "siglip")


def ms(s: pd.Series) -> str:
    s = s.dropna()
    return f"{s.mean():+.2g} ± {s.std(ddof=1) / np.sqrt(len(s)):.2g} (n={len(s)})"


def both(label: str, df: pd.DataFrame, mask: pd.Series, fn) -> None:
    print(f"  {label:<44} all: {fn(df):<36} dropped: {fn(df[~mask])}")


# --- shared shapes ----------------------------------------------------------


def gate(cuts: pd.DataFrame, base: str, arm: str) -> pd.DataFrame:
    """Per-case |admitted-set change| of *arm* against *base*, as a haystack fraction."""
    k = ["cell", "dataset", "embedder", "category", "seed", "style", "kind", "case"]
    b = cuts[cuts.arm == base].set_index(k)
    j = cuts[cuts.arm == arm].set_index(k).join(b[["n_admitted_i0"]], rsuffix="_b", how="inner").reset_index()
    j["f"] = (j.n_admitted_i0 - j.n_admitted_i0_b).abs() / j.n
    return j


def ab_paired(path: Path) -> pd.DataFrame:
    p = pd.read_csv(path)
    q = p[(p.scope == "app_visible") & (p.window == "all_steps")].copy()
    for m in ("cost", "fpr", "fnr"):
        q["d_" + m] = q[m + "_on"] - q[m + "_off"]
    return q


def ab_lines(label: str, q: pd.DataFrame, metrics=("cost",)) -> None:
    raw = q.arm == RAW_ARM
    for m in metrics:
        both(f"{label} Δ{m}", q, raw, lambda d, m=m: ms(d["d_" + m]))
    print(f"  {label} Δcost, the cell alone: {ms(q[raw].d_cost)}")


def changed_pct(j: pd.DataFrame) -> str:
    return f"{100 * (j.f > 0).mean():.2g}% of {len(j)} cases"


# --- studies ----------------------------------------------------------------


def gmm_3585() -> None:
    c = pd.read_csv(G / "gmm-3585/analysis/gate_cuts.csv")
    for arm in ("native", "sklearn_kmeanspp"):
        j = gate(c[c.kind == "fold"], "baseline", arm)
        both(f"fold gate, {arm} changes", j, is_raw(j), changed_pct)
        for sub, sel in (
            ("all sorts", c.kind == "sort"),
            ("text sorts", (c.kind == "sort") & (c["style"] == "text_sort")),
        ):
            j = gate(c[sel], "baseline", arm)
            both(f"{sub}, {arm} mean haystack moved", j, is_raw(j), lambda d: f"{100 * d.f.mean():.2g}% (n={len(d)})")
    ab_lines("A/B native−sklearn", ab_paired(G / "gmm-3585/analysis/ab/agg/ab_paired_cells.csv"))


def anchem_3825() -> None:
    c = pd.read_csv(G / "anchem-3825/analysis/gate3825_cuts.csv")
    for arm in ("ll1e-8", "ll1e-3"):
        j = gate(c, "baseline", arm)
        both(f"gate {arm}: cases moved", j, is_raw(j), changed_pct)
        both(
            f"gate {arm}: p90 / max moved",
            j,
            is_raw(j),
            lambda d: f"{100 * d.f.quantile(0.9):.2g}% / {100 * d.f.max():.2g}%",
        )
    for arm in ("baseline", "ll1e-8"):
        z = c[c.arm == arm]
        both(f"non-convergence, {arm}", z, is_raw(z), lambda d: f"{100 * d.n_unconverged.sum() / d.n_folds.sum():.2g}%")
    for arm in ("ll1e-8", "ll1e-3"):
        path = G / f"anchem-3825/analysis/ab_{arm}/agg/ab_paired_cells.csv"
        if path.exists():
            ab_lines(f"A/B {arm}−baseline", ab_paired(path), ("cost", "fnr"))


def pool(g: pd.DataFrame) -> pd.Series:
    """Pool per-environment (n, mean, sd) rows into one paired distribution."""
    n = g.n.sum()
    m = (g.n * g["mean"]).sum() / n
    ss = ((g.n - 1) * g.sd**2 + g.n * (g["mean"] - m) ** 2).sum()
    sd = np.sqrt(ss / (n - 1))
    return pd.Series({"n": n, "mean": m, "sd": sd, "se": sd / np.sqrt(n)})


def abres_3840() -> None:
    e = pd.read_csv(DOCS / "2026-09-22-ab-resolution-3840/agg/env.csv")
    for pair, g in e.groupby("pair"):
        a, d = pool(g), pool(g[g.arm != RAW_ARM])
        print(
            f"  σ {pair:<30} all: {a.sd:.2g} (n={a.n:.0f})   dropped: {d.sd:.2g} (n={d.n:.0f}), mean {d['mean']:+.2g} ± {d.se:.2g}"
        )


def maxiter_3839() -> None:
    c = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(G / "maxiter-3839/analysis/gate3839_cuts.*.csv"))))
    k = ["cell", "style", "case"]
    sh = c[c.arm == "shipped"].set_index(k)
    lim = c[c.arm == "limit"].set_index(k)
    capped = sh[sh.n_unconverged > 0]
    for tag, idx in (("all", capped.index), ("dropped", capped[~is_raw(capped)].index)):
        parts = [f"{len(idx)} capped cases"]
        for arm in ("shipped", "cap2000", "fallback", "stall1e-4"):
            x = c[c.arm == arm].set_index(k).loc[idx]
            f = 100 * (x.n_admitted_i0 - lim.loc[idx].n_admitted_i0).abs() / x.n
            parts.append(f"{arm} med {f.median():.2g}% p90 {f.quantile(0.9):.2g}%")
        print(f"  distance to converged, {tag}: " + "; ".join(parts))
    ab_lines(
        "A/B cap2000−shipped",
        ab_paired(G / "maxiter-3839/analysis/ab_cap2000/agg/ab_paired_cells.csv"),
        ("cost", "fpr"),
    )


def linhead_2808() -> None:
    d = pd.read_csv(G / "linhead-2808/analysis/agg/trajectories.csv")
    k = ["dataset", "embedder", "category", "seed"]

    def run(df: pd.DataFrame, label: str) -> None:
        a = df[df.arm == "A_shipped"].set_index(k)
        b = df[df.arm == "B_converged"].set_index(k)
        c = df[df.arm == "C_mlp"].set_index(k)
        j = a.join(b, lsuffix="_A", rsuffix="_B", how="inner")
        cols = {"worst-step regret": "max_excess_warm", "final cost": "final_cost", "positives": "n_good_final"}
        parts = [f"{name} {ms(j[col + '_B'] - j[col + '_A'])}" for name, col in cols.items()]
        parts.append(f"deep-spike A/C {a.has_deep.mean() / c.has_deep.mean():.2g}")
        print(f"  B_converged−A_shipped, {label}: " + "; ".join(parts))

    run(d, "all")
    run(d[~is_raw(d)], "dropped")
    run(d[is_raw(d)], "the cell alone")
    run(d[(d.embedder == "siglip") & (d.dataset != "coco_val")], "VG × siglip only")


def cutincl_2865() -> None:
    d = pd.read_csv(DOCS / "2026-08-21-inclusion-cut-rule/cutincl_regret_vs_incumbent.csv")
    d["arm"] = d.arm.str.replace("fold_anchored_w0.3_", "").str.replace("_qmean", "")
    d["harm"] = d.ci_lo > 0.01
    raw = d.env.str.startswith("coco_val/siglip")
    t = pd.DataFrame({"all": d.groupby("arm").harm.sum(), "dropped": d[~raw].groupby("arm").harm.sum()})
    print("  harmful (arm, k) points, whole CI above +0.01:")
    print("    " + t.to_string().replace("\n", "\n    "))
    neg = d[d.arm.isin(["rate", "cross_tilt"]) & (d.inclusion_k < 0)]
    print("  best (most negative) Δregret at k<0 by env, the tilt lead #3557 took up:")
    print("    " + neg.groupby(["arm", "env"]).d_regret.min().round(3).to_string().replace("\n", "\n    "))


def fitq_3329() -> None:
    b = pd.read_csv(G / "struct-3329/analysis/agg/b1_b2_atlas_uniformity.csv")
    both("B1/B2 median KS", b, is_raw(b), lambda x: f"{x.ks_shipped.median():.2g}")
    d = pd.read_csv(G / "struct-3329/results/domainshift_siglip.csv")

    def src(s: str) -> str:
        return "coco" if s.startswith("coco") else "caltech" if s.startswith("caltech") else "vg"

    d["diff"] = [src(a) != src(q) for a, q in zip(d.build_dataset, d.query_dataset)]
    for tag, x in (
        ("all", d),
        ("coco_val dropped", d[(d.build_dataset != "coco_val") & (d.query_dataset != "coco_val")]),
    ):
        fp = x[x.is_self].shifted.mean()
        det = x[~x.is_self & x["diff"]].shifted.mean()
        print(
            f"  siglip domain shift, {tag}: self false-positive {fp:.2g}, different-source caught {det:.2g}, separation {det - fp:.2g}"
        )


def overview_3129() -> None:
    def load(root: Path, tag: str) -> pd.DataFrame:
        want = {"seed", "dataset", "category", "style", "embedder", "t", "n_good", "cost", "rule_inefficiency"}
        out = []
        for f in sorted(glob.glob(str(root / "cells/task_*.csv"))):
            if "__" in os.path.basename(f):
                continue
            try:
                x = pd.read_csv(f, usecols=lambda c: c in want)
            except (pd.errors.EmptyDataError, ValueError):
                continue
            if len(x):
                out.append(x.assign(grid=tag))
        return pd.concat(out, ignore_index=True)

    df = pd.concat(
        [
            load(G / "bench-overview/results", "overview"),
            load(G / "bench-vgbox2/results", "vgbox"),
            load(G / "bench-binary/results", "binary"),
        ]
    )
    key = ["grid", "dataset", "embedder", "style", "category", "seed"]
    # The report's rule-inefficiency headline is over the overview + VG-box grids.
    r = (
        df[(df.t >= 100) & df.grid.isin(["overview", "vgbox"])]
        .dropna(subset=["rule_inefficiency"])
        .groupby(key)
        .rule_inefficiency.mean()
        .reset_index()
    )
    both("rule inefficiency", r, is_raw(r), lambda x: ms(x.rule_inefficiency))
    fin = df.sort_values("t").groupby(key).tail(1)
    both("runs ending on ≤2 positives", fin, is_raw(fin), lambda x: f"{100 * (x.n_good <= 2).mean():.2g}% (n={len(x)})")
    old = load(G / "bench-overview/results", "o")
    new = load(G / "bench-h250/wave1/results", "n")
    k = ["dataset", "embedder", "style", "category", "seed"]
    o = old[(old.t >= 101) & (old.t <= 150)].groupby(k).cost.mean()
    n = new[(new.t >= 201) & (new.t <= 250)].groupby(k).cost.mean()
    j = pd.concat([o.rename("o"), n.rename("n")], axis=1).dropna().reset_index()
    j = j[j.dataset.isin(["visual_genome_m", "coco_val"])]
    j["d"] = j.n - j.o
    both("horizon 250, VG+COCO column", j, is_raw(j), lambda x: ms(x.d))


def blend_3551() -> None:
    t = pd.read_csv(DOCS / "2026-09-22-blend-endpoints-3551/agg/ab_binary_table_ab-binary-corridor_w_0.2.csv")
    t = t[t.weighting == "1:1"]
    for tag, x in (("all", t), ("dropped", t[~t.env.str.startswith("coco_val")])):
        n = x.n.sum()
        m = (x.n * x["mean"]).sum() / n
        se = np.sqrt(((x.n * x.se) ** 2).sum()) / n
        print(f"  binary corridor20−cap50 pooled Δcost, {tag}: {m:+.2g} ± {se:.2g} (n={n}), upper {m + 2 * se:+.2g}")


def hinge_3557() -> None:
    p = pd.read_csv(DOCS / "2026-09-22-hinge-tilt-3557/hinge3557_paired.csv")
    p["env"] = p.dataset + "/" + p["style"]
    for c, q in p.groupby("contrast"):
        print(f"  {c}: harmed stops by env {q[q.harmed].groupby('env').size().to_dict()}")


STUDIES = {
    "gmm-3585": gmm_3585,
    "anchem-3825": anchem_3825,
    "abres-3840": abres_3840,
    "maxiter-3839": maxiter_3839,
    "linhead-2808": linhead_2808,
    "cutincl-2865": cutincl_2865,
    "fitq-3329": fitq_3329,
    "overview-3129": overview_3129,
    "blend-3551": blend_3551,
    "hinge-3557": hinge_3557,
}


def main(argv: list[str]) -> None:
    for name in argv or list(STUDIES):
        print(f"== {name}")
        try:
            STUDIES[name]()
        except FileNotFoundError as e:
            print(f"  SKIPPED, input gone: {e.filename}")


if __name__ == "__main__":
    main(sys.argv[1:])
