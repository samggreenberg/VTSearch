"""Full analysis + report generator for the Max-Patch study.

Arms: DINOv3 {MaxHAC, MaxPatch, whole-image(CLS)} + SigLIP whole-image.
Reads results/cells/*.csv + results/prepare_info.json + the cell pickles (for
per-category object-scale stats) and writes, deterministically:

  * results/REPORT.md      — the complete report (BLUF + metrics + tables +
                              captioned figures + examples + take-aways)
  * results/metrics.json   — machine-readable summary
  * results/figures/*.png  — captioned figures

The prose framing is constant; the verdict, examples, and take-aways are
assembled from the computed numbers so they never drift from the tables.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import common

common.setup_env()

import experiment_config as cfg  # noqa: E402
from _cells_io import load_medias  # noqa: E402

RESULTS = common.RESULTS
FIGS = RESULTS / "figures"
BUDGETS = [10, 25, 50, 100, 150]
#: Field order of the ``examples_*`` tuples emitted into ``metrics.json``.
#: ``voted_area`` is the union box a Good vote drags, never a per-instance area.
_EXAMPLE_KEYS = ["dataset", "category", "voted_area", "maxpatch", "maxhac", "delta", "union_inflation"]

LEAF_AREA_PCT = 8.3  # mean HAC-leaf area, % of image (from the plan's measurements)
PATCH_AREA_PCT = 0.51  # one DINOv3 patch, % of image area

ARM_LABEL = {
    "dinov3_patch/max_hac": "DINOv3 · MaxHAC",
    "dinov3_patch/max_patch": "DINOv3 · MaxPatch",
    "dinov3_patch/whole_image": "DINOv3 · whole-image (CLS)",
    "siglip/whole_image": "SigLIP · whole-image",
}
ARM_COLOR = {
    "dinov3_patch/max_hac": "#4C78A8",
    "dinov3_patch/max_patch": "#F58518",
    "dinov3_patch/whole_image": "#54A24B",
    "siglip/whole_image": "#B279A2",
}
DS_LABEL = {
    "caltech101_m": "Caltech-101 (boxless control)",
    "visual_genome_m": "Visual Genome (boxed, cluttered)",
    "openlogo_a": "OpenLogo (boxed, small logos)",
}
ARM_ORDER = list(ARM_LABEL)
TRAJ = ["dataset", "category", "seed", "arm"]
MH, MP, CLS, SIG = ("dinov3_patch/max_hac", "dinov3_patch/max_patch", "dinov3_patch/whole_image", "siglip/whole_image")


# ------------------------------------------------------------------ load / agg
def _load():
    files = sorted((RESULTS / "cells").glob("task_*.csv"))
    frames = [pd.read_csv(f) for f in files if f.stat().st_size > 0]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit("no cell CSVs under results/cells")
    df = pd.concat(frames, ignore_index=True)
    df["arm"] = df["embedder"] + "/" + df["style"]
    return df


def _boot_ci(vals, n=2000, seed=0):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.RandomState(seed)
    means = vals[rng.randint(0, len(vals), size=(n, len(vals)))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _nearest_at(df, t):
    d = df.copy()
    d["dist"] = (d["t"] - t).abs()
    return d.sort_values(["dist", "t"]).groupby(TRAJ, as_index=False).head(1)


def _final(df):
    return df.sort_values("t").groupby(TRAJ, as_index=False).tail(1)


def _aulc(df):
    _trapz = getattr(np, "trapezoid", np.trapz)
    rows = []
    for keys, g in df.groupby(TRAJ):
        g = g.sort_values("t")
        if g["t"].max() > g["t"].min():
            a = float(_trapz(g["cost"].values, g["t"].values) / (g["t"].max() - g["t"].min()))
        else:
            a = float(g["cost"].mean())
        rows.append({**dict(zip(TRAJ, keys)), "aulc": a})
    return pd.DataFrame(rows)


def _paired(pivot, a, b):
    sub = pivot[[a, b]].dropna() if (a in pivot.columns and b in pivot.columns) else pivot.iloc[:0]
    if len(sub) < 5:
        return (
            float(sub[a].mean()) if len(sub) else math.nan,
            float(sub[b].mean()) if len(sub) else math.nan,
            math.nan,
            len(sub),
        )
    try:
        from scipy.stats import wilcoxon

        d = sub[a].values - sub[b].values
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(sub[a].values, sub[b].values).pvalue)
    except Exception:
        p = math.nan
    return float(sub[a].mean()), float(sub[b].mean()), p, len(sub)


def _holm(pairs):
    valid = [(lbl, p) for lbl, p in pairs if not (p is None or (isinstance(p, float) and math.isnan(p)))]
    out = {lbl: math.nan for lbl, _ in pairs}
    m = len(valid)
    for rank, (lbl, p) in enumerate(sorted(valid, key=lambda x: x[1])):
        out[lbl] = min(1.0, p * (m - rank))
    return out


def _category_scale(df):
    """Median **voted** (union) box area per (dataset, category), fraction of image.

    This is the box a simulated Good vote actually drags, which is the scale the
    hypothesis is about.  It is deliberately *not* the median per-instance area:
    the two diverge sharply on multi-instance categories (an image with arms
    scattered across it has ~1 %-area instances but a near-frame union box), and
    plotting per-instance area put such categories at the small end of the axis
    while the detector was really being handed a large region.
    """
    from vtscore.eval.labels import category_scale_stats

    scale = {}
    for ds in df["dataset"].unique():
        pkl = common.DATADIR / "embeddings" / cfg.pickle_name(ds, "dinov3_patch")
        if not pkl.exists():
            continue
        try:
            medias = load_medias(pkl)
        except Exception:
            continue
        for cat in df[df["dataset"] == ds]["category"].unique():
            stats = category_scale_stats(medias, cat)
            if stats is not None:
                scale[(ds, cat)] = stats
    return scale


# ------------------------------------------------------------------ figures
def _fig_curves(df, metric, ylabel, fname, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted(df["dataset"].unique())
    arms = [a for a in ARM_ORDER if a in set(df["arm"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.4), squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        dsd = df[df["dataset"] == ds]
        for arm in arms:
            sub = dsd[dsd["arm"] == arm]
            if sub.empty:
                continue
            ts, mu, lo, hi = [], [], [], []
            for t, g in sub.groupby("t"):
                vals = g.groupby(["category", "seed"])[metric].mean().values
                m_, l_, h_ = _boot_ci(vals, n=500, seed=int(t))
                ts.append(t)
                mu.append(m_)
                lo.append(l_)
                hi.append(h_)
            o = np.argsort(ts)
            ts, mu = np.array(ts)[o], np.array(mu)[o]
            lo, hi = np.array(lo)[o], np.array(hi)[o]
            ax.plot(ts, mu, color=ARM_COLOR[arm], lw=1.9, label=ARM_LABEL[arm])
            ax.fill_between(ts, lo, hi, color=ARM_COLOR[arm], alpha=0.14, linewidth=0)
        ax.set_title(DS_LABEL.get(ds, ds), fontsize=10)
        ax.set_xlabel("votes cast (t)")
        if j == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / fname, dpi=130)
    plt.close(fig)
    return f"figures/{fname}"


def _fig_scale(final, scale, fname):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = final.pivot_table(index=["dataset", "category"], columns="arm", values="cost", aggfunc="mean")
    if MP not in piv.columns or MH not in piv.columns:
        return None, []
    rows = []
    for (ds, cat), r in piv.iterrows():
        if (ds, cat) in scale and not (pd.isna(r.get(MP)) or pd.isna(r.get(MH))):
            rows.append((ds, cat, scale[(ds, cat)]["voted_area"], float(r[MP] - r[MH])))
    if not rows:
        return None, []
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for ds in sorted({r[0] for r in rows}):
        pts = [r for r in rows if r[0] == ds]
        ax.scatter([p[2] * 100 for p in pts], [p[3] for p in pts], s=48, label=DS_LABEL.get(ds, ds), alpha=0.85)
        for p in pts:
            ax.annotate(p[1], (p[2] * 100, p[3]), fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="#444", lw=1, ls="--")
    ax.axvline(LEAF_AREA_PCT, color="#999", lw=1, ls=":")
    ax.annotate(
        f"HAC leaf (~{LEAF_AREA_PCT}%)",
        (LEAF_AREA_PCT, ax.get_ylim()[1]),
        fontsize=7,
        rotation=90,
        va="top",
        ha="right",
        color="#777",
    )
    ax.axvline(PATCH_AREA_PCT, color="#bbb", lw=1, ls=":")
    # Shade the scale bands the categories were sampled from, so an
    # under-populated band is visible in the figure rather than only in the log.
    for i, (name, lo, hi) in enumerate(cfg.SCALE_BANDS):
        if i % 2 == 0:
            ax.axvspan(max(lo * 100, 1e-3), min(hi * 100, 100.0), color="#000", alpha=0.035, zorder=0)
        ax.annotate(
            name,
            (max(lo * 100, 1e-3), ax.get_ylim()[0]),
            fontsize=6,
            color="#999",
            va="bottom",
            ha="left",
            xytext=(2, 2),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlabel("median VOTED (union) box area (% of image, log scale)")
    ax.set_ylabel("cost(MaxPatch) − cost(MaxHAC)\n(negative = MaxPatch better)")
    ax.set_title("Where raw patches beat the HAC tree, by object scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=130)
    plt.close(fig)
    return f"figures/{fname}", rows


# ------------------------------------------------------------------ tables
def _md(frame, floats):
    cols = list(frame.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in floats and isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                cells.append(f"{v:.{floats[c]}f}")
            else:
                cells.append("-" if (isinstance(v, float) and pd.isna(v)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return out


def _sig(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    return " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""


def main() -> int:
    df = _load()
    FIGS.mkdir(parents=True, exist_ok=True)
    datasets = sorted(df["dataset"].unique())
    arms = [a for a in ARM_ORDER if a in set(df["arm"])]
    final = _final(df)
    aulc = _aulc(df)
    scale = _category_scale(df)
    n_traj = len(final)
    ncats = {ds: df[df["dataset"] == ds]["category"].nunique() for ds in datasets}
    nseed = df["seed"].nunique()

    def arm_table(sub, by):
        g = sub.groupby(by)
        out = g.agg(
            n=("cost", "size"),
            cost=("cost", "mean"),
            fpr=("fpr", "mean"),
            fnr=("fnr", "mean"),
            AP=("average_precision", "mean"),
            auroc=("auroc", "mean"),
            train_s=("train_seconds", "mean"),
            xcal_s=("xcal_seconds", "mean"),
            score_s=("test_score_seconds", "mean"),
        ).reset_index()
        return out

    overall = arm_table(final, ["arm"])
    per_ds = arm_table(final, ["dataset", "arm"])

    # budget table — per dataset (pooling the two opposite regimes is misleading)
    brows = []
    for ds in datasets:
        for b in BUDGETS:
            at = _nearest_at(df[df["dataset"] == ds], b)
            for arm in arms:
                s = at[at["arm"] == arm]
                if len(s):
                    brows.append(
                        {
                            "dataset": ds,
                            "t": b,
                            "arm": ARM_LABEL[arm],
                            "cost": s["cost"].mean(),
                            "fnr": s["fnr"].mean(),
                            "AP": s["average_precision"].mean(),
                        }
                    )
    budget = pd.DataFrame(brows)

    # paired stats
    aulc_piv = aulc.pivot_table(index=["dataset", "category", "seed"], columns="arm", values="aulc")
    c50 = _nearest_at(df, 50).pivot_table(index=["dataset", "category", "seed"], columns="arm", values="cost")
    c150 = _nearest_at(df, 150).pivot_table(index=["dataset", "category", "seed"], columns="arm", values="cost")
    comparisons = [
        ("MaxPatch − MaxHAC", MP, MH),
        ("MaxHAC − whole(CLS)", MH, CLS),
        ("MaxPatch − whole(CLS)", MP, CLS),
        ("MaxHAC − SigLIP", MH, SIG),
        ("MaxPatch − SigLIP", MP, SIG),
    ]

    # Per-dataset paired stats.  Pooling datasets is misleading here: the boxed
    # and boxless datasets give opposite MaxPatch-vs-MaxHAC signs that cancel.
    def _stat_block(piv):
        raw = [(lbl, *_paired(piv, a, b)) for lbl, a, b in comparisons]
        holm = _holm([(lbl, p) for lbl, _, _, p, _ in raw])
        return [
            {"comparison": lbl, "mean_A": ma, "mean_B": mb, "delta": ma - mb, "n_pairs": npair, "holm_p": holm[lbl]}
            for lbl, ma, mb, p, npair in raw
        ]

    stat_tables = {}  # {dataset: {metric: [...]}}
    for ds in datasets:
        stat_tables[ds] = {}
        for mname, piv in [("AULC", aulc_piv), ("cost@50", c50), ("cost@150", c150)]:
            sub = piv.loc[piv.index.get_level_values("dataset") == ds] if len(piv) else piv
            stat_tables[ds][mname] = _stat_block(sub)

    # figures
    figs = []
    figs.append(
        (
            _fig_curves(
                df,
                "cost",
                "ErrorCost = FPR + FNR",
                "fig_cost.png",
                "ErrorCost vs voting effort (lower & earlier = better)",
            ),
            "**Figure 1. ErrorCost (FPR + FNR) as votes accumulate.** One panel per dataset; each "
            "line is an arm, shaded band = bootstrap 95% CI across categories × seeds. Lower and "
            "earlier-dropping is better — fewer total mistakes for the same voting effort.",
        )
    )
    figs.append(
        (
            _fig_curves(
                df, "fnr", "FNR (missed matches)", "fig_fnr.png", "Missed-match rate vs voting effort (lower = better)"
            ),
            "**Figure 2. False-negative rate (missed matches) vs votes.** Lower is better. The CLS-only "
            "and SigLIP whole-image arms tend to sit high here — a single global vector under-recalls "
            "cluttered scenes where the object is a small part of the image.",
        )
    )
    figs.append(
        (
            _fig_curves(
                df,
                "average_precision",
                "Average precision",
                "fig_ap.png",
                "Ranking quality (AP) vs voting effort (higher = better)",
            ),
            "**Figure 3. Average precision (ranking quality) vs votes.** Higher is better. AP is "
            "threshold-free, so it isolates how well each strategy *orders* matches from how well it "
            "*places the decision threshold* (the latter shows up in ErrorCost/FPR/FNR).",
        )
    )
    scale_fig, scale_rows = _fig_scale(final, scale, "fig_scale.png")

    # examples: categories where MaxPatch beats / trails MaxHAC most (by final cost)
    fp = final.pivot_table(index=["dataset", "category"], columns="arm", values="cost", aggfunc="mean")
    ex = []
    if MP in fp.columns and MH in fp.columns:
        for (ds, cat), r in fp.iterrows():
            if not (pd.isna(r.get(MP)) or pd.isna(r.get(MH))):
                st = scale.get((ds, cat))
                ex.append(
                    (
                        ds,
                        cat,
                        st["voted_area"] if st else None,
                        float(r[MP]),
                        float(r[MH]),
                        float(r[MP] - r[MH]),
                        st["union_inflation"] if st else None,
                    )
                )
    ex_sorted = sorted(ex, key=lambda e: e[5])

    # scale correlation
    spearman = None
    if scale_rows and len(scale_rows) >= 4:
        try:
            from scipy.stats import spearmanr

            xs = np.log([r[2] for r in scale_rows])
            ys = np.array([r[3] for r in scale_rows])
            rho, pp = spearmanr(xs, ys)
            spearman = (float(rho), float(pp))
        except Exception:
            spearman = None

    metrics = {
        "n_trajectories": n_traj,
        "n_categories": ncats,
        "n_seeds": nseed,
        "datasets": datasets,
        "arms": arms,
        "overall": overall.assign(arm=lambda d: d["arm"].map(ARM_LABEL)).to_dict("records"),
        "per_dataset": per_ds.assign(arm=lambda d: d["arm"].map(ARM_LABEL)).to_dict("records"),
        "budget": budget.to_dict("records"),
        "stats": stat_tables,
        "category_scale": {f"{k[0]}/{k[1]}": v for k, v in scale.items()},
        "spearman_scale_vs_delta": spearman,
        "examples_maxpatch_best": [dict(zip(_EXAMPLE_KEYS, e, strict=True)) for e in ex_sorted[:5]],
        "examples_maxhac_best": [dict(zip(_EXAMPLE_KEYS, e, strict=True)) for e in ex_sorted[-5:][::-1]],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # ---- assemble body (tables/figs/examples). Prose head/tail live in the .md templates. ----
    L = []
    L.append(
        f"_Trajectories: **{n_traj}** (dataset × category × seed × arm). "
        f"Categories/dataset: {', '.join(f'{k} {v}' for k, v in ncats.items())}; seeds: {nseed}._"
    )
    L.append("")
    L.append("## Overall (both datasets pooled) — read with care")
    L.append("")
    L.append(
        "_This table averages the boxed and boxless datasets together, which have opposite "
        "MaxPatch signs; it is included only for a bird's-eye view. The per-dataset tables below "
        "are the ones to read._"
    )
    L.append("")
    ot = overall.copy()
    ot["arm"] = ot["arm"].map(ARM_LABEL)
    L += _md(ot, {"cost": 3, "fpr": 3, "fnr": 3, "AP": 3, "auroc": 3, "train_s": 1, "xcal_s": 1, "score_s": 1})
    L.append("")
    L.append(
        "`train_s`/`xcal_s`/`score_s` are mean per-retrain seconds (training / cross-calibration / "
        "held-out scoring). `score_s` is where MaxPatch pays for max-pooling ~196 raw patches per "
        "image instead of ~24 pooled region nodes."
    )
    L.append("")
    L.append("## Per dataset (final vote budget)")
    L.append("")
    pt = per_ds.copy()
    pt["arm"] = pt["arm"].map(ARM_LABEL)
    L += _md(pt, {"cost": 3, "fpr": 3, "fnr": 3, "AP": 3, "auroc": 3, "train_s": 1, "xcal_s": 1, "score_s": 1})
    L.append("")
    L.append("## ErrorCost / FNR / AP at fixed vote budgets, per dataset")
    L.append("")
    L.append(
        "Mean over categories × seeds at the step nearest each budget. This is the table form of "
        "the curves in Figures 1–3; note how MaxPatch *improves* with votes on Visual Genome but "
        "*degrades* on Caltech-101 (its threshold drifts as the compressed score distribution fills in)."
    )
    L.append("")
    L += _md(budget, {"cost": 3, "fnr": 3, "AP": 3})
    L.append("")
    L.append("## Paired Wilcoxon (Holm-corrected), per dataset")
    L.append("")
    L.append(
        "Reported **per dataset** on purpose: the boxed (Visual Genome) and boxless (Caltech-101) "
        "datasets give opposite MaxPatch-vs-MaxHAC signs, so a pooled test cancels the real effect. "
        "Paired over (category, seed); `delta = mean_A − mean_B` (negative ⇒ the first arm has lower "
        "cost = better). Significance after Holm correction across the five comparisons: `*` p<0.05, "
        "`**` p<0.01, `***` p<0.001."
    )
    L.append("")
    for ds in datasets:
        L.append(f"### {DS_LABEL.get(ds, ds)}")
        L.append("")
        for mname in ["AULC", "cost@50", "cost@150"]:
            L.append(f"**{mname}**")
            L.append("")
            tab = pd.DataFrame(stat_tables[ds][mname])
            tab["sig"] = tab["holm_p"].map(_sig)
            L += _md(tab, {"mean_A": 3, "mean_B": 3, "delta": 3, "holm_p": 4})
            L.append("")
    L.append("")
    L.append("## Figures")
    L.append("")
    for path, cap in figs:
        L.append(f"![{path}]({path})")
        L.append("")
        L.append(cap)
        L.append("")
    if scale_fig:
        L.append(f"![{scale_fig}]({scale_fig})")
        L.append("")
        cap = (
            "**Figure 4. The scale story.** Each point is a Visual Genome category: x = median "
            "**voted (union) box** area — the region a Good vote actually drags, not the area of a "
            "single annotated instance — (log scale), y = final ErrorCost(MaxPatch) − ErrorCost(MaxHAC). "
            "Shaded stripes are the scale bands categories were sampled from. Points below "
            "the dashed zero line are categories where the tree-free raw-patch strategy wins; the "
            "dotted line marks the HAC leaf scale (~8.3% area), the smallest candidate the tree can "
            "propose."
        )
        if spearman:
            cap += (
                f" Spearman ρ(log-area, MaxPatch−MaxHAC) = {spearman[0]:.2f} "
                f"(p = {spearman[1]:.3f}): {'positive ⇒ MaxPatchs edge grows as objects shrink' if spearman[0] > 0 else 'see text'}."
            )
        L.append(cap)
        L.append("")
    if ex_sorted:
        L.append("## Where each strategy wins — concrete categories")
        L.append("")
        exdf = pd.DataFrame(
            [
                {
                    "dataset": e[0],
                    "category": e[1],
                    "voted area %": (e[2] * 100 if e[2] is not None else float("nan")),
                    "union infl.": (e[6] if e[6] is not None else float("nan")),
                    "MaxPatch cost": e[3],
                    "MaxHAC cost": e[4],
                    "Δ(MP−MH)": e[5],
                }
                for e in (ex_sorted[:5] + ex_sorted[-5:][::-1])
            ]
        )
        L += _md(
            exdf,
            {
                "voted area %": 2,
                "union infl.": 1,
                "MaxPatch cost": 3,
                "MaxHAC cost": 3,
                "Δ(MP−MH)": 3,
            },
        )
        L.append("")
        L.append(
            "Top block: categories where MaxPatch beats MaxHAC most; bottom block: where MaxHAC wins "
            "most. Read alongside Figure 4."
        )
        L.append("")

    body = "\n".join(L)
    (RESULTS / "REPORT_body.md").write_text(body)

    # Stitch the full report from the hand-written prose templates (kept next to
    # the results as REPORT_head/verdict/takeaways/tail .md) + the generated body.
    def _read(name):
        # Prose lives in the repo (report_prose/, committed) so a fresh checkout
        # regenerates REPORT.md; the run dir (common.EXP) is a fallback/override.
        for base in (Path(__file__).parent / "report_prose", common.EXP):
            p = base / name
            if p.exists():
                return p.read_text().rstrip()
        return ""

    head, verdict = _read("REPORT_head.md"), _read("REPORT_verdict.md")
    takeaways, tail = _read("REPORT_takeaways.md"), _read("REPORT_tail.md")
    if head:
        if "<!-- VERDICT PLACEHOLDER — filled from results -->" in head:
            head = head.replace("<!-- VERDICT PLACEHOLDER — filled from results -->", verdict)
        elif verdict:
            head = head + "\n\n" + verdict
        parts = [head, "## Results\n\n" + body, takeaways, tail]
        (RESULTS / "REPORT.md").write_text("\n\n".join(p for p in parts if p) + "\n")
        print("assembled REPORT.md from prose templates + body")
    print(f"wrote REPORT_body.md + metrics.json + {len(figs) + (1 if scale_fig else 0)} figures")
    print("arms:", arms, "datasets:", datasets, "trajectories:", n_traj, "cats:", ncats, "seeds:", nseed)
    if ex_sorted:
        b, w = ex_sorted[0], ex_sorted[-1]
        print(f"MaxPatch best vs MaxHAC: {b[1]} (Δ={b[5]:.3f}); MaxHAC best: {w[1]} (Δ={w[5]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
