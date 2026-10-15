#!/usr/bin/env python3
"""#4216: figures and tables for the report, from starved.py's and cost.py's outputs.

    python figures.py --analysis <analysis dir> --out docs/experiments/2026-10-14-starved-small-4216

Writes ``figures/fig_paired.png``, ``fig_harvest.png``, ``fig_classes.png``, ``fig_cost.png``,
``fig_reach.png`` and ``fig_examples.jpg`` (the images come from the COCO zip each media's
``origin_name`` names), the committed tables (``paired.csv``, ``cost.csv``, ``classes.csv``,
``mechanism.csv``, ``share_by_arm.csv``, ``harvest.csv``, ``reach_by_band.csv``) and ``tables.md``,
the report's tables pre-formatted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

INK, INK2, SURFACE, GRIDC = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # the reports' categorical order (passes the CVD validator)
BLUES = ("#0d3a73", "#2a78d6", "#9cc3f0")  # one hue, dark to light: an ordinal split of one total
MUTED = "#a3a29c"
CHECKPOINTS = (5, 10, 25, 50, 75, 100, 125, 150)

#: The reference arms, in the order every figure draws them (label as starved.REFERENCE writes it).
REFS = [
    ("before #4731 (SotA 2026-10-08, 24 seeds)", "Before #4731\n(24 seeds)"),
    ("today (2026-10-13, 5 seeds)", "Today\n(5 seeds)"),
    ("Binary, the same 2 seeds", "Today, Binary,\nseeds 0-1"),
    ("Region path, today (2 seeds)", "Region path\n(seeds 0-1)"),
    ("EmbeddingGemma 2 (5 seeds)", "EmbeddingGemma 2\n(5 seeds)"),
]


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "lines.solid_capstyle": "round",
        }
    )
    return plt


def fig_paired(P: pd.DataFrame, out: Path) -> None:
    """Every contrast at beta 1 on the small band: the paired change in the starved share."""
    plt = _style()
    p = P[(P.band == "small") & P.contrast.str.contains(r"beta 1$")].copy()
    order = {"opening": 0, "acquisition": 1, "learning": 2, "embedding": 3}
    p["k"] = p.kind.map(order)
    p = p.sort_values(["k", "d_starved"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(p) + 1.4))
    y = np.arange(len(p))[::-1]
    for yi, r in zip(y, p.itertuples()):
        m, se = 100 * r.d_starved, 100 * r.se_starved
        moved = abs(m) > 2 * se if se > 0 else False
        ax.plot([m - 2 * se, m + 2 * se], [yi, yi], color=SLOTS[0] if moved else MUTED, lw=1.6)
        ax.plot([m], [yi], "o", ms=7, color=SLOTS[0] if moved else SURFACE, mec=SLOTS[0] if moved else INK2, mew=1.4)
        ax.text(
            1.02,
            yi,
            f"{r.rescued} rescued, {r.newly_starved} newly starved  ({100 * r.starved_ctl:.0f}% → {100 * r.starved_arm:.0f}%)",
            va="center",
            fontsize=8.5,
            color=INK2,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.kind}: {r.contrast.removesuffix(', beta 1')}" for r in p.itertuples()], fontsize=8.5)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-25, 6)
    ax.grid(axis="x", color=GRIDC, lw=0.8)
    ax.set_xlabel(
        "change in the share of small-object runs that find fewer than 3 positives by vote 150 (points, paired, ±2 SE)"
    )
    ax.set_title(
        "Only the Good-phase exit (#4731) and a different embedder move the starved share", loc="left", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out / "fig_paired.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def harvest_table(R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ref, _ in REFS:
        x = R[R.ref == ref]
        for k in CHECKPOINTS:
            rows.append(
                dict(
                    ref=ref,
                    vote=k,
                    runs=len(x),
                    below3=float((x[f"found_{k}"] < 3).mean()),
                    mean_found=float(x[f"found_{k}"].mean()),
                )
            )
    return pd.DataFrame(rows)


def fig_harvest(H: pd.DataFrame, M: pd.DataFrame, out: Path) -> None:
    plt = _style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    lines = [REFS[0], REFS[1], REFS[3], REFS[4]]
    ends = []
    for (ref, short), color in zip(lines, SLOTS):
        h = H[H.ref == ref]
        ax.plot(h.vote, 100 * h.below3, color=color, lw=2, marker="o", ms=4, label=short.replace("\n", " "))
        ends.append((100 * float(h.below3.iloc[-1]), short.split("\n")[0]))
    ends.sort()
    placed: list[float] = []
    for v, lab in ends:  # labels at the line ends, nudged apart so none overlap
        yv = max(v, placed[-1] + 4.5) if placed else v
        placed.append(yv)
        ax.text(153, yv, lab, color=INK2, fontsize=8.5, va="center")
    ax.set_xlim(0, 175)
    ax.set_ylim(0, 100)
    ax.set_xlabel("votes")
    ax.set_ylabel("small-object runs with < 3 positives found (%)")
    ax.grid(axis="y", color=GRIDC, lw=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("The starved share over the session (beta 1)", loc="left", fontsize=11)

    parts = [
        ("still in the Good phase", "0", "no Good found: no exit can fire"),
        ("still in the Good phase", "1-2", "1-2 Goods, still walking the typed query's sort"),
        ("left the Good phase", "1-2", "1-2 Goods, left the Good phase, still < 3"),
    ]
    m = M.assign(found=M.found.astype(str).replace({"1": "1-2", "2": "1-2"}))
    y = np.arange(len(REFS))[::-1]
    for yi, (ref, short) in zip(y, REFS):
        left = 0.0
        tot = m[m.ref == ref]
        n = int(H[(H.ref == ref)].runs.iloc[0])
        for (where, found, lab), color in zip(parts, BLUES):
            v = 100 * tot[(tot["where"] == where) & (tot.found == found)].runs.sum() / n
            bx.barh(
                yi,
                v,
                left=left,
                color=color,
                height=0.62,
                edgecolor=SURFACE,
                linewidth=2,
                label=lab if yi == y[0] else None,
            )
            left += v
        bx.text(left + 0.8, yi, f"{left:.0f}%", va="center", fontsize=9, color=INK)
    bx.set_yticks(y)
    bx.set_yticklabels([s for _, s in REFS], fontsize=8.5)
    bx.set_xlim(0, 36)
    bx.set_xlabel("small-object runs starved at vote 150 (%)")
    bx.grid(axis="x", color=GRIDC, lw=0.8)
    bx.legend(frameon=False, fontsize=8, loc="lower right")
    bx.set_title("Where the starved runs are at vote 150", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_harvest.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_classes(C: pd.DataFrame, out: Path) -> None:
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE, "#9cc3f0", "#2a78d6", "#0d3a73"])
    w = C.assign(share=C.starved / C.runs).pivot_table(index="cls", columns="ref", values="share")
    k = C.pivot_table(index="cls", columns="ref", values="starved")
    n = C.pivot_table(index="cls", columns="ref", values="runs")
    refs = [r for r, _ in REFS]
    keep = w[refs].fillna(0).max(axis=1) > 0
    order = w[keep].sort_values([refs[0], refs[1]], ascending=False).index
    w, k, n = w.loc[order, refs], k.loc[order, refs], n.loc[order, refs]
    fig, ax = plt.subplots(figsize=(8.6, 0.3 * len(order) + 1.6))
    ax.imshow(w.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(order)):
        for j in range(len(refs)):
            kk, nn, s = k.iat[i, j], n.iat[i, j], w.iat[i, j]
            if np.isfinite(nn):
                ax.text(
                    j,
                    i,
                    f"{int(kk)}/{int(nn)}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=SURFACE if s > 0.55 else INK,
                )
    ax.set_xticks(range(len(refs)))
    ax.set_xticklabels([s for _, s in REFS], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c}@small" for c in order], fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(
        "Starved runs per class (small band, beta 1): runs with < 3 positives by vote 150 / runs",
        loc="left",
        fontsize=10.5,
        pad=34,
    )
    fig.tight_layout()
    fig.savefig(out / "fig_classes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_cost(T: pd.DataFrame, out: Path) -> None:
    plt = _style()
    panels = [
        ("Good phase ends on a dry run (#4731)", "#4731's dry-run exit vs before"),
        ("Region path, with #4731 (same commit, 2 seeds)", "Region path vs Binary (with #4731, 2 seeds)"),
        ("EmbeddingGemma 2 in place of SigLIP (#4837)", "EmbeddingGemma 2 vs SigLIP"),
    ]
    rows = [
        ("large", "every run", "large"),
        ("medium", "every run", "medium"),
        ("small", "every run", "small"),
        ("small", "rescued", "small, rescued runs"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True, sharex=True)
    for ax, (contrast, title) in zip(axes, panels):
        t = T[T.contrast == contrast]
        for yi, (band, group, _) in enumerate(rows):
            for bi, (beta, color) in enumerate(zip((0.25, 1.0, 4.0), SLOTS)):
                r = t[(t.band == band) & (t.group == group) & (t.beta == beta)]
                if r.empty:
                    continue
                m, se = float(r.d_session.iloc[0]), float(r.se.iloc[0])
                yy = yi - (bi - 1) * 0.22
                if np.isfinite(se):
                    ax.plot([m - 2 * se, m + 2 * se], [yy, yy], color=color, lw=1.4)
                lab = {0.25: "beta 1/4", 1.0: "beta 1", 4.0: "beta 4"}[beta]
                ax.plot(
                    [m],
                    [yy],
                    "o",
                    color=color,
                    ms=5.5,
                    mec=SURFACE,
                    mew=1.2,
                    label=lab if (yi == 0 and ax is axes[0]) else None,
                )
        ax.axvline(0, color=INK2, lw=0.8)
        ax.grid(axis="x", color=GRIDC, lw=0.8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("change in the objective, session mean, votes 1-150")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[2] for r in rows])
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_cost.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


EMB_LABEL = {
    "siglip": "SigLIP (default)",
    "siglip2_l": "SigLIP 2 L (premium)",
    "embeddinggemma2": "EmbeddingGemma 2",
    "clip_l": "CLIP L",
    "clip": "CLIP",
}


def reach_table(X: pd.DataFrame) -> pd.DataFrame:
    """Per embedder and band: the share of runs whose typed-query sort holds < 3 positives in its top 150."""
    X = X.assign(
        band=X.category.str.partition("@")[2],
        cls=X.category.str.partition("@")[0],
        short=(X.pos_top150 < 3).astype(int),
    )
    rows = []
    for (emb, band), g in X.groupby(["embedder", "band"]):
        pc = g.groupby("cls").short.mean()
        rows.append(
            dict(
                embedder=emb,
                band=band,
                runs=len(g),
                seeds=g.seed.nunique(),
                short=g.short.mean(),
                se=pc.std(ddof=1) / np.sqrt(len(pc)),
                median_rank3=float(g.rank3[g.rank3 > 0].median()),
            )
        )
    return pd.DataFrame(rows)


def reach_paired(X: pd.DataFrame, base: str = "siglip") -> pd.DataFrame:
    """Each embedder against SigLIP on the same (category, seed): the paired change in the short share."""
    X = X.assign(
        band=X.category.str.partition("@")[2],
        cls=X.category.str.partition("@")[0],
        short=(X.pos_top150 < 3).astype(int),
    )
    b = X[X.embedder == base]
    rows = []
    for emb in X.embedder.unique():
        if emb == base:
            continue
        j = X[X.embedder == emb].merge(b, on=["category", "seed", "band", "cls"], suffixes=("", "_b"))
        for band, g in j.groupby("band"):
            d = g.short - g.short_b
            pc = d.groupby(g.cls).mean()
            rows.append(
                dict(
                    embedder=emb,
                    band=band,
                    pairs=len(g),
                    base_short=g.short_b.mean(),
                    short=g.short.mean(),
                    d=d.mean(),
                    se=pc.std(ddof=1) / np.sqrt(len(pc)),
                    gained=int((d < 0).sum()),
                    lost=int((d > 0).sum()),
                )
            )
    return pd.DataFrame(rows)


def fig_reach(RT: pd.DataFrame, P: pd.DataFrame, out: Path) -> None:
    plt = _style()
    order = ["clip", "clip_l", "siglip", "siglip2_l", "embeddinggemma2"]
    sm = RT[RT.band == "small"].set_index("embedder").loc[order]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(order))[::-1]
    for yi, (emb, r) in zip(y, sm.iterrows()):
        ax.barh(yi, 100 * r.short, color=BLUES[2], height=0.6)
        ax.plot([100 * (r.short - 2 * r.se), 100 * (r.short + 2 * r.se)], [yi, yi], color=INK2, lw=1)
        ax.text(
            100 * (r.short + 2 * r.se) + 1,
            yi,
            f"{100 * r.short:.0f}%  (3rd positive at rank {r.median_rank3:.0f}, median)",
            va="center",
            fontsize=8.5,
            color=INK2,
        )
    # The sessions that exist: SigLIP before and with #4731's exit, EmbeddingGemma 2 with it.
    q = P[
        (P.band == "small")
        & P.contrast.str.startswith("Good phase ends on a dry run (#4731), beta 1")
        & ~P.contrast.str.contains("first run")
    ]
    eg = P[(P.band == "small") & (P.contrast == "EmbeddingGemma 2 in place of SigLIP (#4837), beta 1")]
    obs = [
        ("siglip", float(q.starved_ctl.iloc[0]), "session, before #4731"),
        ("siglip", float(q.starved_arm.iloc[0]), "session, with #4731's exit"),
        ("embeddinggemma2", float(eg.starved_arm.iloc[0]), "session, with #4731's exit"),
    ]
    for emb, v, lab in obs:
        yi = y[order.index(emb)]
        mk = "o" if "before" in lab else "D"
        ax.plot([100 * v], [yi], mk, color=SLOTS[1], ms=7, mec=SURFACE, mew=1.2, label=lab if emb == "siglip" else None)
    ax.set_yticks(y)
    ax.set_yticklabels([EMB_LABEL[e] for e in order])
    ax.set_xlim(0, 75)
    ax.set_xlabel("small-object runs whose typed-query sort holds < 3 positives in its top 150 (%, ±2 SE)")
    ax.grid(axis="x", color=GRIDC, lw=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title(
        "The text tower decides the starved share: the sort's reach on small objects, per embedder",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "fig_reach.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_examples(E: dict, out: Path, n: int = 4) -> None:
    """Seed 0: the first images the Good phase shows (all Bad) and the best-ranked positives, boxed."""
    import zipfile

    from PIL import Image

    plt = _style()
    zips: dict[str, zipfile.ZipFile] = {}

    def load(origin: str) -> Image.Image:
        zpath, member = origin.split("::", 1)
        if zpath not in zips:
            zips[zpath] = zipfile.ZipFile(zpath)
        with zips[zpath].open(member) as f:
            return Image.open(f).convert("RGB")

    cats = [c for c in ("bowl@small", "chair@small", "knife@small") if c in E]
    fig, axes = plt.subplots(2 * len(cats), n, figsize=(2.3 * n, 2.0 * 2 * len(cats) + 0.3))
    for ci, cat in enumerate(cats):
        e = E[cat]
        for j in range(n):
            ax = axes[2 * ci, j]
            t = e["top"][j]
            ax.imshow(load(t["origin"]))
            lab = ", ".join(c.split("@")[0] for c in t["categories"][:2]) or "no class"
            ax.set_title(f"rank {t['rank']}: {lab}", fontsize=7.5, color=INK2)
            ax.axis("off")
            ax = axes[2 * ci + 1, j]
            p = e["positives"][j]
            im = load(p["origin"])
            W, H = im.size
            x0, y0, x1, y1 = p["boxes"][0]
            bw, bh = max((x1 - x0) * W, 24), max((y1 - y0) * H, 24)
            cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
            half = max(bw, bh) * 2.5
            L, T = max(0, cx - half), max(0, cy - half)
            R_, B = min(W, cx + half), min(H, cy + half)
            ax.imshow(im.crop((int(L), int(T), int(R_), int(B))))
            ax.add_patch(
                plt.Rectangle((x0 * W - L, y0 * H - T), (x1 - x0) * W, (y1 - y0) * H, fill=False, ec=SLOTS[1], lw=1.6)
            )
            ax.set_title(
                f"positive at rank {p['rank']} (crop, {int((x1 - x0) * W)}×{int((y1 - y0) * H)} px box)",
                fontsize=7.5,
                color=INK2,
            )
            ax.axis("off")
        axes[2 * ci, 0].text(
            -0.08,
            0.5,
            f'"{e["query"]}"\ntop of the sort',
            transform=axes[2 * ci, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
        )
        axes[2 * ci + 1, 0].text(
            -0.08,
            0.5,
            f"{cat}\npositives",
            transform=axes[2 * ci + 1, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out / "fig_examples.jpg", dpi=110, bbox_inches="tight", pil_kwargs={"quality": 82})
    plt.close(fig)


def fmt(m: float, se: float, pct: bool = False, digits: int = 3) -> str:
    s = 100 if pct else 1
    if not np.isfinite(se) or se == 0:
        return f"{s * m:+.{1 if pct else digits}f}" + (" (identical)" if m == 0 and se == 0 else "")
    bold = abs(m) > 2 * se
    txt = f"{s * m:+.{1 if pct else digits}f} ± {s * se:.{1 if pct else digits}f}"
    return f"**{txt}**" if bold else txt


def tables_md(P: pd.DataFrame, T: pd.DataFrame, H: pd.DataFrame, out: Path) -> None:
    lines = [
        "## Paired starved share, small band (points)",
        "",
        "| contrast | pairs | control | arm | change | rescued | newly starved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in P[P.band == "small"].itertuples():
        lines.append(
            f"| {r.contrast} | {r.pairs} | {100 * r.starved_ctl:.0f}% | {100 * r.starved_arm:.0f}% | {fmt(r.d_starved, r.se_starved, pct=True)} | {r.rescued} | {r.newly_starved} |"
        )
    lines += ["", "## Other bands (points)", "", "| contrast | band | control | change |", "|---|---|---:|---:|"]
    for r in P[P.band != "small"].itertuples():
        if r.starved_ctl > 0 or r.d_starved != 0:
            lines.append(
                f"| {r.contrast} | {r.band} | {100 * r.starved_ctl:.1f}% | {fmt(r.d_starved, r.se_starved, pct=True)} |"
            )
    lines += [
        "",
        "## Objective, session mean over votes 1-150",
        "",
        "| contrast | beta | band | runs | control | change | change at vote 150 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in T.itertuples():
        g = "" if r.group == "every run" else f" ({r.group})"
        lines.append(
            f"| {r.contrast} | {r.beta:g} | {r.band}{g} | {r.runs} | {r.ctl_session:.2f} | {fmt(r.d_session, r.se)} | {r.d_at150:+.3f} |"
        )
    lines += [
        "",
        "## Starved share by vote (small band)",
        "",
        "| arm | " + " | ".join(str(k) for k in CHECKPOINTS) + " |",
        "|---|" + "---:|" * len(CHECKPOINTS),
    ]
    for ref, _ in REFS:
        h = H[H.ref == ref]
        lines.append(f"| {ref} | " + " | ".join(f"{100 * v:.0f}%" for v in h.below3) + " |")
    (out / "tables.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    figs = a.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(a.analysis / "paired.csv")
    T = pd.read_csv(a.analysis / "cost.csv")
    C = pd.read_csv(a.analysis / "classes.csv")
    M = pd.read_csv(a.analysis / "mechanism.csv")
    S = pd.read_csv(a.analysis / "share_by_arm.csv")
    import starved

    R = starved.reference(pd.read_csv(a.analysis / "runs.csv"))
    R = R[R.band == "small"]
    H = harvest_table(R)
    fig_paired(P, figs)
    fig_harvest(H, M, figs)
    fig_classes(C, figs)
    fig_cost(T, figs)
    for name, df in (("paired", P), ("cost", T), ("classes", C), ("mechanism", M), ("share_by_arm", S), ("harvest", H)):
        df.round(4).to_csv(a.out / f"{name}.csv", index=False)
    X = pd.read_csv(a.analysis / "reach.csv")
    RT = reach_table(X)
    fig_reach(RT, P, figs)
    fig_examples(json.loads((a.analysis / "examples.json").read_text()), figs)
    RT.round(4).to_csv(a.out / "reach_by_band.csv", index=False)
    RP = reach_paired(X)
    RP.round(4).to_csv(a.out / "reach_paired.csv", index=False)
    tables_md(P, T, H, a.out)
    with (a.out / "tables.md").open("a") as f:
        f.write(
            "\n## Typed-query reach (share of runs with < 3 positives in the sort's top 150)\n\n| embedder | band | runs | short | median rank of 3rd |\n|---|---|---:|---:|---:|\n"
        )
        for r in RT.itertuples():
            f.write(
                f"| {r.embedder} | {r.band} | {r.runs} | {100 * r.short:.0f}% \u00b1 {100 * r.se:.0f} | {r.median_rank3:.0f} |\n"
            )
        f.write(
            "\n## Typed-query reach, paired against SigLIP (seeds 0-4)\n\n| embedder | band | pairs | SigLIP | embedder | change | cells reached / lost |\n|---|---|---:|---:|---:|---:|---:|\n"
        )
        for r in RP.itertuples():
            f.write(
                f"| {r.embedder} | {r.band} | {r.pairs} | {100 * r.base_short:.0f}% | {100 * r.short:.0f}% | {fmt(r.d, r.se, pct=True)} | {r.gained} / {r.lost} |\n"
            )
    print(f"wrote {figs} and the tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
