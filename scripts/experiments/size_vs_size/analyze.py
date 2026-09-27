#!/usr/bin/env python3
"""Size vs size (#4160): the train-size x test-size table, pooled and per class.

    python analyze.py --exp <exp dir> --out <dir> [--baseline <dir>/text_baseline.csv] [--ts 30,150]

Reads the array's cells (``launch.sh``), one arm per TRAINING size:

    S, M, L        ``<class>@small|medium|large``
    SML=           ``<class>@mix-equal``    (the class's bands at equal shares)
    SMLn           ``<class>@mix-natural``  (at the shares the corpus holds)

and scores each at every TEST size. S, M and L are the per-band cohorts, the
same images under every arm. Test SML is not a cohort of its own: every arm has
one threshold and one negative pool, so the miss rate over any mix of the three
cohorts is that mix's weighted average of the three, and so is the AUROC (an
AUROC is a mean over positives of the share of negatives they outrank). ``SML=``
weights the bands equally and ``SMLn`` at the natural shares, matching the two
training mixes.

Per test size T the metrics are

    recall_T                      the share of T's cohort over the arm's cut
    fpr                           the arm's one false-positive rate
    cost_T = wf*fpr + wn*fnr_T    the shipped operating cost, with T as the positives
    auroc_T                       T's cohort ranked against the held-out negatives
    f1_T                          at the test half's own negative count

**Contrasts are paired and clustered by class.** Two arms are compared on the
same (class, seed) and test cohort; the per-class mean difference is the unit,
so the standard error is over classes (46 for anything touching S: apple,
banana and orange have no small band). A difference under twice its SE is
written "not resolvable".

Writes ``cells_final.csv`` (one row per class x seed x train x test at each
reported click), ``matrix.csv`` (pooled), ``contrasts.csv``, ``per_class.csv``,
``summary.md``, ``figures/*.png``, ``viewer.html``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CALIB = HERE.parent / "calibration"
sys.path.insert(0, str(CALIB))

TRAIN = {"small": "S", "medium": "M", "large": "L", "mix-equal": "SML=", "mix-natural": "SMLn"}
TRAIN_ORDER = ["S", "M", "L", "SML=", "SMLn"]
TEST_ORDER = ["S", "M", "L", "SML=", "SMLn"]
BANDS = {"S": "small", "M": "medium", "L": "large"}
METRICS = ("cost", "auroc", "recall", "f1")
LOWER_IS_BETTER = {"cost": True, "auroc": False, "recall": False, "f1": False, "fpr": True}


def _weights(shares: dict[str, float], present: list[str], mode: str) -> dict[str, float]:
    if mode == "SML=":
        w = {b: 1.0 for b in present}
    else:
        w = {b: float(shares.get(b, 0.0)) for b in present}
    tot = sum(w.values())
    return {b: v / tot for b, v in w.items()} if tot > 0 else {}


def expand(df: pd.DataFrame, shares: dict[str, dict[str, float]], wf: float, wn: float) -> pd.DataFrame:
    """One row per (run step, test size), carrying that test size's metrics."""
    cls_band = df["category"].str.partition("@")
    df = df.assign(cls=cls_band[0], train=cls_band[2].map(TRAIN))
    if df["train"].isna().any():
        bad = sorted(df.loc[df["train"].isna(), "category"].unique())[:5]
        raise SystemExit(f"cells with no train label: {bad}")
    n_neg = df["n_test_neg"].astype(float)
    parts = []
    for test in TEST_ORDER:
        if test in BANDS:
            b = BANDS[test]
            fnr = df[f"fnr_{b}"].astype(float)
            auroc = df[f"auroc_{b}"].astype(float)
            n_pos = df[f"n_test_pos_{b}"].astype(float)
        else:
            fnr = pd.Series(0.0, index=df.index)
            auroc = pd.Series(0.0, index=df.index)
            n_pos = pd.Series(0.0, index=df.index)
            wsum = pd.Series(0.0, index=df.index)
            for cls, idx in df.groupby("cls").groups.items():
                sub = df.loc[idx]
                present = [b for b in BANDS.values() if sub[f"n_test_pos_{b}"].fillna(0).gt(0).all()]
                for b, w in _weights(shares.get(cls, {}), present, test).items():
                    fnr.loc[idx] += w * sub[f"fnr_{b}"].astype(float)
                    auroc.loc[idx] += w * sub[f"auroc_{b}"].astype(float)
                    n_pos.loc[idx] += sub[f"n_test_pos_{b}"].astype(float)
                    wsum.loc[idx] += w
            bad = ~np.isclose(wsum, 1.0)
            fnr[bad] = np.nan
            auroc[bad] = np.nan
        recall = 1.0 - fnr
        fpr = df["fpr"].astype(float)
        tp = recall * n_pos
        fp = fpr * n_neg
        fn = n_pos - tp
        f1 = (2 * tp) / (2 * tp + fp + fn)
        parts.append(
            pd.DataFrame(
                {
                    "dataset": df["dataset"],
                    "embedder": df["embedder"],
                    "cls": df["cls"],
                    "seed": df["seed"],
                    "t": df["t"],
                    "train": df["train"],
                    "test": test,
                    "fpr": fpr,
                    "recall": recall,
                    "fnr": fnr,
                    "cost": wf * fpr + wn * fnr,
                    "auroc": auroc,
                    "f1": f1,
                    "n_pos": n_pos,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


RUN_KEYS = ["embedder", "cls", "seed", "train", "test"]


def at_click(long: pd.DataFrame, t: int, expected: pd.DataFrame, base: pd.DataFrame | None) -> pd.DataFrame:
    """Each run's row at click *t*, or its last row if it ended earlier.

    A run with no row by click *t* has not yet had a Good and a Bad vote, so the
    app had no detector on screen and the user had only the text sort. It is
    scored at the text sort (``starved = 1``) rather than dropped: starvation
    differs by training size, and dropping it would score each arm on the runs
    it happened to get going.
    """
    sub = long[long["t"] <= t].sort_values("t")
    got = sub.groupby(RUN_KEYS, as_index=False).last()
    got["starved"] = 0
    missing = expected.merge(got[RUN_KEYS], on=RUN_KEYS, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    if missing.empty:
        return got
    if base is None:
        raise SystemExit(f"{len(missing)} runs have no row by click {t} and there is no --baseline to score them at")
    b = base.rename(columns={f"text_{m}": m for m in ("fpr", "fnr", "recall", "cost", "auroc", "f1")})
    fill = missing.merge(
        b[["embedder", "cls", "seed", "test", "fpr", "fnr", "recall", "cost", "auroc", "f1"]],
        on=["embedder", "cls", "seed", "test"],
        how="left",
    )
    fill["t"] = 0
    fill["starved"] = 1
    return pd.concat([got, fill], ignore_index=True)


def pooled(final: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean over classes of the per-class seed mean, with its SE over classes."""
    per_cls = final.groupby(["embedder", "train", "test", "cls"])[metric].mean().reset_index()
    g = per_cls.groupby(["embedder", "train", "test"])[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"])
    out["metric"] = metric
    return out.rename(columns={"count": "n_classes"}).drop(columns="std")


def paired(final: pd.DataFrame, metric: str, test: str, a: str, b: str) -> dict:
    """Arm *a* minus arm *b* on test size *test*: paired on (class, seed), clustered by class."""
    sub = final[final["test"] == test]
    wide = sub.pivot_table(index=["embedder", "cls", "seed"], columns="train", values=metric)
    if a not in wide or b not in wide:
        return {}
    d = (wide[a] - wide[b]).dropna()
    per_cls = d.groupby(level="cls").mean()
    n = len(per_cls)
    mean = float(per_cls.mean()) if n else float("nan")
    se = float(per_cls.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    better = (mean < 0) if LOWER_IS_BETTER[metric] else (mean > 0)
    return {
        "metric": metric,
        "test": test,
        "a": a,
        "b": b,
        "diff": mean,
        "se": se,
        "n_classes": n,
        "n_pairs": int(len(d)),
        "resolvable": bool(n > 1 and abs(mean) >= 2 * se),
        "a_better": bool(better) if n > 1 and abs(mean) >= 2 * se else None,
        "classes_a_better": int(((per_cls < 0) if LOWER_IS_BETTER[metric] else (per_cls > 0)).sum()),
    }


def contrasts(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for test in TEST_ORDER:
            # Every training size against the one that matches the test size
            # (for a pure test size) or against the matching mix.
            ref = test
            for a in TRAIN_ORDER:
                if a != ref:
                    r = paired(final, metric, test, a, ref)
                    if r:
                        rows.append({**r, "kind": "vs_matched"})
        # The asymmetry: train small test large against train large test small
        # is not paired (different cohorts), so it is read as two gaps from the
        # diagonal, which are.
    return pd.DataFrame(rows)


def fmt(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    mag = int(np.floor(np.log10(abs(x))))
    return f"{x:.{max(0, digits - 1 - mag)}f}"


def md_matrix(pool: pd.DataFrame, metric: str, embedder: str) -> str:
    sub = pool[(pool["metric"] == metric) & (pool["embedder"] == embedder)]
    lines = ["| train \\ test | " + " | ".join(TEST_ORDER) + " |", "|---|" + "---|" * len(TEST_ORDER)]
    for tr in TRAIN_ORDER:
        cells = []
        for te in TEST_ORDER:
            r = sub[(sub["train"] == tr) & (sub["test"] == te)]
            if r.empty:
                cells.append("")
                continue
            v = float(r["mean"].iloc[0])
            s = f"{v:.2f}"
            cells.append(f"**{s}**" if tr == te else s)
        lines.append(f"| **{tr}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def heatmaps(pool: pd.DataFrame, figdir: Path, t: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for emb in sorted(pool["embedder"].unique()):
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.2))
        for ax, metric in zip(axes, METRICS):
            sub = pool[(pool["metric"] == metric) & (pool["embedder"] == emb)]
            m = np.full((len(TRAIN_ORDER), len(TEST_ORDER)), np.nan)
            for i, tr in enumerate(TRAIN_ORDER):
                for j, te in enumerate(TEST_ORDER):
                    r = sub[(sub["train"] == tr) & (sub["test"] == te)]
                    if not r.empty:
                        m[i, j] = float(r["mean"].iloc[0])
            cmap = "viridis_r" if LOWER_IS_BETTER[metric] else "viridis"
            im = ax.imshow(m, cmap=cmap, aspect="equal")
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if np.isfinite(m[i, j]):
                        # Best train arm per test column is boxed.
                        col = m[:, j]
                        best = np.nanargmin(col) if LOWER_IS_BETTER[metric] else np.nanargmax(col)
                        ax.text(
                            j,
                            i,
                            f"{m[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="white" if im.norm(m[i, j]) < 0.5 else "black",
                            fontweight="bold" if i == best else "normal",
                        )
            ax.set_xticks(range(len(TEST_ORDER)), TEST_ORDER)
            ax.set_yticks(range(len(TRAIN_ORDER)), TRAIN_ORDER)
            ax.set_xlabel("test size")
            ax.set_ylabel("train size")
            arrow = "lower is better" if LOWER_IS_BETTER[metric] else "higher is better"
            ax.set_title(f"{metric} ({arrow})")
            ax.axvline(2.5, color="white", lw=1.5)
            ax.axhline(2.5, color="white", lw=1.5)
        fig.suptitle(
            f"Train size x test size at click {t}, {emb}: mean over classes; bold = best train size per column"
        )
        fig.tight_layout()
        name = f"matrix_t{t}_{emb.replace('+', '_')}.png"
        fig.savefig(figdir / name, dpi=130)
        plt.close(fig)
        written.append(name)
    return written


def per_class_figure(final: pd.DataFrame, figdir: Path, t: int, metric: str = "cost") -> list[str]:
    """Per class, each training size's cost on each pure test size, relative to the matched one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for emb in sorted(final["embedder"].unique()):
        sub = final[(final["embedder"] == emb)]
        per = sub.groupby(["cls", "train", "test"])[metric].mean().unstack("train")
        classes = sorted(sub["cls"].unique())
        fig, axes = plt.subplots(1, 3, figsize=(15, 0.28 * len(classes) + 1.8), sharey=True)
        colors = {"S": "#1f77b4", "M": "#ff7f0e", "L": "#2ca02c", "SML=": "#9467bd", "SMLn": "#d62728"}
        for ax, test in zip(axes, ["S", "M", "L"]):
            for k, cls in enumerate(classes):
                if (cls, test) not in per.index:
                    continue
                row = per.loc[(cls, test)]
                ref = row.get(test)
                if ref is None or not np.isfinite(ref):
                    continue
                for tr in TRAIN_ORDER:
                    if tr == test or tr not in row or not np.isfinite(row[tr]):
                        continue
                    ax.plot(row[tr] - ref, k, "o", color=colors[tr], ms=4, label=tr if k == 0 else None)
            ax.axvline(0, color="grey", lw=1)
            ax.set_title(f"test {test}: {metric} minus train-{test}'s")
            ax.set_xlabel(f"Δ {metric} (right = worse than training at {test})")
        axes[0].set_yticks(range(len(classes)), classes, fontsize=7)
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"train {k}") for k, c in colors.items()]
        fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
        fig.suptitle(f"Per class at click {t}, {emb}: what training at another size costs on each test size", y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        name = f"per_class_{metric}_t{t}_{emb.replace('+', '_')}.png"
        fig.savefig(figdir / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(name)
    return written


def viewer_frame(long: pd.DataFrame) -> pd.DataFrame:
    """The long frame in the shape `curves` and `viewer` read.

    The TEST size rides in the ``dataset`` slot and the TRAINING size in
    ``arm``, so the viewer's "dataset: each" draws one panel per test size with
    one line per training size, and "category" is the class.
    """
    return pd.DataFrame(
        {
            "dataset": "test " + long["test"],
            "embedder": long["embedder"],
            "category": long["cls"],
            "seed": long["seed"],
            "t": long["t"],
            "arm": "train " + long["train"],
            "cost": long["cost"],
            "fpr": long["fpr"],
            "fnr": long["fnr"],
            "recall": long["recall"],
            "f1": long["f1"],
            "auroc": long["auroc"],
        }
    )


def viewer_baseline(base: pd.DataFrame) -> pd.DataFrame:
    return base.rename(columns={"cls": "category"}).assign(dataset="test " + base["test"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=None, help="baseline.py's per-test-size text-sort CSV")
    ap.add_argument("--ts", default="30,150", help="clicks at which to read the table")
    ap.add_argument("--seeds", type=int, default=0, help="analyze seeds < N only (0 = all complete seeds)")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    import os

    shape = json.loads((args.exp / "results" / "grid_shape.json").read_text())
    # The grid's own enumeration, so completeness is read off the cells the
    # array was asked for rather than off the rows that came back.
    os.environ.update(
        CALIB_DATASETS=",".join(shape["datasets"]),
        CALIB_COCO_BETTER_EMBEDDERS=",".join(shape["embedders"]),
        CALIB_N_SEEDS=str(shape["n_seeds"]),
        CALIB_CELL_ORDER=shape["cell_order"],
        CALIB_CATEGORY_MODE="all",
        CALIB_TRAIN_MIXES="equal,natural",
        CALIB_MIX_SHARES=str(args.exp / "mix_shares.json"),
    )
    import _cells_io
    import experiment_config as cfg
    import run_cells
    from vtscore.training.thresholds import inclusion_cost_weights

    args.out.mkdir(parents=True, exist_ok=True)
    figdir = args.out / "figures"
    figdir.mkdir(exist_ok=True)
    wf, wn = inclusion_cost_weights(cfg.INCLUSION)
    shares = json.loads((args.exp / "mix_shares.json").read_text())

    df, prov = _cells_io.load_arm(args.exp / "results")
    print(
        f"read {prov['n_read']}/{prov['n_files']} cell files, {len(df):,} base rows; no-positive {len(prov['no_positive_found'])}"
    )
    keep = [
        c
        for c in df.columns
        if c in {"dataset", "embedder", "category", "seed", "t", "fpr", "n_test_neg"}
        or c.startswith(("fnr_", "auroc_", "n_test_pos_"))
    ]
    df = df[keep]

    # Complete seeds only: a seed missing some of its cells would pool a subset.
    # A cell is done when its file exists, rows or not: a run that never found
    # a positive writes a header and is a result, not a gap.
    prepare = json.loads((args.exp / "results" / "prepare_info.json").read_text())
    cells = cfg.array_cells(run_cells._categories_by_dataset(prepare))
    done = {
        int(p.stem.split("_")[1])
        for p in (args.exp / "results" / "cells").glob("task_[0-9]*.csv")
        if "__" not in p.stem
    }
    grid = pd.DataFrame(cells)
    grid["done"] = [i in done for i in range(len(cells))]
    by_seed = grid.groupby("seed")["done"].all()
    complete = sorted(int(sd) for sd, ok in by_seed.items() if ok)
    if args.seeds:
        complete = [sd for sd in complete if sd < args.seeds]
    print(f"seeds complete: {len(complete)} of {len(by_seed)} ({len(cells) // len(by_seed)} cells each)")
    if not complete:
        raise SystemExit("no seed has all of its cells yet; nothing paired to analyse")
    df = df[df["seed"].isin(complete)]
    grid = grid[grid["seed"].isin(complete)]
    cb = grid["category"].str.partition("@")
    expected = pd.DataFrame(
        {"embedder": grid["embedder"], "cls": cb[0], "seed": grid["seed"], "train": cb[2].map(TRAIN)}
    )
    expected = expected.merge(pd.DataFrame({"test": TEST_ORDER}), how="cross")

    long = expand(df, shares, wf, wn)
    ts = [int(x) for x in args.ts.split(",") if x]
    base = pd.read_csv(args.baseline) if args.baseline and args.baseline.exists() else None
    if base is not None:
        base = base[base["seed"].isin(complete)]
        # The mixed test sizes are a cohort-weighted mean of the bands, so a
        # class without a small band has SML= over two bands, in the baseline
        # as in the arms.

    lines = [
        "# Size vs size (#4160): analysis summary",
        "",
        f"{len(complete)} seeds, {df['category'].nunique()} cells, embedders {sorted(df['embedder'].unique())}; "
        f"cost weights fpr x{wf:g}, fnr x{wn:g}.",
        "",
    ]
    all_final, all_pool, all_con = [], [], []
    for t in ts:
        final = at_click(long, t, expected, base)
        final["click"] = t
        starved = final[final["test"] == "SMLn"].groupby("train")["starved"].mean()
        lines += [
            f"Runs with no detector yet at click {t} (scored at the text sort): "
            + ", ".join(f"{tr} {starved.get(tr, 0):.0%}" for tr in TRAIN_ORDER),
            "",
        ]
        all_final.append(final)
        pools = pd.concat([pooled(final, m) for m in (*METRICS, "fpr")], ignore_index=True)
        pools["click"] = t
        all_pool.append(pools)
        con = contrasts(final)
        con["click"] = t
        all_con.append(con)
        heatmaps(pools, figdir, t)
        per_class_figure(final, figdir, t)
        for emb in sorted(final["embedder"].unique()):
            lines += [f"## Click {t}, {emb}", ""]
            for metric in METRICS:
                lines += [f"### {metric}", "", md_matrix(pools, metric, emb), ""]
            if base is not None:
                b = base.groupby(["test"])[[f"text_{m}" for m in METRICS]].mean()
                lines += [
                    "Text sort alone (click 0): "
                    + "; ".join(
                        f"{te}: cost {b.loc[te, 'text_cost']:.2f}, auroc {b.loc[te, 'text_auroc']:.2f}"
                        for te in TEST_ORDER
                        if te in b.index
                    ),
                    "",
                ]
            c = con[(con["metric"].isin(["cost", "auroc"]))] if not con.empty else con
            lines += [
                "| metric | test | train | minus train | diff | SE | classes | a better in |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for _, r in c.iterrows():
                verdict = "" if r["resolvable"] else " (not resolvable)"
                lines.append(
                    f"| {r['metric']} | {r['test']} | {r['a']} | {r['b']} | {r['diff']:+.3f}{verdict} | {r['se']:.3f} "
                    f"| {r['n_classes']} | {r['classes_a_better']}/{r['n_classes']} |"
                )
            lines.append("")
    final_all = pd.concat(all_final, ignore_index=True)
    final_all.to_csv(args.out / "cells_final.csv", index=False)
    pd.concat(all_pool, ignore_index=True).to_csv(args.out / "matrix.csv", index=False)
    pd.concat(all_con, ignore_index=True).to_csv(args.out / "contrasts.csv", index=False)
    per_cls = final_all.groupby(["click", "embedder", "cls", "train", "test"])[[*METRICS, "fpr"]].mean().reset_index()
    per_cls.to_csv(args.out / "per_class.csv", index=False)
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}/summary.md")

    if not args.no_viewer:
        import curves
        import viewer

        vf = viewer_frame(long)
        vb = viewer_baseline(base) if base is not None else None
        arms = [f"train {a}" for a in TRAIN_ORDER]
        for test in ("S", "M", "L", "SMLn"):
            sub = vf[vf["dataset"] == f"test {test}"]
            curves.quality_vs_clicks(
                sub,
                figdir / f"curves_test_{test}",
                arms=arms,
                metric="cost",
                baseline=vb[vb["dataset"] == f"test {test}"] if vb is not None else None,
            )
        viewer.build_viewer(
            vf,
            args.out / "viewer.html",
            arms=arms,
            baseline=vb,
            skyline=None,
            build={"exp": str(args.exp), "seeds": complete},
            title="Size vs size (#4160)",
            subtitle="coco_better, SigLIP binary. Panel ('dataset') = TEST size; line (arm) = TRAINING size.",
        )
        print(f"wrote {args.out}/viewer.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
