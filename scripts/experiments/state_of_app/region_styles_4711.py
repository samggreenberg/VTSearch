"""#4711: the raw-patch HAC trees against today's tree-free MaxPatch on Region Photo, as curves over the clicks reached.

Reads ``region_styles_4711.sh``'s three arms (Region Photo, siglip+dinov3_patch, 150 clicks, paired by cell and seed),
which differ only in the region geometry (``vtscore/eval/patch_styles.py``):

* ``mp``: the shipped geometry (``max_patch``);
* ``hac``: a raw-patch-leaf HAC tree (``max_patch_hac``);
* ``pca``: the same tree, merged in a per-image PCA space (``max_patch_pca_hac``).

The July study (``docs/experiments/2026-07-29-max-patch/REPORT.md``) found the trees ranked best and lost at the
operating point, so beside what the app shows (the withheld half's F-beta: the typed query's set until the first click
with a detector on screen, then the detector's line, as ``early_check_4664.py`` scores it) this reads the ranking
alone (AP, and the oracle's F-beta: the best cut of the same ranking) and the line's share of the loss, the regret
(oracle minus shown). SigLIP binary, the other production path, is the reference: the ``mp`` arm's Binary cells,
same grid and seeds (not a paired arm: a different path, as ``compare_paths.py`` sets the two side by side).

    python region_styles_4711.py --root <state-of-the-app dir> --seeds 3 --baseline <text_baseline.csv> \\
        --out <dir> [--docs <report dir>/tables] [--tag smoke]
"""

from __future__ import annotations

import argparse
import glob
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from early_check_4664 import MUTED, _axes_style, _headline, area, fbeta, se_over_classes  # noqa: E402

T = 150
GRID_T = np.arange(T + 1)
BETA = 1.0
OCOL = "oracle_fbeta_b1"
CONTROL = "mp"
ARMS = ("mp", "hac", "pca")
CHALLENGERS = ("hac", "pca")
REGION, BINARY = "siglip+dinov3_patch", "siglip"
NAME = {
    "mp": "MaxPatch (shipped)",
    "hac": "MaxPatchHAC",
    "pca": "MaxPatchPcaHAC",
    "bin": "SigLIP binary",
}
COLOR = {"mp": "#8a8985", "hac": "#2a78d6", "pca": "#d65f2a", "bin": "#3a9a5b"}
BANDS = ("small", "medium", "large")
AREAS = ((1, 50), (1, 150))
STEP_COLS = ["category", "seed", "embedder", "t", "phase", "app_trained", "precision", "recall", "average_precision"]
PICK_COLS = ["category", "seed", "embedder", "t", "phase", "picked_label"]


def run_dir(root: Path, date: str, tag: str, arm: str) -> Path:
    return root / f"{date}-regstyles4711{tag}-{arm}-region-b1" / "results" / "cells"


def read_cell(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    head = pd.read_csv(path, nrows=0).columns
    m = _headline(pd.read_csv(path, usecols=[c for c in [*STEP_COLS, OCOL] if c in head], low_memory=False))
    stem, ext = (path[: -len(".csv.gz")], ".csv.gz") if path.endswith(".gz") else (path[: -len(".csv")], ".csv")
    try:
        pk = pd.read_csv(f"{stem}__picks{ext}", usecols=lambda c: c in PICK_COLS, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pk = pd.DataFrame(columns=PICK_COLS)
    return m, pk


def load_arm(d: Path, seeds: int, procs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = sorted(f for f in glob.glob(f"{d}/task_*.csv*") if "__" not in Path(f).name)
    if not cells:
        return pd.DataFrame(columns=STEP_COLS), pd.DataFrame(columns=PICK_COLS)
    with Pool(procs) as pool:
        parts = pool.map(read_cell, cells, chunksize=4)
    steps = pd.concat([p[0] for p in parts if len(p[0])], ignore_index=True)
    picks = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    return steps[steps["seed"] < seeds], picks[picks["seed"] < seeds]


def _on_grid(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """*v* at clicks *t*, carried forward over clicks with no row (NaN before the first)."""
    out = np.full(T + 1, np.nan)
    keep = t <= T
    out[t[keep]] = v[keep]
    return pd.Series(out).ffill().to_numpy()


def per_run(steps: pd.DataFrame, picks: pd.DataFrame, tq: dict, embedder: str) -> dict[str, dict]:
    """Per run of *embedder*: F shown, AP, the oracle's F and the Goods found, on the click grid."""
    steps = steps[steps["embedder"] == embedder]
    picks = picks[picks["embedder"] == embedder]
    out: dict[str, dict] = {}
    for (c, s), g in steps.groupby(["category", "seed"]):
        key = f"{c}|{int(s)}"
        ordinary = g[g["phase"].astype(str) != "check"].drop_duplicates("t", keep="last").sort_values("t")
        t = ordinary["t"].to_numpy(dtype=int)
        det = _on_grid(t, fbeta(ordinary["precision"], ordinary["recall"], BETA))
        on = ordinary.loc[ordinary["app_trained"] == 1, "t"]
        shown = float(on.min()) if len(on) else np.inf
        q = tq.get(key, np.nan)
        out[key] = {
            "f": np.where((GRID_T >= shown) & np.isfinite(det), det, q),
            "ap": _on_grid(t, ordinary["average_precision"].to_numpy(dtype=float)),
            "oracle": _on_grid(t, ordinary[OCOL].to_numpy(dtype=float)) if OCOL in ordinary else np.full(T + 1, np.nan),
            "tq": q,
            "goods": np.zeros(T + 1),
        }
    for (c, s), g in picks.groupby(["category", "seed"]):
        key = f"{c}|{int(s)}"
        if key not in out:
            continue
        ph = g["phase"].astype(str).to_numpy()
        tt = g["t"].to_numpy(dtype=int)
        lab = g["picked_label"].to_numpy(dtype=float)
        click = (ph != "check") & (tt <= T)
        goods = np.zeros(T + 1)
        np.add.at(goods, tt[click], lab[click])
        out[key]["goods"] = np.cumsum(goods)
    return out


def _paired(x: np.ndarray, y: np.ndarray, cls: np.ndarray) -> tuple[float, float, float, float]:
    """Means of both arms, the mean gain, and its SE over classes."""
    d = y - x
    ok = np.isfinite(d)
    return float(np.nanmean(x)), float(np.nanmean(y)), float(np.nanmean(d)), se_over_classes(d[ok], cls[ok])


def analyse(runs: dict[str, dict]) -> dict[str, pd.DataFrame]:
    keys = sorted(set.intersection(*(set(runs[a]) for a in ARMS)))
    keys = [k for k in keys if np.isfinite(runs[CONTROL][k]["tq"])]
    cat = np.array([k.split("|")[0] for k in keys])
    cls = np.array([c.split("@")[0] for c in cat])
    band = np.array([c.split("@")[1] for c in cat])
    have_bin = "bin" in runs and all(k in runs["bin"] for k in keys)
    shown = (*ARMS, "bin") if have_bin else ARMS

    def stack(arm: str, name: str) -> np.ndarray:
        return np.vstack([runs[arm][k][name] for k in keys])

    M = {lab: {a: stack(a, lab) for a in shown} for lab in ("f", "ap", "oracle", "goods")}
    rows, curves = [], []
    groups = [("all", np.ones(len(keys), bool))] + [(f"band:{b}", band == b) for b in BANDS]
    for name, sel in groups:
        if not sel.any():
            continue
        c = cls[sel]
        for arm in CHALLENGERS:
            r: dict = {"group": name, "arm": arm, "runs": int(sel.sum()), "classes": len(set(c))}
            for lo, hi in AREAS:
                r.update(zip([f"meanF_{lo}_{hi}_{x}" for x in ("mp", "arm", "gain", "se")],
                             _paired(area(M["f"][CONTROL][sel], lo, hi), area(M["f"][arm][sel], lo, hi), c)))  # fmt: skip
            for lab, col in (("F", "f"), ("AP", "ap"), ("oracle", "oracle"), ("goods", "goods")):
                for t in (50, 150):
                    r.update(zip([f"{lab}{t}_{x}" for x in ("mp", "arm", "gain", "se")],
                                 _paired(M[col][CONTROL][sel, t], M[col][arm][sel, t], c)))  # fmt: skip
            # The line's share of the loss: F shown falls short of the oracle's F (the best cut of the same ranking) by
            # the regret; the arm's F gain is its oracle gain (the ranking) minus its regret gain (the line).
            for t in (50, 150):
                reg_mp = M["oracle"][CONTROL][sel, t] - M["f"][CONTROL][sel, t]
                reg_arm = M["oracle"][arm][sel, t] - M["f"][arm][sel, t]
                r.update(zip([f"regret{t}_{x}" for x in ("mp", "arm", "gain", "se")], _paired(reg_mp, reg_arm, c)))
            rows.append(r)
        if have_bin:
            r = {"group": name, "arm": "bin", "runs": int(sel.sum()), "classes": len(set(c))}
            for lo, hi in AREAS:
                r.update(zip([f"meanF_{lo}_{hi}_{x}" for x in ("mp", "arm", "gain", "se")],
                             _paired(area(M["f"][CONTROL][sel], lo, hi), area(M["f"]["bin"][sel], lo, hi), c)))  # fmt: skip
            for lab, col in (("F", "f"), ("AP", "ap"), ("oracle", "oracle")):
                for t in (50, 150):
                    r.update(zip([f"{lab}{t}_{x}" for x in ("mp", "arm", "gain", "se")],
                                 _paired(M[col][CONTROL][sel, t], M[col]["bin"][sel, t], c)))  # fmt: skip
            rows.append(r)
        for t in GRID_T[1:]:
            row = {"group": name, "t": int(t)}
            for a in shown:
                row[f"F_{a}"] = float(np.nanmean(M["f"][a][sel, t]))
                row[f"AP_{a}"] = float(np.nanmean(M["ap"][a][sel, t]))
                row[f"oracle_{a}"] = float(np.nanmean(M["oracle"][a][sel, t]))
            for a in CHALLENGERS:
                d = M["f"][a][sel, t] - M["f"][CONTROL][sel, t]
                ok = np.isfinite(d)
                row[f"F_gain_{a}"] = float(np.nanmean(d))
                row[f"F_se_{a}"] = se_over_classes(d[ok], c[ok])
            curves.append(row)
    per = pd.DataFrame({
        "category": cat, "seed": [int(k.split("|")[1]) for k in keys],
        **{f"meanF_1_150_{a}": area(M["f"][a], 1, 150) for a in shown},
        **{f"AP150_{a}": M["ap"][a][:, T] for a in shown}, **{f"oracle150_{a}": M["oracle"][a][:, T] for a in shown},
    })  # fmt: skip
    return {"summary": pd.DataFrame(rows), "curves": pd.DataFrame(curves), "per_run": per}


def figure(res: dict[str, pd.DataFrame], path: Path) -> None:
    cv = res["curves"]
    d = cv[cv["group"] == "all"]
    shown = [a for a in (*ARMS, "bin") if f"F_{a}" in d]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, col, title in (
        (axes[0, 0], "F", "F1 shown on the withheld half"),
        (axes[0, 1], "AP", "Average precision (the ranking)"),
        (axes[1, 0], "oracle", "Oracle F1: the best cut of the same ranking"),
    ):
        for a in shown:
            ax.plot(d["t"], d[f"{col}_{a}"], color=COLOR[a], lw=1.8, label=NAME[a])
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    for a in CHALLENGERS:
        ax.plot(d["t"], d[f"F_gain_{a}"], color=COLOR[a], lw=1.6, label=f"{NAME[a]} − MaxPatch")
        ax.fill_between(d["t"], d[f"F_gain_{a}"] - 2 * d[f"F_se_{a}"], d[f"F_gain_{a}"] + 2 * d[f"F_se_{a}"],
                        color=COLOR[a], alpha=0.12)  # fmt: skip
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("F1 shown, tree minus MaxPatch (±2 SE over classes)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("votes", color=MUTED, fontsize=9)
        _axes_style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_bands(res: dict[str, pd.DataFrame], path: Path) -> None:
    """Mean F1 over clicks 1-150 by size band, each tree's gain over MaxPatch (±2 SE)."""
    s = res["summary"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(BANDS))
    for i, a in enumerate(CHALLENGERS):
        sub = s[(s["arm"] == a) & s["group"].isin([f"band:{b}" for b in BANDS])].set_index("group")
        v = [sub.loc[f"band:{b}", "meanF_1_150_gain"] for b in BANDS]
        e = [2 * sub.loc[f"band:{b}", "meanF_1_150_se"] for b in BANDS]
        ax.bar(x - 0.18 + 0.36 * i, v, width=0.34, yerr=e, color=COLOR[a], label=f"{NAME[a]} − MaxPatch", capsize=3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, BANDS)
    ax.set_title("Mean F1 over clicks 1-150, gain over MaxPatch by object size", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _axes_style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--date", default="2026-10-23")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--docs", type=Path, help="the report's tables dir; figures go beside it")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    base = pd.read_csv(a.baseline)
    base = base[base["seed"] < a.seeds]
    tq = {
        emb: {
            f"{c}|{int(s)}": float(fbeta([p], [r], BETA)[0])
            for c, s, p, r in zip(g["category"], g["seed"], g["text_line_precision_b1"], g["text_line_recall_b1"])
        }  # fmt: skip
        for emb, g in base.groupby("embedder")
    }
    tq_region = tq.get(REGION) or tq.get(BINARY, {})
    runs, counts = {}, {}
    for arm in ARMS:
        steps, picks = load_arm(run_dir(a.root, a.date, a.tag, arm), a.seeds, a.procs)
        runs[arm] = per_run(steps, picks, tq_region, REGION)
        counts[arm] = len(runs[arm])
        if arm == CONTROL:
            runs["bin"] = per_run(steps, picks, tq.get(BINARY, tq_region), BINARY)
            counts["bin"] = len(runs["bin"])
    print(json.dumps(counts), flush=True)
    if min(counts[x] for x in ARMS) == 0:
        print("an arm has no runs yet")
        return 1
    if counts["bin"] == 0:
        del runs["bin"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        res = analyse(runs)
    for o in [a.out] + ([a.docs] if a.docs else []):
        o.mkdir(parents=True, exist_ok=True)
        for name, df in res.items():
            df.to_csv(o / f"rs4711_{name}.csv", index=False)
        (o / "rs4711_counts.json").write_text(json.dumps(counts, indent=1) + "\n")
    for o in [a.out] + ([a.docs.parent / "figures"] if a.docs else []):
        o.mkdir(parents=True, exist_ok=True)
        figure(res, o / "rs4711_over_clicks.png")
        figure_bands(res, o / "rs4711_bands.png")
    s = res["summary"]
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        for cols in (
            [f"meanF_{lo}_{hi}_{x}" for lo, hi in AREAS for x in ("mp", "arm", "gain", "se")],
            [f"{m}150_{x}" for m in ("F", "AP", "oracle") for x in ("mp", "arm", "gain", "se")],
            [f"regret{t}_{x}" for t in (50, 150) for x in ("mp", "arm", "gain", "se")],
            [f"goods{t}_{x}" for t in (50, 150) for x in ("mp", "arm", "gain", "se")],
        ):
            print(s[["group", "arm", "runs", *[c for c in cols if c in s]]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
