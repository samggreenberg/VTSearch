"""#4278: the converged logistic head against today's shipped head on Region Photo, as curves over the clicks reached.

Reads ``logreg_region_4278.sh``'s two arms (Region Photo, siglip+dinov3_patch max_patch, 150 clicks plus the closing
check, paired by cell and seed), which differ only in the head:

* ``ctl``: the shipped head (``linear_svm_prior``);
* ``lr``: logistic regression fitted to convergence (``linear_logreg``, #4114).

#4213 (2026-09-28) read the logistic head's failure on easy classes off its acquisition: the cut drifted down as Bad
images flooded region rows into the fit, admitted more of the pool, and its Hard picks landed among negatives. So
beside what the app shows (the withheld half's F-beta at the run's own beta: the typed query's set until the first
click with a detector on screen, then the detector's line, as ``early_check_4664.py`` scores it) this reads, per arm
and click, where the acquisition cut sits in the pool (``acq_pool_percentile``, a fraction: the share of the pool it
admits is one minus it), how often an acquisition pick is Good (the ``hard`` phase and, after Done, the ``done``
phase, which picks at the same cut), the Goods found, AP and the oracle's F-beta (the ranking alone).

Strata are #4213's (``docs/experiments/2026-09-28-head-switch-4219/region_categories.txt``): the hardest and easiest
quartiles of the 144 cells by an arm independent of both heads here, so a stratum carries no regression to the mean
toward either; the 72 cells between them are ``mid``.

    python logreg_region_4278.py --root <state-of-the-app dir> --seeds 3 --baseline <text_baseline.csv> \\
        --strata <region_categories.txt> --out <dir> [--docs <report dir>/tables] [--tag smoke] [--beta 1]
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
from early_check_4664 import INK, MUTED, _axes_style, _headline, area, fbeta, se_over_classes  # noqa: E402

T = 150
GRID_T = np.arange(T + 1)
CONTROL, ARM = "ctl", "lr"
ARMS = (CONTROL, ARM)
EMBEDDER = "siglip+dinov3_patch"
NAME = {"ctl": "shipped head (SVM from the Goods' centroid)", "lr": "logistic head, converged"}
COLOR = {"ctl": "#8a8985", "lr": "#2a78d6"}
STRATA = ("all", "easy", "mid", "hard")
ACQ_PHASES = ("hard", "done")
WINDOWS = ((1, 50), (51, 100), (101, 150))
AREAS = ((1, 50), (1, 100), (1, 150))
CLICKS = (10, 25, 50, 75, 100, 125, 150)
STEP_COLS = [
    "category", "seed", "embedder", "t", "phase", "app_trained", "precision", "recall", "average_precision",
    "acq_threshold", "acq_pool_percentile", "gmm_variant", "schedule", "pool_variant",
]  # fmt: skip
PICK_COLS = ["category", "seed", "embedder", "t", "phase", "check_kind", "picked_id", "picked_label"]
PHASES = ("hard", "done", "new")
OPENING = ("good", "bad")


def btag(beta: float) -> str:
    return {0.25: "025", 1.0: "1", 4.0: "4"}[float(beta)]


def run_dir(root: Path, date: str, tag: str, arm: str, beta: float) -> Path:
    return root / f"{date}-logreg4278{tag}-{arm}-region-b{btag(beta)}" / "results" / "cells"


def read_cell(args: tuple[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path, ocol = args
    head = pd.read_csv(path, nrows=0).columns
    m = pd.read_csv(path, usecols=[c for c in [*STEP_COLS, ocol] if c in head], low_memory=False)
    m = _headline(m[m["embedder"] == EMBEDDER])
    stem, ext = (path[: -len(".csv.gz")], ".csv.gz") if path.endswith(".gz") else (path[: -len(".csv")], ".csv")
    try:
        pk = pd.read_csv(f"{stem}__picks{ext}", usecols=lambda c: c in PICK_COLS, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pk = pd.DataFrame(columns=PICK_COLS)
    return m, pk[pk["embedder"] == EMBEDDER]


def load_arm(d: Path, seeds: int, procs: int, ocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = sorted(f for f in glob.glob(f"{d}/task_*.csv*") if "__" not in Path(f).name)
    if not cells:
        return pd.DataFrame(columns=STEP_COLS), pd.DataFrame(columns=PICK_COLS)
    with Pool(procs) as pool:
        parts = pool.map(read_cell, [(c, ocol) for c in cells], chunksize=4)
    steps = pd.concat([p[0] for p in parts if len(p[0])], ignore_index=True)
    picks = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    return steps[steps["seed"] < seeds], picks[picks["seed"] < seeds]


def _on_grid(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """*v* at clicks *t*, carried forward over clicks with no row (NaN before the first)."""
    out = np.full(T + 1, np.nan)
    keep = t <= T
    out[t[keep]] = v[keep]
    return pd.Series(out).ffill().to_numpy()


def per_run(steps: pd.DataFrame, picks: pd.DataFrame, tq: dict, beta: float, ocol: str) -> dict[str, dict]:
    """Per run: F shown, AP, the oracle's F, the acquisition cut's pool fraction, Goods found and acquisition picks."""
    out: dict[str, dict] = {}
    for (c, s), g in steps.groupby(["category", "seed"]):
        key = f"{c}|{int(s)}"
        ordinary = g[g["phase"].astype(str) != "check"].drop_duplicates("t", keep="last").sort_values("t")
        t = ordinary["t"].to_numpy(dtype=int)
        det = _on_grid(t, fbeta(ordinary["precision"], ordinary["recall"], beta))
        on = ordinary.loc[ordinary["app_trained"] == 1, "t"]
        shown = float(on.min()) if len(on) else np.inf
        q = tq.get(key, np.nan)
        out[key] = {
            "f": np.where((GRID_T >= shown) & np.isfinite(det), det, q),
            "ap": _on_grid(t, ordinary["average_precision"].to_numpy(dtype=float)),
            "oracle": _on_grid(t, ordinary[ocol].to_numpy(dtype=float)) if ocol in ordinary else np.full(T + 1, np.nan),
            "pct": _on_grid(t, ordinary["acq_pool_percentile"].to_numpy(dtype=float)),
            "tq": q,
            "trained_at": shown,
        }
    for (c, s), g in picks.groupby(["category", "seed"]):
        key = f"{c}|{int(s)}"
        if key not in out:
            continue
        ph = g["phase"].astype(str).to_numpy()
        lab = g["picked_label"].to_numpy(dtype=float)
        tt = g["t"].to_numpy(dtype=int)
        click = (ph != "check") & (tt <= T)
        goods = np.zeros(T + 1)
        np.add.at(goods, tt[click], lab[click])
        acq = np.isin(ph, ACQ_PHASES) & click
        win = {f"{lo}_{hi}": (int((acq & (tt >= lo) & (tt <= hi)).sum()), float(lab[acq & (tt >= lo) & (tt <= hi)].sum()))
               for lo, hi in WINDOWS}  # fmt: skip
        by_phase = {p: (int(((ph == p) & click).sum()), float(lab[(ph == p) & click].sum())) for p in PHASES}
        op = np.isin(ph, OPENING)
        opening = tuple(zip(tt[op].tolist(), ph[op].tolist(), g["picked_id"].to_numpy()[op].tolist()))
        out[key].update(goods=np.cumsum(goods), acq=win, by_phase=by_phase, opening=opening, n_acq=int(acq.sum()))
    return out


def read_strata(path: Path) -> dict[str, str]:
    """``category@band`` -> ``easy`` / ``hard`` from #4213's committed list; every other cell is ``mid``."""
    which, out = None, {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if s in ("# hard", "# easy"):
            which = s[2:]
        elif s and not s.startswith("#") and which:
            out[s] = which
    return out


def _paired(x: np.ndarray, y: np.ndarray, cls: np.ndarray) -> tuple[float, float, float, float]:
    """Means of both arms, the mean gain, and its SE over classes."""
    d = y - x
    ok = np.isfinite(d)
    return float(np.nanmean(x)), float(np.nanmean(y)), float(np.nanmean(d)), se_over_classes(d[ok], cls[ok])


def analyse(runs: dict[str, dict], strata: dict[str, str], beta: float) -> dict[str, pd.DataFrame]:
    keys = sorted(set(runs[CONTROL]) & set(runs[ARM]))
    keys = [k for k in keys if all("goods" in runs[a][k] for a in ARMS) and np.isfinite(runs[CONTROL][k]["tq"])]
    cat = np.array([k.split("|")[0] for k in keys])
    cls = np.array([c.split("@")[0] for c in cat])
    band = np.array([c.split("@")[1] for c in cat])
    stratum = np.array([strata.get(c, "mid") for c in cat])

    def stack(arm: str, name: str) -> np.ndarray:
        return np.vstack([runs[arm][k][name] for k in keys])

    F = {a: stack(a, "f") for a in ARMS}
    AP = {a: stack(a, "ap") for a in ARMS}
    ORC = {a: stack(a, "oracle") for a in ARMS}
    P = {a: stack(a, "pct") for a in ARMS}
    G = {a: stack(a, "goods") for a in ARMS}

    # Pairing: the opening (3 Goods off the typed query's top, then the Bad phase on its sort) reads no head, so it
    # must be the same pick for pick in both arms; the arms may part only from the first trained pick.
    opening_mismatch = int(sum(runs[CONTROL][k]["opening"] != runs[ARM][k]["opening"] for k in keys))
    rows, curves = [], []
    groups = [("all", np.ones(len(keys), bool))] + [(s, stratum == s) for s in STRATA[1:]]
    groups += [(f"band:{b}", band == b) for b in ("small", "medium", "large")]
    for name, sel in groups:
        if not sel.any():
            continue
        c = cls[sel]
        r: dict = {
            "beta": beta,
            "stratum": name,
            "runs": int(sel.sum()),
            "classes": len(set(c)),
            "opening_mismatch": opening_mismatch if name == "all" else np.nan,
        }
        for lo, hi in AREAS:
            r.update(zip([f"meanF_{lo}_{hi}_{x}" for x in ("ctl", "lr", "gain", "se")],
                         _paired(area(F[CONTROL][sel], lo, hi), area(F[ARM][sel], lo, hi), c)))  # fmt: skip
        for lab, M in (("F", F), ("AP", AP), ("oracle", ORC), ("goods", G)):
            for t in (50, 150):
                r.update(zip([f"{lab}{t}_{x}" for x in ("ctl", "lr", "gain", "se")],
                             _paired(M[CONTROL][sel, t], M[ARM][sel, t], c)))  # fmt: skip
        r.update(zip([f"meanAP_1_150_{x}" for x in ("ctl", "lr", "gain", "se")],
                     _paired(area(AP[CONTROL][sel], 1, 150), area(AP[ARM][sel], 1, 150), c)))  # fmt: skip
        for a in ARMS:
            for t in (25, 50, 100, 150):
                r[f"admit{t}_{a}"] = float(np.nanmedian(1.0 - P[a][sel, t]))
            for lo, hi in WINDOWS:
                n = sum(runs[a][keys[i]]["acq"][f"{lo}_{hi}"][0] for i in np.flatnonzero(sel))
                g = sum(runs[a][keys[i]]["acq"][f"{lo}_{hi}"][1] for i in np.flatnonzero(sel))
                r[f"acq_picks_{lo}_{hi}_{a}"] = n / sel.sum()
                r[f"acq_good_{lo}_{hi}_{a}"] = g / n if n else np.nan
            for p in PHASES:
                n = sum(runs[a][keys[i]]["by_phase"][p][0] for i in np.flatnonzero(sel))
                g = sum(runs[a][keys[i]]["by_phase"][p][1] for i in np.flatnonzero(sel))
                r[f"{p}_picks_{a}"] = n / sel.sum()
                r[f"{p}_good_{a}"] = g / n if n else np.nan
            r[f"admit100_mean_{a}"] = float(np.nanmean(1.0 - P[a][sel, 100]))
        rows.append(r)
        if name in STRATA:
            for t in GRID_T[1:]:
                d = F[ARM][sel, t] - F[CONTROL][sel, t]
                ok = np.isfinite(d)
                curves.append({
                    "beta": beta, "stratum": name, "t": int(t),
                    "F_ctl": float(np.nanmean(F[CONTROL][sel, t])), "F_lr": float(np.nanmean(F[ARM][sel, t])),
                    "F_gain": float(np.nanmean(d)), "F_se": se_over_classes(d[ok], c[ok]),
                    "goods_ctl": float(G[CONTROL][sel, t].mean()), "goods_lr": float(G[ARM][sel, t].mean()),
                    "AP_ctl": float(np.nanmean(AP[CONTROL][sel, t])), "AP_lr": float(np.nanmean(AP[ARM][sel, t])),
                    "admit_ctl": float(np.nanmedian(1.0 - P[CONTROL][sel, t])),
                    "admit_lr": float(np.nanmedian(1.0 - P[ARM][sel, t])),
                })  # fmt: skip
    per = pd.DataFrame({
        "category": cat, "seed": [int(k.split("|")[1]) for k in keys], "stratum": stratum,
        **{f"meanF_1_150_{a}": area(F[a], 1, 150) for a in ARMS}, **{f"goods150_{a}": G[a][:, T] for a in ARMS},
        **{f"AP150_{a}": AP[a][:, T] for a in ARMS}, **{f"admit100_{a}": 1.0 - P[a][:, 100] for a in ARMS},
    })  # fmt: skip
    return {"summary": pd.DataFrame(rows), "curves": pd.DataFrame(curves), "per_run": per}


def figure(res: dict[str, pd.DataFrame], path: Path) -> None:
    cv = res["curves"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axes[0, 0]
    for s, ls in (("easy", "-"), ("hard", "--"), ("all", ":")):
        d = cv[cv["stratum"] == s]
        ax.plot(d["t"], d["F_gain"], color=INK, ls=ls, lw=1.6, label=s)
        ax.fill_between(d["t"], d["F_gain"] - 2 * d["F_se"], d["F_gain"] + 2 * d["F_se"], color=INK, alpha=0.08)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("F-beta shown, logistic minus shipped (±2 SE)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for ax, col, title in (
        (axes[0, 1], "goods", "Goods found"),
        (axes[1, 0], "admit", "Share of the pool the acquisition cut admits (median)"),
    ):
        for s, ls in (("easy", "-"), ("hard", "--")):
            d = cv[cv["stratum"] == s]
            for a in ARMS:
                ax.plot(d["t"], d[f"{col}_{a}"], color=COLOR[a], ls=ls, lw=1.6, label=f"{a} {s}")
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8)
    axes[1, 0].set_yscale("log")
    ax = axes[1, 1]
    s = res["summary"].set_index("stratum")
    x = np.arange(len(WINDOWS))
    for j, (st, off) in enumerate((("easy", -0.3), ("hard", 0.1))):
        if st not in s.index:
            continue
        for i, a in enumerate(ARMS):
            v = [s.loc[st, f"acq_good_{lo}_{hi}_{a}"] for lo, hi in WINDOWS]
            ax.bar(x + off + 0.2 * i, v, width=0.18, color=COLOR[a], alpha=1.0 if j == 0 else 0.5, label=f"{a} {st}")
    ax.set_xticks(x, [f"clicks {lo}-{hi}" for lo, hi in WINDOWS])
    ax.set_title("Acquisition picks that are Good", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        _axes_style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--date", default="2026-10-18")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--strata", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--docs", type=Path, help="the report's tables dir; figures go beside it")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    t = btag(a.beta)
    ocol = f"oracle_fbeta_b{t}"
    base = pd.read_csv(a.baseline)
    base = base[(base["embedder"] == EMBEDDER) & (base["seed"] < a.seeds)]
    tq = {f"{c}|{int(s)}": float(fbeta([p], [r], a.beta)[0]) for c, s, p, r in
          zip(base["category"], base["seed"], base[f"text_line_precision_b{t}"], base[f"text_line_recall_b{t}"])}  # fmt: skip
    runs, counts = {}, {}
    for arm in ARMS:
        steps, picks = load_arm(run_dir(a.root, a.date, a.tag, arm, a.beta), a.seeds, a.procs, ocol)
        runs[arm] = per_run(steps, picks, tq, a.beta, ocol)
        counts[arm] = len(runs[arm])
        print(f"beta {a.beta:g} {arm}: {len(runs[arm])} runs", flush=True)
    if min(counts.values()) == 0:
        print("an arm has no runs yet")
        return 1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        res = analyse(runs, read_strata(a.strata), a.beta)
    for o in [a.out] + ([a.docs] if a.docs else []):
        o.mkdir(parents=True, exist_ok=True)
        for name, df in res.items():
            df.to_csv(o / f"lr4278_{name}.csv", index=False)
        (o / "lr4278_counts.json").write_text(json.dumps(counts, indent=1) + "\n")
    for o in [a.out] + ([a.docs.parent / "figures"] if a.docs else []):
        o.mkdir(parents=True, exist_ok=True)
        figure(res, o / "lr4278_over_clicks.png")
    s = res["summary"]
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        for cols in (
            [f"meanF_{lo}_{hi}_{x}" for lo, hi in AREAS for x in ("ctl", "lr", "gain", "se")],
            [f"{m}{t}_{x}" for m in ("F", "AP", "oracle") for t in (150,) for x in ("ctl", "lr", "gain", "se")],
            [f"goods{t}_{x}" for t in (50, 150) for x in ("ctl", "lr", "gain", "se")],
            [f"admit{t}_{a}" for t in (25, 50, 100, 150) for a in ARMS],
            [f"acq_{k}_{lo}_{hi}_{a}" for k in ("picks", "good") for lo, hi in WINDOWS for a in ARMS],
            [f"{p}_{k}_{a}" for p in PHASES for k in ("picks", "good") for a in ARMS] + ["opening_mismatch"],
        ):
            print(s[["stratum", "runs", *cols]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
