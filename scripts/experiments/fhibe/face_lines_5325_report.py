#!/usr/bin/env python
"""Tables and figures for #5325's report (face lines on imagekit's cosines, buffalo_s).

Reads the study's outputs and writes, to ``--out`` (the report directory), only aggregates (no FHIBE ids):

- ``ab.csv`` / ``curves.csv``: ``weak_check_4778.py``'s paired compare at 1024 px (control = the app,
  arm = the centroid's count at bar z5.5);
- ``k1-1024-objective_by_click.csv`` (the arm) and ``k1-1024-objective_by_click-control.csv``: one photo,
  1024 px, the objective at every vote per beta (#5336);
- ``set.csv``: precision, recall and faces returned at 50 votes and after the check, per K x beta (#5336);
- ``click0.csv``: the example sort's line at click 0 per rule (``face_lines_5325_click0.py``'s summary), on
  FHIBE and COCO Better;
- ``pronoun.csv``: each candidate's gain by pronoun, at click 0, over votes 1-25 and 1-150, and for Find;
- ``find.csv``: ``face_lines_5325_find.py report``'s table (``find/report.csv``; ``find/control.csv.gz`` is its input);
- ``fig_tail.png``, ``fig_curves.png``, ``fig_pronoun.png``.

  face_lines_5325_report.py --runs <runs> --work <work dir> --strata person_strata.csv --out <report dir>
"""

from __future__ import annotations

import argparse
import gzip
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CTL, ARM = "2026-10-27-5325ctl-ik", "2026-10-27-5325z55-ik"
BETA_TAG = {0.25: "b025", 1.0: "b1", 4.0: "b4"}


def _pronoun(strata: Path) -> pd.DataFrame:
    st = pd.read_csv(strata)
    st["pronoun"] = st["pronoun"].str.lower().str.extract(r"^(she/her|he/him)", expand=False)
    return st[["category", "pronoun"]]


def _category(run_dir: Path, idx: int) -> str:
    with gzip.open(run_dir / "results" / "cells" / f"task_{idx:04d}.csv.gz", "rt") as fh:
        head = fh.readline().rstrip("\n").split(",")
        return fh.readline().split(",")[head.index("category")]


def _split(h: pd.DataFrame, col: str) -> dict:
    a, b = h.loc[h["pronoun"] == "she/her", col], h.loc[h["pronoun"] == "he/him", col]
    return {"she_her": a.mean(), "he_him": b.mean(), "gap_change": a.mean() - b.mean(),
            "gap_change_se": math.sqrt(a.var() / len(a) + b.var() / len(b))}  # fmt: skip


def tables(a: argparse.Namespace) -> dict[str, pd.DataFrame]:
    runs, work, out = a.runs, a.work, a.out
    pro = _pronoun(a.strata)
    t: dict[str, pd.DataFrame] = {}
    cmp_dir = runs / "2026-10-27-5325-compare-1024"
    t["ab"], t["curves"] = pd.read_csv(cmp_dir / "ab.csv"), pd.read_csv(cmp_dir / "curves.csv")
    t["k1-1024-objective_by_click"] = pd.read_csv(runs / f"{ARM}-analysis" / "k1-1024-objective_by_click.csv")
    t["k1-1024-objective_by_click-control"] = pd.read_csv(runs / f"{CTL}-analysis" / "k1-1024-objective_by_click.csv")

    rows = []
    for k in (1, 4):
        for beta, tag in BETA_TAG.items():
            c = pd.read_csv(runs / f"{CTL}-analysis" / f"k{k}-{tag}-1024" / "cells.csv")
            for at in ("50", "final"):
                rows.append({"K": k, "beta": beta, "at": "50 votes" if at == "50" else "after the check",
                             "fbeta": c[f"thr_fbeta_{at}"].mean(), "precision": c[f"thr_precision_{at}"].mean(),
                             "recall": c[f"thr_recall_{at}"].mean(), "returned": c[f"thr_returned_{at}"].mean()})  # fmt: skip
    t["set"] = pd.DataFrame(rows)

    keep = ["app", "app_z5", "app_z5.5", "app_z6", "drop"]
    c0 = [pd.read_csv(work / d / "summary.csv") for d in ("click0", "click0-coco")]
    c0 = pd.concat(c0)
    t["click0"] = c0[(c0["corpus"] == "unvoted") & (c0["scale"] == 1.0) & c0["rule"].isin(keep)].drop(
        columns=["corpus", "scale"]
    )

    pr = []
    cells = pd.read_csv(work / "click0" / "cells.csv.gz")
    cells = cells[(cells["corpus"] == "unvoted") & (cells["scale"] == 1.0) & cells["rule"].isin(["app", "app_z5.5"])]
    piv = cells.pivot_table(index=["dataset", "k", "beta", "category"], columns="rule", values="fbeta").reset_index()
    piv = piv.merge(pro, on="category")
    piv["d"] = piv["app_z5.5"] - piv["app"]
    for (ds, k, beta), h in piv.groupby(["dataset", "k", "beta"]):
        pr.append({"what": "centroid z5.5, click 0", "size": int(ds.rsplit("_", 1)[1]), "K": k, "beta": beta,
                   **_split(h, "d")})  # fmt: skip
    for size in (1024, 640):
        p = pd.read_csv(runs / f"2026-10-27-5325-compare-{size}" / "per_run.csv").merge(pro, on="category")
        for (k, beta), h in p.groupby(["K", "beta"]):
            for col, what in (("d_1_25", "votes 1-25"), ("d_1_150", "votes 1-150")):
                pr.append({"what": f"centroid z5.5, {what}", "size": size, "K": k, "beta": beta, **_split(h, col)})
    fd = pd.read_csv(work / "find" / "control.csv.gz")
    fd = fd[fd["snap"] == "t50"]
    fd = fd.assign(category=[_category(runs / r, i) for r, i in zip(fd["run"], fd["idx"])]).merge(pro, on="category")
    for c in ("z5", "z5.5"):
        fd = fd.assign(d=fd[f"F_{c}"] - fd["F_app"])
        for (size, k, beta), h in fd.groupby(["size", "k", "beta"]):
            per = h.groupby(["category", "pronoun"], as_index=False)["d"].mean()
            pr.append({"what": f"Find {c}, vote 50", "size": size, "K": k, "beta": beta, **_split(per, "d")})
    t["pronoun"] = pd.DataFrame(pr)
    t["find"] = pd.read_csv(work / "find" / "report.csv")
    out.mkdir(parents=True, exist_ok=True)
    for name, df in t.items():
        df.round(5).to_csv(out / f"{name}.csv", index=False)
    return t


def fig_tail(a: argparse.Namespace) -> None:
    """Where the unvoted media sit above the bulk at click 0, one photo: the person's withheld photos, the rest, a Gaussian."""
    with open(a.work / "click0" / "click0_fhibe_faces_ik_1024.pkl", "rb") as fh:
        cells = [c for c in pickle.load(fh) if c["k"] == 1]  # noqa: S301 - the study's own dump
    pos_z, neg_z = [], []
    for c in cells:
        x = c["scores"][~c["example"]]
        y = c["pos"][~c["example"]]
        med = np.median(x)
        sd = 1.4826 * np.median(np.abs(x - med))
        z = (x - med) / sd
        pos_z.append(z[y])
        neg_z.append(z[~y & (z > 2)])
    pos_z, neg_z = np.concatenate(pos_z), np.concatenate(neg_z)
    n_all = sum(int((~c["example"]).sum()) for c in cells)
    bins = np.arange(2, 14.01, 0.25)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hist(neg_z, bins=bins, histtype="step", lw=1.6, color="#5c636e", label="everyone else")
    ax.hist(pos_z, bins=bins, histtype="step", lw=1.6, color="#2a6fdb", label="the person's other photos")
    from scipy.stats import norm

    mid = (bins[:-1] + bins[1:]) / 2
    ax.plot(mid, n_all * (norm.sf(bins[:-1]) - norm.sf(bins[1:])), color="#d9480f", lw=1.6,
            label="a Gaussian bulk's share")  # fmt: skip
    for z, ls in ((4.0, "-"), (5.5, "--")):
        ax.axvline(z, color="k", ls=ls, lw=0.9)
        ax.text(z + 0.08, 3e4, f"z {z:g}" + (" (app)" if z == 4 else " (arm)"), fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.1, None)
    ax.set_xlabel("cosine to the starting photo, in bulk spreads above the median (z)")
    ax.set_ylabel(f"media, {len(cells)} sessions")
    ax.set_title("Click 0, one photo, 1024 px: the bulk's upper tail is far heavier than a Gaussian's", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(a.out / "fig_tail.png", dpi=130)


def fig_curves(t: dict[str, pd.DataFrame], a: argparse.Namespace) -> None:
    c = t["curves"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True, sharey="row")
    for i, k in enumerate((1, 4)):
        for j, beta in enumerate((0.25, 1.0, 4.0)):
            h = c[(c["K"] == k) & (c["beta"] == beta) & (c["vote"] <= 30)]
            ax = axes[i, j]
            ax.plot(h["vote"], h["control"], color="#8a8f98", label="the app (z 4)")
            ax.plot(h["vote"], h["arm"], color="#2a6fdb", label="bar z5.5")
            ax.set_title(f"{k} photo{'s' if k > 1 else ''}, β {beta:g}", fontsize=9)
            if j == 0:
                ax.set_ylabel("F at the line, withheld")
            if i == 1:
                ax.set_xlabel("vote")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("The paired sessions, 1024 px, votes 1-30 (identical from the head's first line on)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out / "fig_curves.png", dpi=130)


def fig_pronoun(t: dict[str, pd.DataFrame], a: argparse.Namespace) -> None:
    p = t["pronoun"]
    p = p[(p["size"] == 1024) & p["what"].isin(["centroid z5.5, click 0", "Find z5, vote 50"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=False)
    for ax, what in zip(axes, ["centroid z5.5, click 0", "Find z5, vote 50"]):
        h = p[p["what"] == what].reset_index(drop=True)
        lab = [f"K{k} β{b:g}" for k, b in zip(h["K"], h["beta"])]
        x = np.arange(len(h))
        ax.bar(x - 0.2, h["she_her"], 0.4, color="#c2255c", label="she/her")
        ax.bar(x + 0.2, h["he_him"], 0.4, color="#1971c2", label="he/him")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x, lab, fontsize=8)
        ax.set_title(f"{what}: gain in F over the app's line", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(a.out / "fig_pronoun.png", dpi=130)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True, help="the study's work dir (click0/, click0-coco/, find/)")
    ap.add_argument("--strata", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t = tables(a)
    fig_tail(a)
    fig_curves(t, a)
    fig_pronoun(t, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
