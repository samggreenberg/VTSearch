"""#5241: SAM3's zero-shot set as the opening on Binary and Region, against today's SigLIP opening.

Reads ``sam3_opening_5241.sh``'s three arms (seed 0, 144 cells a path, beta 1, 150 clicks):

- ``sig``: today's app; F shown is the untagged row at each click (the typed query's set, then the head).
- ``sam3``: SAM3's set at its 0.5 cut is the untagged row at every click; the head the same session trains is the
  ``hold:head`` row. Picks never depend on what Test shows, so a takeover rule is a choice, per click, between the
  two rows of one session.
- ``sam3gmm``: one click per Binary cell, SAM3's set at the GMM midpoint of its scores (the same set at every click
  and on both paths).

Takeover rules priced on the SAM3 path:

- ``hold``: SAM3's set through click 150 (no takeover);
- ``quota``: the head from the first click it exists (the label quota), as SigLIP's opening on Region hands over;
- ``at<N>``: SAM3's set before click N, the head from N;
- ``best``: per session the better of the two at each click (a bound no label-only rule reaches).

F shown is carried forward over clicks with no row. Means are over cells and SEs over classes (a class's bands are
not independent), as in #4711.

    python sam3_opening_5241.py --root <state-of-the-app> --out <dir> [--date 2026-10-26] [--tag ""]
"""

from __future__ import annotations

import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from early_check_4664 import MUTED, _axes_style, area, fbeta, se_over_classes  # noqa: E402

T = 150
GRID_T = np.arange(T + 1)
HOLD_TAG = "hold:head"
PATHS = {"siglip": "Binary", "siglip+dinov3_patch": "Region"}
FIXED = (10, 20, 30, 50, 75, 100)
AREAS = ((1, 50), (1, 150))
COLS = ["category", "seed", "embedder", "t", "phase", "gmm_variant", "schedule", "pool_variant", "detector_tier",
        "app_trained", "precision", "recall", "average_precision"]  # fmt: skip
COLOR = {"sig": "#8a8985", "hold": "#2a78d6", "quota": "#d65f2a", "head": "#3a9a5b", "best": "#b8b6b0"}


def run_dir(root: Path, date: str, tag: str, arm: str) -> Path:
    return root / f"{date}-sam3open5241{tag}-{arm}-b1" / "results" / "cells"


def read_cell(path: str) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=0).columns
    m = pd.read_csv(path, usecols=[c for c in COLS if c in head], low_memory=False)
    for col in ("gmm_variant", "schedule", "pool_variant"):
        m[col] = m[col].fillna("").astype(str).str.strip().replace({"nan": "", "None": ""}) if col in m else ""
    m = m[(m["schedule"] == "") & m["pool_variant"].isin(("", "max")) & m["gmm_variant"].isin(("", HOLD_TAG))]
    return m[m["phase"].astype(str) != "check"]


def load(d: Path, procs: int) -> pd.DataFrame:
    cells = sorted(f for f in glob.glob(f"{d}/task_*.csv*") if "__" not in Path(f).name)
    if not cells:
        raise SystemExit(f"no cells in {d}")
    with Pool(procs) as pool:
        parts = pool.map(read_cell, cells, chunksize=4)
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def on_grid(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """*v* at clicks *t*, carried forward over clicks with no row (NaN before the first)."""
    out = np.full(T + 1, np.nan)
    keep = t <= T
    out[t[keep]] = v[keep]
    return pd.Series(out).ffill().to_numpy()


def curves(rows: pd.DataFrame, tag: str) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Per (cell, path): (F, AP) of the rows tagged *tag* ('' is the untagged row), on the click grid."""
    out = {}
    sub = rows[rows["gmm_variant"] == tag]
    for (cat, emb), g in sub.groupby(["category", "embedder"]):
        g = g.drop_duplicates("t", keep="last").sort_values("t")
        t = g["t"].to_numpy(dtype=int)
        out[(cat, emb)] = (
            on_grid(t, fbeta(g["precision"], g["recall"], 1.0)),
            on_grid(t, g["average_precision"].to_numpy(dtype=float)),
        )
    return out


def mean_se(x: np.ndarray, cls: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x)
    return float(np.nanmean(x)), se_over_classes(x[ok], cls[ok])


def analyse(sig: pd.DataFrame, sam3: pd.DataFrame, gmm: pd.DataFrame | None, out: Path) -> dict:
    s_shown, q_shown, q_head = curves(sig, ""), curves(sam3, ""), curves(sam3, HOLD_TAG)
    g_line = {k[0]: v[0][1] for k, v in curves(gmm, "").items()} if gmm is not None else {}
    summary, per_click, crossings = [], [], []
    info: dict = {}
    for emb, path in PATHS.items():
        keys = sorted(k for k in set(s_shown) & set(q_shown) & set(q_head) if k[1] == emb)
        cats = [k[0] for k in keys]
        cls = np.array([c.rpartition("@")[0] for c in cats])
        band = np.array([c.rpartition("@")[2] for c in cats])
        sig_f = np.vstack([s_shown[k][0] for k in keys])
        q = np.vstack([q_shown[k][0] for k in keys])  # constant in t: SAM3's set at 0.5
        head = np.vstack([q_head[k][0] for k in keys])
        q0 = q[:, 1]
        gq = np.asarray([g_line.get(c, np.nan) for c in cats])
        info[path] = {"cells": len(keys), "classes": int(len(set(cls))),
                      "sam3_set_constant": bool(np.allclose(q[:, 1:], q0[:, None], equal_nan=True))}  # fmt: skip
        has = np.isfinite(head)
        first = np.where(has.any(axis=1), has.argmax(axis=1), T + 1)
        info[path]["quota_click_median"] = float(np.median(first))
        shown = {
            "sig": sig_f,
            "hold": np.repeat(q0[:, None], T + 1, axis=1),
            "quota": np.where(has, head, q0[:, None]),
            **{f"at{n}": np.where(has & (GRID_T >= n), head, q0[:, None]) for n in FIXED},
            "best": np.where(has, np.fmax(head, q0[:, None]), q0[:, None]),
        }
        if gmm is not None:
            shown["hold_gmm"] = np.repeat(gq[:, None], T + 1, axis=1)
        for groups in (("all", np.ones(len(keys), bool)), *((b, band == b) for b in ("small", "medium", "large"))):
            name, m = groups
            for rule, f in shown.items():
                row = {"path": path, "band": name, "rule": rule, "cells": int(m.sum())}
                for lo, hi in AREAS:
                    a = area(f[m], lo, hi)
                    row[f"F_{lo}_{hi}"], row[f"se_{lo}_{hi}"] = mean_se(a, cls[m])
                    if rule != "sig":
                        d = a - area(sig_f[m], lo, hi)
                        row[f"vs_sig_{lo}_{hi}"], row[f"vs_sig_se_{lo}_{hi}"] = mean_se(d, cls[m])
                for t in (10, 50, 100, 150):
                    row[f"F@{t}"] = float(np.nanmean(f[m][:, t]))
                summary.append(row)
        # The click where the head's mean first reaches SAM3's set, and per cell.
        mh = np.nanmean(head, axis=0)
        reach = np.where(np.isfinite(mh) & (mh >= np.nanmean(q0)))[0]
        info[path]["mean_head_reaches_sam3_at"] = int(reach[0]) if len(reach) else None
        info[path]["sam3_set_F"] = mean_se(q0, cls)
        info[path]["sam3_gmm_F"] = mean_se(gq, cls) if gmm is not None else None
        info[path]["sig_query_F_click1"] = float(np.nanmean(sig_f[:, 1]))
        for t in (25, 50, 100, 150):
            beats = head[:, t] > q0
            info[path][f"cells_head_beats_sam3@{t}"] = float(np.mean(beats[np.isfinite(head[:, t])]))
        # Does SAM3's opening make a better head? Same click, both paths' heads (the head is SigLIP's after the quota).
        sig_head = np.where(GRID_T >= first[:, None], sig_f, np.nan)
        for t in (50, 100, 150):
            d = head[:, t] - sig_head[:, t]
            info[path][f"head_sam3_minus_sig@{t}"] = mean_se(d, cls)
        for k, c in enumerate(cats):
            ok = np.where(np.isfinite(head[k]) & (head[k] > q0[k]))[0]
            crossings.append({"path": path, "category": c, "sam3_F": q0[k], "gmm_F": gq[k],
                              "first_click_head_beats": int(ok[0]) if len(ok) else None,
                              "head_F150": head[k, T], "sig_F150": sig_f[k, T]})  # fmt: skip
        for t in GRID_T[1:]:
            per_click.append({"path": path, "t": int(t), **{r: float(np.nanmean(f[:, t])) for r, f in shown.items()},
                              "head": float(np.nanmean(head[:, t]))})  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "sam3_5241_summary.csv", index=False)
    pc = pd.DataFrame(per_click)
    pc.to_csv(out / "sam3_5241_per_click.csv", index=False)
    pd.DataFrame(crossings).to_csv(out / "sam3_5241_cells.csv", index=False)
    (out / "sam3_5241_info.json").write_text(json.dumps(info, indent=1) + "\n")
    figure(pc, out / "sam3_5241_shown.png")
    return info


def figure(pc: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (name, g) in zip(axes, pc.groupby("path", sort=False), strict=False):
        lines = (("sig", "SigLIP opening (today)"), ("hold", "SAM3 set, held"), ("quota", "SAM3 -> head at quota"),
                 ("head", "head (SAM3 path)"))  # fmt: skip
        for col, label in lines:
            ax.plot(g["t"], g[col], color=COLOR[col], lw=2 if col != "head" else 1.2, label=label)
        ax.text(g["t"].iloc[-1], g["hold"].iloc[-1], "  SAM3", color=COLOR["hold"], va="center", fontsize=9)
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_xlabel("votes", color=MUTED)
        _axes_style(ax)
    axes[0].set_ylabel("F1 shown (mean over cells)", color=MUTED)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/expscratch/sgreenberg/state-of-the-app"))
    ap.add_argument("--date", default="2026-10-26")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    sig = load(run_dir(a.root, a.date, a.tag, "sig"), a.procs)
    sam3 = load(run_dir(a.root, a.date, a.tag, "sam3"), a.procs)
    gdir = run_dir(a.root, a.date, a.tag, "sam3gmm")
    gmm = load(gdir, a.procs) if glob.glob(f"{gdir}/task_*.csv*") else None
    print(json.dumps(analyse(sig, sam3, gmm, a.out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
