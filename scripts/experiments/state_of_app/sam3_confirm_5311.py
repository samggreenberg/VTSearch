"""#5311: #5241's 20th-Good hand-over out of sample (seeds 1-2 at beta 1, seed 1 at beta 1/4 and 4).

Reads ``sam3_gate_5241.py price``'s per-click tables (``<work>/<btag>/gate``) and the run dirs, and writes:

- ``sam3_5311_by_beta.webp``: F-beta shown per vote on each path at each preset: today, SAM3's set held, SAM3 to the
  head at the 20th Good, and the best count at that preset;
- ``sam3_5311_seeds.csv``: ``goods>=N`` against today per seed (the out-of-sample unit);
- ``sam3_5311_set_pr.csv``: precision and recall of SAM3's set at 0.5 and of SigLIP's typed set, per path, at vote 1;
- ``sam3_5311_cells.csv``: per cell at each preset, SAM3's set and today's F over votes 1-150 (the literal examples).

    python sam3_confirm_5311.py --work /expscratch/sgreenberg/sam3-5311 --out DIR
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from early_check_4664 import MUTED, _axes_style, area, se_over_classes
from sam3_opening_5241 import BETA, HOLD_TAG, PATHS, T, btag, curves, load, run_dir

ROOT = Path("/expscratch/sgreenberg/state-of-the-app")
DATE = "2026-11-04"
BETAS = (0.25, 1.0, 4.0)
GOODS = (4, 8, 12, 16, 20, 24, 32, 48)


def goods_by_click(cells: Path) -> dict[tuple[str, str, int], np.ndarray]:
    out = {}
    for pp in sorted(glob.glob(f"{cells}/task_*__picks.csv*")):
        pk = pd.read_csv(pp, usecols=["seed", "category", "embedder", "t", "picked_label"])
        if pk.empty:
            continue
        g = np.zeros(T + 1)
        for t in pk.loc[(pk["picked_label"] == 1) & (pk["t"] <= T), "t"].astype(int):
            g[t:] += 1
        out[(str(pk["category"].iloc[0]), str(pk["embedder"].iloc[0]), int(pk["seed"].iloc[0]))] = g
    return out


def set_pr(rows: pd.DataFrame, arm: str, beta: float) -> list[dict]:
    first = rows[(rows["gmm_variant"] == "") & (rows["t"] == 1)]
    return [{"beta": beta, "arm": arm, "path": PATHS[e], "precision": float(np.nanmean(g["precision"])),
             "recall": float(np.nanmean(g["recall"])), "cells": len(g)} for e, g in first.groupby("embedder")]  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/expscratch/sgreenberg/sam3-5311"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    seeds, prs, cells_rows = [], [], []
    for beta in BETAS:
        BETA["beta"] = beta
        sig = load(run_dir(ROOT, DATE, "", "sig", beta), a.procs)
        sam3 = load(run_dir(ROOT, DATE, "", "sam3", beta), a.procs)
        prs += set_pr(sig, "sig", beta) + set_pr(sam3, "sam3", beta)
        s_shown, q_shown, q_head = curves(sig, ""), curves(sam3, ""), curves(sam3, HOLD_TAG)
        n_goods = goods_by_click(run_dir(ROOT, DATE, "", "sam3", beta))
        for emb, path in PATHS.items():
            keys = sorted(k for k in set(s_shown) & set(q_shown) & set(q_head) & set(n_goods) if k[1] == emb)
            cls = np.array([k[0].rpartition("@")[0] for k in keys])
            seed = np.array([k[2] for k in keys])
            sig_f = np.vstack([s_shown[k][0] for k in keys])
            q0 = np.vstack([q_shown[k][0] for k in keys])[:, 1]
            head = np.vstack([q_head[k][0] for k in keys])
            good = np.vstack([n_goods[k] for k in keys])
            a_sig = area(sig_f, 1, T)
            for k, key in enumerate(keys):
                cells_rows.append({"beta": beta, "path": path, "category": key[0], "seed": key[2],
                                   "sam3_set_F": q0[k], "today_F_1_150": a_sig[k]})  # fmt: skip
            for n in GOODS:
                f = np.where(np.isfinite(head) & (good >= n), head, q0[:, None])
                d = area(f, 1, T) - a_sig
                for s in sorted(set(seed)):
                    m = seed == s
                    seeds.append({"beta": beta, "path": path, "seed": int(s), "N": n, "cells": int(m.sum()),
                                  "vs_sig": float(np.mean(d[m])), "se": se_over_classes(d[m], cls[m])})  # fmt: skip
    pd.DataFrame(seeds).to_csv(a.out / "sam3_5311_seeds.csv", index=False)
    pd.DataFrame(prs).to_csv(a.out / "sam3_5311_set_pr.csv", index=False)
    pd.DataFrame(cells_rows).to_csv(a.out / "sam3_5311_cells.csv", index=False)
    figure(a.work, a.out / "sam3_5311_by_beta.webp")
    return 0


def figure(work: Path, path: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    gate = {b: pd.read_csv(work / btag(b) / "gate" / "sam3_5241_gate.csv") for b in BETAS}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, b in enumerate(BETAS):
        pc = pd.read_csv(work / btag(b) / "gate" / "sam3_5241_gate_per_click.csv")
        for i, p in enumerate(("Binary", "Region")):
            ax, g = axes[i, j], pc[pc["path"] == p]
            gg = gate[b][
                (gate[b]["path"] == p) & (gate[b]["band"] == "all") & gate[b]["rule"].str.startswith("goods>=")
            ]
            best = gg.loc[gg["vs_sig_1_150"].idxmax(), "rule"]
            lines = [("sig", "SigLIP opening (today)", "#8a8985", 2), ("hold", "SAM3 set, held", "#2a78d6", 2),
                     ("goods>=20", "SAM3 -> head at the 20th Good", "#3a9a5b", 2.4)]  # fmt: skip
            if best != "goods>=20":
                lines.append((best, f"SAM3 -> head at {best} (best here)", "#d65f2a", 1.4))
            for col, label, color, lw in lines:
                ax.plot(g["t"], g[col], color=color, lw=lw, label=label)
            ax.set_title(f"{p}, beta {b:g}", loc="left", fontsize=11)
            if i == 1:
                ax.set_xlabel("votes", color=MUTED)
            if j == 0:
                ax.set_ylabel("F-beta shown (mean over cells)", color=MUTED)
            _axes_style(ax)
            ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    raise SystemExit(main())
