#!/usr/bin/env python3
"""#4216: what the contrasts that move the starved share pay or cost, by size band, on the
objective: F-beta of the set the app shows, as a session mean over votes 1-150.

    python cost.py --runs <analysis>/runs.csv --sota /expscratch/$USER/state-of-the-app \\
        --priced-4743 /expscratch/$USER/good-dry-4731/priced-4743 --out <analysis dir>

Each contrast is read off its own study's priced per-run tables, so the numbers are the ones
those studies reported (#4731 re-priced by #4743; #4837's ``analysis-*/curves.csv``):

* ``runs_<arm>_<path>_<beta>.npz`` (``handoff_price_4604.py``): today's curve is the typed
  query's set until the click the app shows the detector, the detector's set from then on.
* ``curves.csv`` (``state_of_app/analyze.py``): ``thr_fbeta`` at every click is already the
  set the app shows.

On the small band the runs are also split by what happened to their harvest: rescued (the
control starved, the arm did not), newly starved, starved in both, fed in both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def npz_session(path: Path) -> pd.DataFrame:
    z = np.load(path, allow_pickle=True)
    det, tq, shown = z["det"], z["tq"], z["shown"]
    grid = np.arange(det.shape[1])
    curve = np.where((grid[None, :] >= shown[:, None]) & np.isfinite(det), det, tq[:, None])
    return pd.DataFrame(
        {
            "category": z["category"],
            "seed": z["seed"].astype(int),
            "session": curve[:, 1:151].mean(axis=1),
            "at150": curve[:, 150],
        }
    )


def curves_session(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path, usecols=["category", "seed", "t", "thr_fbeta"])
    c = c[(c.t >= 1) & (c.t <= 150)]
    s = c.groupby(["category", "seed"]).thr_fbeta.mean().rename("session")
    e = c[c.t == 150].set_index(["category", "seed"]).thr_fbeta.rename("at150")
    return pd.concat([s, e], axis=1).reset_index()


def clustered(df: pd.DataFrame, col: str) -> tuple[float, float]:
    pc = df.groupby("cls")[col].mean()
    return float(df[col].mean()), float(pc.std(ddof=1) / np.sqrt(len(pc))) if len(pc) > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="starved.py's runs.csv")
    ap.add_argument("--sota", type=Path, required=True, help="the state-of-the-app runs root")
    ap.add_argument("--priced-4743", type=Path, required=True, help="#4743's priced dir (good-dry-4731/priced-4743)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    R = pd.read_csv(a.runs)

    def starved_of(arm: str, emb: str | None = None) -> pd.DataFrame:
        x = R[R.arm == arm]
        x = x[x.embedder == emb] if emb else x
        return x[["category", "seed", "starved"]]

    contrasts = []
    for b, beta in (("b025", 0.25), ("b1", 1.0), ("b4", 4.0)):
        contrasts.append(
            (
                "Good phase ends on a dry run (#4731)",
                beta,
                npz_session(a.priced_4743 / f"runs_Q_binary_{b}.npz"),
                npz_session(a.priced_4743 / f"runs_C_binary_{b}.npz"),
                starved_of(f"2026-10-10-gooddry4731Q-{b}"),
                starved_of(f"2026-10-10-gooddry4731C-{b}"),
            )
        )
        eg = a.sota / f"2026-10-13-eg2-4837-{b}"
        contrasts.append(
            (
                "EmbeddingGemma 2 in place of SigLIP (#4837)",
                beta,
                curves_session(eg / "analysis-eg2" / "curves.csv"),
                curves_session(eg / "analysis-binary" / "curves.csv"),
                starved_of(f"2026-10-13-eg2-4837-{b}", "embeddinggemma2"),
                starved_of(f"2026-10-13-eg2-4837-{b}", "siglip"),
            )
        )
    for tag, name in (("Q", "with #4731"), ("C", "before #4731")):
        contrasts.append(
            (
                f"Region path, {name} (same commit, 2 seeds)",
                1.0,
                npz_session(a.priced_4743 / f"runs_{tag}_region_b1.npz"),
                npz_session(a.priced_4743 / f"runs_{tag}_binary_b1.npz"),
                starved_of(f"2026-10-10-gooddry4731{tag}-region-b1"),
                starved_of(f"2026-10-10-gooddry4731{tag}-b1"),
            )
        )

    rows = []
    for label, beta, arm, ctl, sa, sc in contrasts:
        j = arm.merge(ctl, on=["category", "seed"], suffixes=("_a", "_c"))
        j = j.merge(sa, on=["category", "seed"]).merge(sc, on=["category", "seed"], suffixes=("_a", "_c"))
        j["cls"] = j.category.str.partition("@")[0]
        j["band"] = j.category.str.partition("@")[2]
        j["d"] = j.session_a - j.session_c
        j["d150"] = j.at150_a - j.at150_c
        j["fate"] = np.select(
            [
                (j.starved_c == 1) & (j.starved_a == 0),
                (j.starved_c == 0) & (j.starved_a == 1),
                (j.starved_c == 1) & (j.starved_a == 1),
            ],
            ["rescued", "newly starved", "starved in both"],
            "fed in both",
        )
        for band in ("small", "medium", "large", "all"):
            jb = j if band == "all" else j[j.band == band]
            m, se = clustered(jb, "d")
            rows.append(
                dict(
                    contrast=label,
                    beta=beta,
                    band=band,
                    group="every run",
                    runs=len(jb),
                    ctl_session=jb.session_c.mean(),
                    arm_session=jb.session_a.mean(),
                    d_session=m,
                    se=se,
                    d_at150=jb.d150.mean(),
                )
            )
            if band == "small":
                for fate, jf in jb.groupby("fate"):
                    m, se = clustered(jf, "d")
                    rows.append(
                        dict(
                            contrast=label,
                            beta=beta,
                            band=band,
                            group=fate,
                            runs=len(jf),
                            ctl_session=jf.session_c.mean(),
                            arm_session=jf.session_a.mean(),
                            d_session=m,
                            se=se,
                            d_at150=jf.d150.mean(),
                        )
                    )
    T = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    T.to_csv(a.out / "cost.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 60):
        print(T.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
