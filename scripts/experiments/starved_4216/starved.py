#!/usr/bin/env python3
"""#4216: the starved share (fewer than 3 positives found by vote 150) per arm and size band,
paired arm-vs-control differences, and the mechanism checks, from scan_found.py's tables.

    python starved.py --scan <scan dir> --out <analysis dir>

Writes ``runs.csv`` (one row per run), ``share_by_arm.csv``, ``paired.csv`` (every contrast,
every band), ``classes.csv`` (starved runs per class on the four reference arms) and
``mechanism.csv`` (where each starved run was at vote 150).

A cell file with no voting row is a run that never found a Good.  It counts as found = 0 at
every vote, and its (category, seed) comes from its task index: every grid here enumerates
seed-major, then the embedder block, then the 144 cells in one fixed order.  The rule is
derived from, and then checked against, every file that has rows; a mismatch is an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHECKPOINTS = (5, 10, 25, 50, 75, 100, 125, 150)
BETAS = (("b025", "1/4"), ("b1", "1"), ("b4", "4"))


def grid_of(arm: pd.Series) -> pd.Series:
    """Which task layout a run dir uses.  #4837 has its own grid (SigLIP, then EmbeddingGemma 2);
    #4746 and #4671 reuse #4671's binary-only prepare (144 tasks a seed, 1% pool); every other dir
    is on the 288-a-seed grid (144 Binary, then 144 Region tasks, natural prevalence)."""
    return pd.Series(
        np.select(
            [arm.str.contains("eg2-4837"), arm.str.startswith(("acqtarget-4746__", "autopilot-4671__"))],
            ["eg2", "coco144"],
            "coco288",
        ),
        index=arm.index,
    )


PER_SEED = {"coco288": 288, "eg2": 288, "coco144": 144}


def load(scan: Path) -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted(scan.glob("*.csv"))], ignore_index=True)
    d["grid"] = grid_of(d["arm"])
    d["idx"] = d["file"].str.extract(r"task_(\d+)")[0].astype(int)
    d["per_seed"] = d["grid"].map(PER_SEED)
    known = d[d.is_empty == 0].copy()
    cellmap = known.groupby(known["idx"] % 144)["category"].agg(lambda s: sorted(set(s)))
    bad_cell = cellmap[cellmap.map(len) > 1]
    bad_seed = known[known["idx"] // known["per_seed"] != known["seed"].astype(int)]
    if len(bad_cell) or len(bad_seed):
        print("TASK-INDEX RULE FAILS:\n", bad_cell.head(), "\n", bad_seed[["arm", "file", "seed"]].head())
        sys.exit(1)
    block = known.groupby(["grid", (known["idx"] // 144) % 2])["embedder"].agg(lambda s: sorted(set(s)))
    if (block.map(len) > 1).any():
        print("EMBEDDER BLOCK RULE FAILS:\n", block)
        sys.exit(1)
    emp = d[d.is_empty == 1].copy()
    emp["category"] = (emp["idx"] % 144).map(cellmap.map(lambda v: v[0]))
    emp["seed"] = emp["idx"] // emp["per_seed"]
    emp["embedder"] = [block[(g, (i // 144) % 2)][0] for g, i in zip(emp["grid"], emp["idx"])]
    for k in CHECKPOINTS:
        emp[f"found_{k}"] = 0
    emp["found150"] = 0
    emp["found_end"] = 0
    emp["phase150"] = "good (no row)"
    emp["left_good_t"] = -1
    print(f"{len(emp)} cell files with no voting row (no Good all session) mapped to their cells")
    d = pd.concat([known, emp], ignore_index=True)
    d["seed"] = d["seed"].astype(int)
    d["band"] = d["category"].str.partition("@")[2]
    d["cls"] = d["category"].str.partition("@")[0]
    d["starved"] = (d["found150"] < 3).astype(int)
    return d.drop(columns=["idx", "per_seed"])


def pairs() -> list[tuple[str, str, str, str, str | None, str | None]]:
    """(kind, contrast, arm dir, control dir, arm embedder, control embedder): same grid, same commit."""
    p: list[tuple[str, str, str, str, str | None, str | None]] = []
    for b, bl in BETAS:
        p += [
            (
                "opening",
                f"opening without the More walk (#4671), beta {bl}",
                f"autopilot-4671__open_{b}",
                f"autopilot-4671__ctl_{b}",
                None,
                None,
            ),
            (
                "opening",
                f"More walk on the detector's top (#4637), beta {bl}",
                f"2026-10-07-walk4637D-{b}",
                f"2026-10-07-walk4637T-{b}",
                None,
                None,
            ),
            (
                "opening",
                f"Good phase ends on a dry run (#4731), beta {bl}",
                f"2026-10-10-gooddry4731Q-{b}",
                f"2026-10-10-gooddry4731C-{b}",
                None,
                None,
            ),
            (
                "acquisition",
                f"Hard pick in place of the atlas walk (#4671), beta {bl}",
                f"autopilot-4671__nowalk_{b}",
                f"autopilot-4671__ctl_{b}",
                None,
                None,
            ),
            (
                "learning",
                f"typed query as the head's prior, scale 10 (#4384), beta {bl}",
                f"2026-10-13-qp4384arm10-{b}",
                f"2026-10-13-qp4384ctl-{b}",
                None,
                None,
            ),
            (
                "learning",
                f"typed query as the head's prior, scale 3 (#4384), beta {bl}",
                f"2026-10-13-qp4384arm3-{b}",
                f"2026-10-13-qp4384ctl-{b}",
                None,
                None,
            ),
            (
                "embedding",
                f"EmbeddingGemma 2 in place of SigLIP (#4837), beta {bl}",
                f"2026-10-13-eg2-4837-{b}",
                f"2026-10-13-eg2-4837-{b}",
                "embeddinggemma2",
                "siglip",
            ),
        ]
    for b, bl in BETAS[:2]:
        p.append(
            (
                "acquisition",
                f"acquisition target 0.75, not 0.5 (#4746), beta {bl}",
                f"acqtarget-4746__tp75_{b}",
                f"acqtarget-4746__ctl_{b}",
                None,
                None,
            )
        )
    for a, s in (("a8", "1 in 8"), ("a4", "1 in 4"), ("a2", "1 in 2")):
        p.append(
            (
                "acquisition",
                f"uniform band picks, {s} (#4482), beta 1",
                f"2026-10-08-band4482-{a}-b1",
                "2026-10-08-band4482-a0-b1",
                None,
                None,
            )
        )
    p += [
        (
            "opening",
            "Good-phase exit alone, no trained head (#4731 arm D), beta 1",
            "2026-10-09-gooddry4731D-b1",
            "2026-10-09-gooddry4731C-b1",
            None,
            None,
        ),
        (
            "embedding",
            "Region path, before #4731 (same commit), beta 1",
            "2026-10-10-gooddry4731C-region-b1",
            "2026-10-10-gooddry4731C-b1",
            None,
            None,
        ),
        (
            "embedding",
            "Region path, with #4731 (same commit), beta 1",
            "2026-10-10-gooddry4731Q-region-b1",
            "2026-10-10-gooddry4731Q-b1",
            None,
            None,
        ),
        (
            "embedding",
            "Region path, today's app (#4816's runs), beta 1",
            "2026-10-13-overlap4816region-b1",
            "2026-10-13-overlap4816-b1",
            None,
            None,
        ),
    ]
    return p


def clustered(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Mean of *col* over runs, its SE clustered on class (one cluster per class in a band)."""
    pc = df.groupby("cls")[col].mean()
    n = len(pc)
    return float(df[col].mean()), float(pc.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"), n


def select(d: pd.DataFrame, arm: str, emb: str | None) -> pd.DataFrame:
    x = d[d.arm == arm]
    return x[x.embedder == emb] if emb else x


def paired(d: pd.DataFrame) -> pd.DataFrame:
    ks = [f"found_{k}" for k in CHECKPOINTS]
    rows = []
    for kind, label, arm, ctl, ea, ec in pairs():
        A, C = select(d, arm, ea), select(d, ctl, ec)
        if A.empty or C.empty:
            print(f"skip (missing): {label}")
            continue
        j = A.merge(C, on=["category", "seed", "cls", "band"], suffixes=("_a", "_c"))
        j["d_starved"] = j.starved_a - j.starved_c
        j["d_found"] = j.found150_a - j.found150_c
        same = (j[[k + "_a" for k in ks]].to_numpy() == j[[k + "_c" for k in ks]].to_numpy()).all(axis=1)
        for band in ("small", "medium", "large"):
            jb = j[j.band == band]
            if jb.empty:
                continue
            ds, ses, ncls = clustered(jb, "d_starved")
            df_, sef, _ = clustered(jb, "d_found")
            st = (jb.starved_c == 1).to_numpy()
            rows.append(
                dict(
                    kind=kind,
                    contrast=label,
                    arm=arm,
                    control=ctl,
                    band=band,
                    pairs=len(jb),
                    seeds=jb.seed.nunique(),
                    classes=ncls,
                    starved_ctl=jb.starved_c.mean(),
                    starved_arm=jb.starved_a.mean(),
                    d_starved=ds,
                    se_starved=ses,
                    rescued=int(((jb.starved_c == 1) & (jb.starved_a == 0)).sum()),
                    newly_starved=int(((jb.starved_c == 0) & (jb.starved_a == 1)).sum()),
                    ctl_starved_identical_harvest=int(same[(j.band == band).to_numpy()][st].sum()),
                    found_ctl=jb.found150_c.mean(),
                    d_found=df_,
                    se_found=sef,
                )
            )
    return pd.DataFrame(rows)


#: The reference arms the figures and class table read (arm dir, embedder, label, seeds kept).
REFERENCE = [
    ("2026-10-08-b1", "siglip", "before #4731 (SotA 2026-10-08, 24 seeds)", None),
    ("2026-10-13-qp4384ctl-b1", "siglip", "today (2026-10-13, 5 seeds)", None),
    ("2026-10-13-overlap4816region-b1", "siglip+dinov3_patch", "Region path, today (2 seeds)", None),
    ("2026-10-13-overlap4816-b1", "siglip", "Binary, the same 2 seeds", (0, 1)),
    ("2026-10-13-eg2-4837-b1", "embeddinggemma2", "EmbeddingGemma 2 (5 seeds)", None),
]


def reference(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for arm, emb, label, seeds in REFERENCE:
        x = select(d, arm, emb)
        if seeds is not None:
            x = x[x.seed.isin(seeds)]
        out.append(x.assign(ref=label))
    return pd.concat(out, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = load(a.scan)
    d.to_csv(a.out / "runs.csv", index=False)

    g = (
        d.groupby(["arm", "embedder", "band"])
        .agg(
            runs=("starved", "size"),
            seeds=("seed", "nunique"),
            starved=("starved", "mean"),
            found_median=("found150", "median"),
            found_mean=("found150", "mean"),
        )
        .reset_index()
    )
    w = g.pivot_table(index=["arm", "embedder"], columns="band", values=["runs", "seeds", "starved", "found_median"])
    w.columns = [f"{x}_{y}" for x, y in w.columns]
    w = w.reset_index()
    w.to_csv(a.out / "share_by_arm.csv", index=False)

    P = paired(d)
    P.to_csv(a.out / "paired.csv", index=False)

    ref = reference(d)
    sm = ref[ref.band == "small"]
    cls = sm.groupby(["cls", "ref"]).agg(starved=("starved", "sum"), runs=("starved", "size")).reset_index()
    cls.to_csv(a.out / "classes.csv", index=False)
    st = sm[sm.starved == 1]
    mech = st.assign(
        found=st.found150.astype(int).astype(str),
        where=np.where(st.left_good_t > 0, "left the Good phase", "still in the Good phase"),
    )
    mech = mech.groupby(["ref", "where", "found"]).size().rename("runs").reset_index()
    mech.to_csv(a.out / "mechanism.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 70):
        cols = [
            "arm",
            "embedder",
            "runs_small",
            "starved_small",
            "starved_medium",
            "starved_large",
            "found_median_small",
        ]
        print(w[cols].round(3).to_string(index=False))
        print()
        sc = [
            "contrast",
            "pairs",
            "starved_ctl",
            "starved_arm",
            "d_starved",
            "se_starved",
            "rescued",
            "newly_starved",
            "ctl_starved_identical_harvest",
            "d_found",
            "se_found",
        ]
        print(P[P.band == "small"][sc].round(3).to_string(index=False))
        print()
        print(mech.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
