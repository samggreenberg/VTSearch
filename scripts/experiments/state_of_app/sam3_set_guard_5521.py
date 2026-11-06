"""#5521: fall back to SigLIP's opening when SAM3's set is too broad, priced on #5241's COCO and #5490's FGVC sessions.

#5490 found SAM3 reads every aircraft name as "airplane": its 0.5 set covers ~45% of the pool where a variant has
~1%, and #5312's route (SAM3's set held, the head from the 20th Good) costs -0.27 F1 there, while on COCO (#5241,
#5311) it gains +0.12. The app cannot tell in advance whether SAM3 separates the prompt from its siblings, but it
sees the symptom before the first vote: the set's size. A guard opens on SigLIP instead when

- ``frac``: SAM3's set covers more than X of the pool, or
- ``ratio``: SAM3's set is more than K times SigLIP's typed set for the same text (both off the same pool).

Both sets are read off the cells: ``threshold_percentile`` of the click-1 untagged row is the share of the pool
below the shown cut, so 1 minus it is the set's share (the sig arm's row is SigLIP's typed set, the sam3 arm's
SAM3's 0.5 set). Both arms of a study ran the same cells and seeds, so a guarded session is the sig arm's session
when the guard fires and the sam3 arm's ``g20`` (#5312) when it does not: no new sessions.

Studies (beta 1, 150 clicks): ``coco-s0`` (#5241, 2026-10-26), ``coco-s12`` (#5311, 2026-11-04, seeds 1-2), and
``fgvc-<level>`` (#5490, 2026-11-02; family is what a user who cannot name the variant types). F is F1 shown, area
over votes 1-150 and 1-50; SEs over the unit that is independent: COCO's class, FGVC's family.

Two prices. ``cut05`` is the issue's: SAM3's 0.5 set, ``g20``, beta 1. ``shipped`` is the route #5565 shipped
for #5312 (SAM3's cut 0.85 / 0.75 / 0.4 at beta 1/4 / 1 / 4, the head from the 20th Good, held at 4): COCO from
``sam3_cut_5565.py``'s per-cell areas and Test-half sets (today = its hold-0 sig rerun on Binary), FGVC at 0.75
off its Test halves rebuilt by ``sets`` (checked against every session's recorded 0.5 set first). The shipped
guard reads the set at the preset's cut, as the app would show it, and adds the other way a set is the wrong size:
too few images (``k<N`` on the withheld half; #5565's skis@small).

    python sam3_set_guard_5521.py sets --out DIR      # FGVC Test-half sets at every cut (pickle + SAM3 shards)
    python sam3_set_guard_5521.py shipped --out DIR   # guards on the shipped route -> sam3_5521_shipped_*
    python sam3_set_guard_5521.py cut05 --out DIR     # guards on the 0.5 / g20 route -> sam3_5521_cut05_*

``sets`` runs in ``unknown_class_5490.sh``'s environment (``VTSEARCH_DATA_DIR`` = the vts-cache data dir).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "pile"))
import sam3_opening_5241 as so  # noqa: E402
from early_check_4664 import area, se_over_classes  # noqa: E402

ROOT = Path("/expscratch/sgreenberg/state-of-the-app")
HAND_OVER = 20
#: (study, sig run, sam3 run): run dirs under ROOT.
STUDIES = (
    ("coco-s0", "2026-10-26-sam3open5241-sig-b1", "2026-10-26-sam3open5241-sam3-b1"),
    ("coco-s12", "2026-11-04-sam3open5241-sig-b1", "2026-11-04-sam3open5241-sam3-b1"),
    ("fgvc-family", "2026-11-02-uc5490-sig-family-b1", "2026-11-02-uc5490-sam3-family-b1"),
    ("fgvc-variant", "2026-11-02-uc5490-sig-variant-b1", "2026-11-02-uc5490-sam3-variant-b1"),
    ("fgvc-fake", "2026-11-02-uc5490-sig-fake-b1", "2026-11-02-uc5490-sam3-fake-b1"),
)
FRAC = (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
RATIO = (2, 3, 5, 10, 20, 50, 100)
AREAS = ((1, 50), (1, 150))

# The set's share of the pool rides on the click-1 row; read_cell keeps COLS, and the Pool's workers fork after this.
so.COLS.extend(c for c in ("threshold_percentile", "n_haystack", "detector_tier") if c not in so.COLS)


def set_share(rows: pd.DataFrame) -> dict[tuple[str, str, int], float]:
    """Per (cell, path, seed): the share of the pool in the opening's set, off the first untagged row."""
    r = rows[(rows["gmm_variant"] == "") & (rows["detector_tier"].astype(str) == "query")]
    r = r.sort_values("t").drop_duplicates(["category", "embedder", "seed"])
    return {(c, e, int(s)): 1.0 - float(p) for c, e, s, p in
            r[["category", "embedder", "seed", "threshold_percentile"]].itertuples(index=False)}  # fmt: skip


def goods_by_click(cells: Path) -> dict[tuple[str, str, int], np.ndarray]:
    """Goods labelled by each click, per (cell, path, seed), off the session's own picks (check votes included)."""
    out = {}
    for pp in sorted(glob.glob(f"{cells}/task_*__picks.csv*")):
        pk = pd.read_csv(pp, usecols=["seed", "category", "embedder", "t", "picked_label"])
        if pk.empty:
            continue
        g = np.zeros(so.T + 1)
        for t in pk.loc[(pk["picked_label"] == 1) & (pk["t"] <= so.T), "t"].astype(int):
            g[t:] += 1
        out[(str(pk["category"].iloc[0]), str(pk["embedder"].iloc[0]), int(pk["seed"].iloc[0]))] = g
    return out


def unit_of(study: str, cat: str) -> str:
    """The independent unit an SE is taken over: COCO's class, FGVC's family."""
    if study.startswith("fgvc"):
        from pilebuild.loaders import fgvc_aircraft as fa  # noqa: PLC0415

        return fa.family_of(cat)
    return cat.rpartition("@")[0]


def study_cells(study: str, sig_run: str, sam3_run: str, procs: int) -> pd.DataFrame:
    """One row per (cell, path, seed): both sets' shares and the sig / hold / g20 F curves."""
    sig_d, sam3_d = (ROOT / r / "results" / "cells" for r in (sig_run, sam3_run))
    sig, sam3 = so.load(sig_d, procs), so.load(sam3_d, procs)
    s_shown, hold, head = so.curves(sig, ""), so.curves(sam3, ""), so.curves(sam3, so.HOLD_TAG)
    s_share, q_share = set_share(sig), set_share(sam3)
    goods = goods_by_click(sam3_d)
    nan = np.full(so.T + 1, np.nan)
    rows = []
    for k in sorted(set(s_shown) & set(hold)):
        # A session whose head never forms shows SAM3's set throughout (#5490's lesson: keep those cells).
        hd = head.get(k, (nan, nan))[0]
        take = np.isfinite(hd) & (goods.get(k, np.zeros(so.T + 1)) >= HAND_OVER)
        rows.append({"study": study, "category": k[0], "path": so.PATHS[k[1]], "seed": k[2],
                     "unit": unit_of(study, k[0]), "share_sig": s_share[k], "share_sam3": q_share[k],
                     "sig": s_shown[k][0], "hold": hold[k][0], "g20": np.where(take, hd, hold[k][0]),
                     "head": hd, "take": take})  # fmt: skip
    return pd.DataFrame(rows)


def rules() -> dict[str, callable]:
    out = {"g20 (no guard)": lambda d: np.zeros(len(d), bool), "sig (always)": lambda d: np.ones(len(d), bool)}
    for x in FRAC:
        out[f"frac>{x:g}"] = lambda d, x=x: d["share_sam3"].to_numpy() > x
    for k in RATIO:
        out[f"ratio>{k:g}"] = lambda d, k=k: d["ratio"].to_numpy() > k
    for x in (0.05, 0.10):
        for k in (5, 10, 20):
            out[f"frac>{x:g} & ratio>{k:g}"] = lambda d, x=x, k=k: (
                (d["share_sam3"].to_numpy() > x) & (d["ratio"].to_numpy() > k)
            )
    return out


def price(cells: pd.DataFrame) -> pd.DataFrame:
    out = []
    groups = [(s, cells["study"] == s) for s in cells["study"].unique()]
    groups += [("coco-all", cells["study"].str.startswith("coco"))]
    for path in ("Binary", "Region"):
        for name, m in groups:
            d = cells[m.to_numpy() & (cells["path"] == path).to_numpy()]
            if d.empty:
                continue
            sig, g20 = np.vstack(d["sig"]), np.vstack(d["g20"])
            unit = d["unit"].to_numpy()
            for rule, fire in rules().items():
                f = fire(d)
                shown = np.where(f[:, None], sig, g20)
                row = {"path": path, "study": name, "rule": rule, "cells": len(d), "fallback": float(f.mean())}
                for lo, hi in AREAS:
                    a = area(shown, lo, hi)
                    for ref, rf in (("sig", sig), ("g20", g20)):
                        dd = a - area(rf, lo, hi)
                        row[f"vs_{ref}_{lo}_{hi}"] = float(np.mean(dd))
                        row[f"vs_{ref}_se_{lo}_{hi}"] = se_over_classes(dd, unit)
                    row[f"F_{lo}_{hi}"] = float(np.mean(a))
                out.append(row)
    return pd.DataFrame(out)


def figure(cells: pd.DataFrame, path: Path) -> None:
    """SAM3's set share against SigLIP's, per cell, coloured by whether g20 beats SigLIP's opening."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from early_check_4664 import MUTED, _axes_style  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, p in zip(axes, ("Binary", "Region"), strict=True):
        d = cells[cells["path"] == p]
        gain = area(np.vstack(d["g20"]), 1, 150) - area(np.vstack(d["sig"]), 1, 150)
        for fam, marker in (("coco", "o"), ("fgvc", "^")):
            m = d["study"].str.startswith(fam).to_numpy()
            sc = ax.scatter(d["share_sig"][m], d["share_sam3"][m], c=gain[m], cmap="RdBu", vmin=-0.5, vmax=0.5,
                            marker=marker, s=14, lw=0, label={"coco": "COCO (#5241/#5311)", "fgvc": "FGVC (#5490)"}[fam])  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(p, loc="left", fontsize=11)
        ax.set_xlabel("SigLIP typed set, share of pool", color=MUTED)
        _axes_style(ax)
    axes[0].set_ylabel("SAM3 0.5 set, share of pool", color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.colorbar(sc, ax=axes, label="g20 minus SigLIP, F1 1-150")
    fig.savefig(path, dpi=150, bbox_inches="tight")


def load_cells(procs: int) -> pd.DataFrame:
    cells = pd.concat([study_cells(s, sg, sm, procs) for s, sg, sm in STUDIES], ignore_index=True)
    # SigLIP's set can be one image; floor it at one image of the pool so a ratio is finite.
    cells["ratio"] = cells["share_sam3"] / cells["share_sig"].clip(lower=1e-4)
    cells["g20_minus_sig"] = area(np.vstack(cells["g20"]), 1, 150) - area(np.vstack(cells["sig"]), 1, 150)
    return cells


def fgvc_sets(out: Path) -> None:
    """SAM3's set on each FGVC cell's Test half at every cut, per level: ``sam3_cut_5565.py sets`` for #5490's pool."""
    sys.path.insert(0, str(HERE.parent / "calibration"))
    import common  # noqa: PLC0415

    common.setup_env()
    from _cells_io import load_medias  # noqa: PLC0415
    from pilebuild.loaders import fgvc_aircraft as fa  # noqa: PLC0415
    from sam3_cut_5565 import GRID  # noqa: PLC0415
    from sam3_scores import load_sam3  # noqa: PLC0415

    import experiment_config as cfg  # noqa: PLC0415
    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.eval.labels import evaluable_pool, media_is_positive  # noqa: PLC0415
    from vtscore.eval.voting_iterations import _split_media_ids  # noqa: PLC0415

    medias = load_medias(EMBEDDINGS_DIR / "fgvc_aircraft__siglip.pkl")
    rows = []
    for level, prompts in FGVC_LEVELS.items():
        sam3 = load_sam3(FGVC_SCORES, prompts)
        for cat in fa.cells():
            pool = evaluable_pool(medias, cat)
            s, _ = sam3(cat, pool)
            _, test = _split_media_ids(pool, cfg.SIM_FRACTION, np.random.RandomState(0))
            sc = np.array([s[i] for i in test])
            y = np.array([1 if media_is_positive(pool[i], cat) else 0 for i in test])
            for c in GRID:
                keep = sc >= c
                rows.append({"study": f"fgvc-{level}", "category": cat, "seed": 0, "cut": c, "k": int(keep.sum()),
                             "tp": int(y[keep].sum()), "n_pos": int(y.sum()), "n_test": len(y)})  # fmt: skip
        common.log(f"{level}: {len(fa.cells())} cells")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "sam3_5521_fgvc_sets.csv", index=False)


#: #5490's levels as its arms name them, and the prompt set each reads from SAM3's FGVC shards.
FGVC_LEVELS = {"family": "family", "variant": "variant", "fake": "fake_name"}
FGVC_SCORES = Path("/expscratch/sgreenberg/fgvc-5490/scores-sam3")
#: #5565's shipped route: SAM3's cut per preset, the head from the 20th Good (None: SAM3's set held at beta 4).
SHIPPED_CUT = {0.25: 0.85, 1.0: 0.75, 4.0: 0.4}
SHIPPED_GATE = {0.25: HAND_OVER, 1.0: HAND_OVER, 4.0: None}
COCO_5565 = Path("/expscratch/sgreenberg/sam3-5565/out")
SHIPPED = "both: tuned cut, 20th Good (hold at 4)"


def set_f(tp, k, n_pos, beta: float) -> np.ndarray:
    b2 = beta * beta
    return (1 + b2) * np.asarray(tp, float) / (np.asarray(k, float) + b2 * np.asarray(n_pos, float))


def check_fgvc(cells: pd.DataFrame, sets: pd.DataFrame) -> None:
    """The rebuilt 0.5 set must give each FGVC session's recorded click-1 set F (precision and recall)."""
    s = sets[sets["cut"] == 0.5].set_index(["study", "category"])
    bad = []
    for _, r in cells[cells["study"].str.startswith("fgvc")].iterrows():
        x = s.loc[(r["study"], r["category"])]
        f = float(set_f(x["tp"], x["k"], x["n_pos"], 1.0))
        if abs(f - r["hold"][1]) > 1e-5:
            bad.append((r["study"], r["category"], r["path"], f, r["hold"][1]))
    if bad:
        raise SystemExit(f"{len(bad)} FGVC sessions' 0.5 set differs from the rebuilt one, e.g. {bad[:3]}")
    print(f"FGVC: the rebuilt 0.5 set matches all {int(cells['study'].str.startswith('fgvc').sum())} sessions")


def shipped(cells: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Per (study, beta, path, cell): today, the shipped route, and SAM3's set at the shipped cut (k on Test)."""
    sets = pd.read_csv(out / "sam3_5521_fgvc_sets.csv")
    check_fgvc(cells, sets)
    rows = []
    sx = sets.set_index(["study", "category", "cut"])
    for _, r in cells[cells["study"].str.startswith("fgvc")].iterrows():
        x = sx.loc[(r["study"], r["category"], SHIPPED_CUT[1.0])]
        q = float(set_f(x["tp"], x["k"], x["n_pos"], 1.0))
        route = np.where(r["take"], r["head"], q)
        route[0] = np.nan
        rows.append({"study": r["study"], "beta": 1.0, "seed": 0, "path": r["path"], "category": r["category"],
                     "unit": r["unit"], "today": float(area(r["sig"][None], 1, so.T)[0]),
                     "route": float(area(route[None], 1, so.T)[0]), "k": int(x["k"]), "n_test": int(x["n_test"]),
                     "share_sig": r["share_sig"]})  # fmt: skip
    # COCO: #5565's per-cell areas (today = its hold-0 sig rerun on Binary) and its Test-half sets.
    cc = pd.read_csv(COCO_5565 / "sam3_5565_cells.csv")
    cs = pd.read_csv(COCO_5565 / "sam3_5565_sets.csv")
    cs = cs[cs["rule"].str.startswith("fixed:")].set_index(["category", "seed", "cut"])
    for _, r in cc.iterrows():
        x = cs.loc[(r["category"], int(r["seed"]), SHIPPED_CUT[float(r["beta"])])]
        rows.append({"study": "coco", "beta": float(r["beta"]), "seed": int(r["seed"]), "path": r["path"],
                     "category": r["category"], "unit": r["category"].rpartition("@")[0], "today": r["today"],
                     "route": r[SHIPPED], "k": int(x["k"]), "n_test": int(x["n_test"]), "share_sig": np.nan})  # fmt: skip
    d = pd.DataFrame(rows)
    d["share"] = d["k"] / d["n_test"]
    d["route_minus_today"] = d["route"] - d["today"]
    return d


def price_shipped(d: pd.DataFrame) -> pd.DataFrame:
    """Guards on the shipped route's set: too broad (share > X), too narrow (k < N on Test), or both."""
    guards: dict[str, callable] = {"no guard": lambda g: np.zeros(len(g), bool)}
    for x in FRAC:
        guards[f"share>{x:g}"] = lambda g, x=x: g["share"].to_numpy() > x
    for n in (1, 3, 5, 10):
        guards[f"k<{n}"] = lambda g, n=n: g["k"].to_numpy() < n
    for x in (0.075, 0.10, 0.15):
        for n in (1, 3, 5):
            guards[f"share>{x:g} | k<{n}"] = lambda g, x=x, n=n: (g["share"].to_numpy() > x) | (g["k"].to_numpy() < n)
    out = []
    for (study, beta, path), g in d.groupby(["study", "beta", "path"]):
        for seeds, gs in [("all", g), *((str(s), g[g["seed"] == s]) for s in sorted(g["seed"].unique()) if g["seed"].nunique() > 1)]:  # fmt: skip
            for name, fire in guards.items():
                f = fire(gs)
                a = np.where(f, gs["today"], gs["route"])
                unit = gs["unit"].to_numpy()
                dt, dr = a - gs["today"].to_numpy(), a - gs["route"].to_numpy()
                out.append({"study": study, "beta": beta, "path": path, "seeds": seeds, "guard": name,
                            "cells": len(gs), "fallback": float(f.mean()),
                            "vs_today": float(dt.mean()), "vs_today_se": se_over_classes(dt, unit),
                            "vs_route": float(dr.mean()), "vs_route_se": se_over_classes(dr, unit)})  # fmt: skip
    return pd.DataFrame(out)


#: The guard this study recommends, beside no guard and each half of it alone.
GUARD = "share>0.1 | k<3"
HEADLINE = ("no guard", "share>0.1", "k<3", GUARD)


def figure_shipped(d: pd.DataFrame, path: Path) -> None:
    """Binary: each cell's SAM3 set share at the shipped cut against what the route gains over today."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from early_check_4664 import MUTED, _axes_style  # noqa: PLC0415

    d = d[d["path"] == "Binary"]
    style = {("coco", 0.25): ("COCO, beta 1/4 (cut 0.85)", "#9bc1ea", "o"),
             ("coco", 1.0): ("COCO, beta 1 (cut 0.75)", "#2a78d6", "o"),
             ("coco", 4.0): ("COCO, beta 4 (cut 0.4)", "#0d3f78", "o"),
             ("fgvc-family", 1.0): ("FGVC, family typed (cut 0.75)", "#d65f2a", "^"),
             ("fgvc-variant", 1.0): ("FGVC, variant typed", "#e8a33d", "v"),
             ("fgvc-fake", 1.0): ("FGVC, invented name", "#8a5a44", "s")}  # fmt: skip
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    floor = 1e-4
    for (study, beta), (label, color, marker) in style.items():
        g = d[(d["study"] == study) & (d["beta"] == beta)]
        ax.scatter(g["share"].clip(lower=floor), g["route_minus_today"], s=12, color=color, marker=marker, lw=0,
                   alpha=0.75, label=label)  # fmt: skip
    ax.axvline(0.1, color=MUTED, ls="--", lw=1)
    ax.text(0.105, ax.get_ylim()[1] * 0.92, "share > 10%: open on SigLIP", color=MUTED, fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("SAM3's set at the shipped cut, share of the withheld half (empty sets drawn at 0.01%)", color=MUTED)
    ax.set_ylabel("route minus today, F-beta 1-150", color=MUTED)
    _axes_style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("cut05", "sets", "shipped"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "sets":
        fgvc_sets(a.out)
        return 0
    cells = load_cells(a.procs)
    if a.stage == "shipped":
        d = shipped(cells, a.out)
        d.to_csv(a.out / "sam3_5521_shipped_cells.csv", index=False)
        q = (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        dist = d.groupby(["study", "beta", "path"])["share"].quantile(q).unstack()
        dist.to_csv(a.out / "sam3_5521_shipped_shares.csv")
        pr = price_shipped(d)
        pr.to_csv(a.out / "sam3_5521_shipped_price.csv", index=False)
        head = pr[(pr["seeds"] == "all") & pr["guard"].isin(HEADLINE)]
        head.to_csv(a.out / "sam3_5521_shipped_summary.csv", index=False)
        figure_shipped(d, a.out / "sam3_5521_guard.png")
        with pd.option_context("display.width", 250, "display.max_rows", 1000):
            print(dist.round(4).to_string())
            print(head.round(3).to_string(index=False))
        return 0
    flat = cells.drop(columns=["sig", "hold", "g20", "head", "take"])
    flat.to_csv(a.out / "sam3_5521_cut05_cells.csv", index=False)
    q = (0.1, 0.25, 0.5, 0.75, 0.9)
    dist = flat.groupby(["path", "study"])[["share_sam3", "share_sig", "ratio"]].quantile(q).unstack()
    dist.to_csv(a.out / "sam3_5521_cut05_shares.csv")
    pr = price(cells)
    pr.to_csv(a.out / "sam3_5521_cut05_price.csv", index=False)
    figure(cells, a.out / "sam3_5521_cut05_shares.png")
    info = {"cells": flat.groupby(["path", "study"]).size().to_dict()}
    (a.out / "sam3_5521_cut05_info.json").write_text(
        json.dumps({str(k): v for k, v in info["cells"].items()}, indent=1)
    )
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 40):
        print(dist.round(4).to_string())
        cols = ["path", "study", "rule", "cells", "fallback", "vs_sig_1_150", "vs_sig_se_1_150", "vs_g20_1_150",
                "vs_g20_se_1_150"]  # fmt: skip
        print(pr[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
