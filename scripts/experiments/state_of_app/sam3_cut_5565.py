"""#5565: SAM3's opening made right at every preset, by a cut per preset or by a hand-over count per preset.

#5311 found SAM3's fixed 0.5 cut gives the same set at every balance, so one hand-over count (the 20th Good) is
right at beta 1 only. Two fixes are priced on the SAM3 arm's sessions (#5311's and #5565's fresh seed 2):

- **a SAM3 cut per preset**, so the set leans the way the preset does: ``expf`` is label-free (the cut that
  maximises F-beta's expectation when SAM3's scores of the corpus a Test searches are read as probabilities), and
  ``tuned`` is one fixed cut per preset, the grid cut with the best mean set F-beta on the tuning seeds (seeds 0-1
  at beta 1, seed 1 at 1/4 and 4), scored on seed 2;
- **a hand-over count per preset**, fixed before seed 2 ran (the #5565 comment): ``goods>=8`` at 1/4,
  ``goods>=20`` at 1, ``hold`` at 4.

Picks never depend on what Test shows (``sam3_opening_5241.sh``), so SAM3's set at any cut is priced offline: a
cell's Test half is rebuilt as ``sam3_gate_5241.py collect`` rebuilds its Train half (``evaluable_pool``, then
``_split_media_ids`` under ``RandomState(seed)``), the set is the Test media scoring the cut or more
(``step_trainers._query_step``), and ``check`` asserts the 0.5 sets against every session's recorded row first.
The head the session trains is its ``hold:head`` rows, whatever the cut, and a route shows the set until the
Goods count is reached, then the head. Today is the ``sig`` arm: Binary from #5565's rerun (dev shipped hold 0,
#5556, after #5311's tree), Region from the run beside each seed's SAM3 arm (the hold never applied there).

    python sam3_cut_5565.py sets --out DIR      # per (cell, seed, rule): SAM3's set on Test -> sam3_5565_sets.csv
    python sam3_cut_5565.py check --out DIR     # the 0.5 (and 0.3) sets against the sessions' recorded rows
    python sam3_cut_5565.py price --out DIR     # every route against today, per seed and preset; summary + figure

``sets`` loads the SigLIP pickle and SAM3's shards (run it in ``sam3_gate_5241.py collect``'s environment). F is
F-beta at the preset's beta, over votes 1-150; means are over cells and SEs over classes, as in #5311.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAL = HERE.parent / "calibration"
ROOT = Path("/expscratch/sgreenberg/state-of-the-app")
SCORES = Path("/expscratch/sgreenberg/sam3-5213/scores")
BETAS = (0.25, 1.0, 4.0)
SEEDS = (0, 1, 2)
GRID = tuple(round(float(c), 2) for c in np.arange(0.05, 0.951, 0.05))
GOODS = (4, 8, 12, 16, 20, 24, 32, 48)
PATHS = {"siglip": "Binary", "siglip+dinov3_patch": "Region"}
#: Option 2's counts, fixed from #5311's seed 1 before seed 2 ran (None: hold SAM3's set through vote 150).
PRESET_COUNT = {0.25: 8, 1.0: 20, 4.0: None}
#: The seeds a tuned cut is chosen on; the rest score it.
TUNE_SEEDS = {0.25: (1,), 1.0: (0, 1), 4.0: (1,)}
#: The route the two options give together: the tuned cut, handed over at the 20th Good, held at beta 4.
COMBINED = "both: tuned cut, 20th Good (hold at 4)"
COMBINED_GATE = {0.25: "goods>=20", 1.0: "goods>=20", 4.0: "hold"}
#: The validation arms (seed 1, beta 1): the same cells at two lines on #5565's tree.
VALIDATION = {"-v05": "fixed:0.5", "-v03": "fixed:0.3"}


def btag(beta: float) -> str:
    return "b" + (f"{beta:g}".replace(".", "") if beta < 1 else f"{beta:g}")


def cells_dir(date: str, arm: str, beta: float, tag: str = "") -> Path:
    return ROOT / f"{date}-sam3open5241{tag}-{arm}-{btag(beta)}" / "results" / "cells"


def sam3_date(beta: float, seed: int) -> str:
    """The run dir holding the SAM3 arm's sessions at (beta, seed)."""
    if beta == 1.0:
        return "2026-10-26" if seed == 0 else "2026-11-04"
    return "2026-11-04" if seed == 1 else "2026-11-05"


def today_date(beta: float, seed: int, path: str) -> str:
    """The run dir holding today's sessions: #5565's hold-0 rerun on Binary, the SAM3 arm's twin on Region."""
    return "2026-11-05" if path == "Binary" else sam3_date(beta, seed)


def expf_cut(scores: np.ndarray, beta: float) -> float:
    """The cut keeping the top k that maximise ``(1 + b2) S_k / (k + b2 S)``, scores read as probabilities.

    ``S_k`` is the expected Goods in the top k and ``S`` in the corpus: F-beta's ratio of expectations, which
    needs no label. The cut is the k-th score (the set is every score at or above it).
    """
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    if not len(s) or s.sum() <= 0:
        return float("inf")
    b2 = beta * beta
    k = np.arange(1, len(s) + 1)
    e = (1 + b2) * np.cumsum(s) / (k + b2 * s.sum())
    return float(s[int(np.argmax(e))])


def sets(out: Path, seeds: tuple[int, ...]) -> None:
    sys.path.insert(0, str(CAL))
    import common  # noqa: PLC0415

    common.setup_env()
    import experiment_config as cfg  # noqa: PLC0415
    from _cells_io import load_medias  # noqa: PLC0415
    from sam3_scores import load_sam3  # noqa: PLC0415

    from vtscore.config import EMBEDDINGS_DIR  # noqa: PLC0415
    from vtscore.eval.labels import evaluable_pool, media_is_positive  # noqa: PLC0415
    from vtscore.eval.voting_iterations import _split_media_ids  # noqa: PLC0415

    medias = load_medias(EMBEDDINGS_DIR / cfg.text_pickle_name("coco_better", "siglip"))
    sam3 = load_sam3(SCORES, "typed")
    # The grid's 144 cells, off one run's main cell files.
    cats = sorted(set(pd.read_csv(f, usecols=["category"], nrows=1)["category"].iloc[0]
                      for f in glob.glob(f"{cells_dir('2026-11-04', 'sam3', 1.0)}/task_*.csv.gz") if "__" not in f))  # fmt: skip
    rows, calib = [], []
    for cat in cats:
        pool = evaluable_pool(medias, cat)
        s, _ = sam3(cat, pool)
        for seed in seeds:
            _, test = _split_media_ids(pool, cfg.SIM_FRACTION, np.random.RandomState(seed))
            sc = np.array([s[i] for i in test])
            y = np.array([1 if media_is_positive(pool[i], cat) else 0 for i in test])
            cuts = [(f"fixed:{c:g}", c) for c in GRID] + [(f"expf:{btag(b)}", expf_cut(sc, b)) for b in BETAS]
            for rule, c in cuts:
                keep = sc >= c
                rows.append({"category": cat, "seed": seed, "rule": rule, "cut": c, "k": int(keep.sum()),
                             "tp": int(y[keep].sum()), "n_pos": int(y.sum()), "n_test": len(y)})  # fmt: skip
            if seed == seeds[0]:
                calib.append(pd.DataFrame({"category": cat, "score": sc, "y": y}))
        common.log(f"{cat}: {len(pool)} medias")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "sam3_5565_sets.csv", index=False)
    c = pd.concat(calib)
    c["bin"] = pd.cut(c["score"], np.linspace(0, 1, 11), include_lowest=True)
    c.groupby("bin", observed=True).agg(n=("y", "size"), mean_score=("score", "mean"), good_rate=("y", "mean")).to_csv(
        out / "sam3_5565_calibration.csv"
    )
    common.log(f"{len(rows)} set rows over {len(cats)} cells x {len(seeds)} seeds")


def set_table(out: Path) -> pd.DataFrame:
    return pd.read_csv(out / "sam3_5565_sets.csv")


def set_f(st: pd.DataFrame, beta: float) -> pd.Series:
    """F-beta of each set row at *beta* (an empty set scores 0)."""
    b2 = beta * beta
    return (1 + b2) * st["tp"] / (st["k"] + b2 * st["n_pos"])


def check(out: Path, procs: int) -> None:
    """Every SAM3 session's recorded t=1 set against the rebuilt one: precision and recall must match."""
    sys.path.insert(0, str(HERE))
    from sam3_opening_5241 import load  # noqa: PLC0415

    st = set_table(out).set_index(["category", "seed", "rule"])
    runs = [(cells_dir(d, "sam3", b), "fixed:0.5") for d in ("2026-10-26", "2026-11-04", "2026-11-05") for b in BETAS]
    runs += [(cells_dir("2026-11-05", "sam3", 1.0, tag), line) for tag, line in VALIDATION.items()]
    res = []
    for d, line in runs:
        if not d.is_dir() or not glob.glob(f"{d}/task_*.csv*"):
            continue
        m = load(d, procs)
        first = m[(m["gmm_variant"] == "") & (m["t"] == 1)].drop_duplicates(["category", "embedder", "seed"])
        for _, r in first.iterrows():
            ref = st.loc[(r["category"], int(r["seed"]), line)]
            p = ref["tp"] / ref["k"] if ref["k"] else np.nan
            rec = ref["tp"] / ref["n_pos"]
            dp = 0.0 if (np.isnan(p) and np.isnan(r["precision"])) else abs(p - float(r["precision"]))
            res.append({"run": d.parent.parent.name, "path": PATHS[r["embedder"]], "category": r["category"],
                        "seed": int(r["seed"]), "dp": dp, "dr": abs(rec - float(r["recall"]))})  # fmt: skip
    res = pd.DataFrame(res)
    res.to_csv(out / "sam3_5565_check.csv", index=False)
    summary = res.groupby(["run", "path"]).agg(cells=("dp", "size"), max_dp=("dp", "max"), max_dr=("dr", "max"))
    print(summary.to_string())
    bad = res[(res["dp"] > 1e-5) | (res["dr"] > 1e-5)]
    if len(bad):
        raise SystemExit(
            f"{len(bad)} cells' rebuilt set differs from the session's (e.g. {bad.head(3).to_dict('records')})"
        )
    print(f"all {len(res)} sessions' 0.5/0.3 sets match the rebuilt Test half")


def tuned_cuts(st: pd.DataFrame) -> dict[float, float]:
    """Per preset, the grid cut with the best mean set F-beta over its tuning seeds' cells."""
    fixed = st[st["rule"].str.startswith("fixed:")]
    out = {}
    for b in BETAS:
        g = fixed[fixed["seed"].isin(TUNE_SEEDS[b])].assign(f=lambda x, b=b: set_f(x, b))
        out[b] = float(g.groupby("cut")["f"].mean().idxmax())
    return out


def price(out: Path, procs: int) -> None:
    sys.path.insert(0, str(HERE))
    from early_check_4664 import area, se_over_classes  # noqa: PLC0415
    from sam3_confirm_5311 import goods_by_click  # noqa: PLC0415
    from sam3_opening_5241 import BETA, HOLD_TAG, T, curves, load  # noqa: PLC0415

    st = set_table(out)
    tuned = tuned_cuts(st)
    pd.DataFrame([{"beta": b, "tuned_cut": c, "tune_seeds": " ".join(map(str, TUNE_SEEDS[b]))}
                  for b, c in tuned.items()]).to_csv(out / "sam3_5565_tuned_cuts.csv", index=False)  # fmt: skip
    cache: dict[Path, pd.DataFrame] = {}

    def rows_of(d: Path) -> pd.DataFrame:
        if d not in cache:
            cache[d] = load(d, procs)
        return cache[d]

    table, per_vote, set_rows, per_cell = [], [], [], []
    for b in BETAS:
        BETA["beta"] = b
        rules = {"0.5": "fixed:0.5", "expf": f"expf:{btag(b)}", "tuned": f"fixed:{tuned[b]:g}"}
        sf = st.assign(f=set_f(st, b)).set_index(["category", "seed", "rule"])
        for s in SEEDS:
            sdir = cells_dir(sam3_date(b, s), "sam3", b)
            if not sdir.is_dir():
                continue
            sam3 = rows_of(sdir)
            sam3 = sam3[sam3["seed"] == s]
            if sam3.empty:
                continue
            q_head = curves(sam3, HOLD_TAG)
            n_goods = goods_by_click(sdir)
            for emb, path in PATHS.items():
                tdir = cells_dir(today_date(b, s, path), "sig", b)
                sig = rows_of(tdir)
                today = curves(sig[(sig["seed"] == s) & (sig["embedder"] == emb)], "")
                keys = sorted(k for k in set(today) & set(q_head) & set(n_goods) if k[1] == emb and k[2] == s)
                if not keys:
                    continue
                cats = [k[0] for k in keys]
                cls = np.array([c.rpartition("@")[0] for c in cats])
                t_f = np.vstack([today[k][0] for k in keys])
                head = np.vstack([q_head[k][0] for k in keys])
                good = np.vstack([n_goods[k] for k in keys])
                a_today = area(t_f, 1, T)
                routes = {}
                for name, rule in rules.items():
                    q = np.array([sf.loc[(c, s, rule), "f"] for c in cats])
                    set_rows += [{"beta": b, "seed": s, "path": path, "set": name, "rule": rule,
                                  "precision": float(np.nanmean([sf.loc[(c, s, rule), "tp"] / sf.loc[(c, s, rule), "k"]
                                                                 if sf.loc[(c, s, rule), "k"] else np.nan for c in cats])),
                                  "recall": float(np.mean([sf.loc[(c, s, rule), "tp"] / sf.loc[(c, s, rule), "n_pos"]
                                                           for c in cats])),
                                  "set_F": float(np.mean(q))}]  # fmt: skip
                    for n in (*GOODS, None):
                        f = np.repeat(q[:, None], T + 1, axis=1) if n is None else np.where(
                            np.isfinite(head) & (good >= n), head, q[:, None])  # fmt: skip
                        routes[(name, "hold" if n is None else f"goods>={n}")] = f
                for (name, gate), f in routes.items():
                    d = area(f, 1, T) - a_today
                    table.append({"beta": b, "seed": s, "path": path, "set": name, "gate": gate, "cells": len(keys),
                                  "F_1_150": float(np.mean(area(f, 1, T))), "today_F_1_150": float(np.mean(a_today)),
                                  "vs_today": float(np.mean(d)), "se": se_over_classes(d, cls)})  # fmt: skip
                pc = {"today": t_f}
                pc["one count: 0.5, 20th Good"] = routes[("0.5", "goods>=20")]
                n2 = PRESET_COUNT[b]
                pc["option 2: 0.5, count per preset"] = routes[("0.5", "hold" if n2 is None else f"goods>={n2}")]
                pc["option 1a: expf cut, 20th Good"] = routes[("expf", "goods>=20")]
                pc["option 1b: tuned cut, 20th Good"] = routes[("tuned", "goods>=20")]
                pc[COMBINED] = routes[("tuned", COMBINED_GATE[b])]
                a_pc = {label: area(f, 1, T) for label, f in pc.items()}
                for k, key in enumerate(keys):
                    per_cell.append({"beta": b, "seed": s, "path": path, "category": key[0],
                                     **{label: float(v[k]) for label, v in a_pc.items()}})  # fmt: skip
                for label, f in pc.items():
                    for t in range(1, T + 1):
                        per_vote.append({"beta": b, "seed": s, "path": path, "route": label, "t": t,
                                         "F": float(np.nanmean(f[:, t])), "cells": len(keys)})  # fmt: skip
    pd.DataFrame(table).to_csv(out / "sam3_5565_routes.csv", index=False)
    pd.DataFrame(per_vote).to_csv(out / "sam3_5565_per_vote.csv", index=False)
    pd.DataFrame(set_rows).to_csv(out / "sam3_5565_set_pr.csv", index=False)
    pd.DataFrame(per_cell).to_csv(out / "sam3_5565_cells.csv", index=False)
    summary(pd.DataFrame(table), out)
    figure(pd.DataFrame(per_vote), out / "sam3_5565_by_beta.webp")
    print(f"tuned cuts: {tuned}")
    print(f"{len(table)} route rows -> {out / 'sam3_5565_routes.csv'}")


def summary(table: pd.DataFrame, out: Path) -> None:
    """The headline routes on seed 2 (out of sample for every choice made), with each preset's best gate."""
    picks = {"one count (0.5, 20th Good)": lambda b: ("0.5", "goods>=20"),
             "option 2 (0.5, count per preset)": lambda b: ("0.5", "hold" if PRESET_COUNT[b] is None
                                                            else f"goods>={PRESET_COUNT[b]}"),
             "option 1a (expf cut, 20th Good)": lambda b: ("expf", "goods>=20"),
             "option 1b (tuned cut, 20th Good)": lambda b: ("tuned", "goods>=20"),
             COMBINED: lambda b: ("tuned", COMBINED_GATE[b])}  # fmt: skip
    rows = []
    s2 = table[table["seed"] == 2]
    for b in BETAS:
        for path in PATHS.values():
            g = s2[(s2["beta"] == b) & (s2["path"] == path)].set_index(["set", "gate"])
            for name, f in picks.items():
                r = g.loc[f(b)]
                rows.append({"beta": b, "path": path, "route": name, "set": f(b)[0], "gate": f(b)[1],
                             "vs_today": r["vs_today"], "se": r["se"], "F_1_150": r["F_1_150"],
                             "today_F_1_150": r["today_F_1_150"], "cells": r["cells"]})  # fmt: skip
            for st in ("0.5", "tuned"):
                best = g.loc[st]["vs_today"].idxmax()
                rows.append({"beta": b, "path": path, "route": f"best gate on seed 2, {st} set", "set": st,
                             "gate": best, "vs_today": g.loc[(st, best), "vs_today"], "se": g.loc[(st, best), "se"],
                             "F_1_150": g.loc[(st, best), "F_1_150"],
                             "today_F_1_150": g.loc[(st, best), "today_F_1_150"], "cells": g.loc[(st, best), "cells"]})  # fmt: skip
    pd.DataFrame(rows).to_csv(out / "sam3_5565_summary.csv", index=False)


def figure(pv: pd.DataFrame, path: Path) -> None:
    """F-beta shown per vote on seed 2, each path at each preset: today and the routes."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sys.path.insert(0, str(HERE))
    from early_check_4664 import MUTED, _axes_style  # noqa: PLC0415

    style = {"today": ("SigLIP opening (today)", "#8a8985", 2.0),
             "option 2: 0.5, count per preset": ("SAM3 at 0.5, count per preset (option 2)", "#2a78d6", 1.6),
             "option 1b: tuned cut, 20th Good": ("SAM3 at a cut per preset, 20th Good (option 1b)", "#d65f2a", 1.4),
             COMBINED: ("SAM3 at a cut per preset, 20th Good, held at 4 (both)", "#3a9a5b", 2.4)}  # fmt: skip
    pv = pv[pv["seed"] == 2]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, b in enumerate(BETAS):
        for i, p in enumerate(PATHS.values()):
            ax = axes[i, j]
            for route, (label, color, lw) in style.items():
                g = pv[(pv["beta"] == b) & (pv["path"] == p) & (pv["route"] == route)]
                ax.plot(g["t"], g["F"], color=color, lw=lw, label=label)
            ax.set_title(f"{p}, beta {b:g} (seed 2)", loc="left", fontsize=11)
            if i == 1:
                ax.set_xlabel("votes", color=MUTED)
            if j == 0:
                ax.set_ylabel("F-beta shown (mean over cells)", color=MUTED)
            _axes_style(ax)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("sets", "check", "price"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "sets":
        sets(a.out, tuple(a.seeds))
    elif a.stage == "check":
        check(a.out, a.procs)
    else:
        price(a.out, a.procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
