#!/usr/bin/env python3
"""#4582: re-score the Cost-era studies' cells at F-beta 1/4, 1 and 4, no new runs.

Every Cost-era verdict was a paired Δcost at Inclusion 0 (FPR + FNR).  Each
per-step row also carries the withheld half's ``precision`` and ``recall`` at
that step's threshold, so the returned set's F-beta at the app's three presets
is a column away.  This script re-reads what survives of each study, checks it
reproduces the report's own cost delta (so the pairing is the report's), and
then swaps cost for F-beta.

What survives, and where it is read from:

* **cells** (``_cells_io.load_arm``): #4184's 0.44% ladder and the #3826 A/B.
* **committed viewer pages**: the cells of #3287, #3314, #4114 and #4219 were
  deleted (``scratch-deletions.md``), but each ``viewer.html`` keeps the raw
  per-run series of every metric at full click resolution, quantised to 1/1000
  (``viewer._encode``, ``RUNS_SCALE``).  Precision and recall are among them, so
  F-beta is recoverable to about 0.001 per run.
* **per-sort part tables**: #3826's offline study scored each candidate line
  on 1,120 labelled text sorts; ``tp``, ``fp``, ``n_pos`` and ``n`` per sort are
  enough for every metric here.

**The threshold is that era's line, not today's.**  A Cost-era row's precision
and recall are measured above the line that arm drew, so every F-beta here is
"F-beta above the line that arm drew": what the arm decision changed, not the
objective as a balance-era review reads it (the labels line, #4452).

    python rescore_fbeta_4582.py --out OUTDIR [--studies 3287,3314,...]
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "docs" / "experiments"
SCRATCH = Path("/expscratch/sgreenberg")

#: The app's balance presets (#4448), as ``analyze_progression_4184.FBETAS``.
FBETAS: dict[str, float] = {"f025": 0.25, "f1": 1.0, "f4": 4.0}
#: Every metric a table reports.  Cost first: it is the reproduction check.
METRICS = ("cost", *FBETAS, "average_precision")

RUN = ["dataset", "embedder", "category", "seed"]
N_BOOT = 2000


def fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """F-beta from a returned set's precision and recall; 0 where it returned nothing right.

    The same rule as ``analyze_progression_4184.fbeta``: a NaN precision (nothing
    returned) scores 0, as does a returned set with no true positive.
    """
    p = np.nan_to_num(np.asarray(precision, dtype=float))
    r = np.asarray(recall, dtype=float)
    b2 = beta * beta
    den = b2 * p + r
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, (1 + b2) * p * r / den, 0.0)
    return np.where(np.isnan(r), np.nan, out)


# ---------------------------------------------------------------------------
# Sources.  Each returns {metric: DataFrame(index=RUN + ["arm"], columns=t)},
# filled: the click-0 text-sort anchor carries until the run's first row, and
# each row carries until the next (``analyze_progression_4184.filled_matrix``).


def _decode(enc: dict) -> np.ndarray:
    """Inverse of ``viewer._encode`` (mirrors ``selftest_viewer._decode``)."""
    shape = enc["shape"]
    n_t = shape[-1]
    rows = int(np.prod(shape[:-1]))
    deltas = np.frombuffer(gzip.decompress(base64.b64decode(enc["v"])), dtype=np.int16).reshape(rows, n_t)
    mask = np.unpackbits(np.frombuffer(gzip.decompress(base64.b64decode(enc["m"])), dtype=np.uint8))
    vals = np.cumsum(deltas.astype(np.int64), axis=1) / enc["scale"]
    valid = mask[: rows * n_t].reshape(rows, n_t).astype(bool)
    return np.where(valid, vals, np.nan).reshape(shape)


def viewer_payload(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    m = re.search(r'id="payload"[^>]*>(.*?)</script>', html, re.S) or re.search(
        r'type="application/json">(.*?)</script>', html, re.S
    )
    if not m:
        raise SystemExit(f"{path}: no payload")
    return json.loads(m.group(1))


def from_viewer(path: Path, rename: dict[str, str] | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Every run's series from a committed ``viewer.html``, filled, with F-beta derived."""
    d = viewer_payload(path)
    runs = d.get("runs")
    if not runs:
        raise SystemExit(f"{path}: no per-run payload")
    t = list(runs["t"])
    if t != list(range(t[0], t[-1] + 1)):
        raise SystemExit(f"{path}: per-run grid is thinned ({len(t)} clicks); F-beta per click is not recoverable")
    vals = _decode(runs["values"])  # (run, metric, click)
    keys = [m["key"] for m in d["metrics"]]
    need = {"cost", "precision", "recall", "average_precision"}
    if not need <= set(keys):
        raise SystemExit(f"{path}: metrics {keys} lack {sorted(need - set(keys))}")
    rename = rename or {}
    idx = pd.MultiIndex.from_tuples(
        [
            (*d["groups"][gi], int(d["seeds"][si]), rename.get(d["arms"][ai], d["arms"][ai]))
            for gi, ai, si in runs["index"]
        ],
        names=[*RUN, "arm"],
    )
    raw = {k: pd.DataFrame(vals[:, keys.index(k), :], index=idx, columns=t) for k in need}
    shown = raw["cost"].notna()
    shown[t[0]] = False
    # Fill along clicks: t=0 holds the text-sort anchor, so a forward fill is
    # exactly the anchor-until-first-row-then-carry rule.  A shown row whose
    # precision is NaN returned nothing; it must not inherit the previous
    # row's precision, so it is marked through the fill and restored.
    filled = {}
    for k, v in raw.items():
        if k == "precision":
            v = v.where(~(shown & v.isna()), -1.0).ffill(axis=1)
            filled[k] = v.where(v >= 0, np.nan)
        else:
            filled[k] = v.ffill(axis=1)
    out = {"cost": filled["cost"], "average_precision": filled["average_precision"]}
    for tag, b in FBETAS.items():
        out[tag] = pd.DataFrame(
            fbeta(filled["precision"].to_numpy(), filled["recall"].to_numpy(), b), index=idx, columns=t
        )
    out["__precision"], out["__recall"] = filled["precision"], filled["recall"]
    prov = {
        "source": str(path.relative_to(REPO)),
        "runs": int(len(idx)),
        "arms": sorted(set(idx.get_level_values("arm"))),
        "clicks": f"{t[0]}..{t[-1]}",
        "runs_scale": runs["values"]["scale"],
    }
    out["__shown"] = shown
    return out, prov


def from_cells(
    arms: Sequence[tuple[Path, str]],
    baseline_csv: Path | None,
    horizon: int,
    t_col: str = "t",
    app_visible: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Every run's series from surviving cells, filled with the text-sort anchor."""
    import _cells_io  # noqa: PLC0415

    # No baseline: no click-0 anchor, which only the ``filled`` window mode reads.
    acols = ["text_cost", "text_AP", "text_precision", "text_recall"]
    base = _cells_io.legacy_datasets(pd.read_csv(baseline_csv)) if baseline_csv else pd.DataFrame(columns=RUN + acols)
    anchors = base.groupby(RUN)[acols].mean()
    mats: dict[str, list[pd.DataFrame]] = {k: [] for k in ("cost", "average_precision", "precision", "recall")}
    shown_all = []
    prov: dict = {"arms": {}}
    for d, label in arms:
        frame, p = _cells_io.load_arm(d)
        prov["arms"][label] = {"dir": str(d), "load": _cells_io.describe_load(p)}
        if t_col == "n_votes" and "n_votes" not in frame.columns:
            frame = frame.assign(n_votes=frame["n_good"] + frame["n_bad"])  # as ``analyze_ab``
        if app_visible and "app_trained" in frame.columns:
            frame = frame[frame["app_trained"].fillna(0).astype(int) == 1]
        frame = frame[frame[t_col] <= horizon]
        cells = base[RUN].drop_duplicates()
        seen = frame[RUN].drop_duplicates()
        lost = len(p.get("unreadable") or []) + len(p.get("zero_byte") or [])
        prov["arms"][label]["cells"] = int(len(seen))
        prov["arms"][label]["lost"] = lost
        grid = pd.concat([cells, seen]).drop_duplicates()
        idx = pd.MultiIndex.from_frame(grid[RUN])
        anchor_cols = {
            "cost": "text_cost",
            "average_precision": "text_AP",
            "precision": "text_precision",
            "recall": "text_recall",
        }
        wide_cost = frame.pivot_table(index=RUN, columns=t_col, values="cost", aggfunc="mean")
        wide_cost = wide_cost.reindex(index=idx, columns=range(horizon + 1))
        shown = wide_cost.notna()
        shown[0] = False
        for k, a in anchor_cols.items():
            wide = frame.pivot_table(index=RUN, columns=t_col, values=k, aggfunc="mean")
            wide = wide.reindex(index=idx, columns=range(horizon + 1))
            # A precision of NaN on a shown row is "returned nothing"; keep it
            # distinct from "no row" by marking it, filling, and restoring.
            if k == "precision":
                wide = wide.where(~(shown & wide.isna()), -1.0)
            wide[0] = anchors.reindex(idx)[a].to_numpy()
            wide = wide.ffill(axis=1)
            if k == "precision":
                wide = wide.where(wide >= 0, np.nan)
            wide.index = pd.MultiIndex.from_tuples([(*i, label) for i in idx], names=[*RUN, "arm"])
            mats[k].append(wide)
        shown.index = pd.MultiIndex.from_tuples([(*i, label) for i in idx], names=[*RUN, "arm"])
        shown_all.append(shown)
    full = {k: pd.concat(v) for k, v in mats.items()}
    out = {"cost": full["cost"], "average_precision": full["average_precision"]}
    for tag, b in FBETAS.items():
        out[tag] = pd.DataFrame(
            fbeta(full["precision"].to_numpy(), full["recall"].to_numpy(), b),
            index=full["precision"].index,
            columns=full["precision"].columns,
        )
    out["__shown"] = pd.concat(shown_all)
    out["__precision"], out["__recall"] = full["precision"], full["recall"]
    return out, prov


# ---------------------------------------------------------------------------
# Paired reads.


@dataclass
class Window:
    name: str
    lo: int
    hi: int


@dataclass
class Study:
    key: str
    title: str
    report: str
    load: Callable[[], tuple[dict[str, pd.DataFrame], dict]]
    incumbent: str
    #: The arm the Cost-era decision picked (may be the incumbent).
    cost_winner: str
    windows: list[Window]
    #: Strata a report tabled apart: name -> function of the run index frame.
    strata: dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict)
    #: Column of the run index the SE clusters on; None = bootstrap over runs.
    cluster: str | None = None
    #: Restrict to runs the report paired (e.g. drop never-trained cells).
    survivors_only: bool = False
    #: How a run's window mean is taken, matching the report's analyzer:
    #: ``filled`` - every click, text-sort anchor before the first row
    #: (``analyze_progression_4184``); ``common`` - only the clicks at which every
    #: arm of the run has a row (``analyze_stage_b.aulc``); ``own`` - each arm
    #: over its own rows in the window (``analyze_calfrac.cell_means``).
    window_mode: str = "filled"
    arms: list[str] | None = None
    note: str = ""
    #: Also pair each arm against the one before it (a ladder's steps).
    steps: bool = False
    #: The report's own cost deltas, for the reproduction check:
    #: (arm, stratum, window) -> (delta, se).
    reported: dict[tuple[str, str, str], tuple[float, float]] = field(default_factory=dict)


def _se(d: np.ndarray, clusters: np.ndarray | None, rng: np.random.Generator) -> float:
    """Bootstrap over runs, or ``analyze_stage_b.paired``'s SE over cluster means."""
    if d.size < 2:
        return float("nan")
    if clusters is None:
        return float(rng.choice(d, size=(N_BOOT, d.size), replace=True).mean(axis=1).std(ddof=1))
    g = pd.Series(d).groupby(clusters).mean().to_numpy()
    return float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")


def paired_table(study: Study, mats: dict[str, pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    """``arm − incumbent`` per (stratum, window, metric), paired on the run."""
    shown = mats["__shown"]
    arms = study.arms or sorted(set(mats["cost"].index.get_level_values("arm")))
    rows = []
    # The clicks at which every arm of a run has a row (``common`` mode).
    common_mask = None
    for arm in arms:
        sh = shown.xs(arm, level="arm")
        common_mask = sh if common_mask is None else (common_mask & sh.reindex(common_mask.index).fillna(False))
    for metric in METRICS:
        m = mats[metric]
        inc = m.xs(study.incumbent, level="arm")
        inc_shown = shown.xs(study.incumbent, level="arm").any(axis=1)
        for arm in arms:
            if arm == study.incumbent:
                continue
            a = m.xs(arm, level="arm")
            common_idx = a.index.intersection(inc.index)
            if study.survivors_only:
                a_shown = shown.xs(arm, level="arm").any(axis=1)
                keep = a_shown.reindex(common_idx).fillna(False) & inc_shown.reindex(common_idx).fillna(False)
                common_idx = common_idx[keep.to_numpy()]
            ix = common_idx.to_frame(index=False)
            strata = {"all": pd.Series(True, index=ix.index)}
            strata.update({name: fn(ix) for name, fn in study.strata.items()})
            for w in study.windows:
                cols = [c for c in m.columns if w.lo <= c <= w.hi]
                av, iv = a.loc[common_idx, cols], inc.loc[common_idx, cols]
                if study.window_mode == "common":
                    mask = common_mask.loc[common_idx, cols].to_numpy()
                    av, iv = av.where(mask), iv.where(mask)
                elif study.window_mode == "own":
                    av = av.where(shown.xs(arm, level="arm").loc[common_idx, cols].to_numpy())
                    iv = iv.where(shown.xs(study.incumbent, level="arm").loc[common_idx, cols].to_numpy())
                diff = (av.mean(axis=1) - iv.mean(axis=1)).to_numpy()
                for sname, smask in strata.items():
                    if isinstance(smask, pd.Series) and smask.dtype != bool:
                        groups = smask.dropna().unique()
                        parts = [(f"{sname}={g}", (smask == g).to_numpy()) for g in sorted(groups)]
                    else:
                        parts = [(sname, np.asarray(smask, dtype=bool))]
                    for label, mask in parts:
                        dd = diff[mask]
                        ok = np.isfinite(dd)
                        dd = dd[ok]
                        cl = ix.loc[mask, study.cluster].to_numpy()[ok] if study.cluster else None
                        rows.append(
                            {
                                "study": study.key,
                                "arm": arm,
                                "vs": study.incumbent,
                                "stratum": label,
                                "window": w.name,
                                "metric": metric,
                                "n": int(dd.size),
                                "delta": float(dd.mean()) if dd.size else np.nan,
                                "se": _se(dd, cl, rng),
                            }
                        )
    out = pd.DataFrame(rows)
    out["resolved"] = out["delta"].abs() > 2 * out["se"]
    return out


def pooled_iv(t: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Inverse-variance pool across windows (#3287's headline), per *by*."""
    rows = []
    for key, g in t.groupby(list(by), dropna=False):
        w = 1.0 / g["se"].clip(lower=1e-9) ** 2
        rows.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)),
                "window": "pooled",
                "delta": float((w * g["delta"]).sum() / w.sum()),
                "se": float(np.sqrt(1.0 / w.sum())),
                "n": int(g["n"].max()),
            }
        )
    out = pd.DataFrame(rows)
    out["resolved"] = out["delta"].abs() > 2 * out["se"]
    return out


# ---------------------------------------------------------------------------
# The studies.

BANDS_3287 = [
    Window("early 1-25", 1, 25),
    Window("mid 26-60", 26, 60),
    Window("late 61-100", 61, 100),
    Window("deep 101-150", 101, 150),
]
WHOLE = [Window("clicks 1-150", 1, 150)]


def _concat_viewers(pages: Sequence[Path]) -> Callable[[], tuple[dict[str, pd.DataFrame], dict]]:
    def load() -> tuple[dict[str, pd.DataFrame], dict]:
        parts, provs = [], []
        for p in pages:
            m, prov = from_viewer(p)
            parts.append(m)
            provs.append(prov)
        out = {k: pd.concat([pt[k] for pt in parts]) for k in parts[0]}
        for k, v in out.items():
            if v.index.duplicated().any():
                raise SystemExit(f"duplicate runs across pages for {k}")
        return out, {"pages": provs}

    return load


def _size_band(ix: pd.DataFrame) -> pd.Series:
    return ix["category"].astype(str).str.rsplit("@", n=1).str[-1]


def _geometry(ix: pd.DataFrame) -> pd.Series:
    return ix["embedder"].astype(str)


def studies() -> dict[str, Study]:
    s: dict[str, Study] = {}
    cf = EXP / "2026-08-27-calibration-fraction-3287"
    s["3287"] = Study(
        key="3287",
        title="#3287 calibration fraction",
        report="2026-08-27-calibration-fraction-3287/REPORT.md",
        load=_concat_viewers(
            [
                cf / "viewer.html",
                cf / "clip" / "viewer.html",
                cf / "clip_l" / "viewer.html",
                cf / "siglip2l" / "viewer.html",
            ]
        ),
        incumbent="0.50",
        cost_winner="0.30",
        windows=BANDS_3287,
        window_mode="own",
        strata={"geometry": _geometry},
        note="Shipped since #3287: 0.3 for single-vector spaces, 0.5 for `dinov3_patch` (`PRODUCTION_SPLIT_BY_SPACE`).",
    )
    s["3314"] = Study(
        key="3314",
        title="#3314 calibration fold count",
        report="2026-08-28-calibration-fold-count-3310/REPORT.md",
        load=_concat_viewers([EXP / "2026-08-28-calibration-fold-count-3310" / "viewer.html"]),
        incumbent="K=2",
        cost_winner="K=2",
        windows=BANDS_3287,
        window_mode="own",
        strata={"geometry": _geometry},
        note="Counterfactual K re-cut on K=2's trajectory; K=2 kept on wall-clock, not on benefit.",
    )
    s["4114"] = Study(
        key="4114",
        title="#4114 converged logistic head (binary)",
        report="2026-09-27-logreg-head-4114/REPORT.md",
        load=_concat_viewers([EXP / "2026-09-27-logreg-head-4114" / "viewer.html"]),
        incumbent="svm",
        cost_winner="lrconv",
        windows=WHOLE,
        strata={"band": _size_band},
        cluster="category",
        survivors_only=True,
        window_mode="common",
        reported={
            ("lrconv", "all", "clicks 1-150"): (-0.0098, 0.0027),
            ("linear", "all", "clicks 1-150"): (0.026, 0.0032),
        },
    )
    hs = EXP / "2026-09-28-head-switch-4219"
    s["4219"] = Study(
        key="4219",
        title="#4219 head switch (binary)",
        report="2026-09-28-head-switch-4219/REPORT.md",
        load=_concat_viewers([hs / "viewer.html"]),
        incumbent="svm",
        cost_winner="svm",
        windows=WHOLE,
        strata={"band": _size_band},
        cluster="category",
        survivors_only=True,
        window_mode="common",
        reported={
            ("lrconv", "all", "clicks 1-150"): (-0.0099, 0.0028),
            ("svmc01", "all", "clicks 1-150"): (0.0057, 0.0026),
        },
        note="The no-switch decision rested on the region run (below); binary favoured lrconv on cost.",
    )
    s["4219r"] = Study(
        key="4219r",
        title="#4219 head switch (region voting)",
        report="2026-09-28-head-switch-4219/REPORT.md",
        load=_concat_viewers([hs / "region" / "viewer.html"]),
        incumbent="svm",
        cost_winner="svm",
        windows=WHOLE,
        cluster="category",
        survivors_only=True,
        window_mode="common",
        reported={("lrconv", "all", "clicks 1-150"): (-0.0010, 0.0039)},
    )
    prog = SCRATCH / "progression-4184"
    rungs = ["r1_xcal", "r2_gmm", "r3_blend", "r4_rawmean", "r5_anchored", "r6_split70", "r7_acq4"]
    s["4184"] = Study(
        key="4184",
        title="#4184 calibration ladder, 0.44% pool",
        report="2026-09-25-progression-4184/REPORT.md",
        load=lambda: from_cells([(prog / r / "results", r) for r in rungs], prog / "text_baseline.csv", 150),
        incumbent="r1_xcal",
        cost_winner="r4_rawmean",
        windows=[Window("clicks 1-150", 1, 150), Window("click 150", 150, 150)],
        arms=rungs,
        steps=True,
        note="The 5% pool's cells were deleted on 2026-10-01; only the 0.44% (natural) pool survives.",
    )
    ab = SCRATCH / "textcut-ab-3826"
    s["3826ab"] = Study(
        key="3826ab",
        title="#3826 text-sort line A/B (trajectory)",
        report="2026-09-23-text-cut-ab-3826/REPORT.md",
        load=lambda: from_cells(
            [
                (ab / "ab_gmm_midpoint" / "results", "gmm_midpoint"),
                (ab / "ab_guarded_tail" / "results", "guarded_tail"),
            ],
            None,
            400,
            t_col="n_votes",
        ),
        incumbent="gmm_midpoint",
        cost_winner="gmm_midpoint",
        # ``analyze_ab.WINDOWS`` on vote counts, scope app_visible, each arm's own rows.
        windows=[Window("all steps", 2, 400), Window("votes 6-20", 6, 20), Window("votes 21+", 21, 400)],
        window_mode="own",
        reported={("guarded_tail", "all", "all steps"): (0.016, 0.0048)},
    )
    return s


def text_cut_offline(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """#3826's offline study: each candidate line on 1,120 labelled text sorts.

    Read as ``analyze_3826.quality_frame`` / ``paired`` read it: the full-frame
    rows, paired on (dataset, embedder, category), SE clustered on (dataset,
    category) because four embedders see one query.
    """
    keys = ["dataset", "embedder", "category"]
    parts = sorted((SCRATCH / "textcut-3826" / "analysis" / "parts").glob("*_cuts.csv"))
    parts = [p for p in parts if not p.name.startswith("old_")]
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in parts], ignore_index=True)
    df = df[df["frame"] == "full"].copy()
    df["cost"] = df["fp"] / (df["n"] - df["n_pos"]) + (1 - df["tp"] / df["n_pos"])
    prec = np.where(df["n_adm"] > 0, df["tp"] / df["n_adm"].clip(lower=1), np.nan)
    rec = (df["tp"] / df["n_pos"]).to_numpy()
    for tag, b in FBETAS.items():
        df[tag] = fbeta(prec, rec, b)
    inc, alt = "gmm_shipped", "gmm_guarded_z3"
    a = df[df["rule"] == alt].set_index(keys)
    b_ = df[df["rule"] == inc].set_index(keys)
    common_idx = a.index.intersection(b_.index)
    cl = common_idx.get_level_values("dataset") + "|" + common_idx.get_level_values("category")
    rows = []
    for metric in ("cost", *FBETAS):
        d = pd.Series((a.loc[common_idx, metric] - b_.loc[common_idx, metric]).to_numpy(float))
        g = d.groupby(np.asarray(cl))
        sums, k = g.sum(), g.ngroups
        resid = sums - g.size() * d.mean()
        se = float(np.sqrt(k / (k - 1) * (resid**2).sum()) / d.size)
        rows.append(
            {
                "study": "3826",
                "arm": alt,
                "vs": inc,
                "stratum": "all",
                "window": "one sort",
                "metric": metric,
                "n": int(d.size),
                "delta": float(d.mean()),
                "se": se,
            }
        )
    out = pd.DataFrame(rows)
    out["resolved"] = out["delta"].abs() > 2 * out["se"]
    out["cost_winner"] = inc
    print(f"[3826] reproduce Δcost {alt}: report +0.0528 ± 0.0114, here {rows[0]['delta']:+.4f} ± {rows[0]['se']:.4f}")
    return out, {"parts": len(parts), "sorts_paired": int(len(common_idx))}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--studies", default="", help="comma list of study keys; default all")
    ap.add_argument("--seed", type=int, default=4582)
    args = ap.parse_args(list(argv) if argv is not None else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_studies = studies()
    want = [k for k in args.studies.split(",") if k] or [*all_studies, "3826"]
    tables, provs = [], {}
    for key in want:
        if key == "3826":
            t, prov = text_cut_offline(rng)
            tables.append(t)
            provs[key] = prov
            print(t.to_string(index=False))
            continue
        st = all_studies[key]
        mats, prov = st.load()
        provs[key] = prov
        t = paired_table(st, mats, rng)
        if st.steps and st.arms:
            from dataclasses import replace  # noqa: PLC0415

            steps = [
                paired_table(replace(st, incumbent=prev, arms=[prev, cur]), mats, rng)
                for prev, cur in zip(st.arms, st.arms[1:], strict=False)
                if prev != st.incumbent
            ]
            t = pd.concat([t, *steps], ignore_index=True)
        if key in ("3287", "3314"):
            t = pd.concat([t, pooled_iv(t, ["study", "arm", "vs", "stratum", "metric"])], ignore_index=True)
        t["cost_winner"] = st.cost_winner
        tables.append(t)
        # Levels, for the report's context: mean per arm over each window.
        lv = []
        for metric in METRICS:
            m = mats[metric]
            for w in st.windows:
                cols = [c for c in m.columns if w.lo <= c <= w.hi]
                per = m[cols].mean(axis=1).groupby(level="arm").mean()
                for arm, v in per.items():
                    lv.append({"study": key, "arm": arm, "window": w.name, "metric": metric, "mean": float(v)})
        pd.DataFrame(lv).to_csv(out / f"levels_{key}.csv", index=False, float_format="%.5g")
        # Per-click means (the figures) and each run's last click (the examples).
        names = {"__precision": "precision", "__recall": "recall"}
        cur = {names.get(k, k): mats[k].groupby(level="arm").mean() for k in (*METRICS, "__precision", "__recall")}
        pd.concat(cur, names=["metric"]).to_csv(out / f"curves_{key}.csv", float_format="%.4g")
        last = max(c for c in mats["cost"].columns if c <= max(w.hi for w in st.windows))
        last = min(last, mats["cost"].columns.max())
        pd.DataFrame({names.get(k, k): mats[k][last] for k in (*METRICS, "__precision", "__recall")}).assign(
            t=last
        ).to_csv(out / f"runs_{key}.csv", float_format="%.4g")
        for (arm, stratum, window), (dl, se) in st.reported.items():
            got = t[(t.arm == arm) & (t.stratum == stratum) & (t.window == window) & (t.metric == "cost")]
            g = got.iloc[0] if len(got) else None
            print(
                f"[{key}] reproduce Δcost {arm} {stratum} {window}: report {dl:+.4f} ± {se:.4f}, "
                f"here {g.delta:+.4f} ± {g.se:.4f} (n={g.n})"
                if g is not None
                else f"[{key}] reproduce: no row for {arm}/{stratum}/{window}"
            )
        print(t[t.stratum == "all"].to_string(index=False))
    res = pd.concat(tables, ignore_index=True)
    res.to_csv(out / "paired.csv", index=False, float_format="%.5g")
    (out / "provenance.json").write_text(json.dumps(provs, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
