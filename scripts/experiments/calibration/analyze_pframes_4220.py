#!/usr/bin/env python3
"""#4220: can a precision-floor promise be kept from the app's own votes?

The #4223 ruling makes the cut's job "return as much as possible while at
least X of it is right", with X the user's.  A live rule has to *estimate*
precision above a cut from what the app has - its votes and its pool - and
cut where the estimate reaches X.  This prices candidate estimators on the
precision frames the harness records (``CALIB_PFRAME_STEPS``), against the
truth each frame also carries (the test half's labels).

**Estimators** give P(positive | score) for every item of a target corpus;
estimated precision of the top k is the running mean of that over the corpus
sorted by score, and the cut is the largest k whose estimate is >= X:

* ``insample`` - fitted on the votes' final-model scores (in-sample, so
  overconfident: the model trained on exactly those items);
* ``foldrank`` - fitted on the calibration folds' held-out vote scores, each
  mapped to its percentile in its own fold model's haystack, and applied to
  the corpus through the final model's pool percentiles (the rank transfer the
  shipped fused cut uses);
* ``foldraw`` - the same held-out scores applied to final-model scores as
  they are (the scale mismatch r4 won by, kept as a reference).

Each is fitted ``logistic`` (standardized, unpenalized) or ``isotonic``, and
read as a point estimate or a bootstrap lower bound (10th percentile of 30
refits) - a promise has to be cut at a lower bound, or it fails half the time
by construction.  ``+em`` re-estimates the corpus prior from its unlabelled
scores (Saerens EM) before averaging: P(y|s) is a posterior, so it moves when
the corpus's prevalence differs from the pool the votes came from.

**Scenarios.**  ``same``: the corpus has the voted pool's prevalence (the
natural arm's test half; the 5% arm's test half thinned to 5%).  ``shifted``:
the 5% arm's untouched 0.44% test half - a session whose pool was richer than
the corpus the detector is then run over.

**Reference cuts**: the recorded cutdiag candidates and the shipped threshold,
read at the precision and recall they achieve.  None targets a precision.

    python analyze_pframes_4220.py --arm natural=DIR --arm h0.05=DIR --out OUT [--jobs N]
"""

from __future__ import annotations

import argparse
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _cells_paths import main_frame_files, pframe_files, side_frame_files  # noqa: E402

FLOORS = (0.25, 0.5, 0.75, 0.9)
BOOTS = 30
LCB_PCT = 10
ESTIMATORS = ("insample", "foldrank", "foldraw")
FITS = ("logistic", "isotonic")
#: The recorded cutdiag candidates read as reference cuts, plus the shipped one.
REF_TAUS = (
    "tau_mid",
    "tau_cross",
    "tau_priorfree",
    "tau_rate",
    "tau_gumbel_priorfree",
    "tau_tail_a040",
    "tau_tail_a110",
    "tau_tail_a220",
    "tau_tail_a400",
)
#: The corpus prevalence the 5% arm's ``same`` scenario thins its test half to.
THIN_TO = {"h0.05": 0.05}


# --- fitting ---------------------------------------------------------------------------


def fit_posterior(kind: str, s: np.ndarray, y: np.ndarray):
    """P(y=1 | s) as a callable, or None when the labels carry no contrast."""
    if len(s) < 2 or y.min() == y.max():
        return None
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        mu, sd = float(s.mean()), float(s.std()) or 1.0
        m = LogisticRegression(C=1e4, max_iter=2000).fit(((s - mu) / sd)[:, None], y)
        return lambda x: m.predict_proba(((np.asarray(x) - mu) / sd)[:, None])[:, 1]
    from sklearn.isotonic import IsotonicRegression

    m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
    return lambda x: m.predict(np.asarray(x))


def em_prior_shift(p_corpus: np.ndarray, pi_train: float, iters: int = 200) -> np.ndarray:
    """Saerens-Latinne-Decaestecker EM: re-weight posteriors to the corpus's own prior.

    *pi_train* is the prior the posterior was fitted under - the voted pool's,
    estimated as the mean posterior over that pool (the votes are picked by
    score, so the fitted P(y|s) is the pool's posterior).  Needs no labels.
    """
    pi_train = float(np.clip(pi_train, 1e-6, 1 - 1e-6))
    p = np.clip(p_corpus, 1e-9, 1 - 1e-9)
    pi = pi_train
    for _ in range(iters):
        a = (pi / pi_train) * p
        b = ((1 - pi) / (1 - pi_train)) * (1 - p)
        w = a / (a + b)
        new = float(w.mean())
        if abs(new - pi) < 1e-7:
            break
        pi = new
    a = (pi / pi_train) * p
    b = ((1 - pi) / (1 - pi_train)) * (1 - p)
    return a / (a + b)


def percentile_in(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fraction of *ref_sorted* strictly below each x: the rank transfer's coordinate."""
    return np.searchsorted(ref_sorted, x, side="left") / max(len(ref_sorted), 1)


# --- one frame ---------------------------------------------------------------------------


def training_set(est: str, fr: dict) -> tuple[np.ndarray, np.ndarray]:
    if est == "insample":
        return fr["vote_scores"].astype(np.float64), fr["vote_labels"].astype(np.float64)
    s, y, f = fr["fold_cal_scores"].astype(np.float64), fr["fold_cal_labels"].astype(np.float64), fr["fold_cal_fold"]
    if est == "foldraw":
        return s, y
    hs, hf = fr["fold_hay_scores"].astype(np.float64), fr["fold_hay_fold"]
    out = np.empty_like(s)
    for k in np.unique(f):
        hay = np.sort(hs[hf == k])
        out[f == k] = percentile_in(hay, s[f == k])
    return out, y


def corpus_coordinate(est: str, corpus_scores: np.ndarray, pool_sorted: np.ndarray) -> np.ndarray:
    return percentile_in(pool_sorted, corpus_scores) if est == "foldrank" else corpus_scores


def pick(est_prec: np.ndarray, x: float) -> int:
    ok = np.flatnonzero(est_prec >= x)
    return int(ok.max()) if len(ok) else -1


def frame_rows(meta: dict, fr: dict, cutdiag: pd.DataFrame | None, rng: np.random.Generator) -> tuple[list, list]:
    """Estimator rows and reference-cut rows for one frame."""
    test_s = fr["test_scores"].astype(np.float64)
    test_y = fr["test_labels"].astype(np.int64)
    pool_s = fr["pool_scores"].astype(np.float64)
    pool_sorted = np.sort(pool_s)
    corpora = {"same": np.arange(len(test_s))}
    if meta["arm"] in THIN_TO:
        target = THIN_TO[meta["arm"]]
        pos = np.flatnonzero(test_y == 1)
        neg = np.flatnonzero(test_y == 0)
        keep = int(round(len(pos) * (1 - target) / target))
        thin_rng = np.random.default_rng([meta["seed"], 4220, len(test_s)])
        corpora = {
            "same": np.sort(np.concatenate([pos, thin_rng.choice(neg, size=min(keep, len(neg)), replace=False)])),
            "shifted": np.arange(len(test_s)),
        }
    base = {
        **meta,
        "t": int(fr["t"]),
        "n_vote_pos": int(fr["vote_labels"].sum()),
        "n_cal_pos": int(fr["fold_cal_labels"].sum()),
    }
    rows, refs = [], []
    for scen, idx in corpora.items():
        cs, cy = test_s[idx], test_y[idx]
        order = np.argsort(-cs, kind="stable")
        cs_sorted, ys = cs[order], cy[order]
        k = np.arange(1, len(cs) + 1)
        tp = np.cumsum(ys)
        prec, rec = tp / k, tp / max(int(cy.sum()), 1)
        oracle = {x: (float(rec[np.flatnonzero(prec >= x).max()]) if (prec >= x).any() else 0.0) for x in FLOORS}
        scen_base = {**base, "scenario": scen, "corpus_prevalence": float(cy.mean())}
        for est in ESTIMATORS:
            ts, ty = training_set(est, fr)
            coord = corpus_coordinate(est, cs_sorted, pool_sorted)
            pool_coord = corpus_coordinate(est, pool_s, pool_sorted)
            for fit in FITS:
                curves: dict[str, np.ndarray | None] = {}
                f0 = fit_posterior(fit, ts, ty)
                if f0 is None:
                    curves = dict.fromkeys(("point", "lcb", "point+em", "lcb+em"))
                else:
                    p0 = f0(coord)
                    curves["point"] = np.cumsum(p0) / k
                    boots, boots_em = [], []
                    pi_train0 = float(np.mean(f0(pool_coord)))
                    curves["point+em"] = np.cumsum(em_prior_shift(p0, pi_train0)) / k
                    for _ in range(BOOTS):
                        b = rng.integers(0, len(ts), len(ts))
                        fb = fit_posterior(fit, ts[b], ty[b])
                        if fb is None:
                            continue
                        pb = fb(coord)
                        boots.append(np.cumsum(pb) / k)
                        boots_em.append(np.cumsum(em_prior_shift(pb, float(np.mean(fb(pool_coord))))) / k)
                    curves["lcb"] = np.percentile(np.array(boots), LCB_PCT, axis=0) if len(boots) >= 5 else None
                    curves["lcb+em"] = (
                        np.percentile(np.array(boots_em), LCB_PCT, axis=0) if len(boots_em) >= 5 else None
                    )
                for reading, curve in curves.items():
                    for x in FLOORS:
                        i = pick(curve, x) if curve is not None else -1
                        rows.append(
                            {
                                **scen_base,
                                "estimator": est,
                                "fit": fit,
                                "reading": reading,
                                "X": x,
                                "fitted": curve is not None,
                                "returned": i + 1,
                                "achieved": float(prec[i]) if i >= 0 else np.nan,
                                "recall": float(rec[i]) if i >= 0 else 0.0,
                                "oracle_recall": oracle[x],
                                "violated": bool(i >= 0 and prec[i] < x),
                                "empty": i < 0,
                            }
                        )
        # Reference cuts: what the recorded candidates and the shipped cut deliver.
        taus = {"shipped": float(fr["threshold"])}
        if cutdiag is not None and not cutdiag.empty:
            cd = cutdiag[cutdiag["t"] == int(fr["t"])]
            if "geometry" in cd.columns and (cd["geometry"] == "pooled").any():
                cd = cd[cd["geometry"] == "pooled"]
            if not cd.empty:
                for c in REF_TAUS:
                    if c in cd.columns and np.isfinite(cd[c].iloc[0]):
                        taus[c] = float(cd[c].iloc[0])
        for name, tau in taus.items():
            n = int((cs >= tau).sum())
            tpn = int(cy[cs >= tau].sum())
            refs.append(
                {
                    **scen_base,
                    "cut": name,
                    "returned": n,
                    "achieved": tpn / n if n else np.nan,
                    "recall": tpn / max(int(cy.sum()), 1),
                }
            )
    return rows, refs


def cell_rows(job: tuple[str, str, str, str]) -> tuple[list, list]:
    arm, npz_path, main_path, cutdiag_path = job
    first = pd.read_csv(main_path, nrows=1)
    if first.empty:
        # A starved cell (header-only main frame): it never held a Good and a
        # Bad vote at once, so it has no detector and nothing to promise.  It is
        # counted by the caller, not scored.
        return [], []
    meta = {
        "arm": arm,
        "category": str(first["category"].iloc[0]),
        "seed": int(first["seed"].iloc[0]),
        "band": str(first["category"].iloc[0]).split("@")[-1],
    }
    cutdiag = pd.read_csv(cutdiag_path, low_memory=False) if cutdiag_path else None
    z = np.load(npz_path)
    steps = sorted({k.split("/")[0] for k in z.files}, key=lambda s: int(s[1:]))
    # crc32, not hash(): str hashing is salted per process, and reruns must reproduce.
    rng = np.random.default_rng([meta["seed"], zlib.crc32(meta["category"].encode()), 4220])
    rows, refs = [], []
    for st in steps:
        fr = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(st + "/")}
        r, f = frame_rows(meta, fr, cutdiag, rng)
        rows += r
        refs += f
    return rows, refs


def jobs_for(arm: str, cells_dir: Path) -> list[tuple[str, str, str, str]]:
    mains = {p.name.split(".")[0]: p for p in main_frame_files(cells_dir)}
    cds = {p.name.split("__")[0]: p for p in side_frame_files(cells_dir, "__cutdiag")}
    out = []
    for npz in pframe_files(cells_dir):
        stem = npz.name.split("__")[0]
        if stem in mains:
            out.append((arm, str(npz), str(mains[stem]), str(cds[stem]) if stem in cds else ""))
    return out


def summarize(df: pd.DataFrame, refs: pd.DataFrame, out: Path) -> str:
    keys = ["scenario", "estimator", "fit", "reading", "X"]
    g = (
        df.assign(
            reachable=df["oracle_recall"] > 0,
            wasted_empty=df["empty"] & (df["oracle_recall"] > 0),
            rec_ratio=np.where(
                df["oracle_recall"] > 0, df["recall"] / df["oracle_recall"].where(df["oracle_recall"] > 0), np.nan
            ),
        )
        .groupby(["arm", *keys])
        .agg(
            frames=("returned", "size"),
            violation_rate=("violated", "mean"),
            empty_rate=("empty", "mean"),
            empty_though_reachable=("wasted_empty", "mean"),
            recall_share_of_oracle=("rec_ratio", "mean"),
            median_achieved=("achieved", "median"),
        )
        .reset_index()
    )
    g.to_csv(out / "estimators_summary.csv", index=False, float_format="%.4g")
    by_t = (
        df[df["reading"].isin(["lcb", "lcb+em"])]
        .groupby(["arm", "scenario", "estimator", "fit", "reading", "X", "t"])
        .agg(violation_rate=("violated", "mean"), empty_rate=("empty", "mean"), mean_vote_pos=("n_vote_pos", "mean"))
        .reset_index()
    )
    by_t.to_csv(out / "estimators_by_t.csv", index=False, float_format="%.4g")
    r = refs.groupby(["arm", "scenario", "cut", "t"]).agg(
        median_returned=("returned", "median"),
        median_precision=("achieved", "median"),
        median_recall=("recall", "median"),
    )
    r.reset_index().to_csv(out / "reference_cuts.csv", index=False, float_format="%.4g")
    lines = ["# #4220 precision frames - machine summary", ""]
    lines.append(
        f"Frames: {df.groupby(['arm', 'category', 'seed', 't']).ngroups} (cell x step); arms {sorted(df['arm'].unique())}."
    )
    lines.append("")
    top = g[g["reading"].isin(["lcb", "lcb+em"])].sort_values(["arm", "scenario", "X", "violation_rate"])
    lines += ["```", top.round(3).to_string(index=False), "```"]
    lines.append("")
    lines.append("## Reference cuts at t=150")
    lines += ["```", r.reset_index().query("t == 150").round(3).to_string(index=False), "```"]
    text = "\n".join(lines)
    (out / "SUMMARY.md").write_text(text + "\n")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="label=results_dir (holding cells/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for spec in args.arm:
        label, _, d = spec.partition("=")
        jobs += jobs_for(label, Path(d) / "cells")
    if not jobs:
        raise SystemExit("no precision frames found - was the run launched with CALIB_PFRAME_STEPS?")
    rows, refs = [], []
    starved = 0
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for r, f in ex.map(cell_rows, jobs, chunksize=4):
                starved += not r
                rows += r
                refs += f
    else:
        for j in jobs:
            r, f = cell_rows(j)
            starved += not r
            rows += r
            refs += f
    df, rf = pd.DataFrame(rows), pd.DataFrame(refs)
    df.to_csv(args.out / "estimator_rows.csv.gz", index=False, float_format="%.5g")
    rf.to_csv(args.out / "reference_rows.csv.gz", index=False, float_format="%.5g")
    print(summarize(df, rf, args.out))
    (args.out / "provenance.json").write_text(
        json.dumps({"cells": len(jobs), "starved_skipped": starved, "arms": args.arm}, indent=2) + "\n"
    )
    print(f"cells {len(jobs)}, starved (no detector, skipped) {starved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
