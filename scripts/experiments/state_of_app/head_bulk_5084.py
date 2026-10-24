#!/usr/bin/env python3
"""#5084: the head's count of what stands out counts the bulk's heavy upper tail; price bulk models for it offline.

The head's count line (#4786, #5039) keeps a multiple of what stands out in the head's scores: #4603's count, the
corpus logits above ``median + 4 spreads`` less what a Gaussian bulk puts there (``_count_estimate``). The heads'
bulks are heavier than a Gaussian above 3 spreads, so the count reads the bulk's tail as positives (#5084).

  head_bulk_5084.py extract --results DIR [--results DIR ...] --out DIR [--procs N]
  head_bulk_5084.py price --x DIR --out FILE.pkl [--procs N] [--limit N]
  head_bulk_5084.py tables --priced FILE.pkl [FILE.pkl ...] --out TABLES [--figures FIGURES]

``extract`` keeps, for every snapshot of a Binary / Region snapshot run (``fold_scale_4936.sh``,
``CALIB_SAVE_TEST_SCORES_AT``) on the head's count line (the class model holds fewer than ``HEAD_COUNT_GOODS``
held-out Goods, and a typed query on a single-vector space is past its hold), the withheld half's scores best first,
their truth and the folds' held-out Goods' and Bads' scores, in ``head_count_5039.py``'s format (``<run>.npz`` +
``<run>.meta.csv``), so its ``price`` / ``grid`` and this study's pricers read them. It replaces #5039's
``extract``, which needed #4937's per-snapshot table. Face runs go through ``head_count_5039.py extract-faces``.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fold_scale_4936 import PATHS, _ident, load_snapshots  # noqa: E402
from head_count_5039 import _jobs, _save, scenario_corpora  # noqa: E402


def _run_cell(args) -> list:
    """The head-count snapshots of one Binary / Region cell, as ``head_count_5039._save`` takes them."""
    from vtscore.detectors.label_quota import query_holds
    from vtscore.eval.live_threshold_rules import head_count_recall
    from vtscore.training.thresholds.labels_line import HEAD_COUNT_GOODS, _finite_unit, head_count_cut

    npz, run = args
    ident = _ident(os.path.dirname(npz), npz)
    out, seen = [], set()
    for key, snap in sorted(load_snapshots(npz).items()):
        prefix = key.rsplit("/", 1)[0]
        path = PATHS.get(prefix)
        model = snap.get("model")
        step = int(snap["t"])  # a step can be filed under several snapshot clicks; keep one copy of each
        if path is None or not model or int(model["n_pos"]) >= HEAD_COUNT_GOODS or (path, step) in seen:
            continue
        votes = int((np.asarray(snap["sim_vote"]) >= 0).sum())
        if query_holds(votes, 0, patch_space=path == "region"):
            continue
        seen.add((path, step))
        s = _finite_unit(np.asarray(snap["scores"], dtype=np.float64))
        y = np.asarray(snap["labels"], dtype=np.int8)
        if y.sum() == 0:
            continue
        order = np.argsort(-s, kind="stable")
        fl = np.asarray(snap["fold_labels"], dtype=np.int8)
        fs = np.asarray(snap["fold_scores"], dtype=np.float64)
        with head_count_recall("off", "off"):  # #4603's count alone, the line before #5039
            thr = head_count_cut(s[order], float(snap["beta"]))
        meta = {"run": run, **ident, "path": path, "t": step, "votes": votes,
                "k_today": int((s >= thr).sum())}  # fmt: skip
        out.append((meta, s[order], y[order], fs[fl == 1], fs[fl == 0]))
    return out


def cmd_extract(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    for res in a.results:
        run = os.path.basename(os.path.dirname(res.rstrip("/")))
        files = sorted(glob.glob(os.path.join(res, "cells", "task_*__testscores.npz")))
        with ProcessPoolExecutor(max_workers=a.procs) as ex:
            items = [it for part in ex.map(_run_cell, [(f, run) for f in files], chunksize=4) for it in part]
        if not items:
            print(f"{run}: {len(files)} cells, no head-count snapshots", flush=True)
            continue
        meta = _save(a.out, run, items)
        print(f"{run}: {len(files)} cells, {len(meta)} head-count snapshots", flush=True)


BETAS = (0.25, 1.0, 4.0)
E708 = math.log(3 / 8) / math.log(1 / 4)
E13 = 0.18925
ZGRID = np.round(np.arange(1.0, 8.01, 0.05), 2)


def mult(beta: float) -> float:
    return beta ** (E13 if beta > 1 else E708)


def bulk(x: np.ndarray):
    med = float(np.median(x))
    sd = 1.4826 * float(np.median(np.abs(x - med)))
    return med, sd


def surv_counts(x: np.ndarray, med: float, sd: float) -> np.ndarray:
    """#(x > med + z sd) on ZGRID."""
    xs = np.sort(x)
    return x.size - np.searchsorted(xs, med + ZGRID * sd, side="right")


def lower_counts(x: np.ndarray, med: float, sd: float) -> np.ndarray:
    xs = np.sort(x)
    return np.searchsorted(xs, med - ZGRID * sd, side="left")


def zi(z: float) -> int:
    return int(np.argmin(np.abs(ZGRID - z)))


def gauss_tail(n: int, z) -> np.ndarray:
    from scipy.special import ndtr

    return n * ndtr(-np.asarray(z, dtype=np.float64))


def exp_fit(above: np.ndarray, lo: float, hi: float):
    """Fit log #above(z) = a - b z over z in [lo, hi]; (a, b) or None."""
    m = (ZGRID >= lo - 1e-9) & (ZGRID <= hi + 1e-9) & (above > 0)
    if m.sum() < 3:
        return None
    b, a = np.polyfit(ZGRID[m], np.log(above[m]), 1)
    return a, -b


def estimators(x: np.ndarray) -> dict[str, float]:
    n = x.size
    out: dict[str, float] = {}
    if n < 50:
        return out
    med, sd = bulk(x)
    if not sd > 0:
        return out
    up = surv_counts(x, med, sd)
    lo = lower_counts(x, med, sd)
    g = gauss_tail(n, ZGRID)
    clamp = lambda v: float(min(max(v, 1.0), n))  # noqa: E731
    for z in (3.0, 4.0, 5.0, 6.0):
        out[f"gauss_z{z:g}"] = clamp(up[zi(z)] - g[zi(z)])
    for z in (3.0, 3.5, 4.0):
        out[f"mirror_z{z:g}"] = clamp(up[zi(z)] - lo[zi(z)])
    # Exponential upper tail fitted in a band, extrapolated to the bar: what the bulk itself puts above it.
    for flo, fhi in ((1.5, 2.5), (2.0, 3.0), (2.0, 2.5)):
        fit = exp_fit(up, flo, fhi)
        for z in (3.0, 4.0):
            key = f"exp{flo:g}-{fhi:g}_z{z:g}"
            if fit is None:
                out[key] = out[f"gauss_z{z:g}"]
                continue
            a, b = fit
            out[key] = clamp(up[zi(z)] - math.exp(a - b * z))
            # The largest excess over the fitted tail from the bar up.
            if z == 3.0:
                m = ZGRID >= 3.0
                exc = up[m] - np.exp(a - b * ZGRID[m])
                out[f"exp{flo:g}-{fhi:g}_max3"] = clamp(float(exc.max()))
    # The lower tail's shape as the bulk's own (mirror), fitted as an exponential and extrapolated (smoother).
    fit = exp_fit(lo, 2.0, 3.5)
    if fit is not None:
        a, b = fit
        for z in (3.0, 4.0):
            out[f"lexp_z{z:g}"] = clamp(up[zi(z)] - math.exp(a - b * z))
    # A log-concave tail: log #above(z) quadratic (curvature <= 0) over a band, extrapolated up (exponential on a
    # heavy tail, close to a Gaussian's on a Gaussian bulk).
    for flo, fhi in ((1.0, 2.5), (1.5, 2.5), (1.0, 3.0)):
        fit = quad_fit(up, flo, fhi)
        tag = f"quad{flo:g}-{fhi:g}"
        if fit is None:
            for z in ("_z3", "_z4", "_max3"):
                out[tag + z] = out["gauss_z4"]
            continue
        pred = np.exp(np.polyval(fit, ZGRID))
        for z in (3.0, 4.0):
            out[f"{tag}_z{z:g}"] = clamp(up[zi(z)] - pred[zi(z)])
        m = ZGRID >= 3.0
        out[f"{tag}_max3"] = clamp(float((up[m] - pred[m]).max()))
    # The bulk as an exponential tail plus a constant (the positives, above the band): fitted jointly over a band.
    for flo, fhi in ((1.5, 4.5), (1.0, 4.0), (2.0, 5.0)):
        c = expc_fit(up, flo, fhi)
        out[f"expC{flo:g}-{fhi:g}"] = clamp(c) if c is not None else out["gauss_z4"]
    return out


def quad_fit(above: np.ndarray, lo: float, hi: float):
    """Coefficients (c2, c1, c0) of log #above(z) over z in [lo, hi], c2 held <= 0 (a straight line if not)."""
    m = (ZGRID >= lo - 1e-9) & (ZGRID <= hi + 1e-9) & (above > 0)
    if m.sum() < 4:
        return None
    z, ly = ZGRID[m], np.log(above[m])
    c = np.polyfit(z, ly, 2)
    if c[0] > 0:
        b, a = np.polyfit(z, ly, 1)
        c = np.array([0.0, b, a])
    return c


BGRID = np.arange(0.3, 3.01, 0.02)


def expc_fit(above: np.ndarray, lo: float, hi: float):
    """C in #above(z) = A exp(-b z) + C over z in [lo, hi], weighted least squares (weights 1 / #above), b on a grid."""
    m = (ZGRID >= lo - 1e-9) & (ZGRID <= hi + 1e-9) & (above > 0)
    if m.sum() < 5:
        return None
    z, t = ZGRID[m], above[m].astype(np.float64)
    w = 1.0 / t
    best = None
    for b in BGRID:
        X = np.stack([np.exp(-b * z), np.ones_like(z)], 1)
        Xw = X * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(Xw, t * np.sqrt(w), rcond=None)
        if coef[0] < 0 or coef[1] < 0:
            continue
        r = float(((X @ coef - t) ** 2 * w).sum())
        if best is None or r < best[0]:
            best = (r, coef[1])
    return None if best is None else float(best[1])


MULTS = np.exp(np.linspace(np.log(0.1), np.log(10.0), 41))
#: The closed loop's arms (head_bulk_5084.sh), as (scale, exponent): the count keeps scale * beta ** exponent times
#: today's count (``gauss_z4``) at beta <= 1.
ARM_MULTS = {"s71e1": (0.71, 1.0), "s50e1": (0.5, 1.0)}


def excess_peak(x_desc: np.ndarray, beta: float, model: str, z_min: float) -> float:
    """k at the excess curve's expected F-beta peak (#5039's _excess_depth), the bulk Gaussian or a fitted tail."""
    from scipy.special import ndtr

    n = x_desc.size
    med, sd = bulk(x_desc)
    if n < 50 or not sd > 0:
        return float("nan")
    z = (x_desc - med) / sd
    if model == "gauss":
        nb = n * ndtr(-z)
    else:
        lo, hi = map(float, model[4:].split("-"))
        fit = quad_fit(surv_counts(x_desc, med, sd), lo, hi)
        if fit is None:
            nb = n * ndtr(-z)
        else:
            # Below the band's top the bulk is read off as-is (no excess); above, the fitted tail.
            nb = np.where(z >= hi, np.exp(np.polyval(fit, np.clip(z, None, 12.0))), np.arange(1, n + 1))
    k = np.arange(1, n + 1, dtype=np.float64)
    up = z >= z_min
    if not up.any():
        return 1.0
    excess = k - nb
    total = max(float(excess[up].max()), 1.0)
    b2 = beta * beta
    f = np.where(up, (1 + b2) * np.clip(excess, 0, total) / (b2 * total + k), -1.0)
    return float(np.argmax(f) + 1)


XC = {
    "xc_gauss_z2.75": ("gauss", 2.75),
    "xc_gauss_z3.5": ("gauss", 3.5),
    "xc_quad1-2.5_z2.5": ("quad1-2.5", 2.5),
    "xc_quad1-2.5_z3": ("quad1-2.5", 3.0),
    "xc_quad1-3_z3": ("quad1-3", 3.0),
}
GRID_CANDS = (
    *XC,
    "gauss_z4",
    "gauss_z5",
    "exp1.5-2.5_max3",
    "exp2-2.5_max3",
    "quad1-2.5_z3",
    "quad1-2.5_z4",
    "quad1-2.5_max3",
    "quad1.5-2.5_max3",
    "quad1-3_max3",
    "quad1-3_z4",
    "lab_prec_z3",
    "lab_prec_z4",
    "lab_rec_z4",
    "oracle",
)
#: The bulk-tail grid the tail-shape figure reads: the withheld negatives above median + z spreads.
Z_TAIL = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)


def label_counts(x: np.ndarray, med: float, sd: float, xg: np.ndarray, xb: np.ndarray) -> dict[str, float]:
    """Counts read from the labels: what stands out above median + z spreads, thinned by the share of the labelled items
    up there that are Goods (``lab_prec``), and divided again by the share of the held-out Goods up there
    (``lab_rec``, the line's recall as the labels see it). Shares are smoothed (+0.5 / +1) for the few labels a head
    serves on."""
    out = {}
    for z in (3.0, 4.0):
        thr = med + z * sd
        g, b = int(np.count_nonzero(xg > thr)), int(np.count_nonzero(xb > thr))
        prec = (g + 0.5) / (g + b + 1)
        out[f"lab_prec_z{z:g}"] = float(np.count_nonzero(x > thr)) * prec
        if z == 4.0:
            out["lab_rec_z4"] = out["lab_prec_z4"] / ((g + 0.5) / (xg.size + 1))
    return out


def gap_and_count(x_desc: np.ndarray, beta: float, n_hat: float, c: float | None = None) -> int:
    from vtscore.detectors.centroid_head import _top_gap_count

    n = x_desc.size
    gap = _top_gap_count(x_desc, drop="skip25" if beta >= 1.0 else "widest")
    k = min(max(1, round((mult(beta) if c is None else c) * max(n_hat, float(gap)))), n)
    line = min(float(x_desc[k - 1]), float(x_desc[gap - 1]))
    return int(np.count_nonzero(x_desc >= line))


def _snap(args) -> list[dict]:
    from vtscore.training.thresholds.labels_line import _logit

    ident, s, y, goods, bads = args
    rows = []
    for scen, (sv, yv) in scenario_corpora(s, y, ident["key"]).items():
        x = _logit(sv)
        if x.size < 50:
            continue
        est = estimators(x)
        med, sd = bulk(x)
        npos = int(yv.sum())
        xg, xb = _logit(goods), _logit(bads)
        base = {
            **{k: ident[k] for k in ("run", "path", "run_beta", "idx", "t", "class", "seed")},
            "scenario": scen,
            "n": int(x.size),
            "n_pos": npos,
            "pos_above_z4": int(np.count_nonzero(x[yv == 1] > med + 4 * sd)),
            "pos_above_z3": int(np.count_nonzero(x[yv == 1] > med + 3 * sd)),
            "neg_above_z4": int(np.count_nonzero(x[yv == 0] > med + 4 * sd)),
            "neg_above_z3": int(np.count_nonzero(x[yv == 0] > med + 3 * sd)),
            "n_goods": int(xg.size),
            "n_bads": int(xb.size),
            "pos_z_med": float(np.median((x[yv == 1] - med) / sd)) if npos else math.nan,
            "goods_z_med": float(np.median((xg - med) / sd)) if xg.size else math.nan,
            "pos_share_z4": float(np.mean(x[yv == 1] > med + 4 * sd)) if npos else math.nan,
            "goods_share_z4": float(np.mean(xg > med + 4 * sd)) if xg.size else math.nan,
            "neg_tail": np.array([np.count_nonzero(x[yv == 0] > med + z * sd) for z in Z_TAIL], dtype=np.int32),
        }
        for k, v in est.items():
            base[f"n_{k}"] = v
        cands = dict(est)
        if "gauss_z4" in est:
            cands.update(label_counts(x, med, sd, xg, xb))
        cands["oracle"] = float(max(npos, 1))
        for beta in BETAS:
            r = dict(base, beta=beta)
            b2 = beta * beta
            if npos > 0:
                f = (1 + b2) * np.cumsum(yv) / (b2 * npos + np.arange(1, yv.size + 1))
                r["best_f"], r["best_k"] = float(f.max()), int(np.argmax(f)) + 1
            per_beta = dict(cands)
            for name, (model, zm) in XC.items():
                v = excess_peak(x, beta, model, zm)
                per_beta[name] = (v / mult(beta)) if np.isfinite(v) else cands["gauss_z4"]
            if beta <= 1.0 and "gauss_z4" in est:
                for arm, (sc, ex) in ARM_MULTS.items():
                    k = gap_and_count(x, beta, est["gauss_z4"], sc * beta**ex)
                    r[f"k_{arm}"] = k
                    if npos > 0:
                        r[f"F_{arm}"] = float(f[k - 1]) if k > 0 else 0.0
            for name, nh in per_beta.items():
                k = gap_and_count(x, beta, nh)
                r[f"k_{name}"] = k
                if npos > 0:
                    r[f"F_{name}"] = float(f[k - 1]) if k > 0 else 0.0
                if name in GRID_CANDS:
                    ks = np.array([gap_and_count(x, beta, nh, c) for c in MULTS], dtype=np.int32)
                    r[f"K_{name}"] = ks
                    if npos > 0:
                        r[f"G_{name}"] = np.where(ks > 0, f[np.maximum(ks, 1) - 1], 0.0).astype(np.float32)
            rows.append(r)
    return rows


def cmd_price(a) -> None:
    jobs = _jobs(a.x)
    if a.limit:
        jobs = jobs[:: max(1, len(jobs) // a.limit)][: a.limit]
    print(f"{len(jobs)} snapshots", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        for out in ex.map(_snap, jobs, chunksize=4):
            rows.extend(out)
    df = pd.DataFrame(rows)
    df.to_pickle(a.out)
    print(f"wrote {len(df)} rows -> {a.out}")


SCEN = ("native", "pos1/4", "neg0.44")
BENCH = {"binary": "Binary Photo", "region": "Region Photo", "face": "FHIBE faces"}


def _by_class(h: pd.DataFrame, col: str) -> pd.Series:
    """Per-class means of *col* (a scalar or a grid array per row)."""
    if h.empty:
        return pd.Series(dtype=object)
    return h.groupby("class")[col].apply(lambda v: np.mean(np.stack(v.values), 0) if v.dtype == object else v.mean())


def _gain(h: pd.DataFrame, col: str, base: str = "F_gauss_z4") -> tuple[float, float, float, float]:
    """(today, arm, paired gain, its SE over classes), each class weighted once."""
    h = h[h[col].notna()]
    a, b = _by_class(h, base), _by_class(h, col)
    g = b - a
    return float(a.mean()), float(b.mean()), float(g.mean()), float(g.std(ddof=1) / np.sqrt(max(len(g), 1)))


def _grid_gain(h: pd.DataFrame, name: str) -> np.ndarray:
    """Class-mean F-beta gain over today on MULTS for candidate *name*."""
    h = h[h[f"G_{name}"].notna()]
    return np.stack(_by_class(h, f"G_{name}").values).mean(0) - float(_by_class(h, "F_gauss_z4").mean())


def presets_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bench in [b for b in BENCH if b in set(df["path"])]:
        for beta in BETAS:
            d = df[(df["path"] == bench) & (df["beta"] == beta)]
            for scen in (*SCEN, "empty"):
                h = d[d["scenario"] == scen]
                if h.empty:
                    continue
                r = {"bench": bench, "beta": beta, "scenario": scen, "snapshots": len(h), "classes": h["class"].nunique(),
                     "kept_today": float(h["k_gauss_z4"].mean()), "positives": float(h["n_pos"].mean())}  # fmt: skip
                if scen != "empty":
                    r["kept_over_pos_today"] = float(np.median(h["k_gauss_z4"] / h["n_pos"]))
                    r["best_f"] = float(_by_class(h, "best_f").mean())
                for arm in ARM_MULTS if beta <= 1.0 else ():
                    r[f"kept_{arm}"] = float(h[f"k_{arm}"].mean())
                    if scen != "empty":
                        r["F_today"], r[f"F_{arm}"], r[f"gain_{arm}"], r[f"se_{arm}"] = _gain(h, f"F_{arm}")
                        r[f"kept_over_pos_{arm}"] = float(np.median(h[f"k_{arm}"] / h["n_pos"]))
                if beta > 1.0 and scen != "empty":
                    r["F_today"] = float(_by_class(h, "F_gauss_z4").mean())
                rows.append(r)
    return pd.DataFrame(rows)


def candidates_table(df: pd.DataFrame) -> pd.DataFrame:
    """Each candidate count at the one multiplier best over Binary and Region Photo's six preset cells (per beta), what
    the same multiplier does on faces, and the gain each path would take at its own multiplier."""
    rows = []
    for beta in (0.25, 1.0):
        d = df[df["beta"] == beta]
        emp = d[d["scenario"] == "empty"]
        photo = [q for q in ("binary", "region") if q in set(d["path"])]
        for name in GRID_CANDS:
            if f"G_{name}" not in d:
                continue
            cur = {(p, s): _grid_gain(d[(d["path"] == p) & (d["scenario"] == s)], name)
                   for p in photo for s in SCEN}  # fmt: skip
            mean = np.mean(list(cur.values()), 0)
            j = int(np.argmax(mean))
            r = {"beta": beta, "count": name, "c": float(MULTS[j]), "multiplier_vs_today": float(MULTS[j] / mult(beta)),
                 "gain_binary_region": float(mean[j])}  # fmt: skip
            for (p, s), g in cur.items():
                r[f"gain_{p}_{s}"] = float(g[j])
            for p in photo:
                r[f"gain_{p}_own_c"] = float(np.mean([cur[(p, s)] for s in SCEN], 0).max())
            if "face" in set(d["path"]):
                for s in SCEN:
                    r[f"gain_face_{s}"] = float(_grid_gain(d[(d["path"] == "face") & (d["scenario"] == s)], name)[j])
            for p in [b for b in BENCH if b in set(emp["path"])]:
                e = emp[emp["path"] == p]
                r[f"empty_kept_{p}"] = float(np.stack(e[f"K_{name}"].values)[:, j].mean())
                r[f"empty_kept_{p}_today"] = float(e["k_gauss_z4"].mean())
            rows.append(r)
    return pd.DataFrame(rows)


def diagnosis_table(df: pd.DataFrame) -> pd.DataFrame:
    """What the count reads at beta 1: positives, today's count, the withheld negatives above median + 4 spreads (and
    what a Gaussian bulk would put there), and where the held-out Goods sit against the withheld positives."""
    from scipy.special import ndtr

    rows = []
    d = df[df["beta"] == 1.0]
    for bench in [b for b in BENCH if b in set(d["path"])]:
        for scen in (*SCEN, "empty"):
            h = d[(d["path"] == bench) & (d["scenario"] == scen)]
            pos = h["n_pos"].clip(lower=1)
            rows.append({
                "bench": bench, "scenario": scen, "snapshots": len(h), "n": float(h["n"].median()),
                "positives": float(h["n_pos"].median()), "count_today": float(h["n_gauss_z4"].median()),
                "count_over_pos": float(np.median(h["n_gauss_z4"] / pos)) if scen != "empty" else math.nan,
                "kept_over_pos_b1": float(np.median(h["k_gauss_z4"] / pos)) if scen != "empty" else math.nan,
                "pos_above_z4": float(h["pos_above_z4"].median()), "neg_above_z4": float(h["neg_above_z4"].median()),
                "gauss_above_z4": float(np.median(h["n"] * ndtr(-4.0))),
                "pos_z_med": float(h["pos_z_med"].median()), "goods_z_med": float(h["goods_z_med"].median()),
                "pos_share_z4": float(h["pos_share_z4"].mean()), "goods_share_z4": float(h["goods_share_z4"].mean()),
                "share_corr": float(h[["pos_share_z4", "goods_share_z4"]].dropna().corr().iloc[0, 1]),
                "goods": float(h["n_goods"].median()), "bads": float(h["n_bads"].median()),
            })  # fmt: skip
    return pd.DataFrame(rows)


def examples_table(df: pd.DataFrame) -> pd.DataFrame:
    """Literal snapshots at beta 1 and 1/4, native: per bench the one at the median of today's kept / positives."""
    rows = []
    for beta in (0.25, 1.0):
        for bench in [b for b in BENCH if b in set(df["path"])]:
            h = df[(df["path"] == bench) & (df["beta"] == beta) & (df["scenario"] == "native")]
            if h.empty:
                continue
            ratio = h["k_gauss_z4"] / h["n_pos"]
            r = h.loc[(ratio - ratio.median()).abs().idxmin()]
            rows.append({"bench": bench, "beta": beta, "run": r["run"], "idx": r["idx"], "t": r["t"], "class": r["class"],
                         "n": r["n"], "positives": r["n_pos"], "goods": r["n_goods"], "count_today": r["n_gauss_z4"],
                         "neg_above_z4": r["neg_above_z4"], "pos_above_z4": r["pos_above_z4"], "best_k": r["best_k"],
                         "best_f": r["best_f"], "kept_today": r["k_gauss_z4"], "F_today": r["F_gauss_z4"],
                         **{f"{p}_{a}": r[f"{p}_{a}"] for a in ARM_MULTS for p in ("k", "F")}})  # fmt: skip
    return pd.DataFrame(rows)


def figures(df: pd.DataFrame, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from early_check_4664 import INK, MUTED, _axes_style
    from scipy.special import ndtr

    benches = [b for b in BENCH if b in set(df["path"])]
    colors = {"native": "#2a78d6", "pos1/4": "#eb6834", "neg0.44": "#3a9e5c"}
    fig, axes = plt.subplots(2, len(benches), figsize=(4.6 * len(benches), 7.4), squeeze=False)
    for i, beta in enumerate((0.25, 1.0)):
        for j, bench in enumerate(benches):
            ax = axes[i, j]
            d = df[(df["path"] == bench) & (df["beta"] == beta)]
            rel = MULTS / mult(beta)
            on = (rel >= 0.2) & (rel <= 2.0)
            for s in SCEN:
                ax.plot(rel[on], _grid_gain(d[d["scenario"] == s], "gauss_z4")[on], color=colors[s], lw=1.8, label=s)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xscale("log")
            ax.set_xlim(0.2, 2.0)
            ax.set_xticks([0.25, 0.5, 0.71, 1.0, 1.4, 2.0], ["0.25", "0.5", "0.71", "1", "1.4", "2"])
            ax.minorticks_off()
            for arm, (sc, ex) in ARM_MULTS.items():
                ax.axvline(sc * beta**ex / mult(beta), color=MUTED, lw=0.8, ls="--")
                ax.text(sc * beta**ex / mult(beta), ax.get_ylim()[1], f" {arm}", color=MUTED, fontsize=7, va="top")
            ax.set_title(f"{BENCH[bench]}, beta {beta:g}", color=INK, fontsize=10, loc="left")
            ax.set_xlabel("count multiplier / today's", color=MUTED, fontsize=8)
            if j == 0:
                ax.set_ylabel("F-beta gain over today (class mean)", color=MUTED, fontsize=8)
            _axes_style(ax)
            ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "offline_gain_vs_multiplier.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bc = {"binary": "#2a78d6", "region": "#eb6834", "face": "#3a9e5c"}
    for bench in benches:
        h = df[(df["path"] == bench) & (df["beta"] == 1.0) & (df["scenario"] == "native")]
        tail = np.median(np.stack(h["neg_tail"].values), 0).astype(float)
        exp = float(h["n"].median()) * ndtr(-np.asarray(Z_TAIL))
        ok = tail > 0
        ax.plot(np.asarray(Z_TAIL)[ok], tail[ok] / exp[ok], "o-", color=bc[bench], lw=1.8, label=BENCH[bench])
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("median + z spreads (1.4826 MAD)", color=MUTED, fontsize=9)
    ax.set_ylabel("withheld negatives above (median) / a Gaussian bulk's", color=MUTED, fontsize=9)
    ax.set_title("The heads' bulks are heavier than a Gaussian above 3 spreads", color=INK, fontsize=10, loc="left")
    _axes_style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bulk_tail.png"), dpi=150)
    plt.close(fig)


def cmd_tables(a) -> None:
    df = pd.concat([pd.read_pickle(p) for p in a.priced], ignore_index=True)  # noqa: S301 (price wrote them)
    # Each head at the preset its own session ran (the line steered the session that reached it); a bench with runs at
    # one preset only (Region Photo's beta 1 snapshots) lends those heads to every preset.
    one = df.groupby("path")["run_beta"].transform("nunique") == 1
    df = df[one | np.isclose(df["run_beta"], df["beta"])].reset_index(drop=True)
    os.makedirs(a.out, exist_ok=True)
    for name, fn in (("presets", presets_table), ("candidates", candidates_table), ("diagnosis", diagnosis_table),
                     ("examples", examples_table)):  # fmt: skip
        t = fn(df)
        t.to_csv(os.path.join(a.out, f"offline_{name}.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 60):
            print(f"== {name}\n{t.round(3).to_string(index=False)}", flush=True)
    figures(df, a.figures or a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--results", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=8)
    p.set_defaults(func=cmd_extract)
    p = sub.add_parser("price")
    p.add_argument("--x", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="price an even sample of this many snapshots (a test)")
    p.set_defaults(func=cmd_price)
    p = sub.add_parser("tables")
    p.add_argument("--priced", nargs="+", required=True, help="price's pickles, one per bench")
    p.add_argument("--out", required=True, help="the report's tables dir")
    p.add_argument("--figures", help="the report's figures dir (default: --out)")
    p.set_defaults(func=cmd_tables)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
