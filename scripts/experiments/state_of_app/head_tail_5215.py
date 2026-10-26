#!/usr/bin/env python3
"""#5215: tell a head whose positives sit under the bulk's tail before shrinking its count; price a gated multiplier.

#5084 shipped the head's count line at ``0.71 beta`` times what stands out at beta <= 1 (``HEAD_COUNT_PRECISION_SCALE``)
in place of #4603's ``beta ** 0.708``. That pays on heads whose positives stand out above the bulk (Region Photo's
"person": 49 of 50 above 4 spreads, +0.12 F1) and costs heads whose positives sit under the bulk's heavy tail (Binary
Photo's "book": 5 of 50 above, 16 negatives), where a smaller count only drops positives: Binary Photo at 1%
prevalence loses -0.014 F-beta at beta 1/4. This study reads, for every head-count snapshot #5084 extracted, the
signals the app has at the line (the scores' shape, the drop, the held-out Goods' and Bads' placement against the
count's line) and prices a multiplier gated on each against 0.71 beta on the same heads.

  head_tail_5215.py price --x DIR --out FILE.pkl [--procs N] [--limit N]
  head_tail_5215.py tables --priced FILE.pkl [FILE.pkl ...] --out TABLES [--figures FIGURES]

``price`` reads #5084's extracted snapshots (``head_bulk_5084.py extract`` / ``head_count_5039.py extract-faces``) and
writes, per snapshot, prevalence scenario and beta in (1/4, 1): the gate features (label-legal: corpus scores, held-out
Goods / Bads) and the F-beta of the count line on a grid of multipliers relative to #4603's ``beta ** 0.708``.
``tables`` prices gates (shrink to 0.71 beta when the feature says the positives stand out, else keep a second
multiplier) per bench, preset and prevalence, the threshold chosen on Binary + Region Photo and carried to faces.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from head_bulk_5084 import BENCH, bulk, estimators, gap_and_count, mult, quad_fit, surv_counts, zi  # noqa: E402
from head_count_5039 import _jobs, scenario_corpora  # noqa: E402

BETAS = (0.25, 1.0)
#: The shipped count (#5084): 0.71 beta times what stands out, as a multiple of #4603's beta ** 0.708.
S71 = 0.71
#: Multipliers relative to #4603's beta ** 0.708 (1 = the line before #5084); 0.71 beta / beta ** 0.708 is added.
REL = np.round(np.exp(np.linspace(np.log(0.1), np.log(4.0), 38)), 4)


def rel_grid(beta: float) -> np.ndarray:
    return np.unique(np.r_[REL, 1.0, S71 * beta / mult(beta)])


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    if not a.size or not b.size:
        return math.nan
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def features(x: np.ndarray, xg: np.ndarray, xb: np.ndarray) -> dict[str, float]:
    """The signals the app has at the head's count line, beta-free: *x* the corpus logits best first, *xg* / *xb* the
    held-out Goods' / Bads' logits. The line they are read against is the beta-1 line before #5084 (``k1``)."""
    n = x.size
    med, sd = bulk(x)
    est = estimators(x)
    cnt = est.get("gauss_z4", 1.0)
    k1 = gap_and_count(x, 1.0, cnt, 1.0)
    line = float(x[k1 - 1])
    up = surv_counts(x, med, sd)
    from scipy.special import ndtr

    def above(v, thr):
        return int(np.count_nonzero(v >= thr))

    zg, zb = (xg - med) / sd, (xb - med) / sd
    g1, b1 = above(xg, line), above(xb, line)
    rank = lambda v: int(np.count_nonzero(x >= v))  # noqa: E731 (corpus rank of a logit)
    f = {
        "count": cnt,
        "k1": float(k1),
        "count_frac": cnt / n,
        # Tail heaviness: corpus above 3 spreads over a Gaussian bulk's; above 3 over above 4; today's count over
        # a log-concave tail fit's.
        "tail3_over_gauss": float(up[zi(3.0)]) / max(float(n * ndtr(-3.0)), 1e-9),
        "up3_over_up4": float(up[zi(3.0)]) / max(float(up[zi(4.0)]), 1.0),
        "count_over_quad": cnt / max(est.get("quad1-2.5_z4", cnt), 1.0),
        "count_over_mirror": cnt / max(est.get("mirror_z4", cnt), 1.0),
        "top_z": float((x[0] - med) / sd),
        "line_z": float((line - med) / sd),
        "goods": float(xg.size),
        "bads": float(xb.size),
        "goods_above_line": g1 / max(xg.size, 1),
        "bads_above_line": float(b1),
        "bads_above_line_share": b1 / max(xb.size, 1),
        "lab_prec_line": (g1 + 0.5) / (g1 + b1 + 1.0),
        "goods_z_med": float(np.median(zg)) if xg.size else math.nan,
        "goods_z_min": float(zg.min()) if xg.size else math.nan,
        "bads_z_med": float(np.median(zb)) if xb.size else math.nan,
        "bads_z_max": float(zb.max()) if xb.size else math.nan,
        "bads_above_z4": float(np.count_nonzero(zb > 4.0)),
        "auc_goods_bads": _auc(xg, xb),
        # Where the labels sit in the corpus, over the beta-1 count: the deepest Good, the median Good, the top Bad.
        "deep_good_rank": rank(float(xg.min())) / k1 if xg.size else math.nan,
        "med_good_rank": rank(float(np.median(xg))) / k1 if xg.size else math.nan,
        "top_bad_rank": rank(float(xb.max())) / k1 if xb.size else math.nan,
        "bads_over_deep_good": float(np.count_nonzero(xb >= xg.min())) if xg.size else math.nan,
    }
    # The drop: where the widest gap in the top sits against the count, and how wide it is in spreads.
    from vtscore.detectors.centroid_head import CENTROID_GAP_TOP, _top_gap_count

    for drop in ("widest", "skip25"):
        gap = _top_gap_count(x, drop=drop)
        f[f"gap_{drop}_over_k1"] = gap / k1
    head = x[: CENTROID_GAP_TOP + 1]
    d = head[:-1] - head[1:]
    f["gap_width_z"] = float(d.max() / sd) if d.size else math.nan
    fit = quad_fit(up, 1.0, 2.5)
    f["quad_curv"] = float(fit[0]) if fit is not None else math.nan
    return f


def _snap(args) -> list[dict]:
    from vtscore.training.thresholds.labels_line import _logit

    ident, s, y, goods, bads = args
    rows = []
    xg, xb = _logit(goods), _logit(bads)
    for scen, (sv, yv) in scenario_corpora(s, y, ident["key"]).items():
        x = _logit(sv)
        if x.size < 50:
            continue
        med, sd = bulk(x)
        if not sd > 0:
            continue
        feat = features(x, xg, xb)
        npos = int(yv.sum())
        base = {
            **{k: ident[k] for k in ("run", "path", "run_beta", "idx", "t", "class", "seed")},
            "scenario": scen, "n": int(x.size), "n_pos": npos,
            "pos_above_line": int(np.count_nonzero(yv[: int(feat["k1"])])),
            "pos_above_z4": int(np.count_nonzero(x[yv == 1] > med + 4 * sd)),
            **feat,
        }  # fmt: skip
        for beta in BETAS:
            r = dict(base, beta=beta)
            rel = rel_grid(beta)
            ks = np.array([gap_and_count(x, beta, feat["count"], c * mult(beta)) for c in rel], dtype=np.int32)
            r["rel"], r["K"] = rel.astype(np.float32), ks
            if npos > 0:
                b2 = beta * beta
                f = (1 + b2) * np.cumsum(yv) / (b2 * npos + np.arange(1, yv.size + 1))
                r["best_f"] = float(f.max())
                r["G"] = f[np.maximum(ks, 1) - 1].astype(np.float32)
            rows.append(r)
    return rows


def cmd_price(a) -> None:
    jobs = [j[:5] for j in _jobs(a.x)]
    if a.limit:
        jobs = jobs[:: max(1, len(jobs) // a.limit)][: a.limit]
    print(f"{len(jobs)} snapshots", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        for out in ex.map(_snap, jobs, chunksize=8):
            rows.extend(out)
    df = pd.DataFrame(rows)
    df.to_pickle(a.out)
    print(f"wrote {len(df)} rows -> {a.out}")


#: The gate features ``tables`` scans (label-legal: what the app has at the line).
FEATS = (
    "count_frac", "tail3_over_gauss", "up3_over_up4", "count_over_quad", "top_z", "goods_above_line", "bads_above_line",
    "bads_above_line_share", "lab_prec_line", "goods_z_med", "goods_z_min", "bads_z_med", "bads_z_max",
    "bads_above_z4", "auc_goods_bads", "deep_good_rank", "med_good_rank", "top_bad_rank", "bads_over_deep_good",
    "gap_widest_over_k1", "gap_skip25_over_k1", "gap_width_z", "quad_curv",
)  # fmt: skip
#: Where the gate does not shrink: #4603's count (``today``) or 1.4 times it; relative to beta ** 0.708.
ALTS = {"today": 1.0, "1.4x": 1.4}
#: The gate the report prices: keep #4603's count where the head's top score sits under this many bulk spreads.
TOP_Z = 6.1


def _load(priced: list[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_pickle(p) for p in priced], ignore_index=True)  # noqa: S301 (price wrote them)
    # Each head at the preset its own session ran; a bench with runs at one preset only lends its heads to both.
    one = df.groupby("path")["run_beta"].transform("nunique") == 1
    df = df[(one | np.isclose(df["run_beta"], df["beta"])) & (df["scenario"] != "empty")].reset_index(drop=True)
    df["cell"] = df["path"] + "/" + df["beta"].map("{:g}".format) + "/" + df["scenario"]
    for name, rel in (("s71", None), *ALTS.items()):
        df[f"F_{name}"] = [
            float(g[np.argmin(np.abs(r - (S71 * b / mult(b) if rel is None else rel)))])
            for g, r, b in zip(df["G"], df["rel"], df["beta"])
        ]  # fmt: skip
    return df.drop(columns=["G", "K", "rel"])


def _cells(df: pd.DataFrame, v: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Per cell, the class mean of *v* (each class once) and its SE over classes."""
    c = df.assign(v=v).groupby(["cell", "class"])["v"].mean().groupby("cell")
    return c.mean(), c.std(ddof=1) / np.sqrt(c.size())


def gated(df: pd.DataFrame, feat: str, thr: float, alt: str) -> np.ndarray:
    """F-beta of the gate: *alt*'s count where *feat* < *thr*, else 0.71 beta."""
    keep = np.nan_to_num(df[feat].to_numpy(dtype=np.float64), nan=np.inf) < thr
    return np.where(keep, df[f"F_{alt}"], df["F_s71"])


def cmd_tables(a) -> None:
    df = _load(a.priced)
    os.makedirs(a.out, exist_ok=True)
    photo = df["path"].isin(["binary", "region"]).to_numpy()
    d = (df["F_today"] - df["F_s71"]).to_numpy()
    today, se_t = _cells(df, d)
    ceiling, _ = _cells(df, np.maximum(d, 0.0))
    g = gated(df, "top_z", TOP_Z, "today") - df["F_s71"]
    gate, se_g = _cells(df, g.to_numpy())
    kept = df.assign(k=np.nan_to_num(df["top_z"], nan=np.inf) < TOP_Z).groupby("cell")["k"].mean()
    cells = pd.DataFrame({"today_everywhere": today, "today_se": se_t, "gate": gate, "gate_se": se_g,
                          "ceiling": ceiling, "share_kept_today": kept}).reset_index()  # fmt: skip
    # The scan: every feature, threshold (photo quantiles), side and alt; chosen on Binary + Region, read on faces.
    rows = []
    for feat in FEATS:
        v = df[feat].to_numpy(dtype=np.float64)
        for q in np.arange(0.1, 0.91, 0.05):
            thr = float(np.nanquantile(v[photo], q))
            for side in (1.0, -1.0):  # keep alt where v < thr, or where v > thr (the feature negated)
                keep = np.nan_to_num(side * v, nan=np.inf) < side * thr
                for alt in ALTS:
                    m, _ = _cells(df, np.where(keep, df[f"F_{alt}"], df["F_s71"]) - df["F_s71"])
                    pm, fm = (
                        m[[c for c in m.index if not c.startswith("face")]],
                        m[[c for c in m.index if c.startswith("face")]],
                    )
                    rows.append({"feature": feat, "keep_alt_if": "below" if side > 0 else "above", "thr": thr,
                                 "alt": alt, "photo_mean": pm.mean(), "photo_min": pm.min(),
                                 "face_mean": fm.mean() if fm.size else math.nan,
                                 "face_min": fm.min() if fm.size else math.nan,
                                 "binary_1pct_b025": m.get("binary/0.25/neg0.44", math.nan)})  # fmt: skip
    scan = pd.DataFrame(rows)
    best = scan.sort_values("photo_mean", ascending=False).groupby(["feature", "alt"]).head(1)
    best = best.sort_values("photo_mean", ascending=False)
    ex = df[(df["scenario"] == "native") & (df["beta"] == 0.25)]
    examples = ex.groupby(["path", "class"]).agg(snapshots=("top_z", "size"), top_z=("top_z", "median"),
        pos_above_line=("pos_above_line", "median"), positives=("n_pos", "median"), k1=("k1", "median"),
        F_today=("F_today", "mean"), F_s71=("F_s71", "mean")).reset_index()  # fmt: skip
    examples = examples[examples["class"].isin(["book", "banana", "bicycle", "toothbrush", "person", "clock"])]
    for name, t in (("cells", cells), ("scan_best", best), ("examples", examples)):
        t.to_csv(os.path.join(a.out, f"offline_{name}.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 60):
            print(f"== {name}\n{t.round(4).to_string(index=False)}", flush=True)
    zq = df[df["beta"] == 1.0].groupby(["path", "scenario"])["top_z"].quantile([0.25, 0.5, 0.75]).unstack()
    print(f"== top_z quartiles at beta 1\n{zq.round(2)}")
    figures(cells, a.figures or a.out)


def figures(cells: pd.DataFrame, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from early_check_4664 import INK, MUTED, _axes_style

    benches = [b for b in BENCH if any(cells["cell"].str.startswith(b + "/"))]
    fig, axes = plt.subplots(1, len(benches), figsize=(4.4 * len(benches), 3.8), squeeze=False, sharey=True)
    for ax, bench in zip(axes[0], benches):
        c = cells[cells["cell"].str.startswith(bench + "/")]
        lab = [s.split("/", 1)[1].replace("neg0.44", "1%").replace("pos1/4", "0.1%").replace("native", "0.44%")
               .replace("/", ", ") for s in c["cell"]]  # fmt: skip
        y = np.arange(len(c))
        ax.barh(
            y + 0.2, c["today_everywhere"], 0.38, xerr=c["today_se"], color="#b8b8b8", label="beta**0.708 everywhere"
        )
        ax.barh(y - 0.2, c["gate"], 0.38, xerr=c["gate_se"], color="#2a78d6", label=f"gate: top < {TOP_Z:g} spreads")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, [f"beta {s}" for s in lab], fontsize=8)
        ax.set_title(BENCH[bench], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("F-beta over 0.71 beta (class mean)", color=MUTED, fontsize=8)
        _axes_style(ax)
    axes[0][0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gate_vs_shipped.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
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
