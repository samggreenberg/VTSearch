"""#4384: the early head started from the typed query, priced on full sessions.

Two session sets a preset (``queryprior_4384.sh``): the control (today's app, with the shadow
``task_NNNN__queryprior.csv``) and the arm (``CALIB_QUERY_PRIOR=<scale>``: every head fit also takes the typed
query's direction). Per run and vote:

- **shown**: the app's set, the untagged rows' F-beta at the run's beta (the typed query's set while #4736's hold
  lasts, the head's after it). This is the objective.
- **head**: the trained head's set at every vote it is drawn (the hold's ``hold:head`` rows, then the untagged
  ``trained`` rows), so the hold can be re-priced offline: what is shown changes no pick.
- **the shadow's candidates** (control): AP and the F-beta of the top *K*, where *K* is what the app's head keeps
  on the withheld half at that vote, from the positives' ranks under each candidate head.

Every run is in every average (#4631); a vote before a run's first head shows the typed query under every rule.
SEs are clustered on the 48 classes.

    python queryprior_4384.py extract --cells <run>/results/cells --beta 1 --out X.npz
    python queryprior_4384.py shadow --x b025=X025.npz --x b1=X1.npz --x b4=X4.npz --out <dir>
    python queryprior_4384.py pair --pair b1=ARM.npz::CTL.npz [...] --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))
import _cells_io  # noqa: E402

T = 150
GRID = np.arange(T + 1)
HOLD_TAG = "hold:head"
COLS = [
    "category", "seed", "t", "phase", "n_good", "n_bad", "app_trained", "detector_tier", "beta", "precision",
    "recall", "average_precision", "n_flagged", "n_test_pos", "n_test_neg", "gmm_variant", "pool_variant",
    "schedule",
]  # fmt: skip
SPANS = {"1-10": (1, 10), "1-25": (1, 25), "1-50": (1, 50), "51-150": (51, 150), "1-150": (1, 150)}
HOLDS = (0, 10, 15, 20, 25, 30, 40)


def fbeta_pr(p: np.ndarray, r: np.ndarray, beta: float) -> np.ndarray:
    """F-beta from precision and recall; an empty set (precision NaN, recall 0) scores 0 (as #4736's analyzer)."""
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    b2 = beta * beta
    with np.errstate(invalid="ignore", divide="ignore"):
        den = b2 * p + r
        f = np.where(den > 0, (1 + b2) * p * r / den, 0.0)
    return np.where(np.isfinite(p) & np.isfinite(r), f, np.where(r == 0, 0.0, np.nan))


def carry(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Values at votes *t* on 0..T, carried forward over votes with none; NaN before the first."""
    out = np.full(T + 1, np.nan)
    t = np.asarray(t, dtype=int)
    keep = (t >= 0) & (t <= T)
    out[t[keep]] = np.asarray(v, dtype=float)[keep]
    return pd.Series(out).ffill().to_numpy()


def rank_metrics(ranks: str, n_pos: int, k: float, beta: float) -> tuple[float, float, float]:
    """``(AP, F-beta of the top k, best F-beta over every k)`` from the positives' 0-based ranks."""
    if not isinstance(ranks, str) or not ranks.strip() or n_pos <= 0:
        return np.nan, np.nan, np.nan
    r = np.sort(np.array(ranks.split(), dtype=np.int64))
    hits = np.arange(1, len(r) + 1)
    ap = float(np.sum(hits / (r + 1)) / n_pos)
    b2 = beta * beta

    def f_at(kk: np.ndarray) -> np.ndarray:
        tp = np.searchsorted(r, kk, side="left").astype(float)
        fp = kk - tp
        fn = n_pos - tp
        den = (1 + b2) * tp + b2 * fn + fp
        return np.where(den > 0, (1 + b2) * tp / den, 0.0)

    fk = float(f_at(np.array([max(0.0, k)]))[0]) if np.isfinite(k) else np.nan
    best = float(f_at((r + 1).astype(float)).max())
    return ap, fk, best


def read_cell(args: tuple[str, float]) -> dict | None:
    path, beta = args
    try:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
        picks = pd.read_csv(
            path.replace(".csv", "__picks.csv"), usecols=["t", "phase", "picked_label", "n_good", "n_bad"]
        )
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    if df.empty:
        return None
    category, seed = str(df["category"].iloc[0]), int(df["seed"].iloc[0])
    df = df[~_cells_io.check_rows(df) & np.isclose(df["beta"].astype(float), beta) & (df["t"] <= T)]
    df = df.assign(F=fbeta_pr(df["precision"].to_numpy(), df["recall"].to_numpy(), beta))
    base = _cells_io._base_rows(df).drop_duplicates("t", keep="last").sort_values("t")
    tag = df["gmm_variant"].fillna("").astype(str) == HOLD_TAG
    held = df[tag].drop_duplicates("t", keep="last")
    trained = base[base["detector_tier"].astype(str) == "trained"]
    head = pd.concat([held, trained]).drop_duplicates("t", keep="last").sort_values("t")
    query = base[base["detector_tier"].astype(str) == "query"]
    out: dict = {"category": category, "seed": seed}
    out["shown"] = carry(base["t"], base["F"])
    out["shown_ap"] = carry(base["t"], base["average_precision"])
    out["head"] = carry(head["t"], head["F"])
    out["head_ap"] = carry(head["t"], head["average_precision"])
    out["k_head"] = carry(head["t"], head["n_flagged"])
    out["query_F"] = float(query["F"].iloc[0]) if len(query) else np.nan
    out["query_ap"] = float(query["average_precision"].iloc[0]) if len(query) else np.nan
    out["query_k"] = float(query["n_flagged"].iloc[0]) if len(query) else np.nan
    out["first_head"] = float(head["t"].min()) if len(head) else np.inf
    on = base.loc[base["app_trained"] == 1, "t"]
    out["app_from"] = float(on.min()) if len(on) else np.inf
    p = picks[picks["t"] <= T].drop_duplicates("t", keep="last")
    out["n_good"] = carry(p["t"], p["n_good"])
    out["n_bad"] = carry(p["t"], p["n_bad"])
    qp_path = Path(path.replace(".csv", "__queryprior.csv"))
    try:
        qp = pd.read_csv(qp_path) if qp_path.exists() else pd.DataFrame()
    except (pd.errors.EmptyDataError, ValueError):
        qp = pd.DataFrame()
    variants = [c[len("ranks_") :] for c in qp.columns if c.startswith("ranks_")]
    out["variants"] = variants
    qp = qp[qp["t"] <= T].drop_duplicates("t", keep="last") if len(qp) else qp
    k_at = pd.Series(out["k_head"])
    for v in variants:
        ap, fk, fq, best = [], [], [], []
        for t, ranks, npos in zip(qp["t"], qp[f"ranks_{v}"], qp["n_test_pos"]):
            a_, f_, b_ = rank_metrics(ranks, int(npos), float(k_at.iloc[int(t)]), beta)
            _a, fq_, _b = rank_metrics(ranks, int(npos), out["query_k"], beta)
            ap.append(a_)
            fk.append(f_)
            fq.append(fq_)
            best.append(b_)
        out[f"sh_ap_{v}"] = carry(qp["t"], ap)
        out[f"sh_fk_{v}"] = carry(qp["t"], fk)
        out[f"sh_fq_{v}"] = carry(qp["t"], fq)
        out[f"sh_best_{v}"] = carry(qp["t"], best)
    return out


def extract(a: argparse.Namespace) -> None:
    cells = sorted(str(p) for p in Path(a.cells).glob("task_[0-9]*.csv*") if "__" not in p.name)
    with ProcessPoolExecutor(a.workers) as ex:
        got = list(ex.map(read_cell, [(c, a.beta) for c in cells], chunksize=4))
    runs = [g for g in got if g is not None]
    print(f"{a.cells}: {len(cells)} cells, {len(runs)} read")
    variants = sorted({tuple(r["variants"]) for r in runs if r["variants"]})
    if len(variants) > 1:
        raise SystemExit(f"cells disagree on the shadow's variants: {variants}")
    vs = list(variants[0]) if variants else []
    series = ["shown", "shown_ap", "head", "head_ap", "k_head", "n_good", "n_bad"]
    series += [f"sh_{m}_{v}" for v in vs for m in ("ap", "fk", "fq", "best")]
    nan = np.full(T + 1, np.nan)
    arrays = {k: np.stack([r.get(k, nan) for r in runs]) for k in series}
    np.savez_compressed(
        a.out,
        beta=a.beta,
        variants=np.array(vs),
        category=np.array([r["category"] for r in runs]),
        seed=np.array([r["seed"] for r in runs]),
        first_head=np.array([r["first_head"] for r in runs]),
        app_from=np.array([r["app_from"] for r in runs]),
        query_F=np.array([r["query_F"] for r in runs]),
        query_ap=np.array([r["query_ap"] for r in runs]),
        query_k=np.array([r["query_k"] for r in runs]),
        **arrays,
    )
    print(f"wrote {a.out} ({len(runs)} runs, shadow variants: {vs or 'none'})")


def clustered_se(d: np.ndarray, cls: np.ndarray) -> float:
    """SE of the mean of *d* clustered on *cls* (CR0)."""
    ok = np.isfinite(d)
    d, cls = d[ok], cls[ok]
    n = len(d)
    if n < 2:
        return np.nan
    e = d - d.mean()
    sums = pd.Series(e).groupby(cls).sum().to_numpy()
    return float(np.sqrt(np.sum(sums**2)) / n)


def span_mean(x: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    lo, hi = span
    return np.nanmean(x[:, lo : hi + 1], axis=1)


def fill_query(x: np.ndarray, z: dict, key: str = "query_F") -> np.ndarray:
    """*x* with the typed query's value at every vote before the run's first head (what the app shows there)."""
    q = z[key][:, None]
    return np.where(np.isfinite(x), x, np.broadcast_to(q, x.shape))


def load(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["cls"] = np.array([c.split("@")[0] for c in d["category"]])
    return d


def held(z: dict, hold: int) -> np.ndarray:
    """The objective had the typed query's set been held to *hold* labels: the query's F, then the head's."""
    labels = z["n_good"] + z["n_bad"]
    head = fill_query(z["head"], z)
    return np.where(np.nan_to_num(labels, nan=0) >= hold, head, z["query_F"][:, None])


def fmt(x: float, digits: int = 3) -> str:
    return "" if not np.isfinite(x) else f"{x:+.{digits}f}"


def shadow(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# #4384 shadow: candidate heads on the control's votes\n"]
    rows = []
    for spec in a.x:
        preset, path = spec.split("=", 1)
        z = load(path)
        vs = [str(v) for v in z["variants"]]
        n = len(z["category"])
        # Checks: the shadow's ``head`` is the run's own head.
        ap_gap = np.nanmax(np.abs(z["sh_ap_head"] - z["head_ap"]))
        f_gap = np.nanmax(np.abs(z["sh_fk_head"] - z["head"]))
        lines.append(f"\n## {preset}: {n} runs\n\ncheck: |shadow head AP - run head AP| max {ap_gap:.4f}; "
                     f"|shadow head F@K - run head F| max {f_gap:.4f}\n")  # fmt: skip
        for metric, key, fill in (("AP", "ap", "query_ap"), ("F-beta at the app's head count", "fk", "query_F")):
            lines.append(f"\n### {metric}, mean over runs, the typed query before a run's first head\n")
            votes = [7, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150]
            lines.append("| candidate | " + " | ".join(f"v{v}" for v in votes) + " | 1-25 | 1-50 | 1-150 |")
            lines.append("|---" * (len(votes) + 4) + "|")
            base = fill_query(z[f"sh_{key}_head"], z, fill)
            for v in vs:
                x = fill_query(z[f"sh_{key}_{v}"], z, fill)
                cells = [f"{np.nanmean(x[:, t]):.3f}" for t in votes]
                spans = []
                for s in ("1-25", "1-50", "1-150"):
                    m = span_mean(x, SPANS[s])
                    d = m - span_mean(base, SPANS[s])
                    spans.append(
                        f"{np.nanmean(m):.3f}"
                        + ("" if v == "head" else f" ({fmt(np.nanmean(d))}±{clustered_se(d, z['cls']):.3f})")
                    )
                    rows.append({"preset": preset, "metric": key, "candidate": v, "span": s,
                                 "mean": float(np.nanmean(m)), "d_vs_head": float(np.nanmean(d)),
                                 "se": clustered_se(d, z["cls"])})  # fmt: skip
                lines.append(f"| {v} | " + " | ".join(cells) + " | " + " | ".join(spans) + " |")
            q = z[fill]
            lines.append(f"\ntyped query: {np.nanmean(q):.3f}\n")
    pd.DataFrame(rows).to_csv(out / "shadow.csv", index=False)
    (out / "shadow.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def pair(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# #4384 arm against control, paired per run\n"]
    rows = []
    for spec in a.pair:
        preset, rest = spec.split("=", 1)
        arm_p, ctl_p = rest.split("::")
        A, C = load(arm_p), load(ctl_p)
        ka = {(c, int(s)): i for i, (c, s) in enumerate(zip(A["category"], A["seed"]))}
        kc = {(c, int(s)): i for i, (c, s) in enumerate(zip(C["category"], C["seed"]))}
        common = sorted(set(ka) & set(kc))
        ia = np.array([ka[k] for k in common])
        ic = np.array([kc[k] for k in common])
        cls = A["cls"][ia]
        seeds = A["seed"][ia]
        lines.append(f"\n## {preset}: {len(common)} paired runs (arm {len(ka)}, control {len(kc)})\n")
        qgap = np.nanmax(np.abs(A["query_F"][ia] - C["query_F"][ic]))
        lines.append(f"check: typed query's F identical across arms: max gap {qgap:.2e}\n")

        def table(title: str, xa: np.ndarray, xc: np.ndarray, what: str) -> None:
            lines.append(
                f"\n### {title}\n\n| span | control | arm | arm - control | seeds (min..max) |\n|---|---|---|---|---|"
            )
            for s, sp in SPANS.items():
                ma, mc = span_mean(xa, sp), span_mean(xc, sp)
                d = ma - mc
                per_seed = [np.nanmean(d[seeds == sd]) for sd in np.unique(seeds)]
                lines.append(f"| {s} | {np.nanmean(mc):.4f} | {np.nanmean(ma):.4f} | {fmt(np.nanmean(d), 4)} ± "
                             f"{clustered_se(d, cls):.4f} | {fmt(min(per_seed), 4)}..{fmt(max(per_seed), 4)} |")  # fmt: skip
                rows.append({"preset": preset, "what": what, "span": s, "control": float(np.nanmean(mc)),
                             "arm": float(np.nanmean(ma)), "d": float(np.nanmean(d)), "se": clustered_se(d, cls),
                             "seed_min": float(min(per_seed)), "seed_max": float(max(per_seed))})  # fmt: skip

        table("The objective: the set the app shows (each arm with #4736's 30-label hold)",
              A["shown"][ia], C["shown"][ic], "shown")  # fmt: skip
        table("The head's set at every vote it is drawn (hidden during the hold)",
              fill_query(A["head"], A)[ia], fill_query(C["head"], C)[ic], "head")  # fmt: skip
        table("AP of what the app shows", A["shown_ap"][ia], C["shown_ap"][ic], "shown_ap")
        table("AP of the head", fill_query(A["head_ap"], A, "query_ap")[ia],
              fill_query(C["head_ap"], C, "query_ap")[ic], "head_ap")  # fmt: skip
        lines.append("\n### Goods in hand\n\n| vote | control | arm | arm - control |\n|---|---|---|---|")
        for t in (10, 25, 50, 100, 150):
            ga, gc = A["n_good"][ia, t], C["n_good"][ic, t]
            d = ga - gc
            lines.append(
                f"| {t} | {np.nanmean(gc):.2f} | {np.nanmean(ga):.2f} | {fmt(np.nanmean(d), 2)} ± {clustered_se(d, cls):.2f} |"
            )
            rows.append({"preset": preset, "what": "goods", "span": str(t), "control": float(np.nanmean(gc)),
                         "arm": float(np.nanmean(ga)), "d": float(np.nanmean(d)), "se": clustered_se(d, cls)})  # fmt: skip
        lines.append("\n### The hold re-priced (offline; the hold changes no pick): session mean 1-150 and 1-25\n")
        lines.append(
            "| hold (labels) | control 1-25 | arm 1-25 | control 1-150 | arm 1-150 | arm@H - control@30, 1-150 |"
        )
        lines.append("|---|---|---|---|---|---|")
        c30 = span_mean(held(C, 30)[ic], SPANS["1-150"])
        for h in HOLDS:
            hc, ha = held(C, h)[ic], held(A, h)[ia]
            d = span_mean(ha, SPANS["1-150"]) - c30
            lines.append(f"| {h} | {np.nanmean(span_mean(hc, SPANS['1-25'])):.4f} | {np.nanmean(span_mean(ha, SPANS['1-25'])):.4f} | "
                         f"{np.nanmean(span_mean(hc, SPANS['1-150'])):.4f} | {np.nanmean(span_mean(ha, SPANS['1-150'])):.4f} | "
                         f"{fmt(np.nanmean(d), 4)} ± {clustered_se(d, cls):.4f} |")  # fmt: skip
            rows.append({"preset": preset, "what": f"hold{h}", "span": "1-150",
                         "control": float(np.nanmean(span_mean(hc, SPANS['1-150']))),
                         "arm": float(np.nanmean(span_mean(ha, SPANS['1-150']))), "d": float(np.nanmean(d)),
                         "se": clustered_se(d, cls)})  # fmt: skip
        sh_gap = np.nanmax(np.abs(held(A, 30)[ia] - A["shown"][ia]))
        sc_gap = np.nanmax(np.abs(held(C, 30)[ic] - C["shown"][ic]))
        lines.append(
            f"\ncheck: the hold at 30 rebuilt from head + query vs the shown rows: max gap arm {sh_gap:.2e}, control {sc_gap:.2e}\n"
        )
        # Per-vote curves for the figures.
        curves = pd.DataFrame({"vote": GRID})
        for name, x in (("ctl_shown", C["shown"][ic]), ("arm_shown", A["shown"][ia]),
                        ("ctl_head", fill_query(C["head"], C)[ic]), ("arm_head", fill_query(A["head"], A)[ia]),
                        ("ctl_head_ap", fill_query(C["head_ap"], C, "query_ap")[ic]),
                        ("arm_head_ap", fill_query(A["head_ap"], A, "query_ap")[ia])):  # fmt: skip
            curves[name] = np.nanmean(x, axis=0)
        curves["query_F"] = np.nanmean(C["query_F"][ic])
        curves["query_ap"] = np.nanmean(C["query_ap"][ic])
        curves.to_csv(out / f"curves_{preset}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "pair.csv", index=False)
    (out / "pair.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--cells", required=True)
    e.add_argument("--beta", type=float, required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("shadow")
    s.add_argument("--x", action="append", required=True, help="PRESET=X.npz")
    s.add_argument("--out", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--pair", action="append", required=True, help="PRESET=ARM.npz::CTL.npz")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"extract": extract, "shadow": shadow, "pair": pair}[a.cmd](a)


if __name__ == "__main__":
    main()
