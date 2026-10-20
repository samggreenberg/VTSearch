"""#5134: Smart in the objective's units - the candidates priced on today's sessions, then the A/B.

``smart_fbeta_5134.sh`` runs today's app with every Smart candidate recorded (``read``: the trend of each
reading on every row, the light unchanged) and the app with Smart read off one candidate (an arm: green once it
stops rising). Three steps:

* ``read``: per run, every session's per-click arrays - the shown set's F-beta at the run's beta (the typed
  query's set until the first click with a detector on screen, then the detector's line, as #5039's A/B scores
  it), Smart's light, the cost window's slope and t, every reading's slope / mean / t / newest value, the vote
  counts, and the click Done was reported (the vote before the first ``done`` pick).
* ``price``: on the reading pass alone, where each rule would first say "you can likely stop", what the session
  has then, and what it gains after (to click 150 and 200); and #4878's two edges, over a stop at a fixed click and
  over the rule's own stops dealt at random, paired by session with the SE over classes. Rules: today's light
  (re-read off ``smart_slope`` / ``smart_slope_t``, checked against the ``smart`` column), each candidate under the
  rising rule at the shipped thresholds and a grid of others, Done, and a placebo that stops N clicks after Smart can
  first read. The best of each family is chosen on seeds 0-1 and read on seeds 2-4.
* ``ab``: each arm against the reading pass, session for session: where Smart first goes green and what is still to
  come there, where Done fires and what it leaves, and the session's mean F-beta over clicks 1-150 and 1-200.

The cells it reads are the study's, run at 58780e78c, where ``smart_slope`` / ``smart_slope_t`` were the cost light's
margins and the app's light was the cost's. Since the ship they are the margins of the light the run reads.

    python smart_fbeta_5134.py read --root /expscratch/$USER/state-of-the-app --date 2026-10-20 \\
        --baseline <text_baseline.csv> --work <dir> [--arms read,ap] [--betas 0.25,1,4] [--seeds 5]
    python smart_fbeta_5134.py price --work <dir> --out <dir>
    python smart_fbeta_5134.py ab --work <dir> --out <dir> [--arms ap]
"""

from __future__ import annotations

import argparse
import glob
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from vtscore.detectors.cost_trend import SMART_FLAT_THRESHOLD as FLAT
from vtscore.detectors.cost_trend import SMART_SLOPE_T as SLOPE_T
from vtscore.detectors.stability import MIN_PER_CLASS

T = 200
HORIZONS = (150, 200)
GRID_T = np.arange(T + 1)
BETAS = (0.25, 1.0, 4.0)
PRESET = {0.25: "β 1/4", 1.0: "β 1", 4.0: "β 4"}
CONTROL = "read"
READINGS = ("ap", "fbeta", "fbeta_best", "heldout_ap", "heldout_fbeta", "line")
PARTS = ("slope", "mean", "t", "last")
#: The rising rule's grid: the relative-slope cutoff, the t cutoff, and what the slope is relative to - the
#: window's mean (the rule as shipped to the arm) or its gap to 1 (how far the score is from perfect).
THETAS = (0.005, 0.01, 0.015, 0.03)
TS = (1.0, 2.0, 3.0)
NORMS = ("mean", "gap")
DELAYS = tuple(range(0, 125, 5))
SELECT_SEEDS = (0, 1)
OPENING = ("good", "bad", "more")
STEP_COLS = [
    "category", "seed", "embedder", "t", "phase", "app_trained", "precision", "recall", "average_precision",
    "gmm_variant", "schedule", "pool_variant", "n_good", "n_bad", "smart", "stable", "span", "smart_slope",
    "smart_slope_t", "line_head_count", *(f"smart_{r}_{p}" for r in READINGS for p in PARTS),
]  # fmt: skip
PICK_COLS = ["category", "seed", "embedder", "t", "phase", "check_kind", "picked_label"]
LIGHT = {"red": 0, "yellow": 1, "green": 2}


def fbeta(p, r, beta: float) -> np.ndarray:
    """F-beta, an empty set (precision NaN, recall 0) scoring 0 (``early_check_4664.fbeta``)."""
    p, r = np.asarray(p, dtype=float), np.asarray(r, dtype=float)
    b2 = beta * beta
    with np.errstate(invalid="ignore", divide="ignore"):
        den = b2 * p + r
        f = np.where(den > 0, (1 + b2) * p * r / den, 0.0)
    f = np.where(np.isfinite(p) & np.isfinite(r), f, np.nan)
    return np.where(~np.isfinite(p) & (r == 0), 0.0, f)


def _headline(m: pd.DataFrame) -> pd.DataFrame:
    for col in ("gmm_variant", "schedule"):
        if col in m.columns:
            m = m[m[col].isna() | m[col].astype(str).str.strip().isin(("", "nan", "None"))]
    if "pool_variant" in m.columns:
        m = m[m["pool_variant"].fillna("").astype(str).str.strip().isin(("", "max"))]
    return m


def _grid(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """*v* at every click 0..T: the last row at or before it, NaN before the first."""
    out = np.full(T + 1, np.nan)
    out[t] = v
    return pd.Series(out).ffill().to_numpy()


def read_cell(path: str, tq: dict, beta: float) -> list[dict]:
    try:
        head = pd.read_csv(path, nrows=0).columns
        m = pd.read_csv(path, usecols=[c for c in STEP_COLS if c in head], low_memory=False)
    except (pd.errors.EmptyDataError, EOFError, OSError):
        # A cell still being written (an interim read): skipped, and counted by the caller.
        return []
    m = _headline(m[m["embedder"] == "siglip"])
    stem = path[: -len(".csv.gz")] if path.endswith(".gz") else path[: -len(".csv")]
    ext = ".csv.gz" if path.endswith(".gz") else ".csv"
    try:
        pk = pd.read_csv(f"{stem}__picks{ext}", usecols=lambda c: c in PICK_COLS, low_memory=False)
        pk = pk[pk["embedder"] == "siglip"]
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pk = pd.DataFrame(columns=PICK_COLS)
    out = []
    for (c, s), g in m.groupby(["category", "seed"]):
        key = f"{c}|{int(s)}"
        ordinary = g[g["phase"].astype(str) != "check"].drop_duplicates("t", keep="last").sort_values("t")
        ordinary = ordinary[ordinary["t"] <= T]
        t = ordinary["t"].to_numpy(dtype=int)
        det = _grid(t, fbeta(ordinary["precision"], ordinary["recall"], beta))
        on = ordinary.loc[ordinary["app_trained"] == 1, "t"]
        shown = float(on.min()) if len(on) else np.inf
        q = tq.get(key, np.nan)
        rec = {
            "category": c,
            "seed": int(s),
            "tq": q,
            "shown": shown,
            "f": np.where((GRID_T >= shown) & np.isfinite(det), det, q),
            "smart": _grid(t, ordinary["smart"].map(LIGHT).to_numpy(dtype=float)),
            "stable": _grid(t, ordinary["stable"].map(LIGHT).to_numpy(dtype=float)),
            "n_good": _grid(t, ordinary["n_good"].to_numpy(dtype=float)),
            "n_bad": _grid(t, ordinary["n_bad"].to_numpy(dtype=float)),
            "cost_slope": _grid(t, ordinary["smart_slope"].to_numpy(dtype=float)),
            "cost_t": _grid(t, ordinary["smart_slope_t"].to_numpy(dtype=float)),
            "head_count": _grid(t, ordinary["line_head_count"].to_numpy(dtype=float)),
        }
        for r in READINGS:
            for p in PARTS:
                col = f"smart_{r}_{p}"
                v = ordinary[col].to_numpy(dtype=float) if col in ordinary else np.full(len(t), np.nan)
                rec[f"{r}_{p}"] = _grid(t, v)
        picks = pk[(pk["category"] == c) & (pk["seed"] == s)]
        done = picks.loc[picks["phase"].astype(str) == "done", "t"]
        # The vote count when Done was reported: the vote before its first ``done`` pick.
        rec["done"] = float(done.min()) - 1 if len(done) else np.nan
        out.append(rec)
    return out


ARRAYS = ("f", "smart", "stable", "n_good", "n_bad", "cost_slope", "cost_t", "head_count") + tuple(
    f"{r}_{p}" for r in READINGS for p in PARTS
)


def read(root: Path, date: str, arms: list[str], betas: list[float], seeds: int, baseline: Path, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(baseline)
    base = base[base["embedder"] == "siglip"]
    for beta in betas:
        tq = {
            f"{c}|{int(s)}": float(fbeta([p], [r], beta)[0])
            for c, s, p, r in base[["category", "seed", "text_precision", "text_recall"]].itertuples(index=False)
        }
        for arm in arms:
            d = root / f"{date}-smart5134-{arm}-b{btag(beta)}" / "results" / "cells"
            cells = sorted(f for f in glob.glob(f"{d}/task_*.csv*") if "__" not in Path(f).name)
            if not cells:
                print(f"{arm} b{btag(beta)}: no cells", flush=True)
                continue
            with Pool(16) as pool:
                parts = pool.map(partial(read_cell, tq=tq, beta=beta), cells, chunksize=4)
            recs = [r for p in parts for r in p if r["seed"] < seeds]
            recs.sort(key=lambda r: (r["seed"], r["category"]))
            np.savez_compressed(
                work / f"{arm}-b{btag(beta)}.npz",
                category=np.array([r["category"] for r in recs]),
                seed=np.array([r["seed"] for r in recs]),
                tq=np.array([r["tq"] for r in recs]),
                shown=np.array([r["shown"] for r in recs]),
                done=np.array([r["done"] for r in recs]),
                **{k: np.stack([r[k] for r in recs]) for k in ARRAYS},
            )
            print(f"{arm} b{btag(beta)}: {len(cells)} cells, {len(recs)} sessions", flush=True)


def btag(beta: float) -> str:
    return {0.25: "025", 1.0: "1", 4.0: "4"}[beta]


def load(work: Path, arm: str, beta: float) -> dict | None:
    p = work / f"{arm}-b{btag(beta)}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- the rules


def _counts_ok(d: dict) -> np.ndarray:
    return (d["n_good"] >= MIN_PER_CLASS) & (d["n_bad"] >= MIN_PER_CLASS)


def cost_green(d: dict) -> np.ndarray:
    """Today's light re-read off its margins: (sessions, T+1) bool."""
    s, t = d["cost_slope"], d["cost_t"]
    with np.errstate(invalid="ignore"):
        return _counts_ok(d) & np.isfinite(s) & ((s >= FLAT) | (t > -SLOPE_T))


def rising_green(d: dict, reading: str, theta: float = -FLAT, tcut: float = SLOPE_T, norm: str = "mean") -> np.ndarray:
    """A candidate's light under the rising rule, re-read off its recorded trend: (sessions, T+1) bool."""
    rel, mean, t = d[f"{reading}_slope"], d[f"{reading}_mean"], d[f"{reading}_t"]
    with np.errstate(invalid="ignore", divide="ignore"):
        if norm == "gap":
            rel = rel * mean / np.maximum(1.0 - mean, 1e-6)
        return _counts_ok(d) & np.isfinite(rel) & ((rel <= theta) | (t < tcut))


def first_click(green: np.ndarray, frm: np.ndarray | None = None) -> np.ndarray:
    """Per session, the first click 1..T the light is green (at or after *frm*); NaN if never."""
    g = green[:, 1:].copy()
    if frm is not None:
        clicks = np.arange(1, T + 1)
        g &= clicks[None, :] >= np.nan_to_num(frm, nan=np.inf)[:, None]
    hit = g.any(axis=1)
    return np.where(hit, g.argmax(axis=1) + 1, np.nan).astype(float)


def smart_readable(d: dict) -> np.ndarray:
    """The first click today's Smart could fit a trend (its slope is finite): the placebo's origin."""
    return first_click(_counts_ok(d) & np.isfinite(d["cost_slope"]))


def rules(d: dict) -> dict[str, np.ndarray]:
    """Every rule's first stop click per session, NaN where it never fires within T."""
    out = {"done": d["done"].astype(float), "cost": first_click(cost_green(d))}
    for r in READINGS:
        for th in THETAS:
            for tc in TS:
                for nm in NORMS:
                    out[f"{r}:{nm}:th{th:g}:t{tc:g}"] = first_click(rising_green(d, r, th, tc, nm))
    origin = smart_readable(d)
    for n in DELAYS:
        stop = origin + n
        out[f"delay:{n}"] = np.where(stop <= T, stop, np.nan)
    return out


def family(rule: str) -> str:
    if rule in ("done", "cost"):
        return rule
    if rule.startswith("delay:"):
        return "delay"
    return rule.split(":")[0]


def _se(x: np.ndarray, cls: np.ndarray) -> float:
    by = pd.Series(x).groupby(cls).mean().dropna()
    return float(by.std(ddof=1) / np.sqrt(len(by))) if len(by) > 1 else float("nan")


def _cls(category: np.ndarray) -> np.ndarray:
    return np.array([str(c).split("@")[0] for c in category])


def score(f: np.ndarray, stop: np.ndarray, cls: np.ndarray, horizon: int) -> dict:
    """What a stop leaves each session with to *horizon*, and #4878's two edges (paired, SE over classes)."""
    n = f.shape[0]
    fired = np.isfinite(stop) & (stop <= horizon)
    leave = np.where(fired, stop, horizon).astype(int)
    fz = np.nan_to_num(f, nan=0.0)
    at = fz[np.arange(n), leave]
    final = fz[:, horizon]
    gain = final - at
    mean_leave = float(leave.mean())
    lo = int(np.floor(mean_leave))
    hi = min(lo + 1, horizon)
    fixed_at = (1 - (mean_leave - lo)) * fz[:, lo] + (mean_leave - lo) * fz[:, hi]
    dealt = fz[:, leave].mean(axis=1)
    return {
        "fire_rate": float(fired.mean()),
        "mean_leave": mean_leave,
        "median_fire": float(np.nanmedian(np.where(fired, stop, np.nan))) if fired.any() else float("nan"),
        "f_at_leave": float(at.mean()),
        "to_come": float(gain.mean()),
        "se_to_come": _se(gain, cls),
        "share_gain_gt_0.1": float((gain[fired] > 0.1).mean()) if fired.any() else float("nan"),
        "edge_fixed": float((at - fixed_at).mean()),
        "se_edge_fixed": _se(at - fixed_at, cls),
        "edge_dealt": float((at - dealt).mean()),
        "se_edge_dealt": _se(at - dealt, cls),
    }


def price(work: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows, held, checks, overclicks = [], [], [], []
    for beta in BETAS:
        d = load(work, CONTROL, beta)
        if d is None:
            continue
        cls = _cls(d["category"])
        rs = rules(d)
        # Today's light re-read off its margins must be the light the run recorded.
        recorded = (d["smart"] == LIGHT["green"])[:, 1:]
        reread = cost_green(d)[:, 1:]
        valid = np.isfinite(d["smart"][:, 1:])
        checks.append(
            {
                "beta": beta,
                "clicks": int(valid.sum()),
                "cost_light_mismatch": int((recorded != reread)[valid].sum()),
            }
        )
        for h in HORIZONS:
            for name, stop in rs.items():
                rows.append(
                    {"beta": beta, "horizon": h, "rule": name, "family": family(name), **score(d["f"], stop, cls, h)}
                )
        # Held out: each family's best by the dealt edge on seeds 0-1, read on seeds 2-4 (horizon 200).
        pick = np.isin(d["seed"], SELECT_SEEDS)
        for fam in sorted({family(n) for n in rs}) if pick.any() and (~pick).any() else ():
            names = [n for n in rs if family(n) == fam]
            sc = {n: score(d["f"][pick], rs[n][pick], cls[pick], T)["edge_dealt"] for n in names}
            best = max(sc, key=lambda n: (sc[n], n))
            held.append(
                {
                    "beta": beta,
                    "family": fam,
                    "rule": best,
                    "chosen_edge_dealt": sc[best],
                    **score(d["f"][~pick], rs[best][~pick], cls[~pick], T),
                }
            )
        # Each candidate's light over clicks at the shipped thresholds, beside today's and the objective.
        greens = {"cost": cost_green(d)} | {r: rising_green(d, r) for r in READINGS}
        for c in range(5, T + 1, 5):
            row = {"beta": beta, "click": c, "f": float(np.nanmean(d["f"][:, c]))}
            for k, g in greens.items():
                row[f"green_{k}"] = float(g[:, c].mean())
            for r in READINGS:
                row[f"{r}_mean"] = (
                    float(np.nanmean(d[f"{r}_mean"][:, c])) if np.isfinite(d[f"{r}_mean"][:, c]).any() else np.nan
                )
            overclicks.append(row)
    pd.DataFrame(rows).to_csv(out / "rules.csv", index=False, float_format="%.5g")
    pd.DataFrame(held).to_csv(out / "held_out.csv", index=False, float_format="%.5g")
    pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)
    pd.DataFrame(overclicks).to_csv(out / "over_clicks.csv", index=False, float_format="%.5g")
    print(pd.DataFrame(checks).to_string(index=False))
    r = pd.DataFrame(rows)
    shipped = r[
        r["rule"].isin(["done", "cost", "delay:0", *(f"{x}:mean:th0.015:t2" for x in READINGS)]) & (r["horizon"] == T)
    ]
    cols = [
        "beta",
        "rule",
        "fire_rate",
        "median_fire",
        "mean_leave",
        "f_at_leave",
        "to_come",
        "share_gain_gt_0.1",
        "edge_fixed",
        "se_edge_fixed",
        "edge_dealt",
        "se_edge_dealt",
    ]
    print(shipped[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(held).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ---------------------------------------------------------------- the A/B


def ab(work: Path, out: Path, arms: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for beta in BETAS:
        c = load(work, CONTROL, beta)
        if c is None:
            continue
        for arm in arms:
            a = load(work, arm, beta)
            if a is None:
                continue
            key_c = {f"{x}|{s}": i for i, (x, s) in enumerate(zip(c["category"], c["seed"], strict=True))}
            pairs = [
                (key_c[f"{x}|{s}"], j)
                for j, (x, s) in enumerate(zip(a["category"], a["seed"], strict=True))
                if f"{x}|{s}" in key_c
            ]
            ic = np.array([p[0] for p in pairs])
            ia = np.array([p[1] for p in pairs])
            cls = _cls(c["category"][ic])
            fc, fa = np.nan_to_num(c["f"][ic], nan=0.0), np.nan_to_num(a["f"][ia], nan=0.0)
            # Each arm's own light: the control's is the cost window's; an arm's is its reading's, as recorded.
            green_c = first_click(c["smart"][ic] == LIGHT["green"])
            green_a = first_click(a["smart"][ia] == LIGHT["green"])
            row = {"beta": beta, "arm": arm, "sessions": len(pairs)}
            for lo, hi in ((1, 150), (1, 200)):
                dm = fa[:, lo : hi + 1].mean(axis=1) - fc[:, lo : hi + 1].mean(axis=1)
                row[f"d_mean_f_{lo}_{hi}"] = float(dm.mean())
                row[f"se_mean_f_{lo}_{hi}"] = _se(dm, cls)
            for name, (sc, sa) in {
                "smart": (green_c, green_a),
                "done": (c["done"][ic].astype(float), a["done"][ia].astype(float)),
            }.items():
                for tag, stop, f in (("ctl", sc, fc), ("arm", sa, fa)):
                    s = score(f, stop, cls, T)
                    row[f"{name}_{tag}_fire"] = s["fire_rate"]
                    row[f"{name}_{tag}_median"] = s["median_fire"]
                    row[f"{name}_{tag}_mean_leave"] = s["mean_leave"]
                    row[f"{name}_{tag}_f_at"] = s["f_at_leave"]
                    row[f"{name}_{tag}_to_come"] = s["to_come"]
                    row[f"{name}_{tag}_to_come150"] = score(f, stop, cls, 150)["to_come"]
                    row[f"{name}_{tag}_edge_dealt"] = s["edge_dealt"]
                    row[f"{name}_{tag}_edge_fixed"] = s["edge_fixed"]
                both = np.isfinite(sc) & np.isfinite(sa)
                if both.any():
                    row[f"{name}_d_click"] = float((sa - sc)[both].mean())
                    row[f"{name}_se_d_click"] = _se((sa - sc)[both], cls[both])
                lc, la = np.where(np.isfinite(sc), sc, T).astype(int), np.where(np.isfinite(sa), sa, T).astype(int)
                dfa = fa[np.arange(len(ia)), la] - fc[np.arange(len(ic)), lc]
                row[f"{name}_d_f_at"] = float(dfa.mean())
                row[f"{name}_se_d_f_at"] = _se(dfa, cls)
            same = (a["smart"][ia] == c["smart"][ic]) | (np.isnan(a["smart"][ia]) & np.isnan(c["smart"][ic]))
            row["share_clicks_same_light"] = float(same[:, 1:].mean())
            rows.append(row)
            for t in range(1, T + 1):
                d = fa[:, t] - fc[:, t]
                curves.append({"beta": beta, "arm": arm, "click": t, "d_f": float(d.mean()), "se": _se(d, cls)})
    pd.DataFrame(curves).to_csv(out / "ab_curves.csv", index=False, float_format="%.5g")
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "ab.csv", index=False, float_format="%.5g")
    print(frame.T.to_string(float_format=lambda v: f"{v:.4f}"))


INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
#: Today's light in neutral grey; the candidates in the reference palette's fixed order.
SERIES = {
    "cost": ("today: FPR + FNR stops falling", "#8a8985"),
    "ap": ("AP of the window stops rising", "#2a78d6"),
    "fbeta_best": ("F-beta of the window stops rising", "#eb6834"),
    "line": ("the line's expected F-beta stops rising", "#1baf7a"),
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, labelcolor=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)


def figures(price_dir: Path, ab_dir: Path | None, out: Path) -> None:
    """``lights.png``: each light's share of sessions green over clicks, above the objective it should track;
    ``ab.png``: each closed-loop arm minus today's app over clicks (paired, ±1 SE over classes)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    oc = pd.read_csv(price_dir / "over_clicks.csv")
    betas = [b for b in BETAS if (oc["beta"] == b).any()]
    fig, axes = plt.subplots(2, len(betas), figsize=(4.2 * len(betas), 6.2), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, beta in enumerate(betas):
        g = oc[oc["beta"] == beta]
        ax = axes[0, j]
        for key, (label, color) in SERIES.items():
            ax.plot(g["click"], g[f"green_{key}"], color=color, lw=2, label=label)
        ax.set_ylim(0, 1)
        ax.set_title(PRESET[beta], color=INK, fontsize=11)
        if j == 0:
            ax.set_ylabel("share of sessions\nwith the light green", color=INK)
        _style(ax)
        ax = axes[1, j]
        ax.plot(g["click"], g["f"], color=INK, lw=2)
        ax.set_xlabel("click", color=INK)
        if j == 0:
            ax.set_ylabel("withheld F-beta (mean)", color=INK)
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "lights.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    if ab_dir is None or not (ab_dir / "ab_curves.csv").exists():
        return
    cv = pd.read_csv(ab_dir / "ab_curves.csv")
    betas = [b for b in BETAS if (cv["beta"] == b).any()]
    fig, axes = plt.subplots(1, len(betas), figsize=(4.2 * len(betas), 3.4), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, beta in enumerate(betas):
        ax = axes[0, j]
        ax.axhline(0, color=MUTED, lw=1)
        for arm, g in cv[cv["beta"] == beta].groupby("arm", sort=False):
            label, color = SERIES.get(arm, (arm, INK))
            ax.fill_between(g["click"], g["d_f"] - g["se"], g["d_f"] + g["se"], color=color, alpha=0.18, lw=0)
            ax.plot(g["click"], g["d_f"], color=color, lw=2, label=f"Smart: {label}")
        ax.set_title(PRESET[beta], color=INK, fontsize=11)
        ax.set_xlabel("click", color=INK)
        if j == 0:
            ax.set_ylabel("withheld F-beta,\narm minus today", color=INK)
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(out / "ab.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--root", type=Path, required=True)
    r.add_argument("--date", default="2026-10-20")
    r.add_argument("--baseline", type=Path, required=True)
    r.add_argument("--work", type=Path, required=True)
    r.add_argument("--arms", default=CONTROL)
    r.add_argument("--betas", default="0.25,1,4")
    r.add_argument("--seeds", type=int, default=5)
    p = sub.add_parser("price")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("ab")
    b.add_argument("--work", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--arms", default="ap")
    f = sub.add_parser("figures")
    f.add_argument("--price", type=Path, required=True)
    f.add_argument("--ab", type=Path)
    f.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "read":
        read(a.root, a.date, a.arms.split(","), [float(x) for x in a.betas.split(",")], a.seeds, a.baseline, a.work)
    elif a.cmd == "price":
        price(a.work, a.out)
    elif a.cmd == "figures":
        figures(a.price, a.ab, a.out)
    else:
        ab(a.work, a.out, a.arms.split(","))


if __name__ == "__main__":
    main()
