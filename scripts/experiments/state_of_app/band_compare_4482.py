"""#4482: what a uniform band pick is worth against an Autopilot pick, as curves over the click a user has reached.

Reads ``band_4482.sh``'s seven arms (Binary Photo, beta 1, paired by cell and seed):

* ``a0``: Autopilot as shipped, to 200 clicks; ``a8`` / ``a4`` / ``a2``: one in 8 / 4 / 2 of the picks past the
  opening is a band pick (``band_share``), to 200 clicks;
* ``c50`` / ``c100`` / ``c150``: as shipped, stopping at that click, so the run's closing check is the check a user
  would run where they stop.

Per run and click it scores what the app shows on the withheld half: the typed query's set until the first click
with a detector on screen (``app_trained``), then the detector's line (carried over clicks with no row). Every run
ends with the app's check, so each also has an "after the check" value: its last ``check`` row.

    python band_compare_4482.py --root <state-of-the-app dir> --date 2026-10-08 --seeds 5 \\
        --baseline <text_baseline.csv> --out <dir> [--docs <report dir>]
"""

from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BETA = 1.0
T = 200
GRID_T = np.arange(T + 1)
ARMS = {"a0": 200, "a8": 200, "a4": 200, "a2": 200, "c50": 50, "c100": 100, "c150": 150}
SHARE = {"a0": None, "a8": 8, "a4": 4, "a2": 2}
NAME = {
    "a0": "Autopilot as shipped",
    "a8": "1 in 8 picks a band pick",
    "a4": "1 in 4 picks a band pick",
    "a2": "1 in 2 picks a band pick",
}
COLOR = {"a0": "#8a8985", "a8": "#2a78d6", "a4": "#1baf7a", "a2": "#eb6834"}
CLICKS = (25, 50, 75, 100, 125, 150, 175, 200)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OPENING = ("good", "bad", "more")
STEP_COLS = [
    "category", "seed", "embedder", "t", "phase", "app_trained", "precision", "recall", "average_precision",
    "gmm_variant", "schedule", "pool_variant",
]  # fmt: skip
PICK_COLS = ["category", "seed", "embedder", "t", "phase", "picked_label"]


def fbeta(p, r, beta: float = BETA) -> np.ndarray:
    """F-beta, with an empty set (precision NaN, recall 0) scoring 0 (``analyze._fbeta_pr``)."""
    p, r = np.asarray(p, dtype=float), np.asarray(r, dtype=float)
    b2 = beta * beta
    with np.errstate(invalid="ignore", divide="ignore"):
        den = b2 * p + r
        f = np.where(den > 0, (1 + b2) * p * r / den, 0.0)
    f = np.where(np.isfinite(p) & np.isfinite(r), f, np.nan)
    return np.where(~np.isfinite(p) & (r == 0), 0.0, f)


def read_cell(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    head = pd.read_csv(path, nrows=0).columns
    m = pd.read_csv(path, usecols=[c for c in STEP_COLS if c in head], low_memory=False)
    m = m[m["embedder"] == "siglip"]
    # The headline row only: every step also carries ~30 diagnostic cut variants (``gmm_variant``), which are
    # not the line the app shows (``handoff_extract_4604.py`` keeps the same rows).
    for col in ("gmm_variant", "schedule"):
        if col in m.columns:
            m = m[m[col].isna() | m[col].astype(str).str.strip().isin(("", "nan", "None"))]
    if "pool_variant" in m.columns:
        m = m[m["pool_variant"].fillna("").astype(str).str.strip().isin(("", "max"))]
    stem = path[: -len(".csv")]
    pk = (
        pd.read_csv(f"{stem}__picks.csv", usecols=lambda c: c in PICK_COLS)
        if os.path.getsize(f"{stem}__picks.csv")
        else None
    )
    if pk is not None:
        pk = pk[pk["embedder"] == "siglip"]
    return m, pk if pk is not None else pd.DataFrame(columns=PICK_COLS)


def load_arm(root: Path, date: str, arm: str, seeds: int, procs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = sorted(f for f in glob.glob(f"{root}/{date}-band4482-{arm}-b1/results/cells/task_*.csv") if "__" not in f)
    with Pool(procs) as pool:
        parts = pool.map(read_cell, cells, chunksize=16)
    steps = pd.concat([p[0] for p in parts if len(p[0])], ignore_index=True)
    picks = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    return steps[steps["seed"] < seeds], picks[picks["seed"] < seeds]


def per_run(steps: pd.DataFrame, picks: pd.DataFrame, tq: dict, last: int) -> dict[str, dict]:
    """Per run: the shown set's F at every click to *last* (NaN past it), the after-check F, and the pick tallies."""
    out: dict[str, dict] = {}
    for (c, s), g in steps.groupby(["category", "seed"]):
        key = f"{c}|{int(s)}"
        q = tq.get(key, np.nan)
        ordinary = g[g["phase"].astype(str) != "check"].drop_duplicates("t", keep="last").sort_values("t")
        ordinary = ordinary[ordinary["t"] <= last]
        det = np.full(T + 1, np.nan)
        det[ordinary["t"].to_numpy(dtype=int)] = fbeta(ordinary["precision"], ordinary["recall"])
        det = pd.Series(det).ffill().to_numpy()
        ap = np.full(T + 1, np.nan)
        ap[ordinary["t"].to_numpy(dtype=int)] = ordinary["average_precision"].to_numpy(dtype=float)
        ap = pd.Series(ap).ffill().to_numpy()
        on = ordinary.loc[ordinary["app_trained"] == 1, "t"]
        shown = float(on.min()) if len(on) else np.inf
        f = np.where((GRID_T >= shown) & np.isfinite(det), det, q)
        f[GRID_T > last] = np.nan
        chk = g[g["phase"].astype(str) == "check"].sort_values("t")
        after = float(fbeta(chk["precision"].iloc[-1:], chk["recall"].iloc[-1:])[0]) if len(chk) else f[last]
        out[key] = {"f": f, "ap": ap, "after": after, "shown": shown, "tq": q}
    for (c, s), g in picks.groupby(["category", "seed"]):
        key = f"{c}|{int(s)}"
        if key not in out:
            out[key] = {"f": np.where(GRID_T <= last, tq.get(key, np.nan), np.nan), "ap": np.full(T + 1, np.nan),
                        "after": tq.get(key, np.nan), "shown": np.inf, "tq": tq.get(key, np.nan)}  # fmt: skip
        ph = g["phase"].astype(str)
        lab = g["picked_label"].to_numpy(dtype=float)
        tt = g["t"].to_numpy(dtype=int)
        goods = np.zeros(T + 1)
        ordinary = ph.to_numpy() != "check"
        np.add.at(goods, np.clip(tt[ordinary], 0, T), lab[ordinary])
        out[key]["goods"] = np.cumsum(goods)
        band = (ph == "band").to_numpy()
        hard = (~ph.isin(OPENING + ("check", "band", "prompt"))).to_numpy()
        out[key].update(
            band_picks=int(band.sum()), band_goods=float(lab[band].sum()), learned_picks=int(hard.sum()),
            learned_goods=float(lab[hard].sum()), check_picks=int((ph == "check").sum()),
        )  # fmt: skip
    return out


def se_over_classes(x: np.ndarray, cls: np.ndarray) -> float:
    by = pd.Series(x).groupby(cls).mean().dropna()
    return float(by.std(ddof=1) / np.sqrt(len(by))) if len(by) > 1 else float("nan")


def _axes_style(ax) -> None:
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--date", default="2026-10-08")
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--docs", type=Path)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    base = pd.read_csv(a.baseline)
    base = base[(base["embedder"] == "siglip") & (base["seed"] < a.seeds)]
    tq = {f"{c}|{int(s)}": float(fbeta([p], [r])[0]) for c, s, p, r in
          zip(base["category"], base["seed"], base["text_line_precision_b1"], base["text_line_recall_b1"])}  # fmt: skip
    runs = {}
    for arm, last in ARMS.items():
        steps, picks = load_arm(a.root, a.date, arm, a.seeds, a.procs)
        runs[arm] = per_run(steps, picks, tq, last)
        print(f"{arm}: {len(runs[arm])} runs")
    keys = sorted(set.intersection(*(set(r) for r in runs.values())))
    keys = [k for k in keys if np.isfinite(runs["a0"][k]["tq"])]
    cls = np.array([k.split("|")[0].split("@")[0] for k in keys])
    band = np.array([k.split("|")[0].split("@")[1] for k in keys])
    F = {arm: np.vstack([runs[arm][k]["f"] for k in keys]) for arm in ARMS}
    AP = {arm: np.vstack([runs[arm][k]["ap"] for k in keys]) for arm in SHARE}
    G = {arm: np.vstack([runs[arm][k].get("goods", np.zeros(T + 1)) for k in keys]) for arm in SHARE}
    after = {arm: np.array([runs[arm][k]["after"] for k in keys]) for arm in ARMS}
    chk = {arm: np.array([runs[arm][k].get("check_picks", 0) for k in keys]) for arm in ARMS}
    a.out.mkdir(parents=True, exist_ok=True)
    outs = [a.out] + ([a.docs] if a.docs else [])

    # 1. The pick mix: each arm's unchecked curve and its gain over shipped Autopilot, paired.
    rows = []
    for arm in SHARE:
        r = {"arm": arm, "runs": len(keys)}
        for c in CLICKS:
            r[f"F_at_{c}"] = np.nanmean(F[arm][:, c])
            g = F[arm][:, c] - F["a0"][:, c]
            r[f"gain_at_{c}"], r[f"se_at_{c}"] = np.nanmean(g), se_over_classes(g, cls)
            r[f"AP_at_{c}"] = np.nanmean(AP[arm][:, c])
            r[f"goods_at_{c}"] = np.nanmean(G[arm][:, c])
        for lo, hi in ((1, 50), (51, 100), (101, 150), (151, 200), (1, 200)):
            g = np.nanmean(F[arm][:, lo : hi + 1] - F["a0"][:, lo : hi + 1], axis=1)
            r[f"gain_{lo}_{hi}"], r[f"se_{lo}_{hi}"] = np.nanmean(g), se_over_classes(g, cls)
        r["after_check_200"] = np.nanmean(after[arm])
        g = after[arm] - after["a0"]
        r["after_gain_200"], r["after_se_200"] = np.nanmean(g), se_over_classes(g, cls)
        bp = np.array([runs[arm][k].get("band_picks", 0) for k in keys])
        bg = np.array([runs[arm][k].get("band_goods", 0) for k in keys])
        lp = np.array([runs[arm][k].get("learned_picks", 0) for k in keys])
        lg = np.array([runs[arm][k].get("learned_goods", 0) for k in keys])
        r["band_picks"], r["band_good_share"] = bp.mean(), bg.sum() / max(bp.sum(), 1)
        r["autopilot_picks"], r["autopilot_good_share"] = lp.mean(), lg.sum() / max(lp.sum(), 1)
        rows.append(r)
    mix = pd.DataFrame(rows)

    # 2. The check where the user stops: after-check against unchecked at the same click, and per pick.
    crow = []
    for arm, n in (("c50", 50), ("c100", 100), ("c150", 150), ("a0", 200)):
        unchecked = F[arm][:, n]
        g = after[arm] - unchecked
        # What the same number of further Autopilot clicks bought, on a0's sessions (to 200).
        k = np.round(chk[arm]).astype(int)
        further = np.array(
            [F["a0"][i, min(n + k[i], T)] - F["a0"][i, n] if n + k[i] <= T else np.nan for i in range(len(keys))]
        )
        crow.append({
            "stop_at": n, "runs": len(keys), "unchecked": np.nanmean(unchecked), "after_check": np.nanmean(after[arm]),
            "check_gain": np.nanmean(g), "check_se": se_over_classes(g, cls), "check_picks": chk[arm].mean(),
            "check_gain_per_pick": np.nanmean(g) / max(chk[arm].mean(), 1e-9),
            "same_clicks_autopilot_gain": np.nanmean(further), "same_clicks_se": se_over_classes(further, cls),
            "autopilot_gain_per_click": np.nanmean(further) / max(chk[arm].mean(), 1e-9),
        })  # fmt: skip
    stops = pd.DataFrame(crow)

    # 3. By band, the mixes' gain over clicks 1-200.
    brow = []
    for arm in ("a8", "a4", "a2"):
        g = np.nanmean(F[arm][:, 1:] - F["a0"][:, 1:], axis=1)
        for bd in sorted(set(band)):
            m = band == bd
            brow.append({"arm": arm, "band": bd, "runs": int(m.sum()), "gain_1_200": g[m].mean(),
                         "se": se_over_classes(g[m], cls[m]), "better": float(np.mean(g[m] > 1e-9)),
                         "worse": float(np.mean(g[m] < -1e-9))})  # fmt: skip
    bands = pd.DataFrame(brow)

    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
        mix.to_csv(o / "band_mix.csv", index=False)
        stops.to_csv(o / "band_check_at_stop.csv", index=False)
        bands.to_csv(o / "band_by_band.csv", index=False)
        pd.DataFrame({"t": GRID_T, **{arm: np.nanmean(F[arm], axis=0) for arm in ARMS}}).to_csv(
            o / "band_curves.csv", index=False
        )
    figs = [a.out] + ([a.docs.parent / "figures"] if a.docs else [])
    for o in figs:
        o.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        ax = axes[0]
        for arm in SHARE:
            ax.plot(GRID_T, np.nanmean(F[arm], axis=0), color=COLOR[arm], lw=2, label=NAME[arm])
        for arm, n in (("c50", 50), ("c100", 100), ("c150", 150), ("a0", 200)):
            ax.plot([n], [np.nanmean(after[arm])], "o", ms=7, mfc="none", mec=INK, mew=1.4)
        ax.plot(
            [], [], "o", ms=7, mfc="none", mec=INK, mew=1.4, label="as shipped, after the check where the user stops"
        )
        ax.set_title("Mean F1 of the returned set over clicks", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("clicks", color=MUTED, fontsize=9)
        ax.set_ylabel("F1 of the returned set (withheld half)", color=MUTED, fontsize=9)
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax = axes[1]
        for arm in ("a8", "a4", "a2"):
            g = F[arm] - F["a0"]
            m = np.nanmean(g, axis=0)
            by = pd.DataFrame(g).groupby(cls).mean()
            se = by.std(ddof=1).to_numpy() / np.sqrt(len(by))
            ax.fill_between(GRID_T, m - se, m + se, color=COLOR[arm], alpha=0.18, lw=0)
            ax.plot(GRID_T, m, color=COLOR[arm], lw=2, label=NAME[arm])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title("Paired gain over Autopilot as shipped (± 1 SE over classes)", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("clicks", color=MUTED, fontsize=9)
        _axes_style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(o / "band_over_clicks.png", dpi=150)
        plt.close(fig)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(mix[["arm", "runs"] + [f"gain_{lo}_{hi}" for lo, hi in ((1, 50), (51, 100), (101, 150), (151, 200), (1, 200))]
                  + ["after_gain_200", "band_picks", "band_good_share", "autopilot_good_share"]].round(3).to_string(index=False))  # fmt: skip
        print(mix[["arm"] + [f"F_at_{c}" for c in CLICKS]].round(3).to_string(index=False))
        print(
            mix[["arm"] + [f"AP_at_{c}" for c in (50, 100, 150, 200)] + [f"goods_at_{c}" for c in (50, 100, 150, 200)]]
            .round(3)
            .to_string(index=False)
        )
        print(stops.round(4).to_string(index=False))
        print(bands.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
