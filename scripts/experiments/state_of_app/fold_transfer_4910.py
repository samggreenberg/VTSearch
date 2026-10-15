"""#4910: price the quantile transfer inside the labels line.

Reads ``fold_transfer_4910.sh``'s arms on one path (Binary or Region), at the presets given:

* **The gate** (arm ``c``, today's line): every base row also carries the quantile line fitted on the same head
  beside it (``shadow_*``) and how far the transfer moved the folds' held-out scores and the class model
  (``xfer_*``).  Read per row (logit shifts, the share of held-out Goods above their fold's haystack, the class
  model's moves, the Train cut's move, the withheld set's size), and as the objective the same heads would have
  scored with the transferred line: the session mean of F-beta over votes 1-150 and 1-50, today's line's F
  where the shadow has none (the opening, the centroid's rows).  No session ran the transferred line here, so
  this is the line's own effect, not what it does to acquisition, the weak check or Done.
* **The closed loop** (arms ``q`` / ``qt`` against ``c``): the same session means, each arm's own sessions,
  paired by (category, seed).  Pairing check: the picks must match through the opening (Good, Bad, More on the
  typed query's sort), where no labels line is read; an arm leaves the control at its first learned pick at the
  earliest, since acquisition reads the line from the first trained step.  (The rows' first trained Test, at the
  query hold's 30 labels, comes long after.)

F at a click is the line's F-beta on the withheld half at the row's beta (an empty set scores 0), carried
forward between retrains; before the head is first shown it is the typed query's own (``--baseline``), the
same in every arm.  SEs treat the class as the unit (a class's bands and seeds are not independent).

    python fold_transfer_4910.py --root <state-of-the-app dir> --date 2026-10-15 --seeds 5 --path binary \\
        --baseline <text_baseline.csv> --out <dir> [--docs <docs/experiments/.../data>] [--betas 0.25,1,4]
"""

from __future__ import annotations

import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

T = 150
GRID_T = np.arange(T + 1)
WINDOWS = ((1, 150), (1, 50), (51, 150))
EMBEDDER = {"binary": "siglip", "region": "siglip+dinov3_patch"}
BTAG = {0.25: "025", 1.0: "1", 4.0: "4"}
SEPARATION_D = 1.5  # labels_line.WEAK_SEPARATION_D, the weak check's trigger
COLS = [
    "category", "seed", "embedder", "t", "phase", "app_trained", "detector_tier", "fbeta", "precision", "recall",
    "n_flagged", "train_threshold", "train_prevalence", "line_separation", "line_head_count", "n_cal_scores",
    "shadow_train_threshold", "shadow_threshold", "shadow_n_flagged", "shadow_fbeta", "shadow_separation",
    "xfer_dlogit_med", "xfer_dlogit_p90", "xfer_clip_good", "xfer_dmu_pos", "xfer_dmu_neg", "xfer_dsigma",
    "gmm_variant", "schedule", "pool_variant",
]  # fmt: skip


def fbeta(p, r, beta: float) -> np.ndarray:
    """F-beta, with an empty set (precision NaN, recall 0) scoring 0 (``analyze._fbeta_pr``)."""
    p, r = np.asarray(p, dtype=float), np.asarray(r, dtype=float)
    b2 = beta * beta
    with np.errstate(invalid="ignore", divide="ignore"):
        den = b2 * p + r
        f = np.where(den > 0, (1 + b2) * p * r / den, 0.0)
    f = np.where(np.isfinite(p) & np.isfinite(r), f, np.nan)
    return np.where(~np.isfinite(p) & (r == 0), 0.0, f)


def _headline(m: pd.DataFrame) -> pd.DataFrame:
    """The base row only: every step also carries ~30 diagnostic cut variants."""
    for col in ("gmm_variant", "schedule"):
        if col in m.columns:
            m = m[m[col].isna() | m[col].astype(str).str.strip().isin(("", "nan", "None"))]
    return m[m["pool_variant"].fillna("").astype(str).str.strip().isin(("", "max"))]


OPENING = ("good", "bad", "more")


def read_cell(args: tuple[str, str]) -> pd.DataFrame:
    path, emb = args
    head = pd.read_csv(path, nrows=0).columns
    m = pd.read_csv(path, usecols=[c for c in COLS if c in head], low_memory=False)
    m = _headline(m[m["embedder"] == emb])
    # The pick log's opening, for the pairing check: the picked ids in order up to the opening's last pick.
    picks_path = (
        path.replace(".csv.gz", "__picks.csv.gz") if path.endswith(".gz") else path.replace(".csv", "__picks.csv")
    )
    try:
        pk = pd.read_csv(picks_path, usecols=["category", "seed", "embedder", "phase", "picked_id"], low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
        return m
    pk = pk[pk["embedder"] == emb]
    opening = {}
    for (c, sd), g in pk.groupby(["category", "seed"]):
        ph = g["phase"].astype(str).to_numpy()
        last = max((i for i, p in enumerate(ph) if p in OPENING), default=-1)
        opening[(c, int(sd))] = ",".join(str(int(x)) for x in g["picked_id"].to_numpy()[: last + 1])
    return m.assign(opening_picks=[opening.get((c, int(sd)), "") for c, sd in zip(m["category"], m["seed"])])


def run_dir(root: Path, date: str, tag: str, arm: str, path: str, beta: float) -> Path:
    return root / f"{date}-xfer4910{tag}-{arm}-{path}-b{BTAG[beta]}" / "results" / "cells"


def load_arm(d: Path, emb: str, seeds: int, procs: int) -> pd.DataFrame:
    cells = sorted(f for f in glob.glob(f"{d}/task_*.csv*") if "__" not in Path(f).name)
    if not cells:
        return pd.DataFrame(columns=COLS)
    with Pool(procs) as pool:
        parts = pool.map(read_cell, [(c, emb) for c in cells], chunksize=4)
    steps = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return steps[steps["seed"] < seeds]


def per_run(steps: pd.DataFrame, tq: dict, beta: float) -> dict[str, dict]:
    """Per run: F at every click under the line it drew, and under its shadow on the same heads."""
    out: dict[str, dict] = {}
    for (c, s), g in steps.groupby(["category", "seed"]):
        key = f"{c}|{int(s)}"
        rows = g[g["phase"].astype(str) != "check"].drop_duplicates("t", keep="last").sort_values("t")
        rows = rows[rows["t"] <= T]
        t = rows["t"].to_numpy(dtype=int)
        own = rows["fbeta"].to_numpy(dtype=float)
        sh = rows["shadow_fbeta"].to_numpy(dtype=float)
        shadow = np.where(np.isfinite(sh), sh, own)
        on = rows.loc[rows["app_trained"] == 1, "t"]
        shown = float(on.min()) if len(on) else np.inf
        q = tq.get(key, np.nan)
        curves = {}
        for name, v in (("f", own), ("fs", shadow)):
            det = np.full(T + 1, np.nan)
            det[t] = v
            det = pd.Series(det).ffill().to_numpy()
            curves[name] = np.where((GRID_T >= shown) & np.isfinite(det), det, q)
        out[key] = {
            **curves,
            "cls": c,
            "opening": str(g["opening_picks"].iloc[0]) if "opening_picks" in g.columns else "",
        }
    return out


def se_over_classes(x: np.ndarray, cls: np.ndarray) -> float:
    by = pd.Series(x).groupby(cls).mean().dropna()
    return float(by.std(ddof=1) / np.sqrt(len(by))) if len(by) > 1 else float("nan")


def window_mean(f: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.nanmean(f[:, lo : hi + 1], axis=1)


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def gate(steps: pd.DataFrame, runs: dict[str, dict], beta: float, path: str) -> dict:
    """The same-head readings off arm ``c``'s rows: what the transfer moves, and what it would score."""
    s = steps[np.isfinite(steps["xfer_dlogit_med"].to_numpy(dtype=float))]
    trained = s[s["detector_tier"].astype(str) != "centroid"]
    cut = np.abs(_logit(trained["shadow_train_threshold"]) - _logit(trained["train_threshold"]))
    dk = trained["shadow_n_flagged"].to_numpy(dtype=float) - trained["n_flagged"].to_numpy(dtype=float)
    sep_a = trained["line_separation"].to_numpy(dtype=float) < SEPARATION_D
    sep_b = trained["shadow_separation"].to_numpy(dtype=float) < SEPARATION_D
    keys = sorted(runs)
    cls = np.array([runs[k]["cls"].split("@")[0] for k in keys])
    F = np.stack([runs[k]["f"] for k in keys])
    FS = np.stack([runs[k]["fs"] for k in keys])
    row: dict = {"path": path, "beta": beta, "runs": len(keys), "rows_with_folds": int(len(s))}
    row.update(
        dlogit_med=float(np.median(s["xfer_dlogit_med"])),
        dlogit_p90=float(np.median(s["xfer_dlogit_p90"])),
        dlogit_p90_p90=float(np.quantile(s["xfer_dlogit_p90"], 0.9)),
        clip_good_mean=float(np.nanmean(s["xfer_clip_good"])),
        dmu_pos_med=float(np.nanmedian(s["xfer_dmu_pos"])),
        dmu_neg_med=float(np.nanmedian(s["xfer_dmu_neg"])),
        dsigma_med=float(np.nanmedian(s["xfer_dsigma"])),
        cut_dlogit_med=float(np.nanmedian(cut)),
        cut_dlogit_p90=float(np.nanquantile(cut, 0.9)),
        cut_moved_share=float(np.mean(cut > 1e-9)),
        kept_diff_share=float(np.mean(dk != 0)),
        kept_dabs_med=float(np.median(np.abs(dk))),
        kept_d_mean=float(np.mean(dk)),
        weak_flip_share=float(np.mean(sep_a != sep_b)),
        head_count_share=float(np.mean(trained["line_head_count"] == 1)),
    )
    for lo, hi in WINDOWS:
        d = window_mean(FS, lo, hi) - window_mean(F, lo, hi)
        row[f"same_head_{lo}_{hi}"] = float(np.mean(d))
        row[f"same_head_se_{lo}_{hi}"] = se_over_classes(d, cls)
        row[f"f_{lo}_{hi}"] = float(np.mean(window_mean(F, lo, hi)))
    sh = FS - F
    same_curve = pd.DataFrame(
        {"path": path, "beta": beta, "t": GRID_T, "d": np.nanmean(sh, axis=0),
         "se": [se_over_classes(sh[:, t], cls) for t in GRID_T]}
    )  # fmt: skip
    bins = []
    for lo in range(1, T + 1, 10):
        b = s[(s["t"] >= lo) & (s["t"] < lo + 10)]
        bins.append(
            {"path": path, "beta": beta, "t_lo": lo, "t_mid": lo + 4.5, "rows": int(len(b)),
             "clip_good": float(np.nanmean(b["xfer_clip_good"])) if len(b) else np.nan,
             "dmu_pos": float(np.nanmedian(b["xfer_dmu_pos"])) if len(b) else np.nan,
             "dmu_neg": float(np.nanmedian(b["xfer_dmu_neg"])) if len(b) else np.nan}
        )  # fmt: skip
    by_t = []
    for lo, hi in ((1, 25), (26, 50), (51, 100), (101, 150)):
        b = s[(s["t"] >= lo) & (s["t"] <= hi)]
        bt = trained[(trained["t"] >= lo) & (trained["t"] <= hi)]
        bc = np.abs(_logit(bt["shadow_train_threshold"]) - _logit(bt["train_threshold"]))
        by_t.append(
            {
                "path": path,
                "beta": beta,
                "clicks": f"{lo}-{hi}",
                "rows": int(len(b)),
                "dlogit_med": float(np.median(b["xfer_dlogit_med"])) if len(b) else np.nan,
                "clip_good_mean": float(np.nanmean(b["xfer_clip_good"])) if len(b) else np.nan,
                "dmu_pos_med": float(np.nanmedian(b["xfer_dmu_pos"])) if len(b) else np.nan,
                "dmu_neg_med": float(np.nanmedian(b["xfer_dmu_neg"])) if len(b) else np.nan,
                "dsigma_med": float(np.nanmedian(b["xfer_dsigma"])) if len(b) else np.nan,
                "cut_dlogit_med": float(np.nanmedian(bc)) if len(bc) else np.nan,
                "kept_diff_share": float(np.mean(bt["shadow_n_flagged"] != bt["n_flagged"])) if len(bt) else np.nan,
                "dfbeta_mean": float(np.nanmean(bt["shadow_fbeta"] - bt["fbeta"])) if len(bt) else np.nan,
            }
        )
    return {"row": row, "by_t": pd.DataFrame(by_t), "same_curve": same_curve, "bins": pd.DataFrame(bins)}


def closed_loop(control: dict[str, dict], arm: dict[str, dict], name: str, beta: float, path: str) -> dict:
    keys = sorted(set(control) & set(arm))
    cls = np.array([control[k]["cls"].split("@")[0] for k in keys])
    C = np.stack([control[k]["f"] for k in keys])
    A = np.stack([arm[k]["f"] for k in keys])
    # The pairing check: an arm's opening picks are the control's (a prefix of the longer one: a run whose
    # opening ran longer only means it left the opening later, which the line cannot cause).
    checked = [k for k in keys if control[k]["opening"] and arm[k]["opening"]]

    def _prefix(a: str, b: str) -> bool:
        x, y = a.split(","), b.split(",")
        n = min(len(x), len(y))
        return x[:n] == y[:n]

    mism = sum(1 for k in checked if not _prefix(control[k]["opening"], arm[k]["opening"]))
    row = {"path": path, "beta": beta, "arm": name, "pairs": len(keys), "opening_checked": len(checked),
           "opening_mismatch": mism}  # fmt: skip
    for lo, hi in WINDOWS:
        d = window_mean(A, lo, hi) - window_mean(C, lo, hi)
        row[f"control_{lo}_{hi}"] = float(np.mean(window_mean(C, lo, hi)))
        row[f"arm_{lo}_{hi}"] = float(np.mean(window_mean(A, lo, hi)))
        row[f"d_{lo}_{hi}"] = float(np.mean(d))
        row[f"se_{lo}_{hi}"] = se_over_classes(d, cls)
    for t in (25, 50, 100, 150):
        d = A[:, t] - C[:, t]
        row[f"d_at_{t}"] = float(np.nanmean(d))
        row[f"se_at_{t}"] = se_over_classes(d, cls)
    D = A - C
    curve = pd.DataFrame(
        {"path": path, "beta": beta, "arm": name, "t": GRID_T, "control": np.nanmean(C, axis=0),
         "arm_f": np.nanmean(A, axis=0), "d": np.nanmean(D, axis=0),
         "se": [se_over_classes(D[:, t], cls) for t in GRID_T]}
    )  # fmt: skip
    return {"row": row, "curve": curve}


INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
#: The control in neutral gray; the two arms in the reference palette's first two categorical slots.
COLOR = {"c": "#8a8985", "q": "#eb6834", "qt": "#2a78d6"}
NAME = {"c": "today's line (raw)", "q": "quantile transfer", "qt": "quantile, tails kept"}


def _style(ax) -> None:
    ax.grid(color=GRIDC, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRIDC)


def _preset(beta: float) -> str:
    return "beta 1/4" if beta == 0.25 else f"beta {beta:g}"


def figure_curves(curves: pd.DataFrame, path: Path, title: str) -> None:
    """F-beta over clicks per arm (top) and each arm's paired gain over the control with +-1 SE (bottom), per preset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    betas = sorted(curves["beta"].unique())
    fig, axes = plt.subplots(2, len(betas), figsize=(4.2 * len(betas), 6.4), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1.3, 1]})  # fmt: skip
    for j, beta in enumerate(betas):
        cb = curves[curves["beta"] == beta]
        top, bot = axes[0, j], axes[1, j]
        first = cb[cb["arm"] == cb["arm"].iloc[0]]
        top.plot(first["t"], first["control"], color=COLOR["c"], lw=2, label=NAME["c"])
        for arm in ("q", "qt"):
            a = cb[cb["arm"] == arm]
            if not len(a):
                continue
            top.plot(a["t"], a["arm_f"], color=COLOR[arm], lw=2, label=NAME[arm])
            top.annotate(arm, (a["t"].iloc[-1], a["arm_f"].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                         color=MUTED, fontsize=9, va="center")  # fmt: skip
            bot.fill_between(a["t"], a["d"] - a["se"], a["d"] + a["se"], color=COLOR[arm], alpha=0.18, lw=0)
            bot.plot(a["t"], a["d"], color=COLOR[arm], lw=2)
            bot.annotate(arm, (a["t"].iloc[-1], a["d"].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                         color=MUTED, fontsize=9, va="center")  # fmt: skip
        top.annotate("c", (first["t"].iloc[-1], first["control"].iloc[-1]), xytext=(4, 0),
                     textcoords="offset points", color=MUTED, fontsize=9, va="center")  # fmt: skip
        bot.axhline(0, color=MUTED, lw=1)
        top.set_title(_preset(beta), color=INK, fontsize=11, loc="left")
        bot.set_xlabel("clicks", color=MUTED, fontsize=9)
        _style(top)
        _style(bot)
    axes[0, 0].set_ylabel("mean F-beta of the returned set", color=MUTED, fontsize=9)
    axes[1, 0].set_ylabel("paired gain over today's line", color=MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left", y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_mechanism(bins: pd.DataFrame, same: pd.DataFrame, path: Path, title: str) -> None:
    """The control's own heads: the share of held-out Goods above their fold's haystack, the Good mean's move, and
    what the transferred line would score on the same head, over clicks, one line per preset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    betas = sorted(bins["beta"].unique())
    shades = {0.25: "#9fc4ee", 1.0: "#2a78d6", 4.0: "#164a8a"}  # one hue, light to dark: the presets are ordered
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for beta in betas:
        b = bins[bins["beta"] == beta]
        c = same[same["beta"] == beta]
        col = shades.get(beta, "#2a78d6")
        axes[0].plot(b["t_mid"], b["clip_good"], color=col, lw=2, marker="o", ms=4, label=_preset(beta))
        axes[1].plot(b["t_mid"], b["dmu_pos"], color=col, lw=2, marker="o", ms=4, label=_preset(beta))
        axes[2].plot(c["t"], c["d"], color=col, lw=2, label=_preset(beta))
    axes[0].set_title("held-out Goods above their fold's haystack", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("share", color=MUTED, fontsize=9)
    axes[1].set_title("Good mean, transferred minus raw", color=INK, fontsize=10, loc="left")
    axes[1].set_ylabel("logit", color=MUTED, fontsize=9)
    axes[2].set_title("same head: quantile line minus today's", color=INK, fontsize=10, loc="left")
    axes[2].set_ylabel("F-beta", color=MUTED, fontsize=9)
    for ax in axes:
        ax.set_xlabel("clicks", color=MUTED, fontsize=9)
        _style(ax)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[2].axhline(0, color=MUTED, lw=1)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--date", default="2026-10-15")
    ap.add_argument("--tag", default="")
    ap.add_argument("--path", default="binary", choices=sorted(EMBEDDER))
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--betas", default="0.25,1,4")
    ap.add_argument("--arms", default="q,qt", help="closed-loop arms read against c, when their runs exist")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--docs", type=Path)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    emb = EMBEDDER[a.path]
    base = pd.read_csv(a.baseline)
    base = base[(base["embedder"] == emb) & (base["seed"] < a.seeds)]
    gates, by_t, loops, curves, counts, same, bins = [], [], [], [], {}, [], []
    for beta in (float(b) for b in a.betas.split(",")):
        tag = BTAG[beta]
        if f"text_line_precision_b{tag}" in base.columns:
            tq = {f"{c}|{int(s)}": float(fbeta([p], [r], beta)[0]) for c, s, p, r in
                  zip(base["category"], base["seed"], base[f"text_line_precision_b{tag}"], base[f"text_line_recall_b{tag}"])}  # fmt: skip
        else:
            tq = {}
        steps = load_arm(run_dir(a.root, a.date, a.tag, "c", a.path, beta), emb, a.seeds, a.procs)
        if not len(steps):
            print(f"beta {beta:g}: no control runs yet")
            continue
        control = per_run(steps, tq, beta)
        counts[f"c-b{tag}"] = len(control)
        g = gate(steps, control, beta, a.path)
        gates.append(g["row"])
        by_t.append(g["by_t"])
        same.append(g["same_curve"])
        bins.append(g["bins"])
        for arm in (x for x in a.arms.split(",") if x):
            st = load_arm(run_dir(a.root, a.date, a.tag, arm, a.path, beta), emb, a.seeds, a.procs)
            if not len(st):
                continue
            runs = per_run(st, tq, beta)
            counts[f"{arm}-b{tag}"] = len(runs)
            r = closed_loop(control, runs, arm, beta, a.path)
            loops.append(r["row"])
            curves.append(r["curve"])
    if not gates:
        return 1
    outs = [a.out] + ([a.docs] if a.docs else [])
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(gates).to_csv(o / f"gate_{a.path}.csv", index=False)
        pd.concat(by_t).to_csv(o / f"gate_by_click_{a.path}.csv", index=False)
        pd.concat(same).to_csv(o / f"same_head_curve_{a.path}.csv", index=False)
        pd.concat(bins).to_csv(o / f"saturation_{a.path}.csv", index=False)
        if loops:
            pd.DataFrame(loops).to_csv(o / f"closed_loop_{a.path}.csv", index=False)
            pd.concat(curves).to_csv(o / f"curves_{a.path}.csv", index=False)
        (o / f"counts_{a.path}.json").write_text(json.dumps(counts, indent=1) + "\n")
    figs = [a.out] + ([a.docs.parent / "figures"] if a.docs else [])
    label = "Binary Photo" if a.path == "binary" else "Region Photo"
    for o in figs:
        o.mkdir(parents=True, exist_ok=True)
        if curves:
            figure_curves(pd.concat(curves), o / f"transfer_over_clicks_{a.path}.png",
                          f"{label}: the labels line with the folds' scores transferred, against today's")  # fmt: skip
        figure_mechanism(pd.concat(bins), pd.concat(same), o / f"saturation_{a.path}.png",
                         f"{label}: the plain transfer fitted beside today's line, on today's sessions")  # fmt: skip
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(pd.DataFrame(gates).round(4).T.to_string())
        print(pd.concat(by_t).round(4).to_string(index=False))
        if loops:
            print(pd.DataFrame(loops).round(4).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
